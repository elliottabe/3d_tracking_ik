"""Fit per-fly STAC marker offsets ONCE, and let every bout reuse them.

Marker offsets are a per-INDIVIDUAL constant (like body scale): where this
fly's landmarks sit relative to its own skeleton. They are fit on a
stratified, non-consecutive sample of that fly's own high-confidence frames
(`tracking.preprocess.offsets.select_offsets_sample`) and then written to
`offsets_fly<f>.h5`; every later bout of the same fly loads that file rather
than refitting -- refitting per bout would make the marker model a function
of which bout happened to run first, the same class of defect that once let
one arbitrary bout's body scale poison a whole recording by 38x (this
pipeline's documented scale-from-one-bout defect).

The offset solve is a direct `optax` loop; `root_optimization` and
`pose_optimization` delegate their LM solves to `PerFrameSolver`, the same
class the per-bout pose fit uses.

**Units, in order -- the 38x failure's exact path** (`scaled_model_keypoints`
is the one place this happens): `kp3d_units` (0.1 mm world units) is
multiplied by the dimensionless body `scale` from `preprocess.scale` FIRST,
giving model units; then by `anatomy.mocap_scale_factor` (`1` for v1, not
necessarily so for every anatomy) SECOND. Reversing the order, or dropping
either factor, silently fits the offsets to a differently-scaled skeleton with
no error and no NaN.

**The parity gate.** `fit_offsets_from_flat`, re-run on the reference run's
OWN stored `kp_data`, reproduces `offsets_fly{0,1}.h5`'s `offsets` to
`max|Δ| < 5e-4` model units on both flies -- about twice the worst measured
floor (2.38e-4, on fly1; the male, not the female, is the noisy fly here, by
10x). Nothing here is or should be bit-exact: the reference stores float32 and
the solve runs on a GPU.

ONE `PerFrameSolver` is built per fit and threaded through every iteration.
The analyzed-problem cache keys on model STRUCTURE, not site positions, so the
`set_site_pos` copy each outer iteration produces still hits it: the root solve
(`T=1`, partial mask) and the pose solve (`T=n_frames`, full mask) compile once
each. Rebuilding per call costs a compile per iteration.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import h5py
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import optax

from tracking.conventions import plain_mapping
from tracking.inverse_kinematics.solver import (
    PerFrameSolver,
    SolverSettings,
    estimate_orientation_from_keypoints,
    forward,
    get_site_pos,
    get_site_xpos,
    make_qs,
    mjx_load,
    replace_qs,
    set_site_pos,
)
from tracking.io.names import Order, as_order, require_same

__all__ = [
    "OffsetsFit",
    "ensure_offsets",
    "fit_offsets",
    "fit_offsets_from_flat",
    "load_offsets",
    "m_loss",
    "offset_optimization",
    "optimise_offsets",
    "pose_optimization",
    "root_optimization",
    "scaled_model_keypoints",
    "write_offsets_h5",
]

_FREE = int(mujoco.mjtJoint.mjJNT_FREE)
_SLIDE = int(mujoco.mjtJoint.mjJNT_SLIDE)


def scaled_model_keypoints(kp3d_units, *, scale, mocap_scale_factor) -> np.ndarray:
    """`(T, K, 3)` world units -> `(T, K*3)` model units, BOTH multiplies, in
    order: `* scale` (dimensionless body scale) then `* mocap_scale_factor`
    (`anatomy.mocap_scale_factor`). NaN keypoints (a masked/ungapped frame)
    are preserved -- a plain multiply never turns a NaN into a number, so no
    special-casing is needed to keep it that way; do not `nan_to_num` here,
    which would silently fit a marker to the origin.
    """
    kp = np.asarray(kp3d_units, dtype=np.float64)
    t = kp.shape[0]
    flat = (kp * float(scale) * float(mocap_scale_factor)).reshape(t, -1)
    return flat


@dataclass(frozen=True)
class OffsetsFit:
    """The result of one offsets fit -- everything `write_offsets_h5` needs."""

    offsets: np.ndarray  # (K, 3)   model units, kp_order order
    qpos: np.ndarray  # (T, nq)  model units, from the FINAL pose optimization
    marker_sites: np.ndarray  # (T, K, 3) model units
    kp_data: np.ndarray  # (T, K*3) model units -- exactly what was fit
    kp_order: Order
    n_frames: int


def optimise_offsets(
    loss_fn,
    params0,
    *,
    max_iter: int,
    tol: float,
    optimizer: str = "sgd",
    learning_rate=None,
    frozen=None,
):
    """`optax.sgd(5e-4, momentum=0.9, nesterov=False)`, `jaxopt.OptaxSolver`'s
    exact stopping semantics: the loop's continue/stop test is evaluated on
    the PREVIOUS iteration's `error = ||grad||_2` (the pre-update gradient),
    seeded at `+inf` so the FIRST iteration always runs even when the start
    is already inside `tol` -- a solver that checked before stepping would
    return the starting point unchanged on that input, a difference invisible
    on any converging problem and wrong on this one. `tol` is jaxopt's own
    `OptaxSolver` default (1e-3) on the gradient 2-norm; it is NOT `FTOL`
    (0.005 in v1's config), which belongs to the (dropped) `ProjectedGradient`
    pose solver and is a different quantity (relative cost change) entirely.

    `optimizer="sgd"` is the default and is what every parity gate is
    calibrated against. `optimizer="adam"` exists
    because SGD's fixed step is not scale-invariant and `m_loss`'s gradient
    scales linearly with `M_REG_COEF`: measured, `M_REG_COEF` of 100 and 1000
    DIVERGE under SGD while 0, 1 and 10 converge. Adam normalises by the
    gradient's own running scale, so the usable range of `M_REG_COEF` no
    longer depends on the step size. Switching optimizer CHANGES THE FITTED
    OFFSETS and breaks parity -- opt-in for that reason, never a default.

    `frozen` is an optional boolean mask over `params0`; True components are
    held at their starting value. The UPDATE is masked, not the gradient: a
    masked gradient still moves a frozen parameter once momentum carries
    state from earlier steps, so zeroing the delta is the only form that
    actually guarantees no movement under every optimizer.

    Returns `(params, error)`: `params` the optimised point, `error` the
    gradient norm that satisfied (or last failed) the stopping test.

    Raises:
        FloatingPointError: if the optimisation diverged. The check is not
            decoration -- the stopping test is `error > tol`, and `NaN > tol`
            is **False**, so a diverged run EXITS THE LOOP AS IF CONVERGED and
            would otherwise return NaN offsets that `write_offsets_h5` commits
            to disk. Downstream those produce a pose solve that fails in a way
            that looks like a bad fit rather than a failed one. Measured: at
            `M_REG_COEF=100` this path returned all-NaN offsets in 98 s where a
            converging fit takes ~394 s, and nothing reported a problem.
    """
    if optimizer == "sgd":
        opt = optax.sgd(
            learning_rate=5e-4 if learning_rate is None else float(learning_rate),
            momentum=0.9,
            nesterov=False,
        )
    elif optimizer == "adam":
        opt = optax.adam(learning_rate=1e-3 if learning_rate is None else float(learning_rate))
    else:
        raise ValueError(f"optimizer must be 'sgd' or 'adam', got {optimizer!r}")

    free = None if frozen is None else jnp.asarray(~jnp.asarray(frozen, bool), params0.dtype)

    def _body(carry):
        i, params, state, _error = carry
        _value, grad = jax.value_and_grad(loss_fn)(params)
        error = jnp.linalg.norm(grad)
        delta, state = opt.update(grad, state, params)
        if free is not None:
            delta = delta * free
        return i + 1, params + delta, state, error

    def _cond(carry):
        i, _params, _state, error = carry
        return (i < max_iter) & (error > tol)

    # One fused device loop, not `max_iter` dispatched steps. The semantics are
    # unchanged -- `error` starts at inf so the first iteration always runs, and
    # the test is on the PRE-update gradient, exactly as jaxopt's `OptaxSolver`
    # does it -- but the whole loop now costs one host/device round trip instead
    # of up to 2000. The fit calls this once per outer iteration, so a Python
    # loop here was up to 12000 dispatches per fit, each one a launch for a few
    # microseconds of arithmetic.
    params0 = jnp.asarray(params0)
    params = params0
    n_done, params, _state, error = jax.lax.while_loop(
        _cond, _body, (0, params, opt.init(params), jnp.asarray(jnp.inf))
    )
    if not (bool(np.isfinite(np.asarray(params)).all()) and bool(np.isfinite(np.asarray(error)))):
        raise FloatingPointError(
            f"offsets optimisation DIVERGED after {int(n_done)} iterations "
            f"(optimizer={optimizer!r}, learning_rate="
            f"{learning_rate if learning_rate is not None else 'default'}, "
            f"final gradient norm {float(error)!r}).\n"
            f"This is not a slow fit -- it exited EARLY, because the stopping "
            f"test is `error > tol` and `NaN > tol` is False, so a diverged run "
            f"looks converged.\n"
            f"The usual cause is `M_REG_COEF` too large for a fixed SGD step: "
            f"`m_loss`'s gradient scales linearly with it, and 100 diverges "
            f"where 10 converges. Either lower `M_REG_COEF`, or pass "
            f"`optimizer='adam'`, which is scale-invariant."
        )
    return params, error


def m_loss(
    offsets, mjx_model, mjx_data, kp_data, q, initial_offsets, site_idxs, is_regularized, reg_coef
):
    """Summed marker residual over `q.shape[0]` sample frames, plus
    `reg_coef` times an L2 pull of each REGULARIZED site's offset back toward
    `initial_offsets`.
    """
    n = kp_data.shape[1]

    def f(carry, inp):
        qpos, kp = inp
        model_, data_, reg_term, residual = carry
        reg_term = reg_term + jnp.square(offsets - initial_offsets.flatten()) * is_regularized
        data_ = data_.replace(qpos=qpos)
        model_ = set_site_pos(model_, jnp.reshape(offsets, (-1, 3)), site_idxs)
        data_ = forward(model_, data_)
        markers = get_site_xpos(data_, site_idxs).flatten()
        residual = residual + jnp.square(kp - markers)
        return (model_, data_, reg_term, residual), None

    init = (mjx_model, mjx_data, jnp.zeros(n), jnp.zeros(n))
    (_model, _data, reg_term, residual), _ = jax.lax.scan(f, init, (q, kp_data))
    return jnp.sum(residual) + reg_coef * jnp.sum(reg_term)


def _root_dims(anatomy) -> int:
    """4 for a slide root, 7 (translation + quaternion) otherwise.

    A fixed root never reaches here -- the caller skips `root_optimization`
    entirely in that case."""
    jt = anatomy.mj_model.jnt_type
    if len(jt) and int(jt[0]) == _SLIDE:
        return 4
    return 7


def _root_free(anatomy) -> bool:
    """True for a free OR slide root."""
    jt = anatomy.mj_model.jnt_type
    if len(jt) == 0:
        return False
    return int(jt[0]) in (_FREE, _SLIDE)


def _orientation_indices(anatomy) -> tuple[int, int, int, int] | None:
    """`(rear, left, right, front)` KP_NAMES indices from
    `JAXLS_ORIENTATION_KEYPOINTS`, or `None` when that key is absent/empty.

    v1's OWN config sets this (`rear: Scutellum, left: WingL_base, right:
    WingR_base, front: Antenna_Base`), and the pose solve uses it: a per-frame
    orientation warm-start from these four trunk keypoints, applied whenever
    the root is a free joint. `front` maps to `-1` -- "no front
    keypoint" -- only if the `front` sub-key is itself absent, which for v1
    it is not.
    """
    model_cfg = anatomy.cfg.get("model", {}) if anatomy.cfg else {}
    spec = model_cfg.get("JAXLS_ORIENTATION_KEYPOINTS") or {}
    if not spec:
        return None
    rear = anatomy.kp_order.index(spec["rear"])
    left = anatomy.kp_order.index(spec["left"])
    right = anatomy.kp_order.index(spec["right"])
    front_name = spec.get("front")
    front = anatomy.kp_order.index(front_name) if front_name else -1
    return rear, left, right, front


def _root_and_trunk(anatomy) -> tuple[int, np.ndarray]:
    """`(root_kp_idx, trunk_kp_weights)`.

    `root_kp_idx` is `-1` when `ROOT_OPTIMIZATION_KEYPOINT` is absent.
    `trunk_kp_weights` is `(K*3,)`, `1.0` at every coordinate of a keypoint
    named in `TRUNK_OPTIMIZATION_KEYPOINTS`; an empty mapping (v1's shipped
    default) makes every entry `0.0`.
    """
    model_cfg = anatomy.cfg.get("model", {}) if anatomy.cfg else {}
    root_name = model_cfg.get("ROOT_OPTIMIZATION_KEYPOINT")
    root_kp_idx = anatomy.kp_order.index(root_name) if root_name else -1
    trunk_names = model_cfg.get("TRUNK_OPTIMIZATION_KEYPOINTS") or {}
    trunk_mask = np.array(
        [1.0 if n in trunk_names else 0.0 for n in anatomy.kp_order.names], dtype=np.float64
    )
    return root_kp_idx, np.repeat(trunk_mask, 3)


def _solver_settings_from_cfg(solver_cfg) -> SolverSettings:
    """`SolverSettings` for the OFFSETS-fit stage, from `solver_cfg`'s
    `JAXLS_*` keys -- a DIFFERENT tuning from `SolverSettings()`'s own
    defaults, which are the per-bout stage's.

    Measured against the reference `offsets_fly{0,1}.h5`'s own stored
    config: `JAXLS_LAMBDA_INITIAL=1.0`, `JAXLS_COST_TOLERANCE=1e-10`,
    `JAXLS_GRADIENT_TOLERANCE=1e-8`, `JAXLS_PARAMETER_TOLERANCE=1e-10`,
    `N_ITER_Q=500` -- all different from `SolverSettings()`'s `5e-4` /
    `1e-12` / `1e-14` / `1e-16` / `500`.

    `lambda_min` is NOT one of `solver_cfg`'s keys: this stage needs jaxls'
    own `TrustRegionConfig` default of `1e-5`, not `SolverSettings()`'s
    `1e-8`. `1e-8` reproduced the reference to only `1.292e-2` on fly0, 25x
    over the gate.
    """
    return SolverSettings(
        n_iter=int(solver_cfg.get("N_ITER_Q", 500)),
        lambda_initial=float(solver_cfg.get("JAXLS_LAMBDA_INITIAL", 1.0)),
        lambda_min=1e-5,
        cost_tolerance=float(solver_cfg.get("JAXLS_COST_TOLERANCE", 1e-10)),
        gradient_tolerance=float(solver_cfg.get("JAXLS_GRADIENT_TOLERANCE", 1e-8)),
        parameter_tolerance=float(solver_cfg.get("JAXLS_PARAMETER_TOLERANCE", 1e-10)),
        batch=SolverSettings().batch,
        linear_solver=str(solver_cfg.get("JAXLS_LINEAR_SOLVER", "auto")),
    )


def root_optimization(
    anatomy, mjx_model, mjx_data, kp_frame, *, root_kp_idx, kp_weights, settings=None, solver=None
):
    """Warm-start the root pose from ONE frame and a root-only mask.

    Translation always; orientation only insofar as the LM solve moves it.
    `kp_frame` is `(K*3,)`, always frame 0 of the sample. Two LM calls, both
    sharing the caller's `solver`; `q0[:3]` is deliberately not re-seeded
    between them, which the shared root-only mask makes immaterial.
    """
    kp_frame = np.asarray(kp_frame, dtype=np.float64)
    root_dims = _root_dims(anatomy)
    qs_to_opt = np.zeros(int(anatomy.nq), dtype=bool)
    qs_to_opt[:root_dims] = True
    kp_weights = np.asarray(kp_weights, dtype=np.float64)

    solver = PerFrameSolver(settings) if solver is None else solver
    for _ in range(2):  # "Root Optimization" then "Trunk only optimization"
        q0 = np.asarray(mjx_data.qpos, dtype=np.float64).copy()
        q0[:3] = kp_frame[root_kp_idx * 3 : root_kp_idx * 3 + 3]
        q_opt = solver.solve_frame(
            q0,
            mjx_model=mjx_model,
            mjx_data=mjx_data,
            kp_data=kp_frame,
            qs_to_opt=qs_to_opt,
            kp_weights=kp_weights,
            lb=anatomy.lb,
            ub=anatomy.ub,
            site_idxs=anatomy.site_idxs,
        )
        q_merged = make_qs(jnp.asarray(q0), jnp.asarray(qs_to_opt), jnp.asarray(q_opt))
        mjx_data = replace_qs(mjx_model, mjx_data, q_merged)
    return mjx_data


def _fk_batch(mjx_model, mjx_data, qpos_batch, site_idxs):
    def one(q):
        data = mjx_data.replace(qpos=q)
        data = forward(mjx_model, data)
        return get_site_xpos(data, site_idxs)

    return jax.vmap(one)(qpos_batch)


def pose_optimization(
    anatomy,
    mjx_model,
    mjx_data,
    kp_data,
    *,
    root_kp_idx,
    orientation_idx=None,
    settings=None,
    solver=None,
):
    """One independent per-frame LM solve over every frame of `kp_data`
    `(T, K*3)`, warm-started by tiling
    `mjx_data.qpos` and overriding, per frame: (1) the root TRANSLATION from
    `root_kp_idx`'s own keypoint (root free or slide); (2) the root
    ORIENTATION from `orientation_idx` (`(rear, left, right, front)` KP_NAMES
    indices, see `_orientation_indices`) via
    `estimate_orientation_from_keypoints`, ROOT-FREE ONLY (a slide root has no
    quaternion slot to warm-start). v1 sets both `ROOT_OPTIMIZATION_KEYPOINT`
    and `JAXLS_ORIENTATION_KEYPOINTS`; skipping (2) leaves every frame's warm
    start at the tiled default quaternion, a materially worse LM start that
    measurably moves the fitted offsets.

    The solve is unchunked: frames never couple here, so `JAXLS_CHUNK_SIZE`
    (a GPU-memory bound) is set to 0.

    `solver` is REUSED across the whole fit and should be passed by the
    caller. Site positions and the frozen qpos are traced inputs, so a
    `set_site_pos` between outer-loop calls is a data change, not a new
    program -- the compiled solve is valid across every iteration. Building a
    fresh solver here instead costs a full jaxls `.analyze()` and XLA compile
    per iteration: measured 55-107 s each against 0.33 s for a reuse, which
    was the whole of this fit's runtime. A `None` solver builds one, for
    callers that only run a single pass.

    Returns `(mjx_data, qpos (T,nq), marker_sites (T,K,3))`.
    """
    kp_data = np.asarray(kp_data, dtype=np.float64)
    t = int(kp_data.shape[0])
    q_base = np.asarray(mjx_data.qpos, dtype=np.float64)
    q_init = np.tile(q_base, (t, 1))
    if root_kp_idx >= 0 and _root_free(anatomy):
        q_init[:, :3] = kp_data[:, root_kp_idx * 3 : root_kp_idx * 3 + 3]
    if orientation_idx is not None and anatomy.has_freejoint:
        rear, left, right, front = orientation_idx
        quats = np.asarray(estimate_orientation_from_keypoints(kp_data, rear, left, right, front))
        q_init[:, 3:7] = quats
    qs_to_opt = np.ones(int(anatomy.nq), dtype=bool)

    solver = PerFrameSolver(settings) if solver is None else solver
    qpos = solver.solve(
        q_init=q_init,
        mjx_model=mjx_model,
        mjx_data=mjx_data,
        kp_data=kp_data,
        qs_to_opt=qs_to_opt,
        kp_weights=anatomy.kp_weights,
        lb=anatomy.lb,
        ub=anatomy.ub,
        site_idxs=anatomy.site_idxs,
    )
    mjx_data = replace_qs(mjx_model, mjx_data, jnp.asarray(qpos[-1]))
    marker_sites = np.asarray(_fk_batch(mjx_model, mjx_data, jnp.asarray(qpos), anatomy.site_idxs))
    return mjx_data, np.asarray(qpos, dtype=np.float64), marker_sites


def offset_optimization(
    anatomy,
    mjx_model,
    mjx_data,
    kp_data,
    offsets,
    q,
    *,
    n_sample_frames,
    m_reg_coef,
    n_iter_m,
    optimizer="sgd",
    learning_rate=None,
    frozen=None,
):
    """Sample `n_sample_frames` frames, then minimise `m_loss` over the
    flattened site offsets via `optimise_offsets`.

    The sample is reshuffled fresh every outer iteration from a fixed seed,
    not carried over. Ends with `replace_qs` -- kinematics only, NOT
    `forward`'s kinematics+com_pos; only `m_loss`'s internal FK needs the
    latter.
    """
    t = int(kp_data.shape[0])
    n_sample_frames = min(int(n_sample_frames), t)
    key = jax.random.PRNGKey(0)
    shuffled = jax.random.permutation(key, jnp.arange(t), independent=True)
    time_indices = shuffled[:n_sample_frames]

    site_idxs = jnp.asarray(anatomy.site_idxs)
    offset0 = jnp.asarray(get_site_pos(mjx_model, site_idxs)).flatten()
    keypoints = jnp.asarray(kp_data)[time_indices, :]
    q_sub = jnp.asarray(q)[time_indices]
    initial_offsets = jnp.asarray(offsets)
    is_regularized = jnp.asarray(anatomy.is_regularized)

    def loss(params):
        return m_loss(
            params,
            mjx_model,
            mjx_data,
            keypoints,
            q_sub,
            initial_offsets,
            site_idxs,
            is_regularized,
            float(m_reg_coef),
        )

    offset_opt_param, _err = optimise_offsets(
        loss,
        offset0,
        max_iter=int(n_iter_m),
        tol=1e-3,
        optimizer=optimizer,
        learning_rate=learning_rate,
        frozen=frozen,
    )
    mjx_model = set_site_pos(mjx_model, jnp.reshape(offset_opt_param, (-1, 3)), site_idxs)
    mjx_data = replace_qs(mjx_model, mjx_data, mjx_data.qpos)
    return mjx_model, mjx_data, np.asarray(offset_opt_param, dtype=np.float64)


def fit_offsets_from_flat(anatomy, kp_flat, *, solver_cfg) -> OffsetsFit:
    """`(T, K*3)` ALREADY in model units in; alternates root -> N_ITERS x
    (pose, offset) -> a final pose solve, exactly `Stac.fit_offsets`'s own
    sequence. `solver_cfg` is a plain mapping read for `N_ITERS`,
    `N_SAMPLE_FRAMES`, `N_ITER_M`, `M_REG_COEF`, and the pose/root LM
    solve's own `JAXLS_LAMBDA_INITIAL`/`JAXLS_{COST,GRADIENT,PARAMETER}_
    TOLERANCE`/`N_ITER_Q`/`JAXLS_LINEAR_SOLVER` (see
    `_solver_settings_from_cfg`) -- it may carry other keys (e.g. the
    reference run's own stored `USE_JAXLS`/`JAXLS_SMOOTH_WEIGHT`, both
    vestigial here) and they are ignored rather than rejected: this reads what
    it needs out of a larger config, it is not a complete schema.

    The parity gate (`test_fitted_offsets_reproduce_the_reference_run`)
    calls this function directly so it exercises the fit and nothing
    upstream of it.
    """
    solver_cfg = plain_mapping(solver_cfg, what="solver_cfg", api="fit_offsets_from_flat")
    n_iters = int(solver_cfg["N_ITERS"])
    n_sample_frames = int(solver_cfg["N_SAMPLE_FRAMES"])
    n_iter_m = int(solver_cfg["N_ITER_M"])
    m_reg_coef = float(solver_cfg["M_REG_COEF"])
    # `M_OPTIMIZER="adam"` makes the offsets step scale-invariant, which is
    # what lets M_REG_COEF go above ~10 without diverging.
    m_optimizer = str(solver_cfg.get("M_OPTIMIZER", "sgd"))
    m_lr = solver_cfg.get("M_LEARNING_RATE")
    settings = _solver_settings_from_cfg(solver_cfg)
    # ONE solver for the whole fit. Its analyzed-problem cache keys on shape
    # and model STRUCTURE, not on site positions, so the root solve (T=1,
    # partial mask) and the pose solve (T=n_frames, full mask) compile once
    # each and every later iteration reuses them. Rebuilding per call cost a
    # compile per iteration and was this fit's entire runtime.
    solver = PerFrameSolver(settings)

    kp_flat = np.asarray(kp_flat, dtype=np.float64)
    if kp_flat.ndim == 3:
        kp_flat = kp_flat.reshape(kp_flat.shape[0], -1)

    mjx_model, mjx_data = mjx_load(anatomy.mj_model)
    site_idxs = jnp.asarray(anatomy.site_idxs)
    offsets = np.asarray(get_site_pos(mjx_model, site_idxs), dtype=np.float64).copy()
    mjx_model = set_site_pos(mjx_model, jnp.asarray(offsets), site_idxs)
    mjx_data = forward(mjx_model, mjx_data)

    root_kp_idx, trunk_kp_weights = _root_and_trunk(anatomy)
    orientation_idx = _orientation_indices(anatomy)
    if root_kp_idx == -1:
        pass  # ROOT_OPTIMIZATION_KEYPOINT not specified: source skips too.
    elif _root_free(anatomy):
        mjx_data = root_optimization(
            anatomy,
            mjx_model,
            mjx_data,
            kp_flat[0],
            root_kp_idx=root_kp_idx,
            kp_weights=trunk_kp_weights,
            settings=settings,
            solver=solver,
        )
    # else: fixed root -- source skips root_optimization for this case too.

    qpos = np.asarray(mjx_data.qpos, dtype=np.float64)[None, :]
    marker_sites = np.zeros((1, len(anatomy.kp_order), 3))
    for _ in range(n_iters):
        mjx_data, qpos, marker_sites = pose_optimization(
            anatomy,
            mjx_model,
            mjx_data,
            kp_flat,
            root_kp_idx=root_kp_idx,
            orientation_idx=orientation_idx,
            settings=settings,
            solver=solver,
        )
        mjx_model, mjx_data, offsets = offset_optimization(
            anatomy,
            mjx_model,
            mjx_data,
            kp_flat,
            offsets,
            qpos,
            n_sample_frames=n_sample_frames,
            m_reg_coef=m_reg_coef,
            n_iter_m=n_iter_m,
            optimizer=m_optimizer,
            learning_rate=m_lr,
            frozen=anatomy.is_frozen,
        )

    mjx_data, qpos, marker_sites = pose_optimization(
        anatomy,
        mjx_model,
        mjx_data,
        kp_flat,
        root_kp_idx=root_kp_idx,
        orientation_idx=orientation_idx,
        settings=settings,
        solver=solver,
    )

    return OffsetsFit(
        offsets=np.asarray(offsets, dtype=np.float64).reshape(-1, 3),
        qpos=qpos,
        marker_sites=marker_sites,
        kp_data=kp_flat,
        kp_order=anatomy.kp_order,
        n_frames=int(kp_flat.shape[0]),
    )


def fit_offsets(anatomy, kp3d_units, *, scale, solver_cfg) -> OffsetsFit:
    """`(T, K, 3)` world units in; scales via `scaled_model_keypoints`, then
    delegates to `fit_offsets_from_flat`."""
    kp_flat = scaled_model_keypoints(
        kp3d_units, scale=scale, mocap_scale_factor=anatomy.mocap_scale_factor
    )
    return fit_offsets_from_flat(anatomy, kp_flat, solver_cfg=solver_cfg)


def write_offsets_h5(path, fit: OffsetsFit, *, anatomy, cfg) -> None:
    """Write `offsets`/`kp_names` (what `load_offsets` and
    `solve_per_frame_ik` read) plus `kp_data`/`marker_sites`/`qpos`/
    `names_qpos`/`config` for provenance. `xpos`/`xquat`/`names_xpos`/`qvel`
    are deliberately not written -- nothing reads them back.
    """
    cfg = plain_mapping(cfg, what="cfg", api="write_offsets_h5")
    names_qpos = tuple(getattr(anatomy, "names_qpos", ()) or ())
    with h5py.File(str(path), "w") as f:
        f.create_dataset("offsets", data=np.asarray(fit.offsets, dtype=np.float32))
        f.create_dataset("kp_names", data=np.array(fit.kp_order.names, dtype="S"))
        f.create_dataset(
            "kp_data", data=np.asarray(fit.kp_data, dtype=np.float32), compression="gzip"
        )
        f.create_dataset(
            "marker_sites",
            data=np.asarray(fit.marker_sites, dtype=np.float32),
            compression="gzip",
        )
        f.create_dataset("qpos", data=np.asarray(fit.qpos, dtype=np.float32), compression="gzip")
        if names_qpos:
            f.create_dataset("names_qpos", data=np.array(names_qpos, dtype="S"))
        f.create_dataset("config", data=np.bytes_(json.dumps(dict(cfg), indent=2, default=str)))


def load_offsets(path, *, kp_order) -> np.ndarray:
    """`(K, 3)` offsets from `path`, refusing a `kp_names` mismatch against
    `kp_order`. `solve_per_frame_ik` checks the same thing; keeping it here
    too means a reordered offsets file fails loudly instead of fitting every
    marker to the wrong landmark with no NaN and no residual spike.
    """
    order = as_order(kp_order)
    with h5py.File(str(path), "r") as f:
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in f["kp_names"][:]]
        offsets = np.asarray(f["offsets"][:], dtype=np.float64)
    require_same(Order(names), order, what="offsets file kp_names")
    return offsets.reshape(-1, 3)


def ensure_offsets(path, anatomy, kp3d_units, *, scale, solver_cfg, cfg=None) -> np.ndarray:
    """`(K, 3)` offsets for one fly: load `path` if it already exists, else
    fit once (`fit_offsets`) and write it. This is the primary guarantee's
    enforcement point -- every bout of a fly calls this with the SAME
    `path`, so only the FIRST call ever fits; every later call, including
    from a different bout or a different process, loads the file the first
    call wrote rather than refitting.
    """
    path = str(path)
    if os.path.exists(path):
        return load_offsets(path, kp_order=anatomy.kp_order)
    fit = fit_offsets(anatomy, kp3d_units, scale=scale, solver_cfg=solver_cfg)
    write_offsets_h5(path, fit, anatomy=anatomy, cfg=cfg if cfg is not None else solver_cfg)
    return fit.offsets
