"""Per-frame jaxls (Levenberg-Marquardt) IK solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import numpy as np
from mujoco import mjx

__all__ = [
    "FREE_JOINT_NDOF",
    "SolverSettings",
    "PerFrameSolver",
    "mjx_load",
    "set_site_pos",
    "get_site_pos",
    "get_site_xpos",
    "make_qs",
    "replace_qs",
    "forward",
    "estimate_orientation_from_keypoints",
]

# Number of free-joint DOFs in MuJoCo qpos (3 translation + 4 quaternion).
FREE_JOINT_NDOF = 7


def mjx_load(mj_model) -> tuple[Any, Any]:
    """Load a compiled `mujoco.MjModel` into `mjx`. `(mjx_model, mjx_data)`."""
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.make_data(mjx_model)
    return mjx_model, mjx_data


def set_site_pos(mjx_model, offsets, site_idxs):
    """Return `mjx_model` with `site_pos[site_idxs]` set to `offsets`."""
    new_site_pos = mjx_model.site_pos.at[site_idxs].set(offsets)
    return mjx_model.replace(site_pos=new_site_pos)


def get_site_pos(mjx_model, site_idxs):
    """`mjx_model.site_pos[site_idxs]` -- local offset relative to body."""
    return mjx_model.site_pos[site_idxs]


def get_site_xpos(mjx_data, site_idxs):
    """`mjx_data.site_xpos[site_idxs]` -- world-frame site positions."""
    return mjx_data.site_xpos[site_idxs]


def make_qs(q0, qs_to_opt, q):
    """Combine `q0`/`q` by `qs_to_opt`: optimised slots take `q`, rest `q0`."""
    return jnp.copy((1 - qs_to_opt) * q0 + qs_to_opt * jnp.copy(q))


@jax.jit
def _kinematics(mjx_model, mjx_data):
    return mjx.kinematics(mjx_model, mjx_data)


@jax.jit
def _com_pos(mjx_model, mjx_data):
    return mjx.com_pos(mjx_model, mjx_data)


def forward(mjx_model, mjx_data):
    """`mjx.kinematics` then `mjx.com_pos` -- the FK step the marker cost
    needs (site world positions AND subtree center-of-mass)."""
    mjx_data = _kinematics(mjx_model, mjx_data)
    return _com_pos(mjx_model, mjx_data)


def replace_qs(mjx_model, mjx_data, q):
    """Set `mjx_data.qpos = q` and re-run kinematics -- NOT `com_pos`."""
    if q is None:
        print("optimization failed, continuing")
    else:
        mjx_data = mjx_data.replace(qpos=q)
        mjx_data = _kinematics(mjx_model, mjx_data)
    return mjx_data


def _quat_from_frame(rear, left, right, front, has_front):
    """Build a MuJoCo `[w,x,y,z]` quaternion from 3-4 trunk keypoints."""
    lat = left - right
    lat_norm = jnp.linalg.norm(lat)
    lat = jnp.where(lat_norm > 1e-12, lat / lat_norm, jnp.array([0.0, 1.0, 0.0]))

    mid = (left + right) * 0.5
    fwd_raw = jnp.where(has_front, front - rear, mid - rear)
    fwd_raw_norm = jnp.linalg.norm(fwd_raw)
    fwd_raw = jnp.where(fwd_raw_norm > 1e-12, fwd_raw / fwd_raw_norm, jnp.array([1.0, 0.0, 0.0]))

    up = jnp.cross(fwd_raw, lat)
    up_norm = jnp.linalg.norm(up)
    up = jnp.where(up_norm > 1e-12, up / up_norm, jnp.array([0.0, 0.0, 1.0]))
    fwd = jnp.cross(lat, up)
    fwd_norm = jnp.linalg.norm(fwd)
    fwd = jnp.where(fwd_norm > 1e-12, fwd / fwd_norm, jnp.array([1.0, 0.0, 0.0]))

    r = jnp.stack([fwd, lat, up], axis=-1)  # (3, 3): columns = body axes in world frame

    trace = r[0, 0] + r[1, 1] + r[2, 2]
    s0 = jnp.sqrt(jnp.maximum(trace + 1.0, 0.0)) * 2.0
    s1 = jnp.sqrt(jnp.maximum(1.0 + r[0, 0] - r[1, 1] - r[2, 2], 0.0)) * 2.0
    s2 = jnp.sqrt(jnp.maximum(1.0 + r[1, 1] - r[0, 0] - r[2, 2], 0.0)) * 2.0
    s3 = jnp.sqrt(jnp.maximum(1.0 + r[2, 2] - r[0, 0] - r[1, 1], 0.0)) * 2.0

    q0 = jnp.array(
        [s0 / 4.0, (r[2, 1] - r[1, 2]) / s0, (r[0, 2] - r[2, 0]) / s0, (r[1, 0] - r[0, 1]) / s0]
    )
    q1 = jnp.array(
        [(r[2, 1] - r[1, 2]) / s1, s1 / 4.0, (r[0, 1] + r[1, 0]) / s1, (r[0, 2] + r[2, 0]) / s1]
    )
    q2 = jnp.array(
        [(r[0, 2] - r[2, 0]) / s2, (r[0, 1] + r[1, 0]) / s2, s2 / 4.0, (r[1, 2] + r[2, 1]) / s2]
    )
    q3 = jnp.array(
        [(r[1, 0] - r[0, 1]) / s3, (r[0, 2] + r[2, 0]) / s3, (r[1, 2] + r[2, 1]) / s3, s3 / 4.0]
    )

    diag = jnp.array([trace, r[0, 0], r[1, 1], r[2, 2]])
    best = jnp.argmax(diag)
    q = jnp.where(best == 0, q0, jnp.where(best == 1, q1, jnp.where(best == 2, q2, q3)))

    q = q / jnp.linalg.norm(q)
    return jnp.where(q[0] < 0, -q, q)


def estimate_orientation_from_keypoints(kp_flat, rear, left, right, front) -> np.ndarray:
    """Per-frame root orientation from 3-4 trunk keypoints. `(T, 4)` wxyz."""
    kp_flat = jnp.asarray(kp_flat)
    t = kp_flat.shape[0]

    rear_xyz = kp_flat[:, rear * 3 : rear * 3 + 3]
    left_xyz = kp_flat[:, left * 3 : left * 3 + 3]
    right_xyz = kp_flat[:, right * 3 : right * 3 + 3]

    if front >= 0:
        front_xyz = kp_flat[:, front * 3 : front * 3 + 3]
        has_front = jnp.array(True)
    else:
        front_xyz = jnp.zeros((t, 3))
        has_front = jnp.array(False)

    quats = jax.vmap(_quat_from_frame)(
        rear_xyz, left_xyz, right_xyz, front_xyz, jnp.broadcast_to(has_front, (t,))
    )

    def _consistent_sign(prev, q):
        q = jnp.where(jnp.dot(prev, q) < 0, -q, q)
        return q, q

    _, quats = jax.lax.scan(_consistent_sign, quats[0], quats)
    return np.asarray(quats)


@dataclass(frozen=True)
class SolverSettings:
    """LM termination and linear-solver knobs."""

    n_iter: int = 500
    lambda_initial: float = 5e-4
    lambda_min: float = 1e-8
    cost_tolerance: float = 1e-12
    gradient_tolerance: float = 1e-14
    parameter_tolerance: float = 1e-16
    batch: int = 512
    linear_solver: str = "auto"
    penetration_weight: float = 0.0
    """Weight on an ANALYTIC self-intersection penalty. 0.0 (default) leaves
    the cost exactly as it was.

    Above zero, `solve(..., penetration=...)` must also be given a spec from
    `penetration_spec()`, and the cost gains a residual
    `sqrt(w) * relu(-signed_distance)` over sampled surface points.

    It is analytic ON PURPOSE. Adding `mjx.collision` to the cost instead was
    tried and reverted: it reports overlap correctly but its gradient w.r.t.
    qpos is NaN, so the solver returned its input unchanged at 9x the cost.
    Geom POSES come from
    `mjx.kinematics` and differentiate cleanly, so computing the separation
    from those keeps the gradient: measured finite, max 1.97e-3, non-zero on
    18 of 93 DOFs, and strongest on wing roll and pitch -- the two this
    project independently measured as driving the overlap (r=0.80 / 0.75)."""


class _AnalyzedProblem(NamedTuple):
    """One analyzed jaxls problem plus the `jaxls.Var` subclasses used to
    build and read it back."""

    analyzed: Any
    KpVar: type
    SE3Var: type
    JointVar: type
    OffsetVar: type
    FrozenVar: type


def _mask_key(v) -> tuple[bool, ...]:
    return tuple(bool(x) for x in np.asarray(v).ravel())


def _hashable(v) -> tuple[float, ...]:
    return tuple(round(float(x), 6) for x in np.asarray(v).ravel())


def _model_structure_key(mjx_model) -> tuple:
    """Identity of the model parts still baked into the jaxpr."""
    return (
        int(mjx_model.nq),
        int(mjx_model.nbody),
        int(mjx_model.nsite),
        id(mjx_model.body_parentid),
        id(mjx_model.jnt_type),
        id(mjx_model.body_pos),
        id(mjx_model.jnt_axis),
    )


def penetration_spec(mj_model, pairs, *, dofs=None, n_u: int = 5, n_v: int = 9) -> dict:
    """Precompute what the analytic penetration term needs, once per model."""
    import mujoco as _mj

    u = np.linspace(-1.0, 1.0, n_u)
    v = np.linspace(-1.0, 1.0, n_v)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    keep = (uu**2 + vv**2) <= 1.0
    uu, vv = uu[keep], vv[keep]

    blades, targets, locals_, sizes, types = [], [], [], [], []
    for blade_name, target_name in pairs:
        bg = _mj.mj_name2id(mj_model, _mj.mjtObj.mjOBJ_GEOM, blade_name)
        tg = _mj.mj_name2id(mj_model, _mj.mjtObj.mjOBJ_GEOM, target_name)
        if bg < 0 or tg < 0:
            raise ValueError(f"unknown geom in pair {(blade_name, target_name)}")
        ttype = int(mj_model.geom_type[tg])
        if ttype not in (int(_mj.mjtGeom.mjGEOM_CYLINDER), int(_mj.mjtGeom.mjGEOM_SPHERE)):
            raise ValueError(
                f"{target_name} is geom type {ttype}; the analytic term handles "
                f"cylinders and spheres only"
            )
        bs = mj_model.geom_size[bg]
        blades.append(bg)
        targets.append(tg)
        locals_.append(np.stack([np.zeros_like(uu), bs[1] * uu, bs[2] * vv], axis=1))
        sizes.append(np.asarray(mj_model.geom_size[tg]))
        types.append(ttype)
    return {
        "dofs": None if dofs is None else np.asarray(dofs, bool),
        "blade": np.asarray(blades, np.int32),
        "target": np.asarray(targets, np.int32),
        "points": np.asarray(locals_),  # (P, S, 3)
        "size": np.asarray(sizes),  # (P, 3)
        "is_sphere": np.asarray([t == int(_mj.mjtGeom.mjGEOM_SPHERE) for t in types]),
    }


class PerFrameSolver:
    """Per-frame LM IK against one model structure."""

    def __init__(self, settings: SolverSettings | None = None) -> None:
        settings = settings if settings is not None else SolverSettings()
        self.settings = settings
        self._cache: dict[tuple, _AnalyzedProblem] = {}
        self._independent_fns: dict[tuple, Any] = {}
        self._model_id: tuple | None = None
        self._model_ref: Any = None  # keeps the model alive so id()s in the key can't be reused
        self._last_iterations: np.ndarray | None = None

    @property
    def last_iterations(self) -> np.ndarray:
        """`(T,)` int32, LM iterations per frame from the last `solve()`."""
        return self._last_iterations

    def _check_model_identity(self, mjx_model) -> None:
        """Raise if this instance is reused across a different model structure."""
        key = _model_structure_key(mjx_model)
        if self._model_id is None:
            self._model_id = key
            self._model_ref = mjx_model
            return
        if key != self._model_id:
            raise ValueError(
                "PerFrameSolver.solve() called with a different model than this "
                "instance's first solve -- the analyzed-problem cache keys on "
                "model STRUCTURE (body tree, joint axes, body offsets), not site "
                "positions, so reusing a solver across a genuinely different "
                "model would silently run FK against the wrong one. Build a new "
                "PerFrameSolver per model."
            )

    def _pick_linear_solver(self, T: int, tangent_dim: int) -> str:
        """`self.settings.linear_solver`, or `"dense_cholesky"` for `"auto"`."""
        if self.settings.linear_solver != "auto":
            return self.settings.linear_solver
        return "dense_cholesky"

    def _build_se3(
        self,
        T: int,
        nq: int,
        n_kp_dim: int,
        mjx_model,
        mjx_data,
        qs_to_opt: jnp.ndarray,
        kp_weights: jnp.ndarray,
        lb: jnp.ndarray,
        ub: jnp.ndarray,
        site_idxs: jnp.ndarray,
        pen=None,
    ) -> _AnalyzedProblem:
        """Marker-tracking + hinge box-limit costs only: no regularizer and
        no smoothness term.
        """
        n_hinges = nq - FREE_JOINT_NDOF
        dummy_joints = jnp.zeros((n_hinges,))
        dummy_kp = jnp.zeros((n_kp_dim,))
        n_site_dim = int(np.asarray(site_idxs).size) * 3
        dummy_offsets = jnp.zeros((n_site_dim,))
        dummy_frozen = jnp.zeros((nq,))

        class SE3Var(
            jaxls.Var[jaxlie.SE3],
            default_factory=jaxlie.SE3.identity,
            retract_fn=jaxlie.manifold.rplus,
            tangent_dim=6,
        ): ...

        class JointVar(jaxls.Var[jnp.ndarray], default_factory=lambda: dummy_joints): ...

        class KpVar(jaxls.Var[jnp.ndarray], default_factory=lambda: dummy_kp): ...

        class OffsetVar(jaxls.Var[jnp.ndarray], default_factory=lambda: dummy_offsets): ...

        class FrozenVar(jaxls.Var[jnp.ndarray], default_factory=lambda: dummy_frozen): ...

        root_all = SE3Var(jnp.arange(T))
        joint_all = JointVar(jnp.arange(T))
        kp_all = KpVar(jnp.arange(T))
        offset_all = OffsetVar(jnp.arange(T))
        frozen_all = FrozenVar(jnp.arange(T))

        @jaxls.Cost.factory
        def marker_cost(
            var_values: jaxls.VarValues,
            root_var: SE3Var,
            joint_var: JointVar,
            kp_var: KpVar,
            offset_var: OffsetVar,
            frozen_var: FrozenVar,
        ) -> jnp.ndarray:
            t_root = var_values[root_var]
            joints = var_values[joint_var]
            kp = jax.lax.stop_gradient(var_values[kp_var])
            offsets = jax.lax.stop_gradient(var_values[offset_var]).reshape(-1, 3)
            frozen_q = jax.lax.stop_gradient(var_values[frozen_var])

            xyz = t_root.translation()
            wxyz = t_root.rotation().wxyz
            q = jnp.concatenate([xyz, wxyz, joints])

            full_q = jnp.where(qs_to_opt, q, frozen_q)
            data = mjx_data.replace(qpos=full_q)
            model = mjx_model.replace(site_pos=mjx_model.site_pos.at[site_idxs].set(offsets))
            data = forward(model, data)
            markers = get_site_xpos(data, site_idxs).flatten()
            finite = jnp.isfinite(kp)
            kp_clean = jnp.where(finite, kp, 0.0)
            return (kp_clean - markers) * kp_weights * finite

        costs: list[jaxls.Cost] = [marker_cost(root_all, joint_all, kp_all, offset_all, frozen_all)]

        if self.settings.penetration_weight > 0.0 and pen is not None:
            pen_w = float(self.settings.penetration_weight) ** 0.5
            blade_i = jnp.asarray(pen["blade"])
            target_i = jnp.asarray(pen["target"])
            pts_l = jnp.asarray(pen["points"])
            tsize = jnp.asarray(pen["size"])
            is_sph = jnp.asarray(pen["is_sphere"])
            pen_dofs = None if pen.get("dofs") is None else jnp.asarray(pen["dofs"])

            @jaxls.Cost.factory
            def penetration_cost(
                var_values: jaxls.VarValues,
                root_var: SE3Var,
                joint_var: JointVar,
                offset_var: OffsetVar,
                frozen_var: FrozenVar,
            ) -> jnp.ndarray:
                t_root = var_values[root_var]
                joints = var_values[joint_var]
                offsets = jax.lax.stop_gradient(var_values[offset_var]).reshape(-1, 3)
                frozen_q = jax.lax.stop_gradient(var_values[frozen_var])
                q = jnp.concatenate([t_root.translation(), t_root.rotation().wxyz, joints])
                full_q = jnp.where(qs_to_opt, q, frozen_q)
                if pen_dofs is not None:
                    full_q = jnp.where(pen_dofs, full_q, jax.lax.stop_gradient(full_q))
                model = mjx_model.replace(site_pos=mjx_model.site_pos.at[site_idxs].set(offsets))
                data = forward(model, mjx_data.replace(qpos=full_q))

                def one(bi, ti, pl, sz, sph):
                    Rb = data.geom_xmat[bi].reshape(3, 3)
                    world = data.geom_xpos[bi] + pl @ Rb.T
                    Rt = data.geom_xmat[ti].reshape(3, 3)
                    loc = (world - data.geom_xpos[ti]) @ Rt
                    radial = jnp.sqrt(jnp.sum(loc[:, :2] ** 2, axis=1) + 1e-12) - sz[0]
                    axial = jnp.abs(loc[:, 2]) - sz[1]
                    cyl = jnp.maximum(radial, axial)
                    sphere = jnp.sqrt(jnp.sum(loc**2, axis=1) + 1e-12) - sz[0]
                    return jnp.where(sph, sphere, cyl)

                sd = jax.vmap(one)(blade_i, target_i, pts_l, tsize, is_sph)
                return pen_w * jax.nn.relu(-sd).reshape(-1)

            costs.append(penetration_cost(root_all, joint_all, offset_all, frozen_all))

        hinge_lb = lb[FREE_JOINT_NDOF:]
        hinge_ub = ub[FREE_JOINT_NDOF:]

        @jaxls.Cost.factory(kind="constraint_leq_zero")
        def limit_cost(var_values: jaxls.VarValues, joint_var: JointVar) -> jnp.ndarray:
            j = var_values[joint_var]
            return jnp.concatenate([hinge_lb - j, j - hinge_ub])

        costs.append(limit_cost(joint_all))

        variables = [root_all, joint_all, kp_all, offset_all, frozen_all]
        analyzed = jaxls.LeastSquaresProblem(costs=costs, variables=variables).analyze()
        return _AnalyzedProblem(
            analyzed=analyzed,
            KpVar=KpVar,
            SE3Var=SE3Var,
            JointVar=JointVar,
            OffsetVar=OffsetVar,
            FrozenVar=FrozenVar,
        )

    def _get_analyzed(
        self,
        T: int,
        mjx_model,
        mjx_data,
        kp_data: jnp.ndarray,
        qs_to_opt: jnp.ndarray,
        kp_weights: jnp.ndarray,
        lb: jnp.ndarray,
        ub: jnp.ndarray,
        site_idxs: jnp.ndarray,
        pen=None,
    ) -> _AnalyzedProblem:
        """Cache key: `(T, nq, n_kp_dim, _model_structure_key(mjx_model), ...)`."""
        self._check_model_identity(mjx_model)
        nq = int(mjx_model.nq)
        pen_key = (
            None
            if pen is None
            else (
                tuple(int(b) for b in pen["blade"]),
                tuple(int(t) for t in pen["target"]),
                int(np.asarray(pen["points"]).shape[1]),
            )
        )
        n_kp_dim = int(kp_data.shape[-1]) if kp_data.ndim > 1 else int(kp_data.shape[0])

        key = (
            T,
            nq,
            n_kp_dim,
            _model_structure_key(mjx_model),
            _mask_key(qs_to_opt),
            _hashable(kp_weights),
            _hashable(lb),
            _hashable(ub),
            tuple(int(x) for x in np.asarray(site_idxs).ravel()),
            pen_key,
            float(self.settings.penetration_weight),
        )
        if key in self._cache:
            return self._cache[key]
        self._cache[key] = self._build_se3(
            T=T,
            nq=nq,
            n_kp_dim=n_kp_dim,
            mjx_model=mjx_model,
            mjx_data=mjx_data,
            qs_to_opt=qs_to_opt,
            kp_weights=kp_weights,
            lb=lb,
            ub=ub,
            site_idxs=site_idxs,
            pen=pen,
        )
        return self._cache[key]

    def _trust_region(self) -> jaxls.TrustRegionConfig:
        return jaxls.TrustRegionConfig(
            lambda_initial=self.settings.lambda_initial,
            lambda_min=self.settings.lambda_min,
        )

    def _termination(self) -> jaxls.TerminationConfig:
        return jaxls.TerminationConfig(
            max_iterations=self.settings.n_iter,
            cost_tolerance=self.settings.cost_tolerance,
            gradient_tolerance=self.settings.gradient_tolerance,
            parameter_tolerance=self.settings.parameter_tolerance,
        )

    def _single_frame_solver(self, prob: _AnalyzedProblem):
        """`(q_row (nq,), kp_row (n_kp_dim,), offsets (n_site_dim,), frozen_q
        (nq,)) -> (q_opt (nq,), n_lm_iterations)` for the `T=1` problem. SE3
        `offsets`/`frozen_q` are vmapped with
        `in_axes=None` in `_solve_independent` (broadcast, not batched) since
        they are the SAME site offsets and frozen slots for every frame in a
        `solve()` call."""
        kp_var_cls = prob.KpVar
        se3_var_cls, joint_var_cls = prob.SE3Var, prob.JointVar
        offset_var_cls, frozen_var_cls = prob.OffsetVar, prob.FrozenVar
        trust = self._trust_region()
        term = self._termination()
        one = jnp.arange(1)

        def solve_one(q_row, kp_row, offsets, frozen_q):
            wxyz = q_row[3:7]
            qn = jnp.linalg.norm(wxyz)
            wxyz = wxyz / jnp.where(qn > 0, qn, 1.0)
            root = jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(wxyz=wxyz[None]), q_row[None, :3]
            )
            sol, summ = prob.analyzed.solve(
                verbose=False,
                linear_solver="dense_cholesky",
                trust_region=trust,
                termination=term,
                initial_vals=jaxls.VarValues.make(
                    [
                        se3_var_cls(one).with_value(root),
                        joint_var_cls(one).with_value(q_row[None, FREE_JOINT_NDOF:]),
                        kp_var_cls(one).with_value(kp_row[None]),
                        offset_var_cls(one).with_value(offsets[None]),
                        frozen_var_cls(one).with_value(frozen_q[None]),
                    ]
                ),
                return_summary=True,
            )
            r = sol[se3_var_cls(one)]
            j = sol[joint_var_cls(one)]
            q_opt = jnp.concatenate([r.translation()[0], r.rotation().wxyz[0], j[0]])
            return q_opt, summ.iterations

        return solve_one

    def _solve_independent(
        self,
        prob: _AnalyzedProblem,
        q_init: jnp.ndarray,
        kp_data: jnp.ndarray,
        offsets: jnp.ndarray,
        frozen_q: jnp.ndarray,
    ) -> jnp.ndarray:
        """Solve `T` frames as `T` independent single-frame problems, vmapped
        in batches of `settings.batch` (ragged tail padded by repeating its
        last frame, then trimmed). Sets `last_iterations`. `offsets`/
        `frozen_q` is `(T, nq)` and BATCHED like `q_init`: a second pass
        refining a few DOFs needs each frame frozen at ITS OWN pose. Freezing
        every frame at one template and fitting the free DOFs to compensate
        made a wings-only refinement 77x WORSE than its own starting point --
        measured, and the reason this is per-frame. `offsets` stays broadcast."""
        T = int(q_init.shape[0])
        B = max(1, min(self.settings.batch, T))
        key = (id(prob), B)
        if key not in self._independent_fns:
            self._independent_fns[key] = jax.jit(
                jax.vmap(self._single_frame_solver(prob), in_axes=(0, 0, None, 0))
            )
        solve_batch = self._independent_fns[key]

        out, its = [], []
        for s in range(0, T, B):
            qi, ki = q_init[s : s + B], kp_data[s : s + B]
            fi = frozen_q[s : s + B]
            n = int(qi.shape[0])
            if n < B:
                qi = jnp.concatenate([qi, jnp.repeat(qi[-1:], B - n, axis=0)])
                ki = jnp.concatenate([ki, jnp.repeat(ki[-1:], B - n, axis=0)])
                fi = jnp.concatenate([fi, jnp.repeat(fi[-1:], B - n, axis=0)])
            q_b, it_b = solve_batch(qi, ki, offsets, fi)
            out.append(q_b[:n])
            its.append(it_b[:n])
        self._last_iterations = np.asarray(jnp.concatenate(its, axis=0)).astype(np.int32)
        return jnp.concatenate(out, axis=0)

    def _se3_solve_fn(self, prob: _AnalyzedProblem, T: int, linear_solver: str):
        """`(q_init (T,nq), kp_data (T,n_kp_dim), offsets (n_site_dim,),
        frozen_q (nq,)) -> qpos (T,nq)`. `offsets`/`frozen_q` are broadcast to
        `(T, ...)` -- the same site offsets and frozen slots for every frame
        in this batch-of-`T` solve."""
        se3_var_cls, joint_var_cls, kp_var_cls = prob.SE3Var, prob.JointVar, prob.KpVar
        offset_var_cls, frozen_var_cls = prob.OffsetVar, prob.FrozenVar
        trust, term = self._trust_region(), self._termination()

        def solve(q_init, kp_data, offsets, frozen_q):
            xyz_init = q_init[:, :3]
            wxyz_init = q_init[:, 3:7]
            hinges_init = q_init[:, FREE_JOINT_NDOF:]
            qn = jnp.linalg.norm(wxyz_init, axis=-1, keepdims=True)
            wxyz_init = wxyz_init / jnp.where(qn > 0, qn, 1.0)
            roots_init = jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(wxyz=wxyz_init), xyz_init
            )
            n_site_dim = int(offsets.shape[-1])
            sol = prob.analyzed.solve(
                verbose=False,
                linear_solver=linear_solver,
                trust_region=trust,
                termination=term,
                initial_vals=jaxls.VarValues.make(
                    [
                        se3_var_cls(jnp.arange(T)).with_value(roots_init),
                        joint_var_cls(jnp.arange(T)).with_value(hinges_init),
                        kp_var_cls(jnp.arange(T)).with_value(kp_data),
                        offset_var_cls(jnp.arange(T)).with_value(
                            jnp.broadcast_to(offsets, (T, n_site_dim))
                        ),
                        frozen_var_cls(jnp.arange(T)).with_value(frozen_q),
                    ]
                ),
            )
            sol_roots = sol[se3_var_cls(jnp.arange(T))]
            sol_joints = sol[joint_var_cls(jnp.arange(T))]
            return jnp.concatenate(
                [sol_roots.translation(), sol_roots.rotation().wxyz, sol_joints], axis=-1
            )

        return solve

    def _solve_se3(
        self,
        prob: _AnalyzedProblem,
        T: int,
        q_init: jnp.ndarray,
        kp_data: jnp.ndarray,
        offsets: jnp.ndarray,
        frozen_q: jnp.ndarray,
    ) -> jnp.ndarray:
        n_hinges = q_init.shape[1] - FREE_JOINT_NDOF
        linear_solver = self._pick_linear_solver(T, 6 + n_hinges)
        return self._se3_solve_fn(prob, T, linear_solver)(q_init, kp_data, offsets, frozen_q)

    def solve(
        self,
        *,
        q_init: np.ndarray,
        mjx_model,
        mjx_data,
        kp_data: np.ndarray,
        qs_to_opt: np.ndarray,
        kp_weights: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        site_idxs: np.ndarray,
        frozen_qpos: np.ndarray | None = None,
        penetration: dict | None = None,
    ) -> np.ndarray:
        """Solve IK for `T` frames. `q_init` `(T, nq)`; `kp_data` `(T, K*3)`
        (or `(T, K, 3)`) in model units, NaN allowed for a masked keypoint
        coordinate; `qs_to_opt` `(nq,)` bool; `kp_weights` `(K*3,)`.
        """
        kp_data = jnp.asarray(kp_data)
        if kp_data.ndim == 3:
            kp_data = kp_data.reshape(kp_data.shape[0], -1)
        q_init = jnp.asarray(q_init)
        qs_to_opt = jnp.asarray(qs_to_opt)
        kp_weights = jnp.asarray(kp_weights)
        lb = jnp.asarray(lb)
        ub = jnp.asarray(ub)
        site_idxs = jnp.asarray(site_idxs)
        T = int(q_init.shape[0])

        if self.settings.penetration_weight > 0.0 and penetration is None:
            raise ValueError(
                "penetration_weight > 0 needs a `penetration=` spec from "
                "penetration_spec(); without one the term is silently absent "
                "and the solve looks like it ran with the penalty on"
            )
        offsets = jnp.asarray(mjx_model.site_pos)[site_idxs].flatten()
        if frozen_qpos is None:
            frozen_q = jnp.broadcast_to(jnp.asarray(mjx_data.qpos), (T, int(q_init.shape[1])))
        else:
            frozen_q = jnp.asarray(frozen_qpos)
            if frozen_q.shape != (T, int(q_init.shape[1])):
                raise ValueError(
                    f"frozen_qpos must be (T, nq) = {(T, int(q_init.shape[1]))}, "
                    f"got {tuple(frozen_q.shape)}; it supplies the masked-off DOFs "
                    f"for EACH frame, so a single (nq,) vector is not enough"
                )

        if T > 1:
            prob = self._get_analyzed(
                1,
                mjx_model,
                mjx_data,
                kp_data,
                qs_to_opt,
                kp_weights,
                lb,
                ub,
                site_idxs,
                pen=penetration,
            )
            qpos = self._solve_independent(prob, q_init, kp_data, offsets, frozen_q)
        else:
            prob = self._get_analyzed(
                T,
                mjx_model,
                mjx_data,
                kp_data,
                qs_to_opt,
                kp_weights,
                lb,
                ub,
                site_idxs,
                pen=penetration,
            )
            qpos = self._solve_se3(prob, T, q_init, kp_data, offsets, frozen_q)
        return np.asarray(qpos)

    def solve_frame(self, q0: np.ndarray, **kw) -> np.ndarray:
        """One frame. `q0` `(nq,)` -> `(nq,)`. The `T=1` case of `solve()`:"""
        kp_data = jnp.asarray(kw.pop("kp_data"))
        kp_flat = kp_data.flatten() if kp_data.ndim > 1 else kp_data
        result = self.solve(q_init=jnp.asarray(q0)[None], kp_data=kp_flat[None], **kw)
        return result[0]
