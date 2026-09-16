"""Per-frame multi-start STAC IK -- Stage C's solver."""

from __future__ import annotations

import copy
import fnmatch
import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import h5py
import jax.numpy as jnp
import mujoco
import numpy as np

from tracking.conventions import NotFit, plain_mapping
from tracking.inverse_kinematics.solver import (
    FREE_JOINT_NDOF,
    PerFrameSolver,
    SolverSettings,
    estimate_orientation_from_keypoints,
    forward,
    mjx_load,
    set_site_pos,
)
from tracking.io.names import as_order

__all__ = [
    "PerFrameResult",
    "WING_DOF_PATTERNS",
    "chained_starts",
    "dof_mask",
    "frame_costs",
    "qvel_from_qpos",
    "refine_wings",
    "select_by_cost",
    "solve_bout",
    "solver_settings_from_cfg",
    "viterbi_select",
    "warm_start",
    "wing_dof_weights",
    "write_stac_h5",
]

# The six wing DOFs on v1, by NAME glob -- never by the qpos indices 7..12,
# which are v1's alone (v2_3 maps the same slots onto LEGS).
WING_DOF_PATTERNS: tuple[str, ...] = ("wing_*",)

# The implicit fourth candidate: the warm start, solved as-is.
ZERO_START = "zero"

_VALID_STARTS = ("wing_rest_left", "wing_rest_right", "wing_rest_both")


def dof_mask(names_qpos: Sequence[str], patterns: Sequence[str]) -> np.ndarray:
    """`(nq,)` bool: qpos slots whose JOINT NAME matches any glob in `patterns`."""
    return np.array([any(fnmatch.fnmatch(n, p) for p in patterns) for n in names_qpos], dtype=bool)


def wing_dof_weights(names_qpos: Sequence[str], wing_weight: float) -> np.ndarray:
    """`(nq,)` per-DOF multiplier for the Viterbi jump penalty: `wing_weight`
    on the wing DOFs, 1.0 everywhere else.
    """
    w = np.ones(len(names_qpos), dtype=np.float64)
    w[dof_mask(names_qpos, WING_DOF_PATTERNS)] = float(wing_weight)
    return w


def select_by_cost(costs: np.ndarray) -> np.ndarray:
    """`(S, T)` per-candidate per-frame costs -> `(T,)` cheapest candidate."""
    c = np.where(np.isfinite(costs), costs, np.inf)
    idx = np.argmin(c, axis=0)
    idx[~np.isfinite(c).any(axis=0)] = 0
    return idx


def viterbi_select(
    costs: np.ndarray,
    poses: Sequence[np.ndarray],
    mask: np.ndarray,
    switch_weight: float,
    dof_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Temporally consistent candidate choice. `(T,)` candidate index."""
    costs = np.asarray(costs, dtype=np.float64)
    n_cand, n_frames = costs.shape
    if switch_weight <= 0 or n_frames == 1:
        return select_by_cost(costs)
    mask = np.asarray(mask, dtype=bool)
    unary = np.where(np.isfinite(costs), costs, np.inf)
    qm = np.stack([np.where(np.isfinite(q[:, mask]), q[:, mask], 0.0) for q in poses])
    w = (
        np.ones(int(mask.sum()))
        if dof_weights is None
        else np.asarray(dof_weights, dtype=np.float64)[mask]
    )
    jump = np.zeros((n_frames, n_cand, n_cand))
    for t in range(1, n_frames):
        diff = qm[:, t, None, :] - qm[None, :, t - 1, :]  # (S_t, S_{t-1}, M)
        jump[t] = switch_weight * np.sum(w * diff * diff, axis=-1)
    score = unary[:, 0].copy()
    back = np.zeros((n_frames, n_cand), np.int32)
    for t in range(1, n_frames):
        total = score[None, :] + jump[t]  # (S_t, S_{t-1})
        back[t] = np.argmin(total, axis=1)
        score = unary[:, t] + total[np.arange(n_cand), back[t]]
    idx = np.empty(n_frames, np.int32)
    idx[-1] = int(np.argmin(score))
    for t in range(n_frames - 1, 0, -1):
        idx[t - 1] = back[t, idx[t]]
    idx[~np.isfinite(unary).any(axis=0)] = 0
    return idx


def frame_costs(
    mj_model, site_idxs: np.ndarray, q: np.ndarray, kp: np.ndarray, kp_weights: np.ndarray
) -> np.ndarray:
    """`(T,)` marker cost per frame: `sum(((site - kp) * w)^2)` over the FINITE
    keypoint coordinates, via MuJoCo's CPU FK.
    """
    d = mujoco.MjData(mj_model)
    q = np.asarray(q, dtype=np.float64)
    n_frames = q.shape[0]
    out = np.full(n_frames, np.nan)
    kp3 = np.asarray(kp, dtype=np.float64).reshape(n_frames, -1, 3)
    w3 = np.asarray(kp_weights, dtype=np.float64).reshape(-1, 3)
    site_idxs = np.asarray(site_idxs)
    for t in range(n_frames):
        if not np.isfinite(q[t]).all():
            continue
        d.qpos[:] = q[t]
        mujoco.mj_kinematics(mj_model, d)
        r = (d.site_xpos[site_idxs] - kp3[t]) * w3
        r = np.where(np.isfinite(r), r, 0.0)
        out[t] = float(np.sum(r * r))
    return out


def qvel_from_qpos(
    q: np.ndarray, dt: float, *, freejoint: bool = True, max_qvel: float = 20.0
) -> np.ndarray:
    """`(T, nv)` velocities from `(T, nq)` qpos. `dt` in SECONDS."""
    q = np.asarray(q, np.float64)
    qp = np.concatenate([q, q[-1:]], axis=0)
    if not freejoint:
        return np.clip((qp[1:] - qp[:-1]) / dt, -max_qvel, max_qvel)
    v_joint = (qp[1:, FREE_JOINT_NDOF:] - qp[:-1, FREE_JOINT_NDOF:]) / dt
    v_trans = (qp[1:, :3] - qp[:-1, :3]) / dt
    a, b = qp[:-1, 3:7], qp[1:, 3:7]
    # conj(a) * b, quaternions as [w, x, y, z]
    aw, ax, ay, az = a[:, 0], -a[:, 1], -a[:, 2], -a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    dq = np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=1,
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        dq = dq / np.linalg.norm(dq, axis=1, keepdims=True)
        ang = 2.0 * np.arccos(np.clip(dq[:, 0], -1.0, 1.0))
        sin_half = np.sqrt(np.maximum(1.0 - dq[:, 0] ** 2, 0.0))
        ang = (ang + np.pi) % (2 * np.pi) - np.pi  # wrap to (-pi, pi]
        axis = np.where(sin_half[:, None] > 1e-7, dq[:, 1:] / sin_half[:, None], 0.0)
    v_gyro = axis * ang[:, None] / dt
    out = np.concatenate([v_trans, v_gyro, v_joint], axis=1)
    out[:, 6:] = np.clip(out[:, 6:], -max_qvel, max_qvel)
    return out


def warm_start(kp_flat: np.ndarray, kp_order, model_cfg, q_base: np.ndarray) -> np.ndarray:
    """`(T, nq)` production warm start -- the batch solver's own start."""
    kp_order = as_order(kp_order)
    model_cfg = plain_mapping(model_cfg, what="model_cfg", api="warm_start")
    kp_flat = np.asarray(kp_flat, dtype=np.float64)
    n_frames = kp_flat.shape[0]

    root_name = model_cfg.get("ROOT_OPTIMIZATION_KEYPOINT")
    if not root_name:
        raise ValueError(
            "warm_start: model.ROOT_OPTIMIZATION_KEYPOINT is required -- it is "
            "where the root translation comes from, and without it every frame "
            "starts at the origin"
        )
    spec = plain_mapping(
        model_cfg.get("JAXLS_ORIENTATION_KEYPOINTS"), what="model_cfg", api="warm_start"
    )
    if not spec:
        raise ValueError(
            "warm_start: model.JAXLS_ORIENTATION_KEYPOINTS is required -- without "
            "it every frame's warm start keeps the tiled identity quaternion, a "
            "materially worse LM start that measurably moves the fit"
        )

    q = np.tile(np.asarray(q_base, np.float64), (n_frames, 1))
    i_root = kp_order.index(str(root_name))
    q[:, :3] = kp_flat[:, i_root * 3 : i_root * 3 + 3]
    quats = estimate_orientation_from_keypoints(
        jnp.asarray(kp_flat),
        kp_order.index(str(spec["rear"])),
        kp_order.index(str(spec["left"])),
        kp_order.index(str(spec["right"])),
        kp_order.index(str(spec["front"])) if "front" in spec else -1,
    )
    q[:, 3:7] = np.asarray(quats)
    return q


def chained_starts(
    q_solved: np.ndarray,
    names_qpos: Sequence[str],
    qpos_spring: np.ndarray,
    starts: Sequence[str],
) -> dict[str, np.ndarray]:
    """Rest starts built FROM the zero-start solution: everything as solved,
    only the named wing pitch(es) reset to the model's spring rest.
    """
    names = list(names_qpos)
    i_left, i_right = names.index("wing_pitch_left"), names.index("wing_pitch_right")
    qpos_spring = np.asarray(qpos_spring, dtype=np.float64)
    out: dict[str, np.ndarray] = {}
    for name in starts:
        q = np.array(q_solved, dtype=np.float64, copy=True)
        if name == "wing_rest_left":
            q[:, i_left] = qpos_spring[i_left]
        elif name == "wing_rest_right":
            q[:, i_right] = qpos_spring[i_right]
        elif name == "wing_rest_both":
            q[:, i_left] = qpos_spring[i_left]
            q[:, i_right] = qpos_spring[i_right]
        else:
            raise ValueError(
                f"stac.per_frame.starts: unknown start {name!r} (zero is implicit); "
                f"known starts are {list(_VALID_STARTS)}"
            )
        out[name] = q
    return out


def solver_settings_from_cfg(per_frame_cfg) -> SolverSettings:
    """`SolverSettings` from a `stac.per_frame` block."""
    cfg = plain_mapping(per_frame_cfg, what="per_frame_cfg", api="solver_settings_from_cfg")
    return SolverSettings(
        n_iter=int(cfg.get("n_iter", 500)),
        lambda_initial=float(cfg.get("lambda_initial", 5e-4)),
        lambda_min=float(cfg.get("lambda_min", 1e-8)),
        cost_tolerance=float(cfg.get("cost_tolerance", 1e-12)),
        gradient_tolerance=float(cfg.get("gradient_tolerance", 1e-14)),
        parameter_tolerance=float(cfg.get("parameter_tolerance", 1e-16)),
        batch=int(cfg.get("batch", 512)),
    )


def _cpu_model(anatomy, offsets: np.ndarray):
    """A COPY of `anatomy.mj_model` carrying the fitted marker offsets."""
    mj = copy.deepcopy(anatomy.mj_model)
    mj.site_pos[np.asarray(anatomy.site_idxs)] = np.asarray(offsets, np.float64)
    return mj


def _mjx_env(mj, site_idxs, offsets):
    """`(mjx_model, mjx_data)` with the marker offsets set and FK run once."""
    mjx_model, mjx_data = mjx_load(mj)
    mjx_model = set_site_pos(mjx_model, jnp.asarray(offsets), jnp.asarray(site_idxs))
    mjx_data = forward(mjx_model, mjx_data)
    return mjx_model, mjx_data


def refine_wings(
    anatomy,
    qpos: np.ndarray,
    kp3d_model: np.ndarray,
    *,
    offsets: np.ndarray | None = None,
    mjx_model=None,
    mjx_data=None,
    settings: SolverSettings | None = None,
    dof_patterns: Sequence[str] = WING_DOF_PATTERNS,
) -> np.ndarray:
    """Re-solve ONLY the wing DOFs, each frame frozen at ITS OWN pose. `(T, nq)`."""
    qpos = np.asarray(qpos, dtype=np.float64)
    kp3d_model = np.asarray(kp3d_model, dtype=np.float64)
    if kp3d_model.ndim != 3:
        raise ValueError(
            f"refine_wings: kp3d_model must be (T, K, 3), got {kp3d_model.shape}; "
            f"reshape a flat (T, K*3) array with .reshape(T, -1, 3)"
        )
    if qpos.shape[0] != kp3d_model.shape[0]:
        raise ValueError(
            f"refine_wings: qpos has {qpos.shape[0]} frames and kp3d_model {kp3d_model.shape[0]}"
        )
    mask = dof_mask(anatomy.names_qpos, dof_patterns)
    if not mask.any():
        raise ValueError(
            f"refine_wings: no qpos slot matches {list(dof_patterns)} in this "
            f"anatomy ({anatomy.name}); there is nothing to refine"
        )

    if mjx_model is None or mjx_data is None:
        if offsets is None:
            raise ValueError("refine_wings: pass either `offsets` or `mjx_model`/`mjx_data`")
        mjx_model, mjx_data = _mjx_env(
            _cpu_model(anatomy, offsets), anatomy.site_idxs, np.asarray(offsets, np.float64)
        )

    out = qpos.copy()
    ok = np.isfinite(qpos).all(axis=1)
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return out

    kp_flat = kp3d_model.reshape(kp3d_model.shape[0], -1)[idx]
    q_start = qpos[idx]
    # A FRESH solver: this pass runs under a PARTIAL `qs_to_opt` mask, which
    # is a different analyzed problem from the full-mask candidate solves.
    solver = PerFrameSolver(settings if settings is not None else SolverSettings())
    q_solved = np.asarray(
        solver.solve(
            q_init=q_start,
            mjx_model=mjx_model,
            mjx_data=mjx_data,
            kp_data=kp_flat,
            qs_to_opt=mask,
            kp_weights=anatomy.kp_weights,
            lb=anatomy.lb,
            ub=anatomy.ub,
            site_idxs=anatomy.site_idxs,
            frozen_qpos=q_start,  # per frame -- the 77x defect if left to default
        ),
        dtype=np.float64,
    )
    out[idx] = np.where(mask, q_solved, q_start)
    return out


@dataclass(frozen=True)
class PerFrameResult:
    """One bout-fly's per-frame solve, in the order `write_stac_h5` stores it."""

    qpos: np.ndarray
    qpos_selected: np.ndarray
    qvel: np.ndarray
    xpos: np.ndarray
    xquat: np.ndarray
    marker_sites: np.ndarray
    kp_data: np.ndarray
    offsets: np.ndarray
    solved: np.ndarray
    start_idx: np.ndarray
    iterations: np.ndarray
    candidate_names: tuple[str, ...]
    qpos_candidates: np.ndarray | None
    candidate_costs: np.ndarray | None
    summary: Mapping[str, Any]


def solve_bout(
    anatomy,
    kp3d_model: np.ndarray,
    *,
    offsets: np.ndarray,
    settings: SolverSettings | None = None,
    per_frame_cfg: Mapping[str, Any] | None = None,
    solve_mask: np.ndarray | None = None,
) -> PerFrameResult:
    """Solve every frame of `kp3d_model` `(T, K, 3)` independently."""
    t_start = time.time()
    kp3d_model = np.asarray(kp3d_model, dtype=np.float64)
    if kp3d_model.ndim != 3 or kp3d_model.shape[-1] != 3:
        raise ValueError(
            f"solve_bout: kp3d_model must be (T, K, 3), got {kp3d_model.shape}. The "
            f"source's docstring claims a flat (T, K*3) array also works; it does "
            f"not -- its own solvability check indexes axis 1 of a 3-D array and "
            f"raises AxisError on the flat form. Reshape with "
            f".reshape(T, -1, 3) at the call site, where K is known by name."
        )
    if float(anatomy.mocap_scale_factor) != 1.0:
        raise ValueError(
            f"solve_bout: this anatomy sets MOCAP_SCALE_FACTOR="
            f"{anatomy.mocap_scale_factor}, and this path does NOT apply it while "
            f"the offsets fit (offsets_fit.scaled_model_keypoints) DOES -- the "
            f"poses and the marker offsets would be fitted in two different unit "
            f"systems with nothing NaN to show for it. Apply the factor to "
            f"kp3d_model before calling, or set MOCAP_SCALE_FACTOR=1."
        )
    n_frames, n_kp = kp3d_model.shape[0], kp3d_model.shape[1]
    if n_kp != len(anatomy.kp_order):
        raise ValueError(
            f"solve_bout: kp3d_model has {n_kp} keypoints, anatomy "
            f"{len(anatomy.kp_order)} ({list(anatomy.kp_order.names[:3])}...)"
        )

    cfg = plain_mapping(per_frame_cfg, what="per_frame_cfg", api="solve_bout")
    if "dt" not in cfg:
        raise ValueError(
            "solve_bout: per_frame_cfg['dt'] is required (seconds). It is the "
            "stac config's `stac.mujoco.dt` (0.00125 on the reference run), NOT "
            "mj_model.opt.timestep -- load_anatomy leaves that at the XML's "
            "0.0001, so inferring it would scale every qvel by 12.5x with "
            "nothing NaN and no residual moved. Pass "
            "per_frame_cfg={**cfg.stac.per_frame, 'dt': cfg.stac.mujoco.dt}."
        )
    dt = float(cfg["dt"])
    if not 2e-4 <= dt <= 1.0:
        # `dt = 0` is a plausible unset-config value; formatting its rate
        # would raise ZeroDivisionError from inside this guard's own message.
        rate = f" {dt} s is {1 / dt:.0f} Hz." if dt else ""
        raise ValueError(
            f"solve_bout: per_frame_cfg['dt'] = {dt} s is not a plausible frame "
            f"interval (expected 2e-4 .. 1.0 s).{rate}"
            f" If this is mj_model.opt.timestep (1e-4), that is "
            f"the 12.5x trap this key exists to prevent -- `dt` is the RECORDING's "
            f"frame interval, stac.mujoco.dt, 1.25e-3 s on the reference run."
        )
    starts = [str(s) for s in (cfg.get("starts") or [])]
    switch_weight = float(cfg.get("switch_weight", 0.0))
    switch_wing_weight = float(cfg.get("switch_wing_weight", 1.0))
    save_candidates = bool(cfg.get("save_candidates", True))
    do_refine = bool(cfg.get("refine_wings", True))
    max_qvel = float(cfg.get("max_qvel", 20.0))
    settings = settings if settings is not None else solver_settings_from_cfg(cfg)

    offsets = np.asarray(offsets, dtype=np.float64).reshape(-1, 3)
    if offsets.shape[0] != n_kp:
        raise ValueError(f"solve_bout: offsets has {offsets.shape[0]} markers, anatomy {n_kp}")

    nq = int(anatomy.nq)
    site_idxs = np.asarray(anatomy.site_idxs)
    mj = _cpu_model(anatomy, offsets)
    mjx_model, mjx_data = _mjx_env(mj, site_idxs, offsets)

    model_cfg = plain_mapping(
        (anatomy.cfg or {}).get("model"), what="anatomy.cfg['model']", api="solve_bout"
    )
    # Which frames can be STARTED: every warm-start keypoint finite. Resolved
    # by name; the indices never leave this block.
    root_name = model_cfg.get("ROOT_OPTIMIZATION_KEYPOINT")
    if not root_name:
        raise ValueError(
            "solve_bout: anatomy.cfg['model'].ROOT_OPTIMIZATION_KEYPOINT is "
            "required -- it names the keypoint the root translation is warm-"
            "started from, and it also decides which frames are solvable at "
            "all. A bare KeyError here would point at the dict, not at the "
            "anatomy config that is missing the key."
        )
    need_names = [str(root_name)]
    spec = model_cfg.get("JAXLS_ORIENTATION_KEYPOINTS") or {}
    need_names += [str(spec[k]) for k in ("rear", "left", "right", "front") if k in spec]
    need = [anatomy.kp_order.index(n) for n in need_names]
    solvable = np.all([np.isfinite(kp3d_model[:, i]).all(axis=1) for i in need], axis=0)
    if solve_mask is not None:
        solvable = solvable & np.asarray(solve_mask, bool)
    idx = np.flatnonzero(solvable)
    if idx.size == 0:
        raise NotFit(
            "solve_bout: no frame has finite root/orientation keypoints "
            f"({need_names}); every frame would be NaN"
        )

    kp_flat = kp3d_model.reshape(n_frames, -1)
    kp_s = kp_flat[idx]
    q_init = warm_start(kp_s, anatomy.kp_order, model_cfg, np.asarray(mjx_data.qpos))

    solver = PerFrameSolver(settings)
    common = dict(
        mjx_model=mjx_model,
        mjx_data=mjx_data,
        kp_data=kp_s,
        qs_to_opt=np.ones(nq, bool),
        kp_weights=anatomy.kp_weights,
        lb=anatomy.lb,
        ub=anatomy.ub,
        site_idxs=site_idxs,
    )
    cand_names: list[str] = [ZERO_START]
    cand_q: list[np.ndarray] = []
    cand_it: list[np.ndarray] = []
    solve_seconds: dict[str, float] = {}

    t0 = time.time()
    q_zero = np.asarray(solver.solve(q_init=q_init, **common), dtype=np.float64)
    cand_q.append(q_zero)
    cand_it.append(np.asarray(solver.last_iterations))
    solve_seconds[ZERO_START] = round(time.time() - t0, 1)

    for name, q0 in chained_starts(
        q_zero, anatomy.names_qpos, np.asarray(mj.qpos_spring), starts
    ).items():
        t0 = time.time()
        cand_q.append(np.asarray(solver.solve(q_init=q0, **common), dtype=np.float64))
        cand_names.append(name)
        cand_it.append(np.asarray(solver.last_iterations))
        solve_seconds[name] = round(time.time() - t0, 1)

    costs = np.stack([frame_costs(mj, site_idxs, q, kp_s, anatomy.kp_weights) for q in cand_q])
    choice = viterbi_select(
        costs,
        cand_q,
        np.ones(nq, bool),
        switch_weight,
        dof_weights=wing_dof_weights(anatomy.names_qpos, switch_wing_weight),
    )
    q_sel = np.stack([cand_q[choice[t]][t] for t in range(idx.size)])
    it_sel = np.stack([cand_it[choice[t]][t] for t in range(idx.size)])

    q_selected_full = np.full((n_frames, nq), np.nan)
    q_selected_full[idx] = q_sel
    q_full = q_selected_full.copy()
    if do_refine:
        t0 = time.time()
        q_refined = refine_wings(
            anatomy,
            q_sel,
            kp3d_model[idx],
            mjx_model=mjx_model,
            mjx_data=mjx_data,
            settings=settings,
        )
        q_full = np.full((n_frames, nq), np.nan)
        q_full[idx] = q_refined
        solve_seconds["refine_wings"] = round(time.time() - t0, 1)

    d = mujoco.MjData(mj)
    xpos = np.full((n_frames, mj.nbody, 3), np.nan)
    xquat = np.full((n_frames, mj.nbody, 4), np.nan)
    msites = np.full((n_frames, site_idxs.size, 3), np.nan)
    for t in idx:
        d.qpos[:] = q_full[t]
        mujoco.mj_kinematics(mj, d)
        xpos[t] = d.xpos
        xquat[t] = d.xquat
        msites[t] = d.site_xpos[site_idxs]

    qvel = qvel_from_qpos(
        np.nan_to_num(q_full), dt, freejoint=anatomy.has_freejoint, max_qvel=max_qvel
    )
    qvel[~solvable] = np.nan

    start_idx = np.full(n_frames, -1, np.int32)
    start_idx[idx] = choice
    iterations = np.full(n_frames, -1, np.int32)
    iterations[idx] = it_sel

    qpos_candidates = candidate_costs = None
    if save_candidates:
        qpos_candidates = np.full((len(cand_q), n_frames, nq), np.nan, np.float32)
        qpos_candidates[:, idx] = np.stack(cand_q).astype(np.float32)
        candidate_costs = np.full((len(cand_q), n_frames), np.nan, np.float32)
        candidate_costs[:, idx] = costs

    n_cap = int(np.sum(it_sel >= int(settings.n_iter)))
    summary = dict(
        seconds=round(time.time() - t_start, 1),
        solve_seconds=solve_seconds,
        n_frames=int(n_frames),
        n_solved=int(idx.size),
        # by NAME, never a bare candidate index
        chosen_fraction={
            n: round(float(np.mean(choice == i)), 3) for i, n in enumerate(cand_names)
        },
        switches=int(np.sum(choice[1:] != choice[:-1])),
        iterations_median=float(np.median(it_sel)),
        iterations_p95=float(np.percentile(it_sel, 95)),
        frames_at_iteration_cap=n_cap,
        cost_median=float(np.nanmedian(costs[choice, np.arange(idx.size)])),
        refine_wings=do_refine,
        dt=dt,
    )
    return PerFrameResult(
        qpos=q_full,
        qpos_selected=q_selected_full,
        qvel=qvel,
        xpos=xpos,
        xquat=xquat,
        marker_sites=msites,
        kp_data=kp_flat,
        offsets=offsets,
        solved=solvable,
        start_idx=start_idx,
        iterations=iterations,
        candidate_names=tuple(cand_names),
        qpos_candidates=qpos_candidates,
        candidate_costs=candidate_costs,
        summary=summary,
    )


def write_stac_h5(path, result: PerFrameResult, *, anatomy, cfg) -> None:
    """Write `result` as `stac_ik.h5`, atomically (`.tmp` then `os.replace`)."""
    cfg = plain_mapping(cfg, what="cfg", api="write_stac_h5")
    path = str(path)
    tmp = f"{path}.tmp"
    candidates = json.dumps(list(result.candidate_names))
    with h5py.File(tmp, "w") as f:
        f.create_dataset("config", data=np.bytes_(json.dumps(dict(cfg), indent=2, default=str)))
        for key, names in (
            ("kp_names", anatomy.kp_order.names),
            ("names_qpos", anatomy.names_qpos),
            ("names_xpos", anatomy.names_xpos),
        ):
            f.create_dataset(key, data=np.array([str(n) for n in names], dtype="S"))
        for key, value in (
            ("kp_data", result.kp_data),
            ("marker_sites", result.marker_sites),
            ("offsets", result.offsets),
            ("qpos", result.qpos),
            ("xpos", result.xpos),
            ("xquat", result.xquat),
            ("qvel", result.qvel),
        ):
            f.create_dataset(key, data=np.asarray(value, np.float32))
        f.create_dataset("ik_start_idx", data=np.asarray(result.start_idx, np.int32))
        f.create_dataset("ik_iterations", data=np.asarray(result.iterations, np.int32))
        if result.qpos_candidates is not None:
            f.create_dataset("qpos_candidates", data=np.asarray(result.qpos_candidates, np.float32))
            f.create_dataset("candidate_costs", data=np.asarray(result.candidate_costs, np.float32))
        if bool(result.summary.get("refine_wings")):
            f.create_dataset("qpos_selected", data=np.asarray(result.qpos_selected, np.float32))
        f.attrs["ik_solver"] = "per_frame"
        f.attrs["ik_candidates"] = candidates
        f.attrs["ik_summary"] = json.dumps(dict(result.summary), default=str)
    os.replace(tmp, path)
