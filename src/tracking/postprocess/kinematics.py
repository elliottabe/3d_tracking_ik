"""Model forward kinematics and the thorax-egocentric frame."""

from __future__ import annotations

import mujoco
import numpy as np

__all__ = [
    "THORAX_BODY",
    "site_xpos_from_qpos",
    "fitted_site_xpos_from_qpos",
    "body_frames_from_qpos",
    "egocentric_sites",
]

THORAX_BODY = "thorax"


def _forward(anatomy, qpos: np.ndarray):
    """Yield `(t, mujoco.MjData)` for every finite row of `(T, nq)` `qpos`."""
    model = anatomy.mj_model
    data = mujoco.MjData(model)
    for t in range(len(qpos)):
        row = qpos[t]
        if not np.isfinite(row).all():
            continue
        data.qpos[:] = row
        mujoco.mj_forward(model, data)
        yield t, data


def site_xpos_from_qpos(anatomy, qpos: np.ndarray) -> np.ndarray:
    """`(T, nq)` poses -> `(T, nsite, 3)` world-frame site positions, model units."""
    qpos = np.asarray(qpos, np.float64)
    out = np.full((len(qpos), anatomy.nsite, 3), np.nan, np.float64)
    for t, data in _forward(anatomy, qpos):
        out[t] = data.site_xpos
    return out


def fitted_site_xpos_from_qpos(
    anatomy, qpos: np.ndarray, offsets: np.ndarray, *, chunk: int = 512
) -> np.ndarray:
    """`(T, nq)` poses + `(K, 3)` FITTED marker offsets -> `(T, K, 3)` marker
    positions in model units, with those offsets APPLIED. NaN where `qpos` is.
    """
    import jax
    import jax.numpy as jnp

    from tracking.inverse_kinematics.solver import (
        forward,
        get_site_xpos,
        mjx_load,
        set_site_pos,
    )

    qpos = np.asarray(qpos, np.float64)
    offsets = np.asarray(offsets, np.float64).reshape(-1, 3)
    site_idxs = jnp.asarray(anatomy.site_idxs)

    mjx_model, mjx_data = mjx_load(anatomy.mj_model)
    mjx_model = set_site_pos(mjx_model, jnp.asarray(offsets), site_idxs)

    out = np.full((len(qpos), len(offsets), 3), np.nan, np.float64)
    solved = np.flatnonzero(np.isfinite(qpos).all(1))
    if not len(solved):
        return out

    @jax.jit
    def _fk(batch):
        return jax.vmap(
            lambda q: get_site_xpos(forward(mjx_model, mjx_data.replace(qpos=q)), site_idxs)
        )(batch)

    for start in range(0, len(solved), chunk):
        idx = solved[start : start + chunk]
        batch = qpos[idx]
        pad = chunk - len(batch)
        if pad:
            batch = np.concatenate([batch, np.repeat(batch[-1:], pad, axis=0)], axis=0)
        out[idx] = np.asarray(_fk(jnp.asarray(batch)))[: len(idx)]
    return out


def body_frames_from_qpos(anatomy, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(T, nq)` poses -> `((T, nbody, 3) xpos, (T, nbody, 4) xquat)`, model units."""
    qpos = np.asarray(qpos, np.float64)
    xpos = np.full((len(qpos), anatomy.nbody, 3), np.nan, np.float64)
    xquat = np.full((len(qpos), anatomy.nbody, 4), np.nan, np.float64)
    for t, data in _forward(anatomy, qpos):
        xpos[t] = data.xpos
        xquat[t] = data.xquat
    return xpos, xquat


def _quat_to_mat(quat: np.ndarray) -> np.ndarray:
    """`(4,)` wxyz -> `(3, 3)` rotation matrix, via MuJoCo's own conversion."""
    mat = np.zeros(9, np.float64)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, np.float64))
    return mat.reshape(3, 3)


def egocentric_sites(
    site_xpos: np.ndarray,
    xpos: np.ndarray,
    xquat: np.ndarray,
    *,
    anatomy,
    body: str = THORAX_BODY,
) -> np.ndarray:
    """`(T, S, 3)` world points expressed in `body`'s own frame."""
    idx = anatomy.body_idx(body)
    site_xpos = np.asarray(site_xpos, np.float64)
    xpos = np.asarray(xpos, np.float64)
    xquat = np.asarray(xquat, np.float64)
    out = np.full(site_xpos.shape, np.nan, np.float64)
    for t in range(len(site_xpos)):
        if not np.isfinite(xquat[t, idx]).all() or not np.isfinite(xpos[t, idx]).all():
            continue
        rot = _quat_to_mat(xquat[t, idx])
        out[t] = (site_xpos[t] - xpos[t, idx]) @ rot
    return out
