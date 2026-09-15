"""Model forward kinematics and the thorax-egocentric frame.

Everything here is a pure function of `(anatomy, qpos)`: given a pose, where
are the model's sites and bodies, and what do the sites look like from the
thorax's own point of view. No artifact, no IO, no config -- the writer that
persists these lives in `postprocess/outputs.py`.

UNITS. `qpos` and everything derived from it here are in MODEL units, the
same frame `stac_ik.h5`'s `qpos`, `marker_sites` and `kp_data` use. They are
NOT the 0.1 mm world units that `kp3d.npz` and `preprocess/` carry. Crossing
the two needs `inverse_kinematics.bridge`, and nothing in this module does.

The thorax is located by NAME, through `Anatomy.body_idx`. A hard-coded body
index is the same class of defect as a hard-coded keypoint index -- one that
yields confident, self-consistent, completely wrong numbers -- and this
pipeline has already paid for one of those.
"""

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
    """Yield `(t, mujoco.MjData)` for every finite row of `(T, nq)` `qpos`.

    Rows that are not finite are skipped entirely rather than being forced
    through `mj_forward`: an unsolved frame must stay unsolved downstream,
    not acquire a plausible-looking pose from whatever was in `data` last.
    """
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

    This is the array the model -> world bridge must be fitted from, and it is
    NOT `stac_ik.h5`'s `marker_sites` dataset. That dataset is FK'd from the
    INITIAL marker model: measured on the reference run and on this repo's own
    runs, every file, `marker_sites` sits 2.4e-7 model units from an FK with no
    offsets and 2.4e-3 from an FK with them. Its name promises the fitted
    markers and it holds the initial ones, and reading it cost 65-77% of a
    1.5-2x reprojection regression against the reference.

    Why not `site_xpos_from_qpos` with the offsets written into the model:
    MuJoCo's `mj_forward` does not pick up a runtime write to
    `mj_model.site_pos` for these sites -- measured, a 5.7e-3 change in
    `site_pos` moved `site_xpos` by exactly 0.0. The offsets have to go in
    through the MJX model (`solver.set_site_pos`, a functional update), which
    is also the route the solver itself takes, so the sites come out in the
    same arithmetic the pose was solved in.

    Evaluated in chunks of a FIXED size -- the tail is padded and sliced off --
    because `jax.vmap` retraces for every distinct batch length, and a ragged
    final chunk would pay a full compile to save a few frames.
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
    """`(T, S, 3)` world points expressed in `body`'s own frame.

    `p_ego = R_body^T @ (p_world - x_body)`. Both halves matter: dropping the
    rotation leaves a quantity that still looks small, centred and smooth and
    is wrong exactly when the animal turns.

    `body` is a NAME; `Anatomy.body_idx` raises `ValueError` on an unknown
    one rather than silently indexing the wrong body.
    """
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
