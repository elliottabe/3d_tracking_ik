"""Affine (telecentric) camera geometry for the multi-view query model."""

from __future__ import annotations

import jax.numpy as jnp


def affine_rows(cam_mats):
    """(C,4,3) P.T -> M (C,2,3), t (C,2). Raises unless the cameras are affine."""
    P = jnp.swapaxes(jnp.asarray(cam_mats, jnp.float32), 1, 2)  # (C,3,4)
    row3 = P[:, 2, :]
    ok = jnp.all(jnp.abs(row3 - jnp.array([0.0, 0.0, 0.0, 1.0])) < 1e-6)
    if not bool(ok):
        raise ValueError("camera is not affine: projection row 3 != [0,0,0,1]")
    return P[:, :2, :3], P[:, :2, 3]


def local_offset(M, t, center3D, crop_origin):
    """t_local (C,2) so that uv_crop = M @ X_local + t_local."""
    return jnp.einsum("cij,j->ci", M, jnp.asarray(center3D, jnp.float32)) + t - crop_origin


def project_local(X, M, t_local):
    """X (...,3) local -> (...,C,2) crop px."""
    return jnp.einsum("cij,...j->...ci", M, X) + t_local


def ray_from_pixel(M, t_local, uv):
    """Back-project crop pixels to lines. p0 (C,N,3) least-norm point, d (C,3) unit direction."""
    Mt = jnp.swapaxes(M, 1, 2)  # (C,3,2)
    MMt_inv = jnp.linalg.inv(jnp.einsum("cij,ckj->cik", M, M))  # (C,2,2)
    pinv = jnp.einsum("cij,cjk->cik", Mt, MMt_inv)  # (C,3,2)
    p0 = jnp.einsum("cij,cnj->cni", pinv, uv - t_local[:, None, :])
    d = jnp.cross(M[:, 0, :], M[:, 1, :])
    d = d / jnp.linalg.norm(d, axis=-1, keepdims=True)
    return p0, d


def token_pixel_centres(h: int, w: int, patch: int):
    ys, xs = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij")
    return (
        jnp.stack([(xs + 0.5) * patch, (ys + 0.5) * patch], -1).reshape(-1, 2).astype(jnp.float32)
    )


def warp_cameras(M, t_local, A, b):
    return jnp.einsum("cij,cjk->cik", A, M), jnp.einsum("cij,cj->ci", A, t_local) + b


def rotate_world(M, R):
    return jnp.einsum("cij,kj->cik", M, R)  # M @ R.T


def mirror_world(M, t_local, crop: int):
    C = M.shape[0]
    A = jnp.broadcast_to(jnp.diag(jnp.array([-1.0, 1.0], jnp.float32)), (C, 2, 2))
    b = jnp.broadcast_to(jnp.array([crop - 1.0, 0.0], jnp.float32), (C, 2))
    S = jnp.diag(jnp.array([-1.0, 1.0, 1.0], jnp.float32))
    M2, t2 = warp_cameras(M, t_local, A, b)
    return jnp.einsum("cij,jk->cik", M2, S), t2


def px_scale(M):
    return jnp.mean(jnp.sqrt(jnp.sum(M**2, axis=(1, 2)) / 2.0))
