"""DLT triangulation of per-camera 2-D keypoints into 3-D points, and QC."""

from __future__ import annotations

import numpy as np

from tracking.geometry.rig import CameraRig


def view_median_conf(conf: np.ndarray) -> np.ndarray:
    """`conf (T,C,K)` -> `(T,C)` median peak confidence across keypoints per view."""
    conf = np.asarray(conf, np.float32)
    if conf.ndim != 3:
        raise ValueError(f"view_median_conf expects (T,C,K), got {conf.shape}")
    return np.median(conf, axis=2)


def triangulate(
    kp2d: np.ndarray,
    conf: np.ndarray,
    rig: CameraRig,
    *,
    conf_thresh: float = 0.3,
    min_views: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """DLT-triangulate `kp2d (T,C,K,2)` + `conf (T,C,K)` -> `(kp3d (T,K,3), conf3d (T,K))`."""
    kp2d = np.asarray(kp2d, np.float64)
    conf = np.asarray(conf, np.float32)
    T, C, K, _ = kp2d.shape
    if conf.shape != (T, C, K):
        raise ValueError(f"conf shape {conf.shape} does not match kp2d (T,C,K,2)={kp2d.shape}")

    valid = conf >= conf_thresh  # (T,C,K)
    nvalid = valid.sum(axis=1)  # (T,K)

    # Batch over (T*K) points; each has C camera views, per-point validity.
    pts = kp2d.transpose(0, 2, 1, 3).reshape(T * K, C, 2)  # (TK,C,2)
    val = valid.transpose(0, 2, 1).reshape(T * K, C)  # (TK,C)
    pts = np.where(val[..., None], pts, 0.0)

    kp3d_flat = rig.reconstruct_batch(pts, val)  # (TK,3), NaN where < 2 valid views
    ok = nvalid.reshape(T * K) >= min_views
    kp3d_flat = np.where(ok[:, None], kp3d_flat, np.nan)

    conf_perm = conf.transpose(0, 2, 1).reshape(T * K, C)  # (TK,C)
    denom = np.maximum(nvalid.reshape(T * K), 1)
    conf3d_flat = np.where(ok, (conf_perm * val).sum(axis=1) / denom, 0.0)

    kp3d = kp3d_flat.reshape(T, K, 3).astype(np.float32)
    conf3d = conf3d_flat.reshape(T, K).astype(np.float32)
    return kp3d, conf3d


def loo_reprojection(
    kp2d: np.ndarray,
    conf: np.ndarray,
    rig: CameraRig,
    *,
    conf_thresh: float = 0.3,
) -> dict:
    """Leave-one-out reprojection QC: `kp2d (T,C,K,2)`, `conf (T,C,K)` -> report dict."""
    kp2d = np.asarray(kp2d, np.float64)
    conf = np.asarray(conf, np.float32)
    T, C, K, _ = kp2d.shape
    names = rig.cameras.names

    all_errs = []
    per_camera_px: dict[str, float] = {}
    per_camera_n = 0
    for held_out_idx, held_out_name in enumerate(names):
        other_names = [n for i, n in enumerate(names) if i != held_out_idx]
        other_rig = rig.subset(other_names)
        other_idx = [rig.cameras.index(n) for n in other_names]

        kp2d_others = kp2d[:, other_idx]  # (T,C-1,K,2)
        conf_others = conf[:, other_idx]  # (T,C-1,K)
        kp3d_loo, _ = triangulate(kp2d_others, conf_others, other_rig, conf_thresh=conf_thresh)

        uv = rig.project(kp3d_loo.reshape(-1, 3)).reshape(T, K, C, 2)
        uv_held_out = uv[:, :, held_out_idx, :]  # (T,K,2)
        obs_held_out = kp2d[:, held_out_idx]  # (T,K,2)

        held_out_conf_ok = conf[:, held_out_idx] >= conf_thresh  # (T,K)
        good = (
            held_out_conf_ok
            & np.isfinite(kp3d_loo).all(axis=-1)
            & np.isfinite(uv_held_out).all(axis=-1)
            & np.isfinite(obs_held_out).all(axis=-1)
        )
        if good.any():
            errs = np.linalg.norm(uv_held_out[good] - obs_held_out[good], axis=-1)
            per_camera_px[held_out_name] = float(np.median(errs))
            all_errs.append(errs)
            per_camera_n += int(errs.size)
        else:
            per_camera_px[held_out_name] = float("nan")

    if all_errs:
        stacked = np.concatenate(all_errs)
        median_px = float(np.median(stacked))
    else:
        median_px = float("nan")

    return {"median_px": median_px, "per_camera_px": per_camera_px, "n": per_camera_n}
