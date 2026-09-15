"""Reprojection metrics: how far the 3D answer lands from the 2D observations.

Every per-camera number here is keyed by camera NAME. A bare index cannot
point at the camera that is wrong, and this pipeline has two disagreeing camera
index spaces -- using the wrong one plots one camera's keypoints on another
camera's image, which still looks almost plausible.

The IK ratio scores FITTED and MEASURED over ONE frame set. Scored over
different sets it becomes a comparison between two populations: it moves when
coverage moves rather than when fit quality moves, while still reading as a fit
metric.
"""

from __future__ import annotations

import numpy as np

from tracking.geometry.triangulate import loo_reprojection

__all__ = ["per_camera_reproj", "per_frame_reproj", "ik_reproj_report", "loo_report"]


def _reproj_errors(kp2d, conf, xyz, rig, *, conf_thresh):
    """`(T, C, K)` pixel errors with non-confident / non-finite samples NaN."""
    kp2d = np.asarray(kp2d, np.float64)
    conf = np.asarray(conf, np.float32)
    xyz = np.asarray(xyz, np.float64)
    # `CameraRig.project` returns (..., C, 2) -- the camera axis is SECOND TO
    # LAST, not first. Verified against `geometry/rig.py`'s einsum
    # "...k,cjk->...cj". So (T, K, 3) projects to (T, K, C, 2), and comparing
    # it against kp2d's (T, C, K, 2) needs the camera axis moved to position 1.
    # Moving axis 0 instead transposes frames onto cameras: every shape check
    # still passes and every pixel number is wrong.
    proj = np.moveaxis(np.asarray(rig.project(xyz)), -2, 1)  # (T, C, K, 2)
    err = np.linalg.norm(proj - kp2d, axis=-1)
    usable = (conf >= conf_thresh) & np.isfinite(err)
    usable &= np.isfinite(xyz).all(-1)[:, None, :]
    return np.where(usable, err, np.nan)


def per_camera_reproj(kp2d, conf, kp3d_world, rig, *, conf_thresh: float = 0.3) -> dict:
    """Median pixel error overall and per camera, keyed by NAME."""
    err = _reproj_errors(kp2d, conf, kp3d_world, rig, conf_thresh=conf_thresh)
    per = {}
    for i, name in enumerate(rig.cameras.names):
        cam_err = err[:, i, :]
        per[name] = float(np.nanmedian(cam_err)) if np.isfinite(cam_err).any() else float("nan")
    return {
        "median_px": float(np.nanmedian(err)) if np.isfinite(err).any() else float("nan"),
        "per_camera_px": per,
        "n": int(np.isfinite(err).sum()),
    }


def per_frame_reproj(kp2d, conf, kp3d_world, rig, *, conf_thresh: float = 0.3) -> np.ndarray:
    """`(T,)` median pixel error per frame; NaN where a frame has no sample.

    Per frame, not per bout: `qc.posture` conditions its flag on whether THIS
    frame's pose agrees with the cameras, and a bout median cannot answer that.
    """
    err = _reproj_errors(kp2d, conf, kp3d_world, rig, conf_thresh=conf_thresh)
    with np.errstate(invalid="ignore"):
        out = np.nanmedian(err.reshape(len(err), -1), axis=1)
    return np.asarray(out, np.float64)


def ik_reproj_report(
    kp2d, conf, fitted_world, measured_world, rig, *, conf_thresh: float = 0.3
) -> dict:
    """Fitted vs measured reprojection and their ratio, over the SAME frames.

    A frame enters only if BOTH `fitted_world` and `measured_world` are finite
    there. `n_frames_compared` reports how many survived, so a ratio computed
    over a handful of frames cannot be mistaken for one computed over a bout.
    """
    fitted = np.asarray(fitted_world, np.float64)
    measured = np.asarray(measured_world, np.float64)
    frame_ok = np.isfinite(fitted).any((1, 2)) & np.isfinite(measured).any((1, 2))
    idx = np.flatnonzero(frame_ok)

    kp2d = np.asarray(kp2d, np.float64)[idx]
    conf = np.asarray(conf, np.float32)[idx]
    f_rep = per_camera_reproj(kp2d, conf, fitted[idx], rig, conf_thresh=conf_thresh)
    m_rep = per_camera_reproj(kp2d, conf, measured[idx], rig, conf_thresh=conf_thresh)
    ratio = (
        f_rep["median_px"] / m_rep["median_px"]
        if np.isfinite(m_rep["median_px"]) and m_rep["median_px"] > 0
        else float("nan")
    )
    return {
        "fitted_px": f_rep["median_px"],
        "measured_px": m_rep["median_px"],
        "ratio": float(ratio),
        "per_camera_fitted_px": f_rep["per_camera_px"],
        "per_camera_measured_px": m_rep["per_camera_px"],
        "n": int(min(f_rep["n"], m_rep["n"])),
        "n_frames_compared": int(len(idx)),
    }


def loo_report(kp2d, conf, rig, *, conf_thresh: float = 0.3) -> dict:
    """Leave-one-out reprojection, via the implementation Plan A already merged."""
    return loo_reprojection(kp2d, conf, rig, conf_thresh=conf_thresh)
