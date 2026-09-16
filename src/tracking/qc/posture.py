"""Body posture, reported per sex, flagged only when it also misfits."""

from __future__ import annotations

import numpy as np

from tracking.detector.coarse import FloorPlane, fit_floor
from tracking.io.names import as_order

__all__ = [
    "PITCH_DEG",
    "RESIDUAL_MULT",
    "AXIS_PAIR",
    "fly_sex_label",
    "fit_recording_floor",
    "body_axis_pitch_deg",
    "posture_report",
]

PITCH_DEG = 45.0
RESIDUAL_MULT = 3.0
AXIS_PAIR = ("Scutellum", "Antenna_Base")  # posterior, anterior -- NAMES


def fly_sex_label(sex_json: dict, fly: int) -> str:
    """`"female"` / `"male"` / `"unknown"`, read from `sex.json`."""
    if str(sex_json.get("identity")) != "sex":
        return "unknown"
    male = sex_json.get("male_fly")
    if male is None:
        return "unknown"
    return "male" if int(fly) == int(male) else "female"


def _body_centroids(kp3d_world: np.ndarray) -> np.ndarray:
    """`(T, 3)` per-frame mean over finite keypoints."""
    kp = np.asarray(kp3d_world, np.float64)
    with np.errstate(invalid="ignore"):
        return np.nanmean(kp, axis=1)


def fit_recording_floor(bout_kp3d, *, n_fit: int = 2000, up_hint=None) -> FloorPlane:
    """One arena floor plane from several bouts' world-unit keypoints."""
    centroids = np.concatenate([_body_centroids(b) for b in bout_kp3d], axis=0)
    finite = centroids[np.isfinite(centroids).all(-1)]
    if len(finite) < 3:
        raise ValueError(
            f"only {len(finite)} finite body centroids across these bouts; a floor "
            f"plane cannot be fit and must not be guessed"
        )
    return fit_floor(finite, n_fit=min(int(n_fit), len(finite)), up_hint=up_hint)


def body_axis_pitch_deg(kp3d_world, kp_order, floor: FloorPlane, *, pair=AXIS_PAIR):
    """`(T,)` angle in degrees between the body axis and the floor plane."""
    order = as_order(kp_order)
    kp = np.asarray(kp3d_world, np.float64)
    post, ante = (order.index(n) for n in pair)
    axis = kp[:, ante] - kp[:, post]
    norm = np.linalg.norm(axis, axis=-1)
    up = np.asarray(floor.normal, np.float64)
    up = up / np.linalg.norm(up)
    with np.errstate(invalid="ignore", divide="ignore"):
        sin_pitch = (axis @ up) / np.where(norm == 0.0, np.nan, norm)
    return np.degrees(np.arcsin(np.clip(sin_pitch, -1.0, 1.0)))


def _stats(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"median": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def posture_report(
    kp3d_world,
    kp_order,
    floor: FloorPlane,
    *,
    residual_px,
    sex: str,
    pitch_deg: float = PITCH_DEG,
    residual_mult: float = RESIDUAL_MULT,
) -> dict:
    """Pitch and height distributions for one bout-fly, plus the flagged frames."""
    pitch = body_axis_pitch_deg(kp3d_world, kp_order, floor)
    height = floor.height_of(_body_centroids(kp3d_world))
    resid = np.asarray(residual_px, np.float64)

    finite_resid = resid[np.isfinite(resid)]
    baseline = float(np.median(finite_resid)) if finite_resid.size else float("nan")
    resid_thresh = residual_mult * baseline

    pitched = np.isfinite(pitch) & (pitch > pitch_deg)
    misfit = np.isfinite(resid) & (resid > resid_thresh)
    flagged = pitched & misfit

    return {
        "sex": str(sex),
        "pitch_deg": _stats(pitch),
        "height": _stats(height),
        "n_frames": int(len(pitch)),
        "n_pitched": int(np.count_nonzero(pitched)),
        "n_flagged": int(np.count_nonzero(flagged)),
        "flagged_frames": [int(i) for i in np.flatnonzero(flagged)],
        "pitch_threshold_deg": float(pitch_deg),
        "residual_threshold_px": float(resid_thresh),
        "residual_baseline_px": baseline,
        "floor": floor.as_dict(),
    }
