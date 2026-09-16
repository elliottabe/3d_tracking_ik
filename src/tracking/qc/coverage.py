"""Coverage: how much of the bout was actually OBSERVED."""

from __future__ import annotations

import numpy as np

from tracking.io.names import as_order

__all__ = ["OBSERVED_CONF", "gap_runs", "coverage_report"]

OBSERVED_CONF = 0.3


def gap_runs(present: np.ndarray) -> np.ndarray:
    """Lengths of each maximal run of consecutive absent frames in `(T,)`."""
    absent = ~np.asarray(present, bool)
    if not absent.any():
        return np.zeros(0, np.int64)
    padded = np.concatenate(([False], absent, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return (ends - starts).astype(np.int64)


def coverage_report(kp3d_raw, conf3d, kp_order, *, conf_thresh: float = OBSERVED_CONF) -> dict:
    """Missing fraction and gap statistics, per keypoint, keyed by NAME."""
    order = as_order(kp_order)
    kp3d = np.asarray(kp3d_raw, np.float64)
    conf = np.asarray(conf3d, np.float32)
    observed = (conf >= conf_thresh) & np.isfinite(kp3d).all(-1)
    n_frames = int(observed.shape[0])

    per = {
        name: float(1.0 - observed[:, i].mean()) if n_frames else float("nan")
        for i, name in enumerate(order.names)
    }
    runs = (
        np.concatenate([gap_runs(observed[:, i]) for i in range(observed.shape[1])])
        if observed.size
        else np.zeros(0, np.int64)
    )

    worst = max(per, key=per.get) if per else None
    if worst is not None and not per[worst] > 0.0:
        worst = None

    return {
        "missing_fraction": float(1.0 - observed.mean()) if observed.size else float("nan"),
        "per_keypoint_missing": per,
        "gap_runs": {
            "n": int(runs.size),
            "max": int(runs.max()) if runs.size else 0,
            "median": float(np.median(runs)) if runs.size else 0.0,
        },
        "worst_keypoint": worst,
        "n_frames": n_frames,
    }
