"""Coverage: how much of the bout was actually OBSERVED.

Measured from the raw detector output and its confidence, never from a
filtered or gap-filled array. `preprocess.gaps.fill_short_gaps` and the
keypoint filter both leave a never-observed frame finite and plausible, so a
coverage number computed from values alone reports the filter's diligence
instead of the detector's. The mask-free pass's whole advantage on the
reference bout is a coverage number -- female missing 0.007 against 0.154
masked -- and counting an interpolated frame as present erases exactly that.

A frame-keypoint is OBSERVED when its confidence clears `conf_thresh` AND its
position is finite. Either condition failing makes it missing.
"""

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

    # `None` when nothing is missing, rather than whichever keypoint `max`
    # happens to return first. A perfect bout naming a "worst keypoint" at
    # 0.0 missing puts a keypoint name in a scorecard column that reads as a
    # problem; the honest answer there is that there isn't one.
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
