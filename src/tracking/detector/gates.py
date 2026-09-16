"""MVQ bout gates: coarse-pass signals -> gated frame ranges -> a bout table."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tracking.io.artifacts import load_npz
from tracking.io.bouts import BoutSpec, write_bouts_csv
from tracking.io.names import Order

MIN_CAMS = 3  # hard floor on cameras seeing a fly for it to count as tracked
SEP_MIN_PX = 15.0  # below this median 2D centroid separation, identities likely merged
BASELINE_WINDOW = 3000  # coarse frames; unused by v1 gating, see the OPEN ITEM above
EXIST_MIN = 0.5  # typed-slot existence threshold (coarse pass's own TRACKABLE_EXIST)
WING_ANGLE_MIN_DEFAULT = 30.0  # degrees; judgment call, not a measured constant
PROXIMITY_MAX_UNITS_DEFAULT = 30.0  # world units; judgment call, not a measured constant
MALE_SLOT = 1  # fly1 = male, by this pipeline's sexing canonicalization
DEFAULT_MIN_DURATION = 3  # coarse frames
DEFAULT_MAX_GAP = 2  # coarse frames to bridge


def rolling_median(x, window, min_periods=50):
    """Centered rolling median of a 1D signal; NaN wherever fewer than
    `min_periods` finite samples fall in the window.
    """
    return (
        pd.Series(np.asarray(x, dtype=float))
        .rolling(window, center=True, min_periods=min_periods)
        .median()
        .to_numpy()
    )


def compute_gate_signals(tracks, kp_order: Order, *, baseline_window=BASELINE_WINDOW) -> dict:
    """Pull the MVQ gate signals out of a loaded coarse-tracks dict."""
    exist = np.asarray(tracks["exist"], dtype=float)
    n_valid_cams = np.asarray(tracks["n_valid_cams"], dtype=float)
    sep2d_med = np.asarray(tracks["sep2d_med"], dtype=float)
    wing_angle_deg = np.asarray(tracks["wing_angle_deg"], dtype=float)
    sep3d_raw = tracks.get("sep3d")
    sep3d_arr = np.asarray(sep3d_raw, dtype=float) if sep3d_raw is not None else np.array([])
    sep3d = sep3d_arr if sep3d_arr.size else None

    return dict(
        exist=exist,
        n_valid_cams=n_valid_cams,
        sep2d_med=sep2d_med,
        wing_angle_deg=wing_angle_deg,
        sep3d=sep3d,
        num_animals=exist.shape[0],
        kp_order=kp_order,
        baseline_window=baseline_window,
    )


def apply_gates(
    signals: dict,
    *,
    wing_angle_min: float | None = None,
    proximity_max_units: float | None = None,
    exist_min: float = EXIST_MIN,
    min_cams: int = MIN_CAMS,
    sep_min_px: float = SEP_MIN_PX,
    male_slot: int = MALE_SLOT,
) -> np.ndarray:
    """MVQ trackability + behaviour gates -> one bool per coarse frame, `in_bout`."""
    exist = signals["exist"]
    n_valid_cams = signals["n_valid_cams"]
    sep2d_med = signals["sep2d_med"]
    wing_angle_deg = signals["wing_angle_deg"]
    sep3d = signals["sep3d"]
    num_animals, n_frames = exist.shape
    single_fly = num_animals < 2

    wing_angle_min = WING_ANGLE_MIN_DEFAULT if wing_angle_min is None else wing_angle_min
    proximity_max_units = (
        PROXIMITY_MAX_UNITS_DEFAULT if proximity_max_units is None else proximity_max_units
    )

    def _false_to_nan(arr: np.ndarray) -> np.ndarray:
        return np.nan_to_num(arr.astype(float), nan=0.0).astype(bool)

    if single_fly:
        # Nothing to be "not separable" FROM -- see the module docstring.
        separable = np.ones(n_frames, dtype=bool)
    else:
        with np.errstate(invalid="ignore"):
            separable = _false_to_nan(sep2d_med >= sep_min_px)

    with np.errstate(invalid="ignore"):
        per_fly_trackable = _false_to_nan((exist >= exist_min) & (n_valid_cams >= min_cams))
    trackable_both = per_fly_trackable.all(axis=0) if num_animals >= 2 else per_fly_trackable[0]

    trackability_ok = trackable_both & separable

    if single_fly:
        # Neither courtship signal is defined without a second fly (see the
        # module docstring); fall back to trackability alone.
        behaviour_ok = np.ones(n_frames, dtype=bool)
    else:
        if wing_angle_deg.shape[0] > male_slot:
            with np.errstate(invalid="ignore"):
                wing_extension = _false_to_nan(wing_angle_deg[male_slot] >= wing_angle_min)
        else:
            wing_extension = np.zeros(n_frames, dtype=bool)
        if sep3d is not None:
            with np.errstate(invalid="ignore"):
                close_proximity = _false_to_nan(sep3d <= proximity_max_units)
        else:
            close_proximity = np.zeros(n_frames, dtype=bool)
        behaviour_ok = wing_extension | close_proximity

    return trackability_ok & behaviour_ok


def segment_runs(mask, *, min_duration: int, max_gap: int) -> list[tuple[int, int]]:
    """Contiguous True runs of `mask` (coarse-frame index space)."""
    mask = np.asarray(mask, dtype=bool)
    idx = np.nonzero(mask)[0]
    if len(idx) == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = int(idx[0])
    prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - prev - 1 > max_gap:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    return [(s, e) for s, e in runs if (e - s + 1) >= min_duration]


def coarse_to_real(coarse_frame, s: int, e: int) -> tuple[int, int]:
    """Map a `segment_runs` window `[s, e]` (coarse-frame indices, both
    INCLUSIVE) to `(start_frame, end_frame)` real (video) frame numbers via
    `coarse_frame` (T,), the coarse pass's own real-frame-number array.
    """
    n = len(coarse_frame)
    if not (0 <= s < n) or not (0 <= e < n):
        raise IndexError(f"coarse index range [{s},{e}] out of bounds for {n} coarse frames")
    return int(coarse_frame[s]), int(coarse_frame[e])


def gates_to_bouts(
    tracks_path,
    out_csv,
    *,
    kp_order: Order,
    min_duration: int = DEFAULT_MIN_DURATION,
    max_gap: int = DEFAULT_MAX_GAP,
    baseline_window: int = BASELINE_WINDOW,
    wing_angle_min: float | None = None,
    proximity_max_units: float | None = None,
    exist_min: float = EXIST_MIN,
    min_cams: int = MIN_CAMS,
    sep_min_px: float = SEP_MIN_PX,
    male_slot: int = MALE_SLOT,
) -> list[BoutSpec]:
    """Gate a coarse-tracks file into a bout table."""
    tracks = load_npz(tracks_path, kp=kp_order, cams=None)
    signals = compute_gate_signals(tracks, kp_order, baseline_window=baseline_window)
    in_bout = apply_gates(
        signals,
        wing_angle_min=wing_angle_min,
        proximity_max_units=proximity_max_units,
        exist_min=exist_min,
        min_cams=min_cams,
        sep_min_px=sep_min_px,
        male_slot=male_slot,
    )
    runs = segment_runs(in_bout, min_duration=min_duration, max_gap=max_gap)
    coarse_frame = np.asarray(tracks["coarse_frame"])
    windows = [coarse_to_real(coarse_frame, s, e) for s, e in runs]
    bouts = [
        BoutSpec(idx=i, start_frame=sf, end_frame=ef, n_frames=ef - sf + 1, source="gates")
        for i, (sf, ef) in enumerate(windows, start=1)
    ]
    write_bouts_csv(out_csv, bouts)
    return bouts
