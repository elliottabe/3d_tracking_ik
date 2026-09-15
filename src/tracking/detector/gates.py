"""MVQ bout gates: coarse-pass signals -> gated frame ranges -> a bout table.

Two gates decide whether a coarse frame is "in a bout":

  trackability_ok(t) -- both flies exist (`exist >= exist_min`) with enough
      cameras seeing them (`n_valid_cams >= min_cams`), AND the flies are
      separable: median 2D centroid separation (`sep2d_med`, computed
      upstream from the reprojected 3D centroid, not from masks) is not so
      small the two identities have likely merged.

  behaviour_ok(t) -- the male's wing angle clears `wing_angle_min` degrees
      (courtship wing extension, the model's own wing-vs-body-axis angle,
      no mask needed) OR the pair's true 3D separation (`sep3d`) is within
      `proximity_max_units` world units (close following).

`in_bout(t) = trackability_ok(t) AND behaviour_ok(t)`.

A single-fly recording (`num_animals < 2`, `coarse.coarse_pass(...,
num_animals=1)`) has no second fly to be separable from or close to, so
neither courtship signal is defined: `apply_gates` makes `separable` and
`behaviour_ok` both True rather than raising or indexing out of bounds --
a single-fly bout table is driven by trackability alone, the only question
that makes sense without a second fly.

Contiguous `in_bout` runs, after bridging small gaps and dropping too-short
ones (`segment_runs`), become bout windows; `coarse_to_real` maps a
window's coarse-frame indices back to real (video) frame numbers, and
`gates_to_bouts` writes the result through `tracking.io.bouts.
write_bouts_csv` with `source="gates"` -- the same `BoutSpec` a
hand-supplied `bout_summary.csv` produces (`read_bout_summary`,
`source="summary"`), so nothing downstream of bout selection needs to know
which route produced the table.

**Index convention.** `segment_runs` returns `(start, end)` with `end`
INCLUSIVE (`n_frames = end - start + 1`), matching `io.bouts.BoutSpec`'s
inclusive `end_frame` so a gated run needs no off-by-one translation on its
way into a `BoutSpec`. `coarse_to_real` takes the same
inclusive `(s, e)` and reads `coarse_frame[s]`/`coarse_frame[e]` directly.

**Keypoint lookups.** This module never indexes a keypoint by position.
Its own signals (`wing_angle_deg`, `exist`, `n_valid_cams`, `sep2d_med`,
`sep3d`) already carry no keypoint axis -- that reduction (by NAME, against
a keypoint `Order`) happens upstream, in the coarse pass's
`coarse_features`. `kp_order` is threaded through `compute_gate_signals`/
`gates_to_bouts` anyway, so a `tracks_path` is loaded via
`artifacts.load_npz(tracks_path, kp=kp_order, ...)`, which checks the
coarse-track file's order stamp before any gate trusts an array that could
have been written under a different keypoint order -- the same trap
CLAUDE.md's keypoint/camera-order notes describe for every other stage.

**OPEN ITEM: bout gates v2** -- a rest-relative wing feature (gate on wing
angle relative to that fly's own resting baseline rather than an absolute
threshold), a male-behaviour redefinition, and no separate proximity term --
is unfinished. These are the v1 gates: `rolling_median` and `baseline_window`
exist for interface stability with that future work, but v1 baselines nothing
and `wing_angle_min` is an absolute degree threshold. The `bout_summary.csv` (hand-supplied)
route does not depend on this module at all.
"""

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

    Here for a possible future rest-relative feature (see the module
    docstring's OPEN ITEM); the v1 gates below do not call it.
    """
    return (
        pd.Series(np.asarray(x, dtype=float))
        .rolling(window, center=True, min_periods=min_periods)
        .median()
        .to_numpy()
    )


def compute_gate_signals(tracks, kp_order: Order, *, baseline_window=BASELINE_WINDOW) -> dict:
    """Pull the MVQ gate signals out of a loaded coarse-tracks dict.

    `tracks` is the dict `tracking.io.artifacts.load_npz` returns for a
    coarse-tracks file (its order stamp already checked against `kp_order`
    by that call -- see `gates_to_bouts`, which is how a `tracks_path`
    should be turned into `tracks`). Expects `exist` (F,T), `n_valid_cams`
    (F,T), `sep2d_med` (T,), `wing_angle_deg` (F,T) and `sep3d` (T,, all-NaN
    or absent for a single-fly recording).

    `kp_order` is not indexed here (see the module docstring's Keypoint
    lookups note); `baseline_window` is accepted for signature stability
    only (see the OPEN ITEM) and does not affect the returned signals.
    """
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
    """MVQ trackability + behaviour gates -> one bool per coarse frame, `in_bout`.

    See the module docstring for the two gates' definitions and the
    single-fly degenerate case. NaNs in any input signal make that
    frame/fly fail the gate reading them (never silently pass).
    """
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
    """Contiguous True runs of `mask` (coarse-frame index space).

    Bridges gaps of at most `max_gap` False frames and drops runs shorter
    than `min_duration` frames. Returns `[(start, end)]` with `end`
    INCLUSIVE (`n_frames = end - start + 1`) -- see the module docstring.
    """
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

    Raises IndexError on an out-of-range index rather than silently
    wrapping (Python's negative-index semantics) or clamping.
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
    """Gate a coarse-tracks file into a bout table.

    Loads `tracks_path` via `artifacts.load_npz` (checking its order stamp
    against `kp_order`), applies the gates, segments the result into
    windows, maps them to real frame numbers, and writes them to `out_csv`
    via `write_bouts_csv` with `source="gates"` -- the same `BoutSpec` a
    hand-supplied `bout_summary.csv` produces.
    """
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
