"""The five filtering stages the pipeline actually enables, as pure functions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import signal
from scipy.interpolate import PchipInterpolator

__all__ = [
    "despike_isolated_spikes",
    "detect_bone_length_outliers",
    "interpolate_nan_gaps",
    "mask_low_confidence",
    "savgol_smooth",
]


def mask_low_confidence(
    kp: np.ndarray,
    confidence: np.ndarray,
    threshold: float,
    kp_names: Sequence[str],
    exclude_patterns: Sequence[str] = (),
) -> tuple[np.ndarray, np.ndarray, int]:
    """NaN out keypoints whose confidence is below `threshold`."""
    bad_mask = np.asarray(confidence) < threshold  # (T, K)

    n_excluded = 0
    if len(kp_names) and len(exclude_patterns):
        for idx, name in enumerate(kp_names):
            for pat in exclude_patterns:
                if pat in name:
                    n_excluded += int(np.sum(bad_mask[:, idx]))
                    bad_mask[:, idx] = False
                    break

    kp_masked = kp.copy().astype(float)
    kp_masked[bad_mask] = np.nan  # broadcast over xyz
    return kp_masked, bad_mask, n_excluded


def detect_bone_length_outliers(
    kp: np.ndarray,
    edges: np.ndarray,
    threshold_std: float = 3.0,
    kp_names: Sequence[str] = (),
    exclude_patterns: Sequence[str] = (),
) -> tuple[np.ndarray, dict[int, int], int]:
    """NaN both endpoints of a bone whose length is a robust-sigma outlier."""
    kp_flagged = kp.copy()
    report: dict[int, int] = {}
    n_edges_skipped = 0

    excluded: set[int] = set()
    if len(kp_names) and len(exclude_patterns):
        for idx, name in enumerate(kp_names):
            for pat in exclude_patterns:
                if pat in name:
                    excluded.add(idx)
                    break

    edges = np.asarray(edges, dtype=int).reshape(-1, 2)
    if len(edges) == 0:
        return kp_flagged, report, n_edges_skipped

    idx_i, idx_j = edges[:, 0], edges[:, 1]
    all_lengths = np.linalg.norm(kp[:, idx_i, :] - kp[:, idx_j, :], axis=2)  # (T, E)

    for ei in range(len(edges)):
        i, j = int(edges[ei, 0]), int(edges[ei, 1])
        if i in excluded or j in excluded:
            n_edges_skipped += 1
            continue

        lengths = all_lengths[:, ei]
        valid = np.isfinite(lengths)
        if np.sum(valid) < 5:
            continue

        median_len = np.median(lengths[valid])
        mad = np.median(np.abs(lengths[valid] - median_len))
        sigma = mad / 0.6745  # MAD -> sigma for a Gaussian
        if sigma < 1e-8:
            continue

        outlier = (np.abs(lengths - median_len) > threshold_std * sigma) & valid
        n = int(np.sum(outlier))
        if n > 0:
            kp_flagged[outlier, i, :] = np.nan
            kp_flagged[outlier, j, :] = np.nan
            report[ei] = n

    return kp_flagged, report, n_edges_skipped


def despike_isolated_spikes(
    arr: np.ndarray,
    threshold_factor: float = 10.0,
    max_iterations: int = 1,
) -> tuple[np.ndarray, int]:
    """Replace single-frame velocity reversals with the mean of their neighbours."""
    if arr.ndim == 0 or arr.shape[0] < 3:
        return arr.copy(), 0

    T = arr.shape[0]
    orig_shape = arr.shape
    flat = arr.reshape(T, -1).astype(np.float64, copy=True)
    total_fixed = 0

    for c in range(flat.shape[1]):
        x = flat[:, c]

        v0 = np.diff(x)
        abs_v0 = np.abs(v0)
        finite = np.isfinite(abs_v0)
        if finite.sum() < 3:
            continue
        med_v = float(np.median(abs_v0[finite]))
        if med_v < 1e-12:
            continue
        thresh = threshold_factor * med_v

        for _iteration in range(max_iterations):
            v = np.diff(x)
            big = np.abs(v) > thresh
            reversal = (v[:-1] * v[1:]) < 0
            spike = big[:-1] & big[1:] & reversal

            idx = np.nonzero(spike)[0] + 1
            if idx.size == 0:
                break

            n_this = 0
            for t in idx:
                left, right = x[t - 1], x[t + 1]
                if np.isfinite(left) and np.isfinite(right):
                    x[t] = (left + right) / 2.0
                    n_this += 1
            total_fixed += n_this
            if n_this == 0:
                break

    return flat.reshape(orig_shape), total_fixed


def _nan_interp_1d(
    vals: np.ndarray,
    use_spline: bool = True,
    max_edge_extrap: int = 0,
    edge_fit_window: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill NaN gaps in one 1-D signal; returns `(filled, phantom_mask)`."""
    T = len(vals)
    phantom_mask = np.zeros(T, dtype=bool)
    nans = np.isnan(vals)
    if not np.any(nans):
        return vals, phantom_mask

    finite_idx = np.where(~nans)[0]
    if finite_idx.size == 0:
        phantom_mask[:] = True
        return vals, phantom_mask

    # Record the original edge-NaN positions as phantom even if extrapolation
    # later fills them.
    x_min, x_max = finite_idx[0], finite_idx[-1]
    if x_min > 0:
        phantom_mask[:x_min] = True
    if x_max < T - 1:
        phantom_mask[x_max + 1 :] = True

    if np.mean(~nans) < 0.5 or np.sum(~nans) < 4:
        # Too sparse -- leave as NaN rather than interpolate wildly.
        return vals, phantom_mask

    x_good = finite_idx
    v_good = vals[~nans]
    x_bad = np.where(nans)[0]
    out = vals.copy()

    x_interior = x_bad[(x_bad >= x_min) & (x_bad <= x_max)]
    if len(x_interior) > 0:
        if use_spline:
            try:
                pchip = PchipInterpolator(x_good, v_good, extrapolate=False)
                out[x_interior] = pchip(x_interior)
            except Exception:
                out[x_interior] = np.interp(x_interior, x_good, v_good)
        else:
            out[x_interior] = np.interp(x_interior, x_good, v_good)

    if max_edge_extrap > 0:
        fit_n = min(edge_fit_window, len(x_good))

        # Leading edge: fill the `max_edge_extrap` frames closest to x_min.
        if x_min > 0:
            n_fill = int(min(x_min, max_edge_extrap))
            fill_idx = np.arange(x_min - n_fill, x_min)
            if fit_n >= 2:
                slope, intercept = np.polyfit(x_good[:fit_n].astype(float), v_good[:fit_n], 1)
                out[fill_idx] = slope * fill_idx.astype(float) + intercept
            elif fit_n == 1:
                out[fill_idx] = v_good[0]

        # Trailing edge: fill the `max_edge_extrap` frames closest to x_max.
        if x_max < T - 1:
            n_fill = int(min(T - 1 - x_max, max_edge_extrap))
            fill_idx = np.arange(x_max + 1, x_max + 1 + n_fill)
            if fit_n >= 2:
                slope, intercept = np.polyfit(x_good[-fit_n:].astype(float), v_good[-fit_n:], 1)
                out[fill_idx] = slope * fill_idx.astype(float) + intercept
            elif fit_n == 1:
                out[fill_idx] = v_good[-1]

    return out, phantom_mask


def interpolate_nan_gaps(
    kp: np.ndarray,
    use_spline: bool = True,
    max_edge_extrap_frames: int = 0,
    edge_fit_window: int = 5,
) -> tuple[np.ndarray, np.ndarray, int]:
    """`_nan_interp_1d` over every coordinate column of a `(T, K, 3)` array."""
    T, K, _ = kp.shape
    kp_flat = kp.reshape(T, -1).copy()
    phantom_flat = np.zeros(kp_flat.shape, dtype=bool)
    nan_counts = np.isnan(kp_flat).sum(axis=0)
    total_filled = 0

    for c in np.where(nan_counts > 0)[0]:
        n_nan_before = int(nan_counts[c])
        filled, phantom = _nan_interp_1d(
            kp_flat[:, c],
            use_spline=use_spline,
            max_edge_extrap=max_edge_extrap_frames,
            edge_fit_window=edge_fit_window,
        )
        kp_flat[:, c] = filled
        phantom_flat[:, c] = phantom
        total_filled += n_nan_before - int(np.sum(np.isnan(filled)))

    kp_out = kp_flat.reshape(T, K, 3)
    edge_nan_mask = np.any(phantom_flat.reshape(T, K, 3), axis=2)
    return kp_out, edge_nan_mask, total_filled


def savgol_smooth(kp: np.ndarray, window_length: int = 11, polyorder: int = 3) -> np.ndarray:
    """Savitzky-Golay smoothing along time, skipping columns that still hold NaN."""
    T, K, _ = kp.shape
    if window_length % 2 == 0:
        window_length += 1
    window_length = min(window_length, T if T % 2 != 0 else T - 1)
    if window_length <= polyorder:
        return kp

    kp_out = kp.copy()
    for n in range(K):
        for d in range(3):
            vals = kp_out[:, n, d]
            if np.any(np.isnan(vals)):
                continue  # a NaN anywhere would poison the whole column
            kp_out[:, n, d] = signal.savgol_filter(vals, window_length, polyorder)
    return kp_out
