"""Rigid invariants: quantities anatomy forbids from changing."""

from __future__ import annotations

import numpy as np

from tracking.io.names import Order, as_order
from tracking.preprocess.scale import rigid_segment_pairs

__all__ = ["EYE_PAIR", "COLLAPSE_FRAC", "segment_lengths", "invariant_report"]

EYE_PAIR = ("EyeL", "EyeR")

COLLAPSE_FRAC = 0.2


def segment_lengths(kp3d, kp_order, pairs) -> dict[str, np.ndarray]:
    """`{"A--B": (T,) distances}` for each NAME pair present in `kp_order`."""
    order = as_order(kp_order)
    kp3d = np.asarray(kp3d, np.float64)
    out: dict[str, np.ndarray] = {}
    for a, b in pairs:
        if a not in order or b not in order:
            continue
        d = kp3d[:, order.index(a)] - kp3d[:, order.index(b)]
        out[f"{a}--{b}"] = np.linalg.norm(d, axis=-1)
    return out


def invariant_report(kp3d, kp_order, *, include_thorax: bool = False) -> dict:
    """Per-invariant median length, CV and collapse flag, keyed by NAME."""
    order: Order = as_order(kp_order)
    pairs = list(rigid_segment_pairs(order, include_thorax=include_thorax))
    kinds = {f"{a}--{b}": "rigid_segment_length" for a, b in pairs}
    if EYE_PAIR[0] in order and EYE_PAIR[1] in order:
        pairs.append(EYE_PAIR)
        kinds[f"{EYE_PAIR[0]}--{EYE_PAIR[1]}"] = "eye_spacing"

    lengths = segment_lengths(kp3d, order, pairs)
    medians = {k: float(np.nanmedian(v)) for k, v in lengths.items()}
    finite = [m for m in medians.values() if np.isfinite(m) and m > 0]
    reference = float(np.median(finite)) if finite else float("nan")

    rows: dict[str, dict] = {}
    for key, values in lengths.items():
        finite_vals = values[np.isfinite(values)]
        med = medians[key]
        cv = (
            float(np.std(finite_vals) / np.mean(finite_vals))
            if finite_vals.size and np.mean(finite_vals) > 0
            else float("nan")
        )
        rows[key] = {
            "invariant": kinds[key],
            "median": med,
            "cv": cv,
            "n": int(finite_vals.size),
            "collapsed": bool(
                np.isfinite(med) and np.isfinite(reference) and med < COLLAPSE_FRAC * reference
            ),
        }

    scored = {k: v["cv"] for k, v in rows.items() if np.isfinite(v["cv"])}
    return {
        "invariants": rows,
        "worst_cv_invariant": max(scored, key=scored.get) if scored else None,
        "n_collapsed": int(sum(1 for v in rows.values() if v["collapsed"])),
    }
