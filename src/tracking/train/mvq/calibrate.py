"""Temperature scaling for MVQ's existence/visibility logits.

New module (spec 7.3): one scalar temperature each, fit by minimising NLL
on held-out logits. `MVQRunner` divides logits by these at inference
(`detector/mvq/runner.py:250-252,327-328`); `1.0` must be the identity.
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.optimize import minimize_scalar

LOG_T_BOUNDS = (-5.0, 5.0)
# Generous, and in log-space: on separable logits the NLL reaches 0 and goes
# FLAT well before the bound, so the search stops in the plateau (~-4.66) and
# an exact-endpoint test would miss the very case it is there to catch. Any
# |log T| > 4.5 (T outside [0.011, 90]) is degenerate whatever the reason.
_BOUND_ATOL = 0.5


def fit_temperature(logits, targets, label="") -> float:
    """Scalar `T` minimising mean binary-cross-entropy of
    `sigmoid(logits / T)` against `targets`, searched in log-space over
    `T in [exp(-5), exp(5)]`.

    A fit that fails to converge, or that saturates against either bound,
    WARNS: the search is bounded, so a degenerate fit returns a plausible
    scalar that would then divide `MVQRunner`'s logits unremarked.
    """
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    def nll(log_t):
        z = logits / np.exp(log_t)
        return np.mean(np.maximum(z, 0) - z * targets + np.log1p(np.exp(-np.abs(z))))

    result = minimize_scalar(nll, bounds=LOG_T_BOUNDS, method="bounded")
    who = f"[mvq] {label or 'temperature'} fit"
    if not result.success:
        print(f"{who} did NOT converge: {getattr(result, 'message', '')!r}", flush=True)
    for bound in LOG_T_BOUNDS:
        if abs(result.x - bound) < _BOUND_ATOL:
            print(
                f"{who} SATURATED against the log-T bound {bound:+g}: log T={result.x:.4f} "
                f"(T={np.exp(result.x):.4g}, n={logits.size}, positives={int(targets.sum())}) "
                f"-- this temperature is the edge of the search range, not a fit",
                flush=True,
            )
    return float(np.exp(result.x))


def write_calibration(run_dir, exist_temperature, vis_temperature, n_val=None):
    """Merge `meta["calibration"] = {exist_temperature, vis_temperature, n_val}`
    into `<run_dir>/mvq_run.json`, preserving any existing content.

    `n_val` (the number of validation windows the fit saw) is what makes a
    degenerate fit diagnosable after the fact; two bare scalars are not.
    """
    path = os.path.join(str(run_dir), "mvq_run.json")
    meta = {}
    if os.path.exists(path):
        with open(path) as f:
            meta = json.load(f)
    meta["calibration"] = {
        "exist_temperature": float(exist_temperature),
        "vis_temperature": float(vis_temperature),
        "n_val": None if n_val is None else int(n_val),
    }
    with open(path, "w") as f:
        json.dump(meta, f, indent=1)
