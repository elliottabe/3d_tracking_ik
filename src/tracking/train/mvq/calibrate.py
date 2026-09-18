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


def fit_temperature(logits, targets) -> float:
    """Scalar `T` minimising mean binary-cross-entropy of
    `sigmoid(logits / T)` against `targets`, searched in log-space over
    `T in [exp(-5), exp(5)]`.
    """
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    def nll(log_t):
        z = logits / np.exp(log_t)
        return np.mean(np.maximum(z, 0) - z * targets + np.log1p(np.exp(-np.abs(z))))

    result = minimize_scalar(nll, bounds=(-5.0, 5.0), method="bounded")
    return float(np.exp(result.x))


def write_calibration(run_dir, exist_temperature, vis_temperature):
    """Merge `meta["calibration"] = {exist_temperature, vis_temperature}`
    into `<run_dir>/mvq_run.json`, preserving any existing content."""
    path = os.path.join(str(run_dir), "mvq_run.json")
    meta = {}
    if os.path.exists(path):
        with open(path) as f:
            meta = json.load(f)
    meta["calibration"] = {
        "exist_temperature": float(exist_temperature),
        "vis_temperature": float(vis_temperature),
    }
    with open(path, "w") as f:
        json.dump(meta, f, indent=1)
