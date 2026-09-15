"""Every QC check for one bout-fly, merged into the `qc.json` the scorecard reads.

Structural defects RAISE; quality numbers are recorded. The line is whether the
artifact is readable and self-consistent, not whether it is good: a collapsed
keypoint pair is anatomy that cannot exist, while a 3 px reprojection is a
number a human weighs. Failing on quality would make the pipeline stop on the
female -- the fly it exists to measure.
"""

from __future__ import annotations

import json
import os

import numpy as np

from tracking.qc.coverage import coverage_report
from tracking.qc.invariants import invariant_report
from tracking.qc.posture import posture_report
from tracking.qc.reproj import ik_reproj_report, loo_report, per_frame_reproj

__all__ = ["QCStructuralError", "bout_qc", "write_qc_json"]


class QCStructuralError(Exception):
    """The artifact is not self-consistent -- not merely poor."""


def bout_qc(
    *,
    kp2d,
    conf2d,
    kp3d_raw,
    conf3d,
    fitted_world,
    kp_order,
    rig,
    floor,
    sex: str,
    conf_thresh: float = 0.3,
) -> dict:
    """Run every check for one bout-fly and merge the reports."""
    kp3d_raw = np.asarray(kp3d_raw, np.float64)
    fitted_world = np.asarray(fitted_world, np.float64)
    if not np.isfinite(fitted_world).any():
        raise QCStructuralError(
            "every fitted frame is non-finite; this bout-fly has no pose to score"
        )

    inv = invariant_report(kp3d_raw, kp_order)
    if inv["n_collapsed"]:
        collapsed = [k for k, v in inv["invariants"].items() if v["collapsed"]]
        raise QCStructuralError(
            f"collapsed rigid segment(s) {collapsed}: a rigid segment's length "
            f"cannot be near zero, and a collapsed pair is SMOOTHER than a real "
            f"landmark -- every jitter and confidence metric will rate it as good"
        )

    rep = ik_reproj_report(kp2d, conf2d, fitted_world, kp3d_raw, rig, conf_thresh=conf_thresh)
    resid = per_frame_reproj(kp2d, conf2d, fitted_world, rig, conf_thresh=conf_thresh)
    return {
        "reproj": rep,
        "loo": loo_report(kp2d, conf2d, rig, conf_thresh=conf_thresh),
        "invariants": inv,
        "coverage": coverage_report(kp3d_raw, conf3d, kp_order, conf_thresh=conf_thresh),
        "posture": posture_report(fitted_world, kp_order, floor, residual_px=resid, sex=sex),
        "structural_ok": True,
    }


def write_qc_json(path, report: dict) -> None:
    """Write `qc.json` atomically."""
    path = str(path)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    os.replace(tmp, path)
