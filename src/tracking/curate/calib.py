"""Calibration identity: fingerprint a calibration dir, label recordings by group."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import numpy as np

from tracking.geometry.rig import CameraRig

_ROUND = 6


def calib_fingerprint(calib_dir: str | Path) -> str:
    """16-hex digest of a calibration's rounded coefficients.

    Rounded numerics, not file bytes: the same calibration written with fewer
    decimal places must fingerprint equal.

    >>> calib_fingerprint("calibrations/A")            # doctest: +SKIP
    '3f1a9c2b7d4e5061'
    """
    rig = CameraRig.from_calib_dir(calib_dir)
    payload = repr(np.round(rig.matrices_f64, _ROUND).tolist())
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def group_calibrations(dirs: dict[str, str | Path]) -> dict[str, str]:
    """Map recording name -> group label; the largest group is 'A'.

    Raises if the directories disagree on camera names.

    >>> group_calibrations({"r1": "calib/x", "r2": "calib/y"})    # doctest: +SKIP
    {'r1': 'A', 'r2': 'A'}
    """
    cameras = {rec: CameraRig.from_calib_dir(d).cameras.names for rec, d in dirs.items()}
    distinct = set(cameras.values())
    if len(distinct) > 1:
        raise ValueError(f"calibration dirs disagree on camera names: {sorted(distinct)}")
    fp = {rec: calib_fingerprint(d) for rec, d in dirs.items()}
    counts = Counter(fp.values())
    ordered = sorted(counts, key=lambda h: (-counts[h], min(r for r in fp if fp[r] == h)))
    label = {h: chr(ord("A") + i) for i, h in enumerate(ordered)}
    return {rec: label[h] for rec, h in fp.items()}
