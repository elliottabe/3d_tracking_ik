"""The Stage-B gate signature stamped into an mvq-lifted kp3d.npz."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import warnings

from tracking.detector.mvq.checkpoint import checkpoint_sha256, resolved_step


def _identity_sha(checkpoint, step):
    """The checkpoint's own on-disk sha256 when it is a real, readable orbax
    checkpoint; a hash of the path string itself otherwise.
    """
    try:
        return checkpoint_sha256(checkpoint, step)
    except (FileNotFoundError, OSError) as exc:
        msg = (
            f"gate_string: {checkpoint!r} carries no readable orbax checkpoint metadata "
            f"({exc}); falling back to a hash of the path string. This gate signature will "
            f"NOT detect a weight change at the same path -- it is a checkpoint identity in "
            f"name only."
        )
        print(f"[mvq] {msg}", file=sys.stderr, flush=True)
        warnings.warn(msg, stacklevel=2)
        return hashlib.sha256(str(checkpoint).encode("utf-8")).hexdigest()[:16]


def gate_string(
    checkpoint,
    *,
    step,
    exist_thresh,
    window_pref,
    placement=None,
    placement_lag=None,
    no_merge=None,
    min_vis=None,
):
    """The stable gate string for one `(checkpoint, step)` and its knobs."""
    step_r = resolved_step(step)
    g = {
        "checkpoint": os.path.abspath(str(checkpoint)),
        "step": "final" if step_r is None else step_r,
        "sha256": _identity_sha(checkpoint, step_r),
        "exist_thresh": float(exist_thresh),
        "window_pref": str(window_pref),
    }
    if placement is not None:
        g["placement"] = str(placement)
        g["placement_lag"] = None if placement_lag is None else int(placement_lag)
        g["no_merge"] = bool(no_merge)
        g["min_vis"] = min_vis
    return json.dumps(g, sort_keys=True)
