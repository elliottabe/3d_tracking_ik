"""The Stage-B gate signature stamped into an mvq-lifted kp3d.npz.

A short stable string a run compares against a bout's stored gate to decide
whether the file is stale -- produced by a different checkpoint, a different
existence threshold, or a different window-selection rule.
"""

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

    The fallback exists so `gate_string` stays a pure function of its
    arguments and never requires a real checkpoint directory on disk just to
    be called (a synthetic path in a unit test, say). A real `MVQRunner`
    always has a loadable checkpoint by the time it calls `gates_string()`
    (`load_mvq_model` already required it in `__init__`), so this branch is a
    safety net for a path that was never a real checkpoint, not a silent
    swallow of a real one going missing -- but it IS a weaker staleness
    guarantee (a path hash cannot tell a retrained checkpoint from the one it
    replaced), so a caller that hits it needs to be told: a plain
    misconfiguration (wrong path, deleted metadata, a partially-copied
    checkpoint directory) must not silently downgrade into an
    indistinguishable gate string.

    Both a `warnings.warn` (for test capture / a caller with `-W error`) AND
    a plain stderr print fire on every call that falls back -- NOT once per
    process. `warnings.warn` dedupes by (message, category, module, lineno)
    under Python's default filters, so a batch driver that builds one
    `MVQRunner` per bout against the same misconfigured checkpoint path would
    warn on the first bout and fall silent on every bout after it, in the
    same process, with each bout's log tailed separately -- materially the
    same silent-weak-gate failure this fallback exists to surface, just
    moved one bout down the batch. The stderr print (`_report_unrestored`'s
    convention elsewhere in this package) has no dedup registry, so it is
    what must fire every time.
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
    """The stable gate string for one `(checkpoint, step)` and its knobs.

    `step="latest"` is refused by name (`resolved_step`): it names a
    different step every time the training job saves, so a gate built from
    it would compare equal to a kp3d.npz produced by different weights.

    `placement`/`placement_lag`/`no_merge`/`min_vis` are mask-free
    (coarse/fine-pass) window-placement knobs -- added to the signature ONLY
    when `placement` is named, so a masked-route runner's gate string (which
    never has a placement rule) is unaffected by their existence.
    """
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
