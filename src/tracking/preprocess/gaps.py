"""Decide which frames the IK solver is allowed to see."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from tracking.conventions import announce, plain_mapping
from tracking.io.names import Order, OrderMismatch, as_order

__all__ = [
    "MAX_GAP",
    "MIN_SEGMENT",
    "contiguous_segments",
    "fill_short_gaps",
    "finite_frame_mask",
    "marker_validity_mask",
    "required_keypoints",
]


# In frames.
MAX_GAP = 10
MIN_SEGMENT = 30


def marker_validity_mask(kp3d) -> np.ndarray:
    """`(T, K)` bool: which individual keypoints are entirely finite."""
    a = np.asarray(kp3d)
    return np.isfinite(a).all(axis=tuple(range(2, a.ndim)))


def finite_frame_mask(
    kp3d,
    *,
    min_keypoints: int | None = None,
    required: Sequence[str] | None = None,
    kp_order: Order | Sequence[str] | None = None,
) -> np.ndarray:
    """`(T,)` bool: frames usable for the IK solve."""
    per_kp = marker_validity_mask(kp3d)
    if min_keypoints is None:
        return per_kp.all(axis=1)
    ok = per_kp.sum(axis=1) >= int(min_keypoints)
    if isinstance(required, str):
        raise TypeError(
            f"required must be a sequence of names, got the bare string {required!r}; "
            "a string iterates character by character, so it would silently gate on "
            "single-letter keypoint names"
        )
    if required is not None and len(required) > 0:
        ok = ok & per_kp[:, _required_indices(required, kp_order)].all(axis=1)
    else:
        announce(
            "gaps",
            "finite_frame_mask",
            f"min_keypoints={int(min_keypoints)} relaxes the frame gate but no "
            "`required` names were given, so the solver's warm-start keypoints are "
            "NOT gated: a frame missing the root or an orientation keypoint passes "
            "and hands the solver a NaN initial guess, which freezes every joint "
            "while the residual still reports finite. Pass "
            "required=required_keypoints(kp_order, anatomy_cfg) to enable that check.",
        )
    return ok


def _required_indices(required: Sequence[str], kp_order) -> list[int]:
    if kp_order is None:
        raise OrderMismatch(
            f"finite_frame_mask(required={list(required)!r}) needs kp_order: "
            "required keypoints are names, and resolving them without the "
            "keypoint order would gate on whatever keypoint happened to share "
            "that position"
        )
    order = as_order(kp_order)
    return [order.index(name) for name in required]


def fill_short_gaps(kp3d, *, max_gap: int = MAX_GAP) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate NaN runs of at most `max_gap` frames."""
    a = np.array(kp3d, dtype=float, copy=True)
    ok = finite_frame_mask(a)
    filled = np.zeros(len(a), bool)
    if ok.all() or not ok.any():
        return a, filled
    good = np.flatnonzero(ok)
    # Every maximal run of bad frames strictly between two good ones.
    for lo, hi in zip(good[:-1], good[1:], strict=True):
        n = hi - lo - 1
        if n <= 0 or n > max_gap:
            continue
        w = (np.arange(1, n + 1) / (n + 1.0)).reshape((-1,) + (1,) * (a.ndim - 1))
        a[lo + 1 : hi] = a[lo] * (1.0 - w) + a[hi] * w
        filled[lo + 1 : hi] = True
    return a, filled


def contiguous_segments(ok, *, min_len: int = MIN_SEGMENT) -> list[tuple[int, int]]:
    """Half-open `[start, stop)` runs of True in `ok` with `stop - start >= min_len`."""
    a = np.asarray(ok, bool)
    if a.ndim != 1:
        raise ValueError(
            f"contiguous_segments takes a per-frame (T,) mask; got shape {a.shape}. "
            "A (T,K) marker mask (marker_validity_mask) flattened to (T*K,) would "
            "yield spans in a flattened index space -- the wrong index space, with "
            "no symptom. Use finite_frame_mask(kp3d) to reduce a (T,K,3) track to "
            "one bool per frame."
        )
    if a.size == 0:
        return []
    edges = np.diff(a.astype(np.int8))
    starts = np.flatnonzero(edges == 1) + 1
    stops = np.flatnonzero(edges == -1) + 1
    if a[0]:
        starts = np.concatenate(([0], starts))
    if a[-1]:
        stops = np.concatenate((stops, [a.size]))
    return [(int(s), int(e)) for s, e in zip(starts, stops, strict=True) if e - s >= min_len]


def required_keypoints(kp_order, anatomy_cfg) -> tuple[str, ...]:
    """Keypoint **names** a partial frame must still carry to be solvable."""
    order = as_order(kp_order)
    model = _mapping(anatomy_cfg, what="anatomy_cfg").get("model")
    model = _mapping(model, what="anatomy_cfg['model']")
    names: list[str] = []

    # Root warm-start: gated on key PRESENCE, like the solver; a name outside
    # the order is skipped rather than raised on -- but not silently.
    if "ROOT_OPTIMIZATION_KEYPOINT" in model:
        root = model["ROOT_OPTIMIZATION_KEYPOINT"]
        if root in order:
            names.append(root)
        else:
            _announce("ROOT_OPTIMIZATION_KEYPOINT", f"{root!r} is not in the keypoint order")

    orient = _mapping(
        model.get("JAXLS_ORIENTATION_KEYPOINTS"),
        what="anatomy_cfg['model']['JAXLS_ORIENTATION_KEYPOINTS']",
    )
    if orient:
        wanted = ["rear", "left", "right"] + (["front"] if "front" in orient else [])
        missing = [k for k in wanted if k not in orient]
        picked = [orient[k] for k in wanted if k in orient]
        absent = [n for n in picked if n not in order]
        if missing:
            _announce("JAXLS_ORIENTATION_KEYPOINTS", f"has no {'/'.join(missing)} entry")
        if absent:
            _announce(
                "JAXLS_ORIENTATION_KEYPOINTS",
                ", ".join(repr(n) for n in absent) + " is not in the keypoint order",
            )
        if not missing and not absent:
            names.extend(picked)

    # `rear` is usually the root keypoint itself; dedupe, keep config order.
    return tuple(dict.fromkeys(names))


def _announce(key: str, reason: str) -> None:
    """One stderr line per config key that is present but does not resolve."""
    announce(
        "gaps",
        "required_keypoints",
        f"model.{key} {reason}; that warm-start keypoint is NOT gated, so a "
        "partial frame can hand the solver a NaN initial guess. Expected if this "
        "anatomy was pruned; otherwise it is a config typo.",
    )


def _mapping(value, *, what: str) -> Mapping:
    """A mapping view of a config node, tolerating None and refusing a Hydra one."""
    return plain_mapping(value, what=what, api="required_keypoints")
