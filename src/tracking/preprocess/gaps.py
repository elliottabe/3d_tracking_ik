"""Decide which frames the IK solver is allowed to see.

Triangulation leaves holes: a keypoint the cameras could not agree on comes
back NaN. The batched pose solve is one problem over all frames chained by
smoothness costs, so a single NaN frame propagates NaN gradients along the
chain, no joint DOF is ever updated, and every joint stays at its
initialisation while only the separately-solved root moves. Nothing errors --
the solve reports a finite mean error and the bout is marked done -- and the
failure is invisible to reprojection (NaN), to leave-one-out and to IoU. On one
60-bout-fly recording the split was total: the 9 frozen bout-flies had 40..1393
NaN input frames, the 51 healthy ones had 0..12.

This module is the gate in front of that. Short dropouts are bridged by linear
interpolation (`fill_short_gaps`), long ones are left NaN and the remaining
finite runs are cut into solvable segments (`contiguous_segments`); a run
shorter than `MIN_SEGMENT` is not worth a solve. `fill_short_gaps` also returns
which frames it invented, so a later stage can re-NaN poses that were solved
from interpolated rather than measured input.

`required_keypoints` exists because relaxing the frame gate with
`min_keypoints` protects only the IK *residual*, which is already NaN-safe per
marker (a NaN marker contributes zero cost and zero gradient). It does not
protect the solver's per-frame **warm-start**, which reads the root keypoint
and the four trunk orientation keypoints straight out of the array with no
finite check. A partial frame missing exactly one of those hands the solver a
NaN initial guess -- reproducing the very frozen-joint failure this gate is
meant to avoid. The names come from the same two anatomy-config keys the solver
resolves them from, never hard-coded, so a re-rooted anatomy cannot silently
diverge; for v1 they resolve to `Scutellum`, `WingL_base`, `WingR_base`,
`Antenna_Base` (the `rear` orientation key is `Scutellum` too, so the set has
four members, not five).

Positions are `(T, K, 3)` world coordinates; nothing here is scale-dependent.
Everything is pure numpy and takes no config object except `required_keypoints`,
which reads an already-loaded anatomy mapping.
"""

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
    """`(T, K)` bool: which individual keypoints are entirely finite.

    The per-marker counterpart to `finite_frame_mask`, and exactly the markers
    the IK residual already treats as absent (NaN in, zero residual and zero
    gradient out). A keypoint is valid only when every one of its coordinates
    is finite -- half a keypoint is no keypoint.
    """
    a = np.asarray(kp3d)
    return np.isfinite(a).all(axis=tuple(range(2, a.ndim)))


def finite_frame_mask(
    kp3d,
    *,
    min_keypoints: int | None = None,
    required: Sequence[str] | None = None,
    kp_order: Order | Sequence[str] | None = None,
) -> np.ndarray:
    """`(T,)` bool: frames usable for the IK solve.

    `min_keypoints=None` (the default) means **every** keypoint must be finite
    -- the all-or-nothing rule, and the one other stages share rather than
    reimplement. In that branch `required` is **ignored**, and deliberately so:
    when every keypoint is already required, an additional required-subset
    check can only ever agree with it, so it must not be given the chance to
    disagree. It is not merely unused there but never even resolved, so passing
    `required` without a `kp_order` is legal under `min_keypoints=None`.

    `min_keypoints=N` accepts a frame with at least N of its K keypoints
    entirely finite. That is safe for the residual, which ignores a NaN marker
    per frame and per keypoint; it is **not** safe for the solver's warm-start,
    so pass `required=required_keypoints(kp_order, anatomy_cfg)` alongside it.
    Those names must ALSO be finite, whatever the count says. Omitting it is
    legal -- a caller may genuinely have no anatomy config -- but it is
    ANNOUNCED to stderr, every call (ruling R11): a relaxed gate with no
    required set is the warm-start guarantee switched off, and a guard that is
    off by default and silent about it is how this repo's worst bug class
    recurs.

    `required` is a sequence of keypoint **names** and is resolved through
    `kp_order`, which is therefore mandatory whenever a non-empty `required` is
    used with an integer `min_keypoints`. Indices are not accepted: the two
    keypoint index spaces in this pipeline disagree, and an integer here would
    read the wrong body part with no symptom. A name that does not resolve
    raises rather than being dropped -- unlike `required_keypoints`, where
    tolerance is the pruned-anatomy affordance, an explicit caller-supplied
    requirement must never be silently satisfied. An empty `required` (list or
    ndarray) gates nothing; a bare string is rejected rather than iterated
    character by character.
    """
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
    """Linearly interpolate NaN runs of at most `max_gap` frames.

    Returns `(filled (T, K, 3), filled_mask (T,) bool)`. The mask is not a
    convenience: a pose solved from an interpolated frame is not a measurement,
    and a later stage re-NaNs exactly those frames so that an invented
    trajectory never leaves this pipeline as an observation.

    Short dropouts are the common case and bridging them keeps the smoothness
    chain intact, which is what stops one bad frame freezing the whole solve.
    Two runs are left NaN for the segmenter: one longer than `max_gap`, and one
    not bracketed by a good frame on both sides -- a leading or trailing run has
    only one anchor, so filling it would invent a trajectory rather than bridge
    one.

    The unit here is the whole frame, not the keypoint: a frame is a gap if any
    keypoint is missing, and a bridged frame is written from the two bracketing
    frames in full. Per-keypoint gap filling is `tracking.preprocess.filter`'s
    job and runs earlier.
    """
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
    """Half-open `[start, stop)` runs of True in `ok` with `stop - start >= min_len`.

    Half-open because that is what slices the arrays: `kp3d[start:stop]` is the
    segment, and `stop` is the first frame NOT in it. A run shorter than
    `min_len` is dropped -- too little context to be worth a solve.

    `ok` must be one bool PER FRAME. A `(T,K)` marker mask is refused rather
    than flattened: ravelling it would return spans in a `(T*K,)` index space
    that are self-consistent, `K` times too long, and wrong. Reduce a `(T,K,3)`
    track with `finite_frame_mask` first.
    """
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
    """Keypoint **names** a partial frame must still carry to be solvable.

    `model.ROOT_OPTIMIZATION_KEYPOINT` plus the values of
    `model.JAXLS_ORIENTATION_KEYPOINTS` -- the keypoints the solver's per-frame
    warm-start reads without a finite check (see the module docstring). Read
    from the config rather than hard-coded so that a re-rooted anatomy cannot
    diverge from what the solver actually warm-starts from. `anatomy_cfg` is an
    already-loaded mapping, so this stays cheap and builds no model.

    Tolerant by construction, mirroring the solver's own fallbacks: a missing
    key, or a configured name absent from `kp_order`, contributes no requirement
    rather than raising -- which is what keeps a pruned or amputated anatomy
    working, and matches the solver simply skipping the warm-start it cannot
    resolve. The orientation set is all-or-nothing for the same reason: the
    solver disables the whole orientation warm-start if any one of its names
    fails to resolve, so a partial set must not become a partial requirement.

    Tolerant is not silent. An ABSENT key says nothing -- not configuring the
    feature is a legitimate choice. A key that is PRESENT but whose value does
    not resolve (a name outside `kp_order`, a missing `rear`/`left`/`right`
    entry) drops the requirement AND announces one line to stderr naming the key
    and the name: with no requirement the gate is a no-op, `min_keypoints`
    admits partial frames, and the frozen-joint failure this module exists to
    prevent comes back reporting a finite error on a bout marked done. A pruned
    anatomy is a rare deliberate act, so the line costs nothing; a one-character
    config typo is exactly what it is for.
    """
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

    # Orientation warm-start: rear/left/right required, front only if given. If
    # any one fails to resolve the solver disables the WHOLE orientation
    # warm-start, so a partial set must not become a partial requirement.
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
    """One stderr line per config key that is present but does not resolve.

    Format and the stderr-not-`warnings.warn` rule are
    `tracking.preprocess._conventions.announce`'s, shared with every other
    announcement in this package.
    """
    announce(
        "gaps",
        "required_keypoints",
        f"model.{key} {reason}; that warm-start keypoint is NOT gated, so a "
        "partial frame can hand the solver a NaN initial guess. Expected if this "
        "anatomy was pruned; otherwise it is a config typo.",
    )


def _mapping(value, *, what: str) -> Mapping:
    """A mapping view of a config node, tolerating None and refusing a Hydra one.

    A `DictConfig` is refused here rather than converted (decision D6, ruling
    R10): `required_keypoints` is one of the three config surfaces in this
    package, and all three must answer that question the same way or the
    boundary stops being visible. Converting it silently -- which this function
    used to do -- is what made `gaps` the odd one out.
    """
    return plain_mapping(value, what=what, api="required_keypoints")
