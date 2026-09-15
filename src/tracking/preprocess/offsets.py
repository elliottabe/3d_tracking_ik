"""Per-fly STAC marker-offset sample, pooled over a recording.

Marker offsets (`offsets_fly<f>.h5`) are a per-INDIVIDUAL constant, like body
scale: they say where this animal's markers sit relative to its skeleton. They
used to be fit once per run root on whichever bout-fly reached the stage first
and then shared with both flies; on Session0/2025_10_20_13_20_04 that bout-fly
was bout 28 fly0 (the female) frames 0..499, so the male ran IK with her
offsets -- his abdomen fitted 1.10-1.12x too long, wings 1.01-1.06x, and his
abdomen joints sat at rest because her offsets had already absorbed the
difference. Nothing errored and the residual looked fine; the offsets simply
described the wrong fly.

This module builds the sample the offsets fit should see for ONE fly:

1. pool every triangulated bout of that fly under the run root
   (`tracking.preprocess.scale.bout_kp3d_paths`: `kp3d_filt.npz`, else
   `kp3d.npz`);
2. keep a frame only if every keypoint is finite (the shared gate,
   `tracking.preprocess.gaps.finite_frame_mask` -- never a second copy of that
   rule), its per-frame MINIMUM `conf3d` is >= `min_conf`, and its implied body
   scale lies within `mad_k` MADs of the fly's pooled median. The scale gate is
   not redundant with confidence: a bad triangulation can be geometrically
   impossible at a confidence of 0.96, which is how a 38x body-scale error once
   propagated into marker offsets that absorbed it;
3. take `n_frames` stratified round-robin across bouts, best confidence first
   within each bout, so no single bout defines the offsets.

The frames are NOT a time series -- consecutive rows can be seconds apart -- so
the fit must run with the solver's temporal smoothness term off: see
`offsets_fit_cfg`.

Per-fly is only meaningful when fly0/fly1 names a stable individual across
bouts (sex-canonicalized: every bout carries a usable sex.json).
`resolve_offsets_path` therefore REFUSES to fall back to a shared file unless
`stac.allow_shared_offsets` is set deliberately.

Positions are `(T, K, 3)` in world units (0.1 mm each, `WORLD_UNITS_TO_MM`);
nothing here is scale-dependent except the scale gate, which only ever compares
a fly against its own median.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tracking.conventions import announce, refuse_config_object
from tracking.io.names import Order, OrderMismatch, as_order
from tracking.preprocess.gaps import finite_frame_mask

# Reused, never reimplemented: a second copy of either is how the two stages
# of one recording end up disagreeing about which bout, or which body part,
# they read.
from tracking.preprocess.scale import bout_index, bout_kp3d_paths, read_bout_kp3d

__all__ = [
    "SHARED_OFFSETS_NAME",
    "OffsetsSample",
    "aligned_per_frame_scales",
    "load_fly_bouts",
    "offsets_fit_cfg",
    "per_fly_offsets_name",
    "resolve_offsets_path",
    "select_offsets_sample",
]


SHARED_OFFSETS_NAME = "offsets.h5"

# Consistency-constant of the normal distribution: 1.4826 * MAD estimates sigma,
# so `mad_k` is read in standard deviations like every other outlier cut here.
MAD_TO_SIGMA = 1.4826


def per_fly_offsets_name(fly: int) -> str:
    """`offsets_fly<f>.h5` -- the per-INDIVIDUAL file, never shared."""
    return f"offsets_fly{int(fly)}.h5"


@dataclass(frozen=True)
class OffsetsSample:
    """The frames the offsets fit runs on, and the record of how they were chosen.

    `kp3d` is `(N, K, 3)` with row `i` being the frame named by `frames[i]`;
    `frames` is `(bout_idx, frame_idx)` per row, so a sample can be traced back
    to the bout it came from. `provenance` is JSON-serialisable and is written
    beside the h5: it is the only place the gate counts survive.
    """

    kp3d: np.ndarray
    frames: list[tuple[int, int]]
    provenance: dict


def load_fly_bouts(
    run_root,
    fly: int,
    *,
    kp_order: Order | Sequence[str] | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray | None]]:
    """`{bout index: (kp3d (F,K,3), conf3d (F,K) or None)}` for ONE fly.

    `kp3d_filt.npz` is preferred per bout -- it is what the IK consumes, so the
    offsets are fit on the same frames the solve will see -- with `kp3d.npz` as
    the fallback. `conf3d` comes from whichever file was chosen, and is `None`
    when that file has no such key; the confidence gate is then skipped for
    that bout and `select_offsets_sample` records the skip.

    `kp_order`, when given, checks each file's keypoint axis BY NAME. A file
    that names its axis and disagrees raises; a file that does not name it is
    accepted with a line to stderr (see `scale.read_bout_kp3d`) rather than
    refused, because the reference recording's own `kp3d_filt.npz` carries
    `['conf3d', 'kp3d']` and nothing else -- refusing would make this module
    unable to read the very recording it is gated against.

    Omitting `kp_order` is legal -- a caller may genuinely have no anatomy
    config -- but it is ANNOUNCED to stderr, every call (ruling R11): with no
    order the by-name check does not happen at all, and a file whose `kp_names`
    are wrong AND whose keypoint count is wrong comes back without complaint.
    Indexing a keypoint axis in the wrong space is this repo's most expensive
    bug class, and it was found only after a human looked at a picture.
    """
    order = as_order(kp_order) if kp_order is not None else None
    if order is None:
        announce(
            "offsets",
            "load_fly_bouts",
            f"kp_order was not given for fly{int(fly)} under {run_root}, so each "
            "bout's keypoint axis is accepted UNCHECKED -- a file whose kp_names "
            "disagree, or whose keypoint count disagrees, is returned without "
            "complaint and every later measurement reads the wrong body part with "
            "healthy-looking metrics. Pass kp_order=<the model's Order> to enable "
            "the by-name check.",
        )
    out: dict[int, tuple[np.ndarray, np.ndarray | None]] = {}
    source: dict[int, str] = {}
    for path in bout_kp3d_paths(Path(run_root), int(fly)):
        # allow_pickle=False: only the numeric `conf3d`/`kp3d` are read here.
        # `kp_names`, when present, is read by `read_bout_kp3d` below.
        with np.load(path, allow_pickle=False) as z:
            conf = np.asarray(z["conf3d"], dtype=np.float64) if "conf3d" in z.files else None
            kp = None if order is not None else np.asarray(z["kp3d"], dtype=np.float64)
        if order is not None:
            kp = read_bout_kp3d(path, order)
        out[bout_index(path)] = (kp, conf)
        source[bout_index(path)] = path.name
    _announce_mixed_sources(source, fly=int(fly), run_root=run_root)
    return out


def _announce_mixed_sources(source: Mapping[int, str], *, fly: int, run_root) -> None:
    """One stderr line when a fly's bouts did not all come from the same file.

    `bout_kp3d_paths` falls back per BOUT, so a part-way-processed recording
    hands the offsets fit a MIXTURE of filtered and raw tracks -- and the two are
    not of equal quality: the filter degrades the female's rigid-bone CV, and on
    the reference bout the filtered and raw implied scales are 0.011711 vs
    0.011725. `estimate_fly_scale` already announces exactly this mixture, and
    the same mixture reaching the offsets sample must not be the one that goes
    unsaid. Silent when the sources are uniform; recording the per-bout source
    into the provenance is Phase 5a's wiring, this is the free half.
    """
    if len(set(source.values())) <= 1:
        return
    by_source: dict[str, list[int]] = {}
    for bout, name in sorted(source.items()):
        by_source.setdefault(name, []).append(bout)
    announce(
        "offsets",
        "load_fly_bouts",
        f"fly{fly}'s pooled bouts under {run_root} MIXES artifacts -- "
        + "; ".join(f"{name}: bouts {bouts}" for name, bouts in sorted(by_source.items()))
        + ". kp3d_filt.npz and kp3d.npz are not of equal quality (the filter degrades "
        "the female's rigid-bone CV, and on the reference bout the two imply 0.011711 "
        "vs 0.011725), so this sample averages two kinds of measurement into one "
        "per-individual constant. Finish preprocessing the fly's bouts to remove it.",
    )


def _frame_gates(
    kp3d: np.ndarray, conf: np.ndarray | None, min_conf: float
) -> tuple[np.ndarray, np.ndarray, dict]:
    """`(candidate mask, per-frame min conf, counts)` for one bout.

    The finite gate is `finite_frame_mask` with its default `min_keypoints=None`
    -- "every keypoint finite", the same rule the IK frame gate uses. The
    offsets fit reads every marker of every sampled frame, so a partial frame
    has nothing to relax to.

    A `conf` whose shape does not match `kp3d`'s `(T, K)` RAISES, as
    `filter_kp3d` already does for the same mismatch. `min(axis=1)` reduces to
    `(T,)` for any keypoint count, so a truncated or re-ordered confidence array
    would otherwise gate on a keypoint set that is not this array's and report
    `low_conf: 0, conf_gate_applied: True` -- verbatim the sentence the per-bout
    gate record was added to eliminate, through a different door.
    """
    positions = np.asarray(kp3d, dtype=np.float64)
    finite = finite_frame_mask(positions)
    if conf is not None:
        confidences = np.asarray(conf, dtype=np.float64)
        if confidences.shape != positions.shape[:2]:
            raise OrderMismatch(
                f"conf3d has shape {confidences.shape} but kp3d is {positions.shape}; "
                f"expected confidences of {positions.shape[:2]}. A per-frame minimum "
                "over the wrong keypoint axis still reduces to (T,), so the gate "
                "would run on a keypoint set that is not this array's and record "
                "itself as applied."
            )
        min_over_kp = np.where(np.isfinite(confidences), confidences, -np.inf).min(axis=1)
        conf_ok = min_over_kp >= float(min_conf)
    else:
        min_over_kp = np.zeros(len(positions), dtype=np.float64)
        conf_ok = np.ones(len(positions), dtype=bool)
    counts = {
        "n_frames": int(len(positions)),
        "nonfinite": int((~finite).sum()),
        "low_conf": int((finite & ~conf_ok).sum()),
        # Per BOUT, not per recording: a bout whose file carried no conf3d
        # contributes completely ungated frames, and its `low_conf: 0` is
        # otherwise indistinguishable from "every frame was confident".
        "conf_gate_applied": conf is not None,
    }
    return finite & conf_ok, min_over_kp, counts


def select_offsets_sample(
    kp3d_by_bout: Mapping[int, np.ndarray],
    conf_by_bout: Mapping[int, np.ndarray | None],
    *,
    n_frames: int,
    min_conf: float,
    scale_by_bout: Mapping[int, np.ndarray] | None = None,
    mad_k: float = 3.0,
) -> OffsetsSample:
    """Gate every frame of every bout, then draw a stratified sample.

    `scale_by_bout` (optional) maps a bout to its per-frame implied body scale,
    aligned with that bout's frames and NaN where unavailable
    (`aligned_per_frame_scales`). The gate pools the surviving candidates'
    scales, takes their median and MAD, and rejects a frame further than
    `mad_k * 1.4826 * MAD` from the median -- a bad triangulation must not enter
    the fit even when the detector was confident. Zero spread rejects nothing: a
    constant scale is a good scale, not 100% outliers. A candidate whose scale
    is NaN once the gate is running is rejected too, and counted separately as
    `scale_unknown`: unknown is not the same as passing.

    The confidence gate is recorded PER BOUT (`bouts[b]["conf_gate_applied"]`,
    summarised by `conf_gate_bouts`), because on a pooled recording one bout
    written without `conf3d` contributes entirely ungated frames while its
    siblings are gated, and a single recording-wide boolean would claim the
    whole sample was gated.

    Deterministic: no randomness anywhere. Within a bout the most confident
    frames go first, ties broken by frame index; across bouts the draw is
    round-robin, and a bout that runs out simply stops contributing.
    """
    bouts = sorted(kp3d_by_bout)
    candidate: dict[int, np.ndarray] = {}
    min_conf_per_frame: dict[int, np.ndarray] = {}
    counts: dict[int, dict] = {}
    any_conf = False
    for b in bouts:
        conf = conf_by_bout.get(b)
        any_conf |= conf is not None
        candidate[b], min_conf_per_frame[b], counts[b] = _frame_gates(
            kp3d_by_bout[b], conf, min_conf
        )
        counts[b]["scale_outlier"] = 0
        counts[b]["scale_unknown"] = 0

    scale_gate = {"applied": False, "median": None, "mad": None, "mad_k": float(mad_k)}
    if scale_by_bout is not None:
        pooled = (
            np.concatenate(
                [np.asarray(scale_by_bout[b], dtype=np.float64)[candidate[b]] for b in bouts]
            )
            if bouts
            else np.zeros(0)
        )
        pooled = pooled[np.isfinite(pooled)]
        if pooled.size:
            median = float(np.median(pooled))
            mad = float(np.median(np.abs(pooled - median)))
            scale_gate.update(applied=True, median=median, mad=mad)
            tol = float(mad_k) * MAD_TO_SIGMA * mad
            for b in bouts:
                scales = np.asarray(scale_by_bout[b], dtype=np.float64)
                known = np.isfinite(scales)
                # An unknown scale is not a passing scale. In the intended
                # wiring this set is empty -- `aligned_per_frame_scales` returns
                # NaN exactly where `finite_frame_mask` is False, and those
                # frames are already gated out -- but `scale_by_bout` is a
                # public parameter, and an estimator that returns NaN on a
                # measurable-but-degenerate frame would otherwise make the gate
                # most permissive on the frames it is least sure about.
                unknown = candidate[b] & ~known
                counts[b]["scale_unknown"] = int(unknown.sum())
                # Zero spread rejects nothing: a constant scale is a good scale,
                # not 100% outliers.
                bad = (
                    candidate[b] & known & (np.abs(scales - median) > tol)
                    if mad > 0
                    else np.zeros(len(scales), dtype=bool)
                )
                counts[b]["scale_outlier"] = int(bad.sum())
                candidate[b] &= ~(bad | unknown)

    # Per-bout queues, best confidence first, stable on frame index.
    queues: dict[int, list[int]] = {}
    for b in bouts:
        idx = np.flatnonzero(candidate[b])
        order = np.lexsort((idx, -min_conf_per_frame[b][idx]))
        queues[b] = [int(i) for i in idx[order]]
        counts[b]["candidates"] = int(len(idx))

    chosen: list[tuple[int, int]] = []
    while len(chosen) < int(n_frames) and any(queues[b] for b in bouts):
        for b in bouts:
            if queues[b] and len(chosen) < int(n_frames):
                chosen.append((int(b), queues[b].pop(0)))
    for b in bouts:
        counts[b]["selected"] = sum(1 for bb, _ in chosen if bb == b)

    if not chosen:
        raise ValueError(
            "offsets sample: no frame passed the gates "
            f"(min_conf={min_conf}, mad_k={mad_k}); per bout: "
            + "; ".join(f"bout {b}: {counts[b]}" for b in bouts)
        )

    kp3d = np.stack([np.asarray(kp3d_by_bout[b], dtype=np.float64)[f] for b, f in chosen])
    provenance = {
        "n_bouts": len(bouts),
        "n_requested": int(n_frames),
        "n_selected": len(chosen),
        "min_conf": float(min_conf),
        # True when ANY bout was gated; `conf_gate_bouts` and each bout's own
        # `conf_gate_applied` say WHICH, because a mixed pool is the case a
        # single recording-wide boolean gets wrong.
        "conf_gate_applied": bool(any_conf),
        "conf_gate_bouts": {
            "gated": [b for b in bouts if counts[b]["conf_gate_applied"]],
            "ungated": [b for b in bouts if not counts[b]["conf_gate_applied"]],
        },
        "scale_gate": scale_gate,
        "bouts": {str(b): counts[b] for b in bouts},
        "frames": [[b, f] for b, f in chosen],
    }
    return OffsetsSample(kp3d=kp3d, frames=chosen, provenance=provenance)


def aligned_per_frame_scales(
    kp3d: np.ndarray, per_frame_scales_fn: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """`(T,)` implied body scale aligned to `kp3d`'s frames, NaN where unusable.

    `per_frame_scales_fn` is `tracking.preprocess.scale.per_frame_scales`
    partially applied with a keypoint order and a model XML. It is passed in
    rather than imported so the sampling above stays testable without a model:
    the only thing this module needs is one number per frame.

    That function DROPS frames it cannot measure, so its output is not aligned
    with its input in general. It is therefore called only on the all-finite
    subset -- the same `finite_frame_mask` rule the frame gate uses -- where one
    value per input frame is expected, and a short return raises rather than
    silently shifting every later frame's scale by one.
    """
    positions = np.asarray(kp3d, dtype=np.float64)
    out = np.full(len(positions), np.nan)
    finite = finite_frame_mask(positions)
    n_finite = int(finite.sum())
    if n_finite:
        scales = np.asarray(per_frame_scales_fn(positions[finite]), dtype=np.float64)
        if scales.shape[0] != n_finite:
            raise RuntimeError(
                f"per_frame_scales dropped frames from an all-finite input "
                f"({scales.shape[0]} scales for {n_finite} frames); cannot align "
                f"scales to frames, and a misaligned scale gate rejects the wrong ones"
            )
        out[finite] = scales
    return out


def resolve_offsets_path(
    run_root, fly: int, identity: str, *, allow_shared: bool, reason: str = ""
) -> tuple[str, str]:
    """`(path, mode)` with mode `'per_fly'` or `'shared'`.

    Per-fly needs `identity == 'canonical'` -- every bout has a usable sex.json,
    so fly0/fly1 names one individual across the recording. Otherwise a single
    file would blend two animals' anatomy, so this REFUSES and names the cause;
    `stac.allow_shared_offsets` is the deliberate override.
    """
    if identity == "canonical":
        return os.path.join(str(run_root), per_fly_offsets_name(fly)), "per_fly"
    if not allow_shared:
        raise RuntimeError(
            f"fly{fly}: refusing SHARED marker offsets: fly identity is not canonical "
            f"({reason or 'unknown'}), so fly0/fly1 is not a stable individual and one "
            f"{SHARED_OFFSETS_NAME} would blend both animals -- measured 2026-09-04, "
            f"the shared file was fit on the female and the male's abdomen came out "
            f"1.10x too long. Fix the cause (canonicalize sex / apply the id review), "
            f"or set stac.allow_shared_offsets=true to accept a shared fit deliberately."
        )
    return os.path.join(str(run_root), SHARED_OFFSETS_NAME), "shared"


def _plain_branch(container: dict, key: str, *, path: str) -> dict:
    """`container[key]` as a plain dict, creating or replacing it when absent.

    A key present but null is how Hydra spells a disabled group, so `None` is
    treated as absent rather than as an `AttributeError` waiting to happen. Any
    other non-dict is refused by `_refuse_non_dict`.
    """
    branch = container.get(key)
    if branch is None:
        branch = {}
        container[key] = branch
    else:
        _refuse_non_dict(branch, what=f"cfg[{path!r}]")
    return branch


def _refuse_non_dict(value, *, what: str) -> None:
    """Refuse anything but a plain `dict`, naming the type and the conversion.

    Decision D6, now shared by all three config surfaces of this package
    (`tracking.preprocess._conventions.refuse_config_object`, ruling R10). This
    one narrows the rule to `dict` alone, because `offsets_fit_cfg` deep-copies
    its input and a `MappingProxyType` cannot be deep-copied.
    """
    refuse_config_object(value, what=what, api="offsets_fit_cfg", allow=(dict,))


def offsets_fit_cfg(cfg) -> dict:
    """A deep copy of `cfg` for the offsets fit only, temporal terms OFF.

    The sample is a set of non-consecutive frames, so the solver's temporal
    smoothness term (and any per-DOF smoothness multiplier) must be off --
    otherwise it couples frames that are seconds apart and drags the offsets
    toward the average of unrelated poses. With the smoothness chain gone the
    frames are independent and the solver takes its vmapped per-frame path
    automatically, so chunking would only add Python loop overhead: hence
    `JAXLS_CHUNK_SIZE = 0`, the whole sample in one call.

    Keys the caller never set are created, and the caller's config is left
    untouched.

    Takes a plain `dict` only, and refuses anything else by name: see
    `_refuse_non_dict`.
    """
    _refuse_non_dict(cfg, what="cfg")
    fit = copy.deepcopy(cfg)
    model = _plain_branch(
        _plain_branch(fit, "anatomy", path="anatomy"), "model", path="anatomy.model"
    )
    model["JAXLS_SMOOTH_WEIGHT"] = 0.0
    model["JAXLS_SMOOTH_Q_MULT"] = None
    model["JAXLS_CHUNK_SIZE"] = 0
    return fit
