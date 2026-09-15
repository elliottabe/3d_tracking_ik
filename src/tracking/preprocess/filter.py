"""Clean triangulated 3D keypoints before the IK solve -- without losing coverage.

Raw per-frame triangulation occasionally mistriangulates a distal leg keypoint
(`*_TaTip` especially), which jumps far in one frame and back in the next. STAC
then bends the whole leg to chase the outlier, which is the jittery/curling-leg
failure this stage exists to remove. The pipeline runs
confidence mask -> bone-length outlier reject -> isolated-spike removal ->
PCHIP gap fill -> Savitzky-Golay, and writes the result as `kp3d_filt.npz`.

**The guarantee is that filtering never takes an observation away.** A keypoint
the raw triangulation had is finite in the output: an outlier flagged
mid-sequence is replaced by the interpolant, and anything interpolation could
not reach (a flagged frame inside a leading/trailing run past the edge budget)
falls back to its raw value. Without that fallback STAC would be starved and
the frame's qpos would come back NaN. Coverage only ever goes up -- on the
reference bout, 641 raw-NaN points come back finite and 0 raw-finite points are
lost.

**Wings keep their raw values exactly.** The spike and savgol stages are not
wing-aware (only the confidence and bone-length gates exclude wings), so they
flatten the rapid wing extension of a singing male -- about a 2x loss of wing
speed. Every keypoint matching `preserve_raw_patterns` is restored from the raw
triangulation wherever raw is finite, so legs and body get the full smoothing
while wing motion survives intact.

Positions are `(T, K, 3)` in **0.1 mm world units**, not millimetres: a fly's
head-to-abdomen-tip is about 23.8 of them. Confidences are the caller's and are
neither modified nor returned.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from tracking.conventions import refuse_config_object, unknown_key_error
from tracking.io.names import Order, OrderMismatch, as_order
from tracking.preprocess.gaps import marker_validity_mask
from tracking.preprocess.stages import (
    despike_isolated_spikes,
    detect_bone_length_outliers,
    interpolate_nan_gaps,
    mask_low_confidence,
    savgol_smooth,
)

__all__ = [
    "DEFAULT_FILTER_CFG",
    "FilterReport",
    "default_filter_cfg",
    "filter_kp3d",
    "skeleton_edges",
]


# The pipeline's real settings, from `configs/pipeline.yaml`. A caller who
# passes nothing gets exactly today's behaviour.
#
# `savgol.window_length` must keep that name: the stage reads `window_length`,
# so a config writing `window` is silently ignored and falls back to 11.
#
# Private, because a process-wide mutable default is action at a distance.
# Read it through `DEFAULT_FILTER_CFG`; take `default_filter_cfg()` to edit.
_DEFAULT_FILTER_CFG: dict[str, Any] = {
    "enabled": True,
    "preserve_raw_patterns": ["Wing"],
    "confidence": {"enabled": True, "threshold": 0.3, "exclude_keypoint_patterns": ["Wing"]},
    "bone_length": {"enabled": True, "threshold_std": 5.0, "exclude_keypoint_patterns": ["Wing"]},
    "isolated_spike": {"enabled": True, "threshold_factor": 10.0, "max_iterations": 1},
    "interpolation": {
        "enabled": True,
        "use_spline": True,
        "max_edge_extrap_frames": 5,
        "edge_fit_window": 5,
    },
    "savgol": {"enabled": True, "window_length": 7, "polyorder": 2},
    # Present and false in the real config; the keys stay so that turning one on
    # is a config change rather than a code change.
    "centroid_jump": {"enabled": False},
    "identity_relink": {"enabled": False},
    "medfilt_despike": {"enabled": False},
    "medfilt": {"enabled": False},
    "confidence_smooth": {"enabled": False},
}

DEFAULT_FILTER_CFG: Mapping[str, Any] = MappingProxyType(_DEFAULT_FILTER_CFG)
"""The shipped filter settings, as a read-only view. Pass it to `filter_kp3d`.

`DEFAULT_FILTER_CFG["savgol"] = ...` raises `TypeError` instead of silently
rewriting the process-wide defaults. **The nested dicts are still plain
mutable dicts** -- `DEFAULT_FILTER_CFG["savgol"]["enabled"] = False` succeeds
and does poison every later call -- so this proxy catches the common accident,
not every one. When you intend to modify a config, call `default_filter_cfg()`
and edit that.
"""


def default_filter_cfg() -> dict[str, Any]:
    """A fresh deep copy of the shipped filter settings, safe to edit.

    Use this, not `DEFAULT_FILTER_CFG`, whenever the config will be modified --
    an A/B that flips one threshold, a test that enables one stage. Each call
    returns an independent object, so edits cannot leak into another caller.
    """
    return copy.deepcopy(_DEFAULT_FILTER_CFG)


# Stages the real config keeps off, and which are therefore unimplemented.
# The keys stay in the schema, but asking for one RAISES rather than doing
# nothing: a silently ignored knob turns an A/B into "no effect", which reads
# as a real scientific result.
_UNPORTED_STAGES: dict[str, str] = {
    "centroid_jump": (
        "it IS a live stage in the source orchestrator (Step 2b, identity-switch "
        "masking on the raw input), so enabling it here would silently differ from "
        "the reference pipeline rather than merely do nothing"
    ),
    "identity_relink": (
        "it was never a filter stage at all -- the source consumes it in the caller, "
        "not inside filter_keypoints"
    ),
    "medfilt_despike": "the median-filter despike stage was not ported",
    "medfilt": "the median-filter spike/interpolate stage was not ported",
    "confidence_smooth": "the confidence-weighted smoothing stage was not ported",
}


@dataclass(frozen=True)
class FilterReport:
    """Per-stage counts. **The unit differs per field** -- they are not commensurable:

    - `n_confidence_masked`, `n_wing_excluded`: **keypoint-frames** (one
      `(frame, keypoint)` pair).
    - `n_bone_flagged`: **edge-frames** (one `(frame, skeleton edge)` pair).
      Both endpoints of a flagged edge go NaN, so this is not a keypoint-frame
      count and the two are not related by a clean factor of 2 -- an endpoint
      another edge already flagged is not counted twice. On the reference
      female, 1802 edge-frames are 2802 keypoint-frames, 1.55x.
    - `n_spikes_fixed`, `n_interp_values`: **coordinate-frames** -- the spike and
      interpolation stages work per x/y/z column, so one keypoint-frame can
      contribute up to 3. Comparing `n_spikes_fixed` with
      `n_confidence_masked` as if both were keypoint-frames overstates the
      spike stage by up to 3x.
    - `n_edge_phantom_frames`: **frames** (any keypoint edge-phantom in them).
    - `n_bones_flagged`, `n_bone_edges_skipped`: **skeleton edges**.
    """

    n_confidence_masked: int = 0
    n_wing_excluded: int = 0
    n_bone_flagged: int = 0
    n_bones_flagged: int = 0
    n_bone_edges_skipped: int = 0
    n_spikes_fixed: int = 0
    n_interp_values: int = 0
    n_edge_phantom_frames: int = 0


# Per-leg proximal->distal joint order, and the legs that use it.
_LEG_SEGS = ("ThxCx", "Tro", "FeTi", "TiTa", "TaT1", "TaT3", "TaTip")
_LEGS = ("T1L", "T2L", "T3L", "T1R", "T2R", "T3R")


def skeleton_edges(kp_order: Order | Sequence[str]) -> tuple[tuple[str, str], ...]:
    """The segments whose length the bone-length gate checks, as NAME pairs.

    Head/thorax/abdomen, both wing chains, and each leg's proximal->distal
    chain (with the leg's most proximal joint also tied to `Scutellum`).
    Derived from names so it holds for any keypoint ordering, which is the
    whole point: an integer edge table built against the wrong order measures a
    leg where a wing was meant and reports it as a perfectly rigid bone.
    Keypoints the order does not contain are skipped.
    """
    order = as_order(kp_order)

    def e(a: str, b: str) -> tuple[str, str] | None:
        return (a, b) if a in order and b in order else None

    edges: list[tuple[str, str] | None] = [
        e("EyeL", "Antenna_Base"),
        e("EyeR", "Antenna_Base"),
        e("Antenna_Base", "Scutellum"),
        e("Scutellum", "WingL_base"),
        e("WingL_base", "WingL_V12"),
        e("WingL_V12", "WingL_V13"),
        e("Scutellum", "WingR_base"),
        e("WingR_base", "WingR_V12"),
        e("WingR_V12", "WingR_V13"),
        e("Scutellum", "Abd_A4"),
        e("Abd_A4", "Abd_tip"),
    ]
    for leg in _LEGS:
        chain = [f"{leg}_{s}" for s in _LEG_SEGS if f"{leg}_{s}" in order]
        for k in range(len(chain) - 1):
            edges.append((chain[k], chain[k + 1]))
        if chain and "Scutellum" in order:
            edges.append(("Scutellum", chain[0]))
    return tuple(x for x in edges if x is not None)


def _sub(cfg: Mapping[str, Any], key: str) -> dict[str, Any]:
    """One stage's sub-config, as a mapping. A missing or null key means `{}`,
    so every stage falls back to its own per-stage defaults.

    Safe only because `_validate_cfg` has already refused every key that is not
    in the schema: absent here now means "the caller did not set this stage",
    never "the caller misspelled it".
    """
    value = cfg.get(key) or {}
    return value if isinstance(value, dict) else dict(value)


def _validate_cfg(cfg: Mapping[str, Any]) -> None:
    """Refuse any key not in `DEFAULT_FILTER_CFG`'s schema, at every level.

    **Raise where a complete schema exists** (ruling R10). `_DEFAULT_FILTER_CFG`
    enumerates every valid filter key, top level and per stage, so a key that is
    not in it is unambiguously a typo -- and a typo here is silent and
    expensive: `_sub` maps an unknown key to `{}` and each stage then reads
    `enabled` with a default of `False`, so ONE misspelled character turns a
    live stage off. Measured on the reference bout, renaming `bone_length` to
    `bone_lenght` gives `n_bone_flagged = 0` instead of 1802 and moves 128458 of
    300873 finite entries off `kp3d_filt.npz`, with no exception and no stderr
    line; STAC then solves on uncleaned distal leg keypoints, which is the
    jittery/curling-leg failure this module exists to remove.

    This is the symmetric case of the `_UNPORTED_STAGES` rule: that one refuses
    a knob that will not take effect because the stage is missing, this one
    refuses a knob that will not take effect because the NAME is missing.
    """
    refuse_config_object(cfg, what="cfg", api="filter_kp3d")
    for key in cfg:
        if key not in _DEFAULT_FILTER_CFG:
            raise unknown_key_error(key, _DEFAULT_FILTER_CFG, where="cfg", api="filter_kp3d")
        schema = _DEFAULT_FILTER_CFG[key]
        value = cfg[key]
        if not isinstance(schema, dict) or value is None:
            continue
        refuse_config_object(value, what=f"cfg[{key!r}]", api="filter_kp3d")
        if key in _UNPORTED_STAGES:
            # Deferred, not skipped: `_validate_unported_stage_keys` checks these
            # blocks AFTER the stage guard, so a present-and-true `enabled` still
            # reports the missing STAGE rather than a key typo.
            continue
        for sub in value:
            if sub not in schema:
                raise unknown_key_error(sub, schema, where=f"cfg[{key!r}]", api="filter_kp3d")


def _validate_unported_stage_keys(cfg: Mapping[str, Any]) -> None:
    """The same schema check, inside the five unported stage blocks.

    Run AFTER the stage guard, so precedence is preserved: a present-and-true
    `enabled` raises `NotImplementedError` naming the stage, because a stage
    that does not exist is the more fundamental fact and R4's message is the one
    that helps. What this closes is the gap BETWEEN the two rules -- with
    `centroid_jump: {"enabld": True}` the stage guard sees no `enabled` and says
    nothing, and the schema check used to decline to look, so a one-character
    typo asking for a live source stage vanished with no symptom at all. Only
    `enabled` is modelled for these blocks, so `enabled` is the only key they
    accept.
    """
    for key in _UNPORTED_STAGES:
        value = cfg.get(key)
        if not isinstance(value, Mapping):
            continue
        for sub in value:
            if sub not in _DEFAULT_FILTER_CFG[key]:
                raise unknown_key_error(
                    sub, _DEFAULT_FILTER_CFG[key], where=f"cfg[{key!r}]", api="filter_kp3d"
                )


def filter_kp3d(
    kp3d: np.ndarray,
    conf3d: np.ndarray | None,
    kp_order: Order | Sequence[str],
    cfg: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, FilterReport]:
    """Return a cleaned copy of `kp3d (T, K, 3)` plus the stage ledger.

    `kp3d` is in 0.1 mm world units; `conf3d (T, K)` may be None, in which case
    confidence masking is skipped. `cfg` is a plain dict: Hydra nodes are
    converted at the boundary and a `DictConfig` here raises `TypeError` naming
    the conversion -- or None for
    `DEFAULT_FILTER_CFG`. `cfg["enabled"]` false returns an unmodified copy, so
    callers can gate on config without branching.

    `conf3d` is not returned and not modified; it stays the caller's array.

    Raises, in this order and all of them regardless of `cfg["enabled"]`,
    because a config that asks for something it will not get is wrong whether or
    not this call happens to act on it:

    - `TypeError` if `cfg` (or a stage block) is not a plain mapping;
    - `ValueError` if any key is not in `DEFAULT_FILTER_CFG`'s schema, at the
      top level or inside an implemented stage block -- the schema is
      complete, so an
      unknown key is a typo, and an unnoticed typo turns a live stage off
      (`_validate_cfg`);
    - `OrderMismatch` if `kp_order` does not describe `kp3d`'s keypoint axis;
    - `NotImplementedError` if the config enables one of the unimplemented
      stages (see `_UNPORTED_STAGES`);
    - `ValueError` again, last, for an unknown key INSIDE an unported stage
      block (`_validate_unported_stage_keys`). Last on purpose: a present-and-
      true `enabled` there is a missing stage, not a typo, and must report as
      one -- but `{"enabld": True}` fell between the two rules and said nothing.
    """
    kp3d = np.asarray(kp3d, np.float64)
    if cfg is None:
        cfg = DEFAULT_FILTER_CFG
    _validate_cfg(cfg)
    order = as_order(kp_order)

    # The keypoint axis is indexed BY NAME everywhere below, so an order that
    # does not describe this array is refused rather than quietly half-applied:
    # a 30-name order against a 50-column array never reaches the wings and
    # returns an innocent all-zero ledger.
    if kp3d.ndim != 3 or kp3d.shape[1] != len(order):
        raise OrderMismatch(
            f"kp3d has shape {kp3d.shape} but the keypoint order names {len(order)} "
            f"keypoints; expected (T, {len(order)}, 3). Indexing a keypoint axis "
            "against the wrong order measures the wrong body part, and every "
            "jitter, confidence and residual metric rates it as good."
        )

    requested = [k for k in _UNPORTED_STAGES if _sub(cfg, k).get("enabled", False)]
    if requested:
        detail = "; ".join(f"{k}: {_UNPORTED_STAGES[k]}" for k in requested)
        raise NotImplementedError(
            f"filter stage(s) {requested} are present in this config's schema but "
            f"have no ported stage, so enabling one changes nothing -- {detail}. "
            "Port the stage or remove the key; do not leave it on expecting an effect."
        )
    _validate_unported_stage_keys(cfg)

    if not bool(cfg.get("enabled", False)):
        return kp3d.copy(), FilterReport()

    kp_names = list(order)
    kp = kp3d.copy()
    counts: dict[str, int] = {}

    conf_cfg = _sub(cfg, "confidence")
    if conf_cfg.get("enabled", False) and conf3d is not None:
        kp, conf_mask, n_excluded = mask_low_confidence(
            kp,
            np.asarray(conf3d),
            conf_cfg.get("threshold", 0.5),
            kp_names=kp_names,
            exclude_patterns=list(conf_cfg.get("exclude_keypoint_patterns", [])),
        )
        counts["n_confidence_masked"] = int(np.sum(conf_mask))
        counts["n_wing_excluded"] = n_excluded

    bone_cfg = _sub(cfg, "bone_length")
    if bone_cfg.get("enabled", False):
        edge_names = skeleton_edges(order)
        edges = np.array([[order.index(a), order.index(b)] for a, b in edge_names], dtype=int)
        kp, bone_report, n_skipped = detect_bone_length_outliers(
            kp,
            edges,
            bone_cfg.get("threshold_std", 3.0),
            kp_names=kp_names,
            exclude_patterns=list(bone_cfg.get("exclude_keypoint_patterns", [])),
        )
        counts["n_bone_flagged"] = int(sum(bone_report.values()))
        counts["n_bones_flagged"] = len(bone_report)
        counts["n_bone_edges_skipped"] = n_skipped

    spike_cfg = _sub(cfg, "isolated_spike")
    if spike_cfg.get("enabled", True):
        kp, n_spikes = despike_isolated_spikes(
            kp,
            threshold_factor=spike_cfg.get("threshold_factor", 10.0),
            max_iterations=spike_cfg.get("max_iterations", 1),
        )
        counts["n_spikes_fixed"] = n_spikes

    interp_cfg = _sub(cfg, "interpolation")
    if interp_cfg.get("enabled", True):
        kp, edge_nan_mask, n_filled = interpolate_nan_gaps(
            kp,
            use_spline=interp_cfg.get("use_spline", True),
            max_edge_extrap_frames=int(interp_cfg.get("max_edge_extrap_frames", 0)),
            edge_fit_window=int(interp_cfg.get("edge_fit_window", 5)),
        )
        counts["n_interp_values"] = n_filled
        counts["n_edge_phantom_frames"] = int(np.any(edge_nan_mask, axis=1).sum())

    savgol_cfg = _sub(cfg, "savgol")
    if savgol_cfg.get("enabled", False):
        kp = savgol_smooth(
            kp,
            window_length=savgol_cfg.get("window_length", 11),
            polyorder=savgol_cfg.get("polyorder", 3),
        )

    kp = np.asarray(kp, np.float64)

    # Coverage fallback. Smoothing must never DELETE a keypoint the raw
    # triangulation had: an outlier flagged inside a leading/trailing run that
    # interpolation cannot extrapolate would otherwise stay NaN, starve STAC
    # and NaN the whole frame's qpos, after which the bridge FK has NaN sites.
    # `marker_validity_mask` is imported, not open-coded -- two copies is how
    # one gets relaxed and the other does not.
    dropped = (~marker_validity_mask(kp)) & marker_validity_mask(kp3d)  # (T, K)
    kp[dropped] = kp3d[dropped]

    # Preserve raw kinematics for named keypoints (wings by default). Gaps the
    # filter interpolated (raw NaN) are left as the interpolant.
    patterns = list(cfg.get("preserve_raw_patterns", ["Wing"]) or [])
    if patterns:
        keep = [i for i, n in enumerate(kp_names) if any(p in n for p in patterns)]
        if keep:
            keep = np.array(keep)
            raw_keep = kp3d[:, keep]
            finite = marker_validity_mask(raw_keep)  # (T, n_keep)
            sub = kp[:, keep]
            sub[finite] = raw_keep[finite]
            kp[:, keep] = sub

    return kp, FilterReport(**counts)
