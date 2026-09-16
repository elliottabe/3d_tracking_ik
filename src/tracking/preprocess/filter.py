"""Clean triangulated 3D keypoints before the IK solve -- without losing coverage."""

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
    """A fresh deep copy of the shipped filter settings, safe to edit."""
    return copy.deepcopy(_DEFAULT_FILTER_CFG)


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
    """Per-stage counts. **The unit differs per field** -- they are not commensurable:"""

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
    """The segments whose length the bone-length gate checks, as NAME pairs."""
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
    """
    value = cfg.get(key) or {}
    return value if isinstance(value, dict) else dict(value)


def _validate_cfg(cfg: Mapping[str, Any]) -> None:
    """Refuse any key not in `DEFAULT_FILTER_CFG`'s schema, at every level."""
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
            continue
        for sub in value:
            if sub not in schema:
                raise unknown_key_error(sub, schema, where=f"cfg[{key!r}]", api="filter_kp3d")


def _validate_unported_stage_keys(cfg: Mapping[str, Any]) -> None:
    """The same schema check, inside the five unported stage blocks."""
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
    """Return a cleaned copy of `kp3d (T, K, 3)` plus the stage ledger."""
    kp3d = np.asarray(kp3d, np.float64)
    if cfg is None:
        cfg = DEFAULT_FILTER_CFG
    _validate_cfg(cfg)
    order = as_order(kp_order)

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
