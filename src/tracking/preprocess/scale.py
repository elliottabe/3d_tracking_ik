"""One body scale per fly per recording, and a guard that a wrong one raises."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import mujoco
import numpy as np

from tracking.conventions import announce
from tracking.io.names import Order, OrderMismatch, as_order, require_same
from tracking.preprocess.gaps import finite_frame_mask, marker_validity_mask

__all__ = [
    "BODY_LENGTH_MM_MAX",
    "BODY_LENGTH_MM_MIN",
    "BODY_LENGTH_REF_PAIR",
    "DEFAULT_TRUNK_KEYPOINTS",
    "LEG_JOINT_CHAIN",
    "LEG_NAMES",
    "THORAX_PAIRS",
    "WITHIN_BONE_CV_WARN_THRESH",
    "WORLD_UNITS_TO_MM",
    "assert_plausible_body_scale",
    "bout_index",
    "bout_kp3d_paths",
    "estimate_fly_scale",
    "implied_body_length_mm",
    "per_bout_segment_scale",
    "per_frame_scales",
    "read_bout_kp3d",
    "rigid_segment_pairs",
    "robust_scale",
    "segment_scale_diagnostics",
    "warn_if_estimator_ignored",
]


WORLD_UNITS_TO_MM = 0.1

BODY_LENGTH_MM_MIN = 2.0
BODY_LENGTH_MM_MAX = 3.0

BODY_LENGTH_REF_PAIR: tuple[str, str] = ("Antenna_Base", "Abd_tip")

LEG_NAMES = ("T1L", "T1R", "T2L", "T2R", "T3L", "T3R")
# Consecutive joints along one leg's kinematic chain, thorax to tarsus.
LEG_JOINT_CHAIN = ("ThxCx", "Tro", "FeTi", "TiTa", "TaT1")

THORAX_PAIRS: tuple[tuple[str, str], ...] = (
    ("WingL_base", "WingR_base"),
    ("Scutellum", "WingL_base"),
    ("Scutellum", "WingR_base"),
)

WITHIN_BONE_CV_WARN_THRESH = 0.15

# The trunk markers the cloud-fit estimators use.
DEFAULT_TRUNK_KEYPOINTS: tuple[str, ...] = (
    "Scutellum",
    "WingL_base",
    "WingR_base",
    "Abd_A4",
    "Abd_tip",
)

_BOUT_DIR_RE = re.compile(r"^bout_(\d+)$")

_SITE_REST_CACHE: dict[str, dict[str, np.ndarray]] = {}


def _announce(func: str, message: str) -> None:
    """One line to stderr, tagged `[scale] <func>: ...`."""
    announce("scale", func, message)


def _tracking_site_rest(model_xml) -> dict[str, np.ndarray]:
    """`{keypoint name: (3,) rest-pose position}` for every `tracking[...]` site."""
    key = str(model_xml)
    cached = _SITE_REST_CACHE.get(key)
    if cached is not None:
        return cached
    mj = mujoco.MjModel.from_xml_path(key)
    data = mujoco.MjData(mj)
    mujoco.mj_forward(mj, data)
    rest: dict[str, np.ndarray] = {}
    for i in range(mj.nsite):
        name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_SITE, i)
        if name and name.startswith("tracking[") and name.endswith("]"):
            rest[name[len("tracking[") : -1]] = np.array(data.site_xpos[i], dtype=np.float64)
    _SITE_REST_CACHE[key] = rest
    return rest


def implied_body_length_mm(
    scale: float, model_xml, *, ref_pair: tuple[str, str] = BODY_LENGTH_REF_PAIR
) -> float:
    """Body length in mm implied by a candidate world->model `scale`."""
    if not np.isfinite(scale) or scale <= 0:
        return float("nan")
    rest = _tracking_site_rest(model_xml)
    a, b = ref_pair
    missing = [n for n in (a, b) if n not in rest]
    if missing:
        raise OrderMismatch(
            f"implied_body_length_mm: reference keypoints {missing} are not "
            f"tracking[...] sites in {model_xml}; the body-length anchor cannot "
            f"be measured and must not be guessed"
        )
    model_length = float(np.linalg.norm(rest[a] - rest[b]))
    return model_length / float(scale) * WORLD_UNITS_TO_MM


def assert_plausible_body_scale(scale: float, model_xml, *, context: str = "") -> float:
    """Raise unless `scale` implies a plausible *D. melanogaster* body length."""
    mm = implied_body_length_mm(scale, model_xml)
    if not np.isfinite(mm) or not (BODY_LENGTH_MM_MIN <= mm <= BODY_LENGTH_MM_MAX):
        a, b = BODY_LENGTH_REF_PAIR
        where = f" ({context})" if context else ""
        raise ValueError(
            f"implausible body scale{where}: scale={scale!r} implies a body "
            f"length of {mm:.3f} mm via {a}->{b} (model_dist / scale * "
            f"{WORLD_UNITS_TO_MM} mm/world-unit), outside the plausible "
            f"D. melanogaster range [{BODY_LENGTH_MM_MIN}, {BODY_LENGTH_MM_MAX}] mm. "
            f"This is the check that would have caught the historical 38x "
            f"scale-from-first-bout defect (0.000338 vs a good 0.010994) before "
            f"it poisoned a whole recording -- see WORLD_UNITS_TO_MM's anchors."
        )
    return mm


def rigid_segment_pairs(
    kp_order: Order | Sequence[str], *, include_thorax: bool = False
) -> list[tuple[str, str]]:
    """Keypoint NAME pairs spanning one rigid skeletal segment each."""
    names = set(as_order(kp_order).names)
    pairs: list[tuple[str, str]] = []
    for leg in LEG_NAMES:
        chain = [f"{leg}_{joint}" for joint in LEG_JOINT_CHAIN]
        for a, b in zip(chain[:-1], chain[1:], strict=True):
            if a in names and b in names:
                pairs.append((a, b))
    if include_thorax:
        pairs += [(a, b) for a, b in THORAX_PAIRS if a in names and b in names]
    return pairs


def _check_fits(kp3d: np.ndarray, order: Order, *, what: str) -> None:
    if kp3d.ndim != 3 or kp3d.shape[1] != len(order):
        raise OrderMismatch(
            f"{what}: kp3d has shape {kp3d.shape} but the keypoint order names "
            f"{len(order)} keypoints; an order that does not fit the array would "
            f"measure a different body part than it reports"
        )


def _segment_pair_measurements(
    kp3d, kp_order: Order | Sequence[str], model_xml, *, include_thorax: bool = False
) -> list[tuple[tuple[str, str], np.ndarray, float]]:
    """`[((name_a, name_b), observed distances, model rest distance), ...]`."""
    order = as_order(kp_order)
    pairs = rigid_segment_pairs(order, include_thorax=include_thorax)
    if not pairs:
        raise ValueError(
            f"no rigid-segment pairs available for the given keypoint order "
            f"(include_thorax={include_thorax}); it names none of the leg chains "
            f"{LEG_NAMES} joint by joint"
        )
    positions = np.asarray(kp3d, dtype=np.float64)
    _check_fits(positions, order, what="rigid segment measurement")
    rest = _tracking_site_rest(model_xml)
    valid = marker_validity_mask(positions)  # (T, K)

    measurements: list[tuple[tuple[str, str], np.ndarray, float]] = []
    for a, b in pairs:
        if a not in rest or b not in rest:
            continue  # not a model tracking site -- no model length to compare to
        ia, ib = order.index(a), order.index(b)
        both = valid[:, ia] & valid[:, ib]
        if not both.any():
            continue
        observed = np.linalg.norm(positions[both, ia, :] - positions[both, ib, :], axis=-1)
        model_dist = float(np.linalg.norm(rest[a] - rest[b]))
        if model_dist <= 0:
            continue
        measurements.append(((a, b), observed, model_dist))

    if not measurements:
        raise ValueError(
            "no rigid-segment pair had a single frame with both of its keypoints "
            "finite; this bout-fly cannot be measured"
        )
    return measurements


def segment_scale_diagnostics(
    kp3d, kp_order: Order | Sequence[str], model_xml, *, include_thorax: bool = False
) -> dict:
    """Rigid-segment scale for ONE bout-fly, with the physics checks on it."""
    measurements = _segment_pair_measurements(
        kp3d, kp_order, model_xml, include_thorax=include_thorax
    )
    per_pair_scale: list[float] = []
    within_cvs: list[float] = []
    for _pair, observed, model_dist in measurements:
        mean = float(np.mean(observed))
        if mean > 0:
            within_cvs.append(float(np.std(observed)) / mean)
        median = float(np.median(observed))
        if median > 0:
            per_pair_scale.append(model_dist / median)

    if not per_pair_scale:
        raise ValueError(
            "every usable rigid-segment pair had a degenerate (non-positive) "
            "median observed distance"
        )

    scales = np.asarray(per_pair_scale, dtype=np.float64)
    mean_scale = float(np.mean(scales))
    return {
        "scale": float(np.median(scales)),
        "per_pair_scale": scales,
        "within_bone_cv": float(np.mean(within_cvs)) if within_cvs else float("nan"),
        "across_bone_cv": float(np.std(scales) / mean_scale) if mean_scale > 0 else float("nan"),
        "n_pairs_used": int(scales.size),
    }


def per_bout_segment_scale(
    kp3d, kp_order: Order | Sequence[str], model_xml, *, include_thorax: bool = False
) -> np.ndarray:
    """`(n_pairs,)` implied body scales for ONE bout-fly, one per rigid segment."""
    return segment_scale_diagnostics(kp3d, kp_order, model_xml, include_thorax=include_thorax)[
        "per_pair_scale"
    ]


def _umeyama_scale_per_frame(positions: np.ndarray, ref_centered: np.ndarray) -> np.ndarray:
    """`(F,)` Umeyama least-squares similarity scale, one per frame."""
    centered = positions - positions.mean(axis=1, keepdims=True)  # (F, n, 3)
    h = np.einsum("ni,fnj->fij", ref_centered, centered)  # (F, 3, 3) = ref^T data
    _u, singular, _vt = np.linalg.svd(h)
    sign = np.sign(np.linalg.det(h))
    sign[sign == 0] = 1.0
    trace_ds = singular[:, 0] + singular[:, 1] + sign * singular[:, 2]
    denom = (centered**2).sum(axis=(1, 2))
    return trace_ds / np.maximum(denom, 1e-12)


def per_frame_scales(
    kp3d,
    kp_order: Order | Sequence[str],
    model_xml,
    *,
    trunk_names: Sequence[str] | None = None,
    estimator: str = "umeyama",
) -> np.ndarray:
    """`(F,)` per-frame scale estimates for ONE bout-fly, from a marker cloud."""
    order = as_order(kp_order)
    names = list(trunk_names) if trunk_names else list(DEFAULT_TRUNK_KEYPOINTS)
    rest = _tracking_site_rest(model_xml)
    present = [n for n in names if n in order and n in rest]
    if len(present) < 3:
        raise ValueError(
            f"per_frame_scales needs >=3 trunk markers present in both the keypoint "
            f"order and the model's tracking[...] sites; requested {names}, found "
            f"{present}"
        )

    ref = np.array([rest[n] for n in present], dtype=np.float64)
    ref_centered = ref - ref.mean(axis=0)
    ref_spread = float(np.sqrt((ref_centered**2).sum()))

    positions = np.asarray(kp3d, dtype=np.float64)
    _check_fits(positions, order, what="per_frame_scales")
    selected = positions[:, [order.index(n) for n in present], :]
    selected = selected[finite_frame_mask(selected)]
    if selected.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)

    if estimator == "norm_ratio":
        spreads = np.sqrt(((selected - selected.mean(axis=1, keepdims=True)) ** 2).sum(axis=(1, 2)))
        with np.errstate(divide="ignore", invalid="ignore"):
            return ref_spread / spreads
    if estimator == "umeyama":
        return _umeyama_scale_per_frame(selected, ref_centered)
    raise ValueError(f"per_frame_scales: unknown estimator {estimator!r}")


def warn_if_estimator_ignored(estimator: str, *, caller: str = "estimate_fly_scale") -> None:
    """Announce that a non-default `estimator` is about to be ignored."""
    if estimator != "umeyama":
        _announce(
            caller,
            f"estimator={estimator!r} is ignored for "
            f"scale_keypoints='rigid_segment' -- a rigid segment's length is "
            f"measured directly, not fit.",
        )


def robust_scale(per_bout: Mapping, *, mad_k: float = 3.0) -> dict:
    """Pool one fly's per-bout scale samples and reject outlier BOUTS."""
    usable: dict = {}
    for bout, samples in per_bout.items():
        arr = np.asarray(samples, dtype=np.float64).ravel()
        arr = arr[np.isfinite(arr) & (arr > 0)]
        if arr.size:
            usable[bout] = arr

    if not usable:
        raise ValueError("robust_scale: no usable (finite, positive) scale samples in any bout")

    bouts = sorted(usable)
    per_bout_median = {b: float(np.median(usable[b])) for b in bouts}
    medians = np.array([per_bout_median[b] for b in bouts], dtype=np.float64)

    pooled_median = float(np.median(np.concatenate([usable[b] for b in bouts])))
    med_of_medians = float(np.median(medians))
    mad = float(np.median(np.abs(medians - med_of_medians)))

    if mad <= 0:
        outlier_bouts: list = []
    else:
        outlier_bouts = [b for b in bouts if abs(per_bout_median[b] - pooled_median) > mad_k * mad]

    if outlier_bouts and len(outlier_bouts) == len(bouts):
        _announce(
            "robust_scale",
            f"MAD outlier rejection flagged ALL {len(bouts)} bouts as "
            f"outliers (near-zero genuine cross-bout spread -- float-precision noise "
            f"alone crossed the {mad_k} * MAD threshold); pooling all bouts "
            f"UNFILTERED instead of discarding every one.",
        )

    keep = [b for b in bouts if b not in outlier_bouts] or bouts
    final_scale = float(np.median(np.concatenate([usable[b] for b in keep])))

    if len(medians) > 1 and med_of_medians:
        p10, p90 = np.percentile(medians, [10, 90])
        spread_pct = float((p90 - p10) / med_of_medians * 100)
    else:
        spread_pct = 0.0
    mean_of_medians = float(np.mean(medians))
    scale_cv = (
        float(np.std(medians) / mean_of_medians) if len(medians) > 1 and mean_of_medians else 0.0
    )

    return {
        "scale": final_scale,
        "n_bouts": len(bouts),
        "n_samples": int(sum(a.size for a in usable.values())),
        "per_bout_median": per_bout_median,
        "outlier_bouts": outlier_bouts,
        "spread_pct": spread_pct,
        "scale_cv_across_bouts": scale_cv,
    }


def bout_kp3d_paths(run_root, fly: int) -> list[Path]:
    """Every `<run_root>/bouts/bout_*/fly<fly>/kp3d_filt.npz`, sorted by bout."""
    bouts_dir = Path(run_root) / "bouts"
    if not bouts_dir.is_dir():
        return []
    found: list[tuple[int, Path]] = []
    for d in sorted(bouts_dir.iterdir()):
        m = _BOUT_DIR_RE.match(d.name) if d.is_dir() else None
        if not m:
            continue
        fly_dir = d / f"fly{fly}"
        for name in ("kp3d_filt.npz", "kp3d.npz"):
            if (fly_dir / name).exists():
                found.append((int(m.group(1)), fly_dir / name))
                break
    return [p for _, p in sorted(found, key=lambda t: t[0])]


def bout_index(path: Path) -> int:
    m = _BOUT_DIR_RE.match(path.parent.parent.name)
    if not m:
        raise ValueError(f"cannot parse a bout index from {path}")
    return int(m.group(1))


def read_bout_kp3d(path: Path, order: Order) -> np.ndarray:
    """`(T, K, 3)` from a bout artifact, checked against `order` BY NAME."""
    with np.load(path, allow_pickle=True) as z:
        if "kp3d" not in z.files:
            raise KeyError(f"{path} has no 'kp3d' array (it has {sorted(z.files)})")
        kp3d = np.asarray(z["kp3d"], dtype=np.float64)
        named = "kp_names" in z.files
        if named:
            require_same(Order([str(n) for n in z["kp_names"]]), order, what="keypoint")
    _check_fits(kp3d, order, what=str(path))
    if not named:
        _announce(
            "read_bout_kp3d",
            f"{path} carries no kp_names -- ASSUMING its {kp3d.shape[1]} "
            f"keypoints are the caller's order (sha {order.sha}, starting "
            f"{list(order.names[:3])}). Nothing in the file confirms this; an "
            f"array written in a different keypoint order would be measured as "
            f"this one, with healthy-looking rigidity and a plausible body length.",
        )
    return kp3d


def _load_per_bout_scales(
    run_root,
    fly: int,
    kp_order: Order | Sequence[str],
    model_xml,
    *,
    scale_keypoints: str = "rigid_segment",
    include_thorax: bool = False,
    trunk_names: Sequence[str] | None = None,
    estimator: str = "umeyama",
) -> tuple[dict[int, np.ndarray], dict[int, str]]:
    """`({bout: scale samples}, {bout: file name it was read from})` for one fly."""
    order = as_order(kp_order)
    if scale_keypoints == "rigid_segment":
        warn_if_estimator_ignored(estimator, caller="estimate_fly_scale")
    elif scale_keypoints == "trunk":
        trunk_names = list(trunk_names) if trunk_names else list(DEFAULT_TRUNK_KEYPOINTS)
    elif scale_keypoints == "all":
        trunk_names = list(order.names)
    else:
        raise ValueError(
            f"scale_keypoints must be 'rigid_segment', 'trunk' or 'all', got {scale_keypoints!r}"
        )

    per_bout: dict[int, np.ndarray] = {}
    per_bout_source: dict[int, str] = {}
    for path in bout_kp3d_paths(run_root, fly):
        bout = bout_index(path)
        kp3d = read_bout_kp3d(path, order)
        if scale_keypoints == "rigid_segment":
            try:
                diag = segment_scale_diagnostics(
                    kp3d, order, model_xml, include_thorax=include_thorax
                )
            except ValueError as exc:
                _announce(
                    "estimate_fly_scale",
                    f"bout {bout} fly{fly} is not measurable, skipped -- {exc}",
                )
                continue
            if diag["within_bone_cv"] > WITHIN_BONE_CV_WARN_THRESH:
                _announce(
                    "estimate_fly_scale",
                    f"bout {bout} fly{fly} within_bone_cv "
                    f"{diag['within_bone_cv'] * 100:.1f}% > "
                    f"{WITHIN_BONE_CV_WARN_THRESH * 100:.0f}% -- its keypoints break "
                    f"rigidity (a bone changing length), so this bout's scale of "
                    f"{diag['scale']:.6f} is not trustworthy.",
                )
            samples = diag["per_pair_scale"]
        else:
            samples = per_frame_scales(
                kp3d, order, model_xml, trunk_names=trunk_names, estimator=estimator
            )
        if samples.size:
            per_bout[bout] = samples
            per_bout_source[bout] = path.name
    return per_bout, per_bout_source


def estimate_fly_scale(
    run_root,
    fly: int,
    kp_order: Order | Sequence[str],
    model_xml,
    *,
    mad_k: float = 3.0,
    scale_keypoints: str = "rigid_segment",
    include_thorax: bool = False,
    trunk_names: Sequence[str] | None = None,
    estimator: str = "umeyama",
) -> dict:
    """One body scale for ONE fly, pooled over every bout of that fly."""
    per_bout, per_bout_source = _load_per_bout_scales(
        run_root,
        fly,
        kp_order,
        model_xml,
        scale_keypoints=scale_keypoints,
        include_thorax=include_thorax,
        trunk_names=trunk_names,
        estimator=estimator,
    )
    if not per_bout:
        raise ValueError(
            f"estimate_fly_scale: no measurable bout for fly{fly} under {run_root}; "
            f"a scale cannot be pooled from nothing and must not be guessed"
        )
    out = robust_scale(per_bout, mad_k=mad_k)
    out["fly"] = int(fly)
    out["scale_keypoints"] = scale_keypoints
    out["per_bout_source"] = dict(sorted(per_bout_source.items()))
    out["sources_mixed"] = len(set(per_bout_source.values())) > 1
    if out["sources_mixed"]:
        by_source: dict[str, list[int]] = {}
        for bout, name in sorted(per_bout_source.items()):
            by_source.setdefault(name, []).append(bout)
        _announce(
            "estimate_fly_scale",
            f"fly{int(fly)}'s pooled scale MIXES artifacts -- "
            + "; ".join(f"{name}: bouts {bouts}" for name, bouts in sorted(by_source.items()))
            + ". kp3d_filt.npz and kp3d.npz are not of equal quality (the filter "
            "degrades the female's rigid-bone CV, and on the reference bout the two "
            "differ by 0.011711 vs 0.011725), so this number is an average over two "
            "kinds of measurement. Finish preprocessing the fly's bouts to remove it.",
        )
    out["implied_body_length_mm"] = assert_plausible_body_scale(
        out["scale"], model_xml, context=f"{run_root} fly{fly}"
    )
    return out
