"""Per-fly STAC marker-offset sample, pooled over a recording."""

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
    """The frames the offsets fit runs on, and the record of how they were chosen."""

    kp3d: np.ndarray
    frames: list[tuple[int, int]]
    provenance: dict


def load_fly_bouts(
    run_root,
    fly: int,
    *,
    kp_order: Order | Sequence[str] | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray | None]]:
    """`{bout index: (kp3d (F,K,3), conf3d (F,K) or None)}` for ONE fly."""
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
    """One stderr line when a fly's bouts did not all come from the same file."""
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
    """`(candidate mask, per-frame min conf, counts)` for one bout."""
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
    """Gate every frame of every bout, then draw a stratified sample."""
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
    """`(T,)` implied body scale aligned to `kp3d`'s frames, NaN where unusable."""
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
    """`(path, mode)` with mode `'per_fly'` or `'shared'`."""
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
    """`container[key]` as a plain dict, creating or replacing it when absent."""
    branch = container.get(key)
    if branch is None:
        branch = {}
        container[key] = branch
    else:
        _refuse_non_dict(branch, what=f"cfg[{path!r}]")
    return branch


def _refuse_non_dict(value, *, what: str) -> None:
    """Refuse anything but a plain `dict`, naming the type and the conversion."""
    refuse_config_object(value, what=what, api="offsets_fit_cfg", allow=(dict,))


def offsets_fit_cfg(cfg) -> dict:
    """A deep copy of `cfg` for the offsets fit only, temporal terms OFF."""
    _refuse_non_dict(cfg, what="cfg")
    fit = copy.deepcopy(cfg)
    model = _plain_branch(
        _plain_branch(fit, "anatomy", path="anatomy"), "model", path="anatomy.model"
    )
    model["JAXLS_SMOOTH_WEIGHT"] = 0.0
    model["JAXLS_SMOOTH_Q_MULT"] = None
    model["JAXLS_CHUNK_SIZE"] = 0
    return fit
