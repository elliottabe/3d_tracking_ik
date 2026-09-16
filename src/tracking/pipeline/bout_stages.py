"""The bout-fly stages -- the committed form of what was a scratch script."""

from __future__ import annotations

import functools
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from tracking.conventions import NotFit, announce
from tracking.inverse_kinematics.offsets_fit import (
    fit_offsets,
    scaled_model_keypoints,
    write_offsets_h5,
)
from tracking.inverse_kinematics.perframe import (
    solve_bout,
    solver_settings_from_cfg,
    write_stac_h5,
)
from tracking.io.names import Order, OrderMismatch, as_order
from tracking.postprocess.outputs import (
    build_fly_outputs,
    read_stac_h5,
    write_fitted_npz,
    write_outputs_h5,
)
from tracking.preprocess.filter import filter_kp3d
from tracking.preprocess.offsets import (
    aligned_per_frame_scales,
    offsets_fit_cfg,
    per_fly_offsets_name,
    select_offsets_sample,
)
from tracking.preprocess.scale import (
    assert_plausible_body_scale,
    per_frame_scales,
    segment_scale_diagnostics,
)
from tracking.qc.bout import bout_qc, write_qc_json

__all__ = [
    "OFFSETS_MAD_K",
    "OFFSETS_MIN_CONF",
    "OFFSETS_N_FRAMES",
    "fit_fly_constants",
    "ik_bout_fly",
    "load_bout_kp2d",
    "load_bout_kp3d",
    "postprocess_bout_fly",
    "preprocess_bout_fly",
]


OFFSETS_N_FRAMES = 500
OFFSETS_MIN_CONF = 0.8
OFFSETS_MAD_K = 3.0


def load_bout_kp3d(path, *, kp_order: Order) -> tuple[np.ndarray, np.ndarray | None]:
    """`(kp3d, conf3d)` from a `kp3d[_filt].npz`, keypoint axis resolved BY NAME."""
    order = as_order(kp_order)
    with np.load(path, allow_pickle=True) as z:
        missing = [k for k in ("kp3d", "kp_names") if k not in z.files]
        if missing:
            raise KeyError(f"{path} has no {missing}; found {sorted(z.files)}")
        kp3d = np.asarray(z["kp3d"], dtype=np.float64)
        conf3d = np.asarray(z["conf3d"], dtype=np.float64) if "conf3d" in z.files else None
        file_order = Order([str(n) for n in z["kp_names"]])
    try:
        perm = file_order.permutation_to(order)
    except OrderMismatch as exc:
        raise ValueError(f"{path}: {exc}") from exc
    kp3d = kp3d[:, perm]
    if conf3d is not None:
        conf3d = conf3d[:, perm]
    return kp3d, conf3d


def load_bout_kp2d(path, *, kp_order: Order, cameras: Order) -> tuple[np.ndarray, np.ndarray]:
    """`(kp2d, conf)` from a `kp2d.npz`, BOTH axes resolved BY NAME."""
    kp = as_order(kp_order)
    cams = as_order(cameras)
    with np.load(path, allow_pickle=True) as z:
        missing = [k for k in ("kp2d", "kp_names", "cameras", "conf") if k not in z.files]
        if missing:
            raise KeyError(f"{path} has no {missing}; found {sorted(z.files)}")
        kp2d = np.asarray(z["kp2d"], dtype=np.float64)
        conf = np.asarray(z["conf"], dtype=np.float64)
        file_kp_order = Order([str(n) for n in z["kp_names"]])
        file_cam_order = Order([str(c) for c in z["cameras"]])
    try:
        kp_perm = file_kp_order.permutation_to(kp)
        cam_perm = file_cam_order.permutation_to(cams)
    except OrderMismatch as exc:
        raise ValueError(f"{path}: {exc}") from exc
    kp2d = kp2d[:, cam_perm][:, :, kp_perm]
    conf = conf[:, cam_perm][:, :, kp_perm]
    return kp2d, conf


def _solver_cfg_from_anatomy(anatomy) -> dict[str, Any]:
    """The offsets fit's `solver_cfg`: `anatomy.cfg['model']`, temporal terms off."""
    raw = {"anatomy": {"model": dict(anatomy.cfg["model"])}}
    return offsets_fit_cfg(raw)["anatomy"]["model"]


def fit_fly_constants(
    anatomy,
    *,
    fly: int,
    kp3d_by_bout: Mapping[int, np.ndarray],
    conf_by_bout: Mapping[int, np.ndarray | None],
    run_root,
    model_xml,
    n_frames: int = OFFSETS_N_FRAMES,
    min_conf: float = OFFSETS_MIN_CONF,
    mad_k: float = OFFSETS_MAD_K,
    include_thorax: bool = False,
) -> dict[str, Any]:
    """Fit ONE fly's body scale and marker offsets, pooled across every bout."""
    kp_order = anatomy.kp_order
    bouts = sorted(kp3d_by_bout)

    # -- Body scale: pooled across every bout's frames, not per-bout-then-picked.
    pooled_kp3d = np.concatenate([np.asarray(kp3d_by_bout[b], dtype=np.float64) for b in bouts])
    diag = segment_scale_diagnostics(
        pooled_kp3d, kp_order, model_xml, include_thorax=include_thorax
    )
    scale = float(diag["scale"])
    implied_mm = assert_plausible_body_scale(scale, model_xml, context=f"fly{fly} at {run_root}")

    # -- Offsets sample: stratified across every bout, gated by within-fly scale
    #    consistency (never by the physical plausibility checked above).
    per_frame_fn = functools.partial(per_frame_scales, kp_order=kp_order, model_xml=model_xml)
    scale_by_bout: dict[int, np.ndarray] | None = {}
    try:
        for b in bouts:
            scale_by_bout[b] = aligned_per_frame_scales(
                np.asarray(kp3d_by_bout[b], dtype=np.float64), per_frame_fn
            )
    except ValueError as exc:
        announce(
            "pipeline.bout_stages",
            "fit_fly_constants",
            f"fly{fly}: per-frame scale gate unavailable ({exc}); the offsets "
            "sample proceeds WITHOUT the scale-outlier gate (confidence gating "
            "still applies).",
        )
        scale_by_bout = None

    sample = select_offsets_sample(
        kp3d_by_bout,
        conf_by_bout,
        n_frames=n_frames,
        min_conf=min_conf,
        scale_by_bout=scale_by_bout,
        mad_k=mad_k,
    )

    solver_cfg = _solver_cfg_from_anatomy(anatomy)
    fit = fit_offsets(anatomy, sample.kp3d, scale=scale, solver_cfg=solver_cfg)

    offsets_path = Path(run_root) / per_fly_offsets_name(fly)
    write_offsets_h5(
        offsets_path,
        fit,
        anatomy=anatomy,
        cfg={
            "scale": scale,
            "solver_cfg": solver_cfg,
            "sample_provenance": sample.provenance,
        },
    )

    scale_path = Path(run_root) / "scale.json"
    existing: dict[str, Any] = {}
    if scale_path.exists():
        existing = json.loads(scale_path.read_text())
    scale_by_fly = dict(existing.get("scale_by_fly", {}))
    scale_by_fly[str(int(fly))] = {
        "scale": scale,
        "n_pairs_used": int(diag["n_pairs_used"]),
        "implied_body_length_mm": implied_mm,
        "bouts_offered": bouts,
    }
    payload = {**existing, "scale_by_fly": scale_by_fly}
    tmp_path = scale_path.with_suffix(scale_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(tmp_path, scale_path)

    return {
        "fly": int(fly),
        "scale": scale,
        "n_pairs_used": int(diag["n_pairs_used"]),
        "implied_body_length_mm": implied_mm,
        "scale_path": str(scale_path),
        "offsets_path": str(offsets_path),
        "bouts_offered": bouts,
        "bouts_drawn": sorted({b for b, _frame in sample.frames}),
        "n_offset_frames": len(sample.frames),
    }


def preprocess_bout_fly(
    anatomy,
    *,
    bout_dir,
    kp_order: Order,
    cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Filter ONE bout-fly's `kp3d.npz`, writing ONLY `kp3d_filt.npz`."""
    bout_dir = Path(bout_dir)
    kp_order = as_order(kp_order)
    kp3d, conf3d = load_bout_kp3d(bout_dir / "kp3d.npz", kp_order=kp_order)
    filtered, report = filter_kp3d(kp3d, conf3d, kp_order, cfg if cfg is not None else None)

    out_path = bout_dir / "kp3d_filt.npz"
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    arrays: dict[str, np.ndarray] = {
        "kp3d": filtered,
        "kp_names": np.array(list(kp_order), dtype=object),
    }
    if conf3d is not None:
        arrays["conf3d"] = conf3d
    np.savez(tmp_path, **arrays)
    produced = tmp_path if tmp_path.exists() else tmp_path.with_suffix(tmp_path.suffix + ".npz")
    os.replace(produced, out_path)

    return {
        "bout_dir": str(bout_dir),
        "kp3d_filt_path": str(out_path),
        "n_frames": int(kp3d.shape[0]),
        "n_confidence_masked": report.n_confidence_masked,
        "n_wing_excluded": report.n_wing_excluded,
        "n_bone_flagged": report.n_bone_flagged,
        "n_bones_flagged": report.n_bones_flagged,
        "n_bone_edges_skipped": report.n_bone_edges_skipped,
        "n_spikes_fixed": report.n_spikes_fixed,
        "n_interp_values": report.n_interp_values,
        "n_edge_phantom_frames": report.n_edge_phantom_frames,
    }


def ik_bout_fly(
    anatomy,
    *,
    bout_dir,
    kp3d_filt: np.ndarray,
    offsets: np.ndarray,
    scale: float,
    per_frame_cfg: Mapping[str, Any],
    solve_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Solve ONE bout-fly's per-frame IK, writing `stac_ik.h5`."""
    bout_dir = Path(bout_dir)
    kp3d_filt = np.asarray(kp3d_filt, dtype=np.float64)
    n_frames, n_kp = kp3d_filt.shape[0], kp3d_filt.shape[1]

    settings = solver_settings_from_cfg(per_frame_cfg)
    kp3d_model = scaled_model_keypoints(
        kp3d_filt, scale=scale, mocap_scale_factor=anatomy.mocap_scale_factor
    ).reshape(n_frames, n_kp, 3)

    result = solve_bout(
        anatomy,
        kp3d_model,
        offsets=offsets,
        settings=settings,
        per_frame_cfg=per_frame_cfg,
        solve_mask=solve_mask,
    )

    stac_path = bout_dir / "stac_ik.h5"
    write_stac_h5(stac_path, result, anatomy=anatomy, cfg=per_frame_cfg)

    summary: dict[str, Any] = {
        "bout_dir": str(bout_dir),
        "stac_path": str(stac_path),
        "n_frames": n_frames,
    }
    solved = getattr(result, "solved", None)
    if solved is not None:
        summary["n_solved"] = int(np.count_nonzero(solved))
    return summary


def postprocess_bout_fly(
    anatomy,
    *,
    bout_dir,
    kp_order: Order,
    cameras: Order,
    rig,
    floor,
    sex: str,
    kp_scale: float,
    source: str = "kp3d_filt",
    conf_thresh: float = 0.3,
) -> dict[str, Any]:
    """Derive and score ONE bout-fly's `outputs.h5`/`fitted.npz`/`qc.json`."""
    bout_dir = Path(bout_dir)
    kp_order = as_order(kp_order)
    cameras = as_order(cameras)

    stac_path = bout_dir / "stac_ik.h5"
    if not stac_path.exists():
        raise NotFit(
            f"postprocess: {stac_path} does not exist -- `ik` did not fit this "
            f"fly (see the run's not-fit list)"
        )
    stac = read_stac_h5(stac_path)
    kp3d_filt, conf3d_filt = load_bout_kp3d(bout_dir / "kp3d_filt.npz", kp_order=kp_order)
    kp3d_raw, conf3d_raw = load_bout_kp3d(bout_dir / "kp3d.npz", kp_order=kp_order)
    kp2d, conf2d = load_bout_kp2d(bout_dir / "kp2d.npz", kp_order=kp_order, cameras=cameras)

    out = build_fly_outputs(
        anatomy,
        stac,
        kp3d_world=kp3d_filt,
        conf3d=conf3d_filt,
        kp_scale=kp_scale,
        source=source,
    )
    write_outputs_h5(bout_dir / "outputs.h5", out, anatomy=anatomy, kp_order=kp_order)
    write_fitted_npz(bout_dir / "fitted.npz", out, kp_order=kp_order)

    report = bout_qc(
        kp2d=kp2d,
        conf2d=conf2d,
        kp3d_raw=kp3d_raw,
        conf3d=conf3d_raw,
        fitted_world=out.kp3d_mm,
        kp_order=kp_order,
        rig=rig,
        floor=floor,
        sex=sex,
        conf_thresh=conf_thresh,
    )
    write_qc_json(bout_dir / "qc.json", report)

    return {
        "bout_dir": str(bout_dir),
        "n_frames": int(out.qpos.shape[0]),
        "n_bridge_ok": int(np.count_nonzero(out.bridge_ok)),
        "structural_ok": bool(report.get("structural_ok", False)),
    }
