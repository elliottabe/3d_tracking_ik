"""The bout-fly stages -- the committed form of what was a scratch script.

This module is the point of the whole plan: the integration chain that used to
live in `/tmp`, written three times and lost twice, now lives here instead.
Nothing below is new numerics -- every call delegates to `tracking.preprocess`,
`tracking.inverse_kinematics` or `tracking.postprocess` -- this module only
decides WHAT gets called, WITH WHAT, and AT WHICH SCOPE.

**The one thing that matters most: two different scopes.** Spec 5.1 records
that fitting a fly's body scale and marker offsets from ONE bout that happened
to run first produced a male fitted with the female's offsets (abdomen 1.10-
1.12x too long) and, earlier, a 38x body-scale error -- both silent, because
the residual simply absorbed the wrong constant. This module keeps the two
scopes apart on purpose:

- `fit_fly_constants` runs ONCE per fly per recording, over a sample POOLED
  across every bout that fly appears in, and writes `scale.json` and
  `offsets_fly<f>.h5` at the RUN ROOT (beside `bouts/`, not inside one).
- `preprocess_bout_fly`, `ik_bout_fly` and `postprocess_bout_fly` each run for
  ONE bout-fly and write only into that bout-fly's own directory. None of them
  may write a per-fly artifact -- that would be the fit-once-on-whichever-
  bout-arrived-first defect again, reached through a different door.

**The other thing that matters: both axes are resolved by NAME.** `kp3d.npz`'s
own keypoint order and a `kp2d.npz`'s own camera order are not guaranteed to
match this run's canonical orders (`anatomy.kp_order`, `rig.cameras`) -- a
detector or an upstream stage can write either in a different, self-consistent
order, and indexing it positionally gives confident, healthy-looking numbers
for the wrong body part or the wrong camera. `load_bout_kp3d`/`load_bout_kp2d`
resolve both axes through `Order.permutation_to`, never through a bare shape
match.
"""

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


# `select_offsets_sample` requires these explicitly (no defaults of its own,
# so a caller cannot forget to think about them); these are fallbacks for a
# caller that passes none. The driver does NOT reach them -- it passes
# `cfg.preprocess.offsets.*` -- so they are kept EQUAL to
# `configs/preprocess/default.yaml` on purpose: two defaults for one quantity
# that disagree is how a value gets "changed" in the config and silently not
# changed for everyone else.
#
# `OFFSETS_N_FRAMES = 500` is traced to source (`configs/anatomy/v1.yaml:258`
# `N_FRAMES_PER_CLIP: 500`, the reference run's recorded
# `n_frames_per_clip: 500`, and its `offsets_fly*.h5` holding 500 rows), not
# chosen. It read 300 until 2026-09-13, described here as a "sane default":
# a number that moves the offsets fit, the IK and every downstream metric,
# picked for plausibility rather than provenance. See that config file.
#
# `mad_k=3.0` matches `tracking.preprocess.scale.robust_scale`'s own default,
# so the offsets sample's outlier gate and the body-scale pooling use the same
# tolerance unless a caller deliberately diverges them. NOTE: the reference
# records no offsets `mad_k` at all (`robust: none`), so this one is an
# internal choice rather than a ported value -- still open.
OFFSETS_N_FRAMES = 500
OFFSETS_MIN_CONF = 0.8
OFFSETS_MAD_K = 3.0


def load_bout_kp3d(path, *, kp_order: Order) -> tuple[np.ndarray, np.ndarray | None]:
    """`(kp3d, conf3d)` from a `kp3d[_filt].npz`, keypoint axis resolved BY NAME.

    The file's own `kp_names` -- NOT `kp_order` -- says what order its
    `kp3d`/`conf3d` arrays are actually in; a file written by an upstream stage
    in a different (but self-consistent) order is silently correct-shaped and
    wrong, which is this repo's most expensive bug class. `conf3d` is `None`
    when the file carries none (a raw `kp3d.npz` may not).

    A `kp_names` set that disagrees with `kp_order` -- an extra or a missing
    name -- raises `ValueError` naming the offending keypoint(s), never drops
    them: a silently intersected keypoint axis is a shorter one every
    downstream shape check still accepts.
    """
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
    """`(kp2d, conf)` from a `kp2d.npz`, BOTH axes resolved BY NAME.

    A `kp2d.npz`'s own stored `cameras` order need not match `cameras`
    (`rig.cameras`/the canonical calibration-glob order) any more than its
    `kp_names` need match `kp_order` -- CLAUDE.md records this exact failure:
    the npz's own camera order disagreeing with the canonical one plotted one
    camera's keypoints on another camera's image, and the result still looked
    almost plausible. Both axes are therefore resolved the same way, by
    `Order.permutation_to`, never assumed from position.

    The confidence array is stored under the key `conf` (NOT `conf2d` -- that
    key has never existed in a real `kp2d.npz`; `conf3d` is the analogous key
    in `kp3d.npz`, a different, genuinely-optional file). `conf` is REQUIRED
    here, unlike `conf3d` in `load_bout_kp3d`: `bout_qc`/`ik_reproj_report`,
    its sole consumer, cannot accept `None` -- it indexes straight into the
    array -- so treating a missing `conf` as optional would only turn a clear
    `KeyError` at load time into an obscure `IndexError` deep inside QC,
    hundreds of lines from the actual cause.
    """
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
    """The offsets fit's `solver_cfg`: `anatomy.cfg['model']`, temporal terms off.

    `offsets_fit_cfg` is the ONLY thing that turns the smoothness terms off
    (the offsets sample is frames pooled from all over the recording, not a
    time series -- coupling them would drag the fit toward the average of
    unrelated poses). Reusing it here rather than re-deriving the same edit is
    what keeps this call and the solver's config wiring from silently
    disagreeing about which terms are off.
    """
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
    """Fit ONE fly's body scale and marker offsets, pooled across every bout.

    This is the Primary guarantee's enforcement point. Both quantities are
    fit from a sample spanning every bout `kp3d_by_bout` offers -- never from
    whichever bout happens to be first -- and both are written at the RUN
    ROOT (`scale.json`, `offsets_fly<f>.h5`), never into a bout directory:
    that split is what stops a per-bout caller from re-deriving these
    constants from one bout's data, which is the historic defect (spec 5.1).

    Body scale: `segment_scale_diagnostics` on every bout's `kp3d`
    CONCATENATED (frames from every bout vote on the same median, so this
    "pools", it does not "pick"), then `assert_plausible_body_scale` checks
    the result against physics and RAISES `ValueError` if it fails -- on
    purpose, not softened to an announcement: a wrong body scale silently
    poisons a whole recording (the historic 38x scale-from-first-bout defect),
    and letting a fly with an implausible scale proceed to a full IK solve and
    a QC report would make it read as a poor fit rather than as the broken
    measurement it is. Nothing is written for this fly when this raises --
    `scale.json` is untouched, so a later stage cannot read a scale that was
    never actually validated.

    Marker offsets: `select_offsets_sample` draws a stratified, round-robin
    sample across every bout (never letting one bout dominate), gated by an
    OPTIONAL per-frame scale consistency check (`aligned_per_frame_scales` +
    `per_frame_scales`) that rejects a bout-frame whose implied scale is a
    MAD-outlier against the fly's own pooled median -- this is a relative,
    within-fly consistency gate, independent of the physical-plausibility
    check above. `fit_offsets` (a GPU solve) then fits the per-marker offset
    on that sample, in WORLD units -- it applies `scaled_model_keypoints`
    itself.

    `scale.json` is a shared, per-recording file -- one call per fly updates
    its own `scale_by_fly[str(fly)]` entry, read-modify-write, so calling this
    once per fly (never once per run) does not clobber the other fly's entry.

    Returns a dict of the numbers the driver logs: `fly`, `scale`,
    `n_pairs_used`, `implied_body_length_mm`, `scale_path`, `offsets_path`,
    `bouts_offered`, `bouts_drawn`, `n_offset_frames`.
    """
    kp_order = anatomy.kp_order
    bouts = sorted(kp3d_by_bout)

    # -- Body scale: pooled across every bout's frames, not per-bout-then-picked.
    pooled_kp3d = np.concatenate([np.asarray(kp3d_by_bout[b], dtype=np.float64) for b in bouts])
    diag = segment_scale_diagnostics(
        pooled_kp3d, kp_order, model_xml, include_thorax=include_thorax
    )
    scale = float(diag["scale"])
    # RAISES ValueError on an implausible scale -- deliberately not caught. See
    # this function's docstring: a wrong scale must stop the fly's fit here,
    # not degrade to a flag a later stage might not check.
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
    """Filter ONE bout-fly's `kp3d.npz`, writing ONLY `kp3d_filt.npz`.

    Never writes `scale.json` or an `offsets_fly<f>.h5` -- those are
    `fit_fly_constants`'s artifacts, fit once per fly over every bout; a
    per-bout function that wrote them would be re-deriving a per-fly constant
    from one bout, the Primary-guarantee defect reached through a different
    door.

    `cfg` defaults to `DEFAULT_FILTER_CFG` (the shipped settings) via
    `filter_kp3d` itself. `kp3d_filt.npz` carries `kp_names` alongside
    `kp3d`/`conf3d` -- CLAUDE.md's governing rule is that a keypoint axis is
    resolved BY NAME, never positionally, because that exact assumption once
    produced "a correct measurement of the wrong thing, and the best-looking
    keypoint in the table" while every jitter/confidence/residual metric
    rated it as excellent. Every downstream stage
    (`fit_fly_constants`/`ik_bout_fly`/`postprocess_bout_fly`) reads this file
    through `load_bout_kp3d`, which is the STRICT loader and requires
    `kp_names`; omitting it here previously forced a choice between crashing
    every downstream stage or reading through the TOLERANT loader
    (`tracking.preprocess.scale.read_bout_kp3d`) in the one place the axis
    must be checked, not assumed. Earlier this omitted `kp_names` to keep
    `kp3d_filt.npz` byte-comparable with the reference recording's own
    filtered artifacts (which predate this convention); that trade does not
    hold up -- nothing asserts byte equality of our own `kp3d_filt.npz` (the
    parity check reads the REFERENCE's files and compares array VALUES, never
    ours), and the tolerant reader
    VERIFIES `kp_names` against the expected order when present, falling back
    to assuming only when it is absent -- so adding the key only moves our
    files from its assumed path onto its checked path. The written `kp_names`
    is `list(kp_order)`, never re-derived from the input file: `kp3d`/`conf3d`
    were already permuted INTO `kp_order` by `load_bout_kp3d` above, and
    `filter_kp3d` preserves that order, so `kp_order` is the one name for what
    axis order the arrays below are actually in.
    """
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
    """Solve ONE bout-fly's per-frame IK, writing `stac_ik.h5`.

    `kp3d_filt` is WORLD units (0.1 mm); `scaled_model_keypoints` converts it
    to MODEL units (`* scale * anatomy.mocap_scale_factor`) before it reaches
    `solve_bout`, which expects model units and refuses a `kp3d_model` that is
    not `(T, K, 3)`. Passing `kp3d_filt` straight through -- world units where
    the solver expects model units -- is the 38x-class defect: nothing raises
    or goes NaN, the fit is merely worse, and the offsets quietly absorb the
    difference.

    `offsets` is the fly's ALREADY-FIT `(K, 3)` marker offsets (from
    `fit_fly_constants`, loaded by the caller) -- this function never fits
    them itself, which is what keeps the per-bout scope from re-deriving a
    per-fly constant.
    """
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
    """Derive and score ONE bout-fly's `outputs.h5`/`fitted.npz`/`qc.json`.

    Reads `stac_ik.h5` (this bout-fly's IK solve) plus its `kp3d_filt.npz`
    (what the solve was fitted to), `kp3d.npz` (the raw triangulation QC
    scores against) and `kp2d.npz` (for the 2D reprojection and LOO checks),
    all resolved by name (`load_bout_kp3d`/`load_bout_kp2d`). `rig`, `floor`
    and `sex` are recording-level inputs the caller supplies -- `rig` (camera
    calibration) and `floor` (`tracking.qc.posture.fit_recording_floor`,
    pooled over the WHOLE recording's fly root positions, per spec 5.3) are
    not bout-fly-scoped quantities, and computing either from one bout alone
    would reintroduce the same one-bout-defines-it defect this module exists
    to avoid, just for a different constant. `kp_scale` is this fly's
    `scale.json` value.
    """
    bout_dir = Path(bout_dir)
    kp_order = as_order(kp_order)
    cameras = as_order(cameras)

    stac_path = bout_dir / "stac_ik.h5"
    if not stac_path.exists():
        # The `ik` stage raised `NotFit` for this fly, so there is nothing to
        # postprocess. Re-raising the same class keeps the whole bout-fly one
        # outcome rather than turning it into a FileNotFoundError two stages
        # later, which reads as a pipeline bug instead of an untracked fly.
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
