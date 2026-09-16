"""The mask-free FINE pass: video frames -> per-fly `kp2d.npz`/`kp3d.npz`."""

from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import numpy as np

from tracking.detector.centerdetect.detector import cluster_centres, lift_peaks_to_centres
from tracking.detector.mvq.gates import gate_string
from tracking.detector.mvq.policy import SLOT_FEMALE, SLOT_MALE
from tracking.detector.mvq.runner import (
    COLLAPSE_DIST_UNITS,
    HALLUCINATION_VIS_THRESH,
    SEX_UNKNOWN,
    _passes_vis_guard,
    pick_typed_pair,
    prefer_own_window,
)
from tracking.detector.tracking_state import (
    DEFAULT_MAX_REUSE_FRAMES,
    WINDOW_SOURCE_CODE,
    WINDOW_SOURCE_NONE,
    TrackedPlacementState,
    resolved_placement_lag,
)
from tracking.io.artifacts import save_npz
from tracking.io.names import Order

DEFAULT_MAX_GAP_FRAMES = 40

DEFAULT_NO_MERGE = True

MAX_WINDOWS_PER_FRAME = 4

MERGE_DIST_UNITS = 30.0
MIN_VIEWS = 3
MAX_RESID_PX = 25.0

# A bare string, so this module needs no `sexing` import.
FINE_SEX_METHOD = "mvq_typed_slots"


def resolve_min_vis(min_vis):
    """`"auto"` -> the existence hallucination guard (this pass has exactly
    one placement mode, tracked, so there is no "off under CD placement"
    branch to resolve against). `None` is a no-op; anything else is coerced
    to `float`.
    """
    if min_vis == "auto":
        return HALLUCINATION_VIS_THRESH
    return None if min_vis is None else float(min_vis)


def fine_gate_string(runner, *, placement_lag=None, no_merge=DEFAULT_NO_MERGE, min_vis="auto"):
    """The Stage-B `gates` STRING stamped into a fine-lifted `kp3d.npz`."""
    lag = resolved_placement_lag(placement_lag)
    mv = resolve_min_vis(min_vis)
    return gate_string(
        runner.checkpoint,
        step=runner.step,
        exist_thresh=runner.exist_thresh,
        window_pref=runner.window_pref,
        placement="tracked",
        placement_lag=lag,
        no_merge=no_merge,
        min_vis=mv,
    )


def bout_is_current(out_dir, gate_string, n_flies=2):
    """True iff every `<out_dir>/fly<f>/kp3d.npz` exists and carries
    `gates == gate_string` exactly.
    """
    for fly in range(int(n_flies)):
        p = Path(out_dir) / f"fly{fly}" / "kp3d.npz"
        if not (p.exists() and p.stat().st_size > 0):
            return False
        try:
            with np.load(str(p), allow_pickle=True) as z:
                if "gates" not in z.files or str(z["gates"]) != str(gate_string):
                    return False
        except (OSError, ValueError):
            return False
    return True


def coarse_init_centroid(tracks_npz, start_frame, *, max_dist_frames=None):
    """Each fly's seed centroid for a bout, from a recording's coarse pass."""
    with np.load(str(tracks_npz), allow_pickle=True) as z:
        frame = np.asarray(z["coarse_frame"], np.int64)
        centroid = np.asarray(z["centroid"], np.float64)  # (F, T, 3)
    if centroid.ndim != 3 or centroid.shape[-1] != 3:
        raise ValueError(
            f"{tracks_npz}: expected a (F, T, 3) 'centroid' array, got shape {centroid.shape}"
        )
    if centroid.shape[1] != frame.shape[0]:
        raise ValueError(
            f"{tracks_npz}: coarse_frame has {frame.shape[0]} samples but centroid's "
            f"time axis (axis 1, (F, T, 3)) has {centroid.shape[1]}"
        )

    if max_dist_frames is None:
        step = int(np.median(np.diff(frame))) if frame.size > 1 else 1
        max_dist_frames = 4 * max(int(step), 1)

    F = centroid.shape[0]
    out = np.full((F, 3), np.nan, np.float64)
    d = np.abs(frame - int(start_frame))
    order = np.argsort(d, kind="stable")
    for fi in range(F):
        for j in order:
            if d[j] > int(max_dist_frames):
                break
            if np.isfinite(centroid[fi, j]).all():
                out[fi] = centroid[fi, j]
                break
    if not np.isfinite(out).any():
        return None
    return out


def gap_runs(missing):
    """`[(start_index, length), ...]` for every maximal run of True in `missing`."""
    m = np.asarray(missing, bool)
    if not m.any():
        return []
    idx = np.flatnonzero(np.diff(np.concatenate(([False], m, [False])).astype(np.int8)))
    return [(int(a), int(b - a)) for a, b in zip(idx[0::2], idx[1::2], strict=True)]


def _mean_or_none(a):
    a = np.asarray(a, np.float64)
    a = a[np.isfinite(a)]
    return None if a.size == 0 else float(a.mean())


def _mvq_sex_json(sex_prob, exist, n_missing, T, *, sex_head_disagree_frac):
    """`sex.json`, mask-free-route schema only (identity is always "sex" here
    -- there are no masks and no human id review on this route).
    """
    n_flies = int(np.asarray(sex_prob).shape[0])
    pf = {f"fly{f}": _mean_or_none(sex_prob[f]) for f in range(n_flies)}
    ex = {f"fly{f}": _mean_or_none(exist[f]) for f in range(n_flies)}
    if n_flies == 1:
        return {
            "male_fly": None,
            "original_male_fly": None,
            "applied_swap": False,
            "confidence": "low",
            "method": FINE_SEX_METHOD,
            "authority": FINE_SEX_METHOD,
            "heuristic_male_fly": None,
            "heuristic_method": "not_run (mask-free fine pass)",
            "heuristic_agrees": None,
            "review_file": None,
            "review_reviewed_at": None,
            "wing_cv_original": {"fly0": None},
            "cv_ratio": None,
            "mask_song_cv": None,
            "note": (
                "MASK-FREE fine pass, single fly (free-running, num_animals=1): fly0 "
                "is whichever typed slot the model produced each frame (the SEX_UNKNOWN "
                "rule); there is no partner fly to sex against, so identity is reported "
                "unknown rather than assumed"
            ),
            "identity": "unknown",
            "sex_prob": pf,
            "exist": ex,
            "sex_head_disagree_frac": dict(sex_head_disagree_frac or {}),
            "n_frames": int(T),
            "n_missing": dict(n_missing),
        }
    agree = pf["fly0"] is not None and pf["fly1"] is not None and pf["fly0"] > 0.5 > pf["fly1"]
    confidence = "high" if agree else "low"
    return {
        "male_fly": 1,
        "original_male_fly": 1,
        "applied_swap": False,
        "confidence": confidence,
        "method": FINE_SEX_METHOD,
        "authority": FINE_SEX_METHOD,
        "heuristic_male_fly": None,
        "heuristic_method": "not_run (mask-free fine pass)",
        "heuristic_agrees": None,
        "review_file": None,
        "review_reviewed_at": None,
        "wing_cv_original": {"fly0": None, "fly1": None},
        "cv_ratio": None,
        "mask_song_cv": None,
        "note": (
            "MASK-FREE fine pass: fly0 = the mvq FEMALE typed slot, fly1 = the mvq "
            "MALE typed slot, per frame; windows follow each fly's own tracked "
            "centroid (no SAM3 masks anywhere in this bout's provenance)"
        ),
        "identity": "sex",
        "sex_prob": pf,
        "exist": ex,
        "sex_head_disagree_frac": dict(sex_head_disagree_frac or {}),
        "n_frames": int(T),
        "n_missing": dict(n_missing),
    }


def _check_eye_invariant(
    kp3d_mvq_order, kp3d_pipeline_order, mvq_order: Order, pipeline_order: Order
):
    """EyeL-EyeR spacing is a RIGID head landmark pair: a by-name permutation
    cannot change it, and a by-index one almost certainly does (CLAUDE.md).
    """
    if not (
        {"EyeL", "EyeR"} <= set(mvq_order.names) and {"EyeL", "EyeR"} <= set(pipeline_order.names)
    ):
        return
    iL, iR = mvq_order.index("EyeL"), mvq_order.index("EyeR")
    jL, jR = pipeline_order.index("EyeL"), pipeline_order.index("EyeR")
    d0 = np.nan_to_num(
        np.linalg.norm(kp3d_mvq_order[..., iL, :] - kp3d_mvq_order[..., iR, :], axis=-1)
    )
    d1 = np.nan_to_num(
        np.linalg.norm(kp3d_pipeline_order[..., jL, :] - kp3d_pipeline_order[..., jR, :], axis=-1)
    )
    if not np.allclose(d0, d1, atol=1e-4):
        bad = int(np.argmax(np.abs(d0 - d1)))
        raise RuntimeError(
            f"EyeL-EyeR distance moved through the keypoint permutation "
            f"({d0.flat[bad]:.4f} -> {d1.flat[bad]:.4f} units); the written kp3d.npz is "
            f"NOT the same anatomy in a different order"
        )


def _atomic_write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


def _concat_windows(batch):
    """Concatenate several frames'/bouts' `MVQRunner.windows(...)` dicts
    along the leading (window) axis into one dict for a single `infer` call."""
    keys = batch[0].keys()
    return {k: np.concatenate([b[k] for b in batch], axis=0) for k in keys}


def _pick_single_fly(out, off, nb, runner, assign_local, *, min_vis=None):
    """The `num_animals=1` analogue of `pick_typed_pair`: ONE fly, read via
    `SEX_UNKNOWN` (`MVQRunner.read_typed`'s single-fly rule -- take whichever
    typed slot the model produced, rather than assuming a sex), the same
    own-window preference and highest-existence tie-break `pick_typed_pair`
    uses -- but no collapse guard, because there is no second fly to
    collapse against.
    """
    cands = []
    for b in range(off, off + nb):
        r = runner.read_typed(out, b, want_sex=SEX_UNKNOWN)
        if r is not None and _passes_vis_guard(r, min_vis=min_vis):
            cands.append((-float(r["exist"]), int(b), r))
    if not cands:
        return {}, False, None, {}
    own_row = -1
    if assign_local is not None and int(assign_local[0]) >= 0:
        own_row = int(off) + int(assign_local[0])
    _ne, best_b, best = prefer_own_window(cands, own_row, "own", lambda c: c[1])
    return {0: (best, best_b)}, False, None, {0: own_row >= 0 and best_b == own_row}


class _BoutRun:
    """One bout's in-flight state and output arrays."""

    def __init__(
        self,
        spec,
        runner,
        reader,
        detector,
        *,
        num_animals=2,
        merge_dist_units,
        max_reuse_frames,
        no_merge,
        placement_lag,
        min_vis,
        min_views,
        max_resid_px,
        collapse_dist_units,
    ):
        self.spec, self.runner, self.reader, self.detector = spec, runner, reader, detector
        self.n_flies = int(num_animals)
        self.placement_lag = int(placement_lag)
        self.min_vis, self.min_views = min_vis, int(min_views)
        self.max_resid_px = float(max_resid_px)
        self.collapse_dist_units = float(collapse_dist_units)
        T, C, K = int(spec.n_frames), int(runner.C), int(runner.K)
        self.T, self.C, self.K = T, C, K
        F = self.n_flies
        self.kp3d = np.full((F, T, K, 3), np.nan, np.float32)
        self.kp2d = np.full((F, T, C, K, 2), np.nan, np.float32)
        self.vis = np.zeros((F, T, C, K), np.float32)
        self.conf_raw = np.zeros((F, T, K), np.float32)
        self.exist = np.full((F, T), np.nan, np.float32)
        self.sex_prob = np.full((F, T), np.nan, np.float32)
        self.slot = np.full((F, T), -1, np.int8)
        self.own_window = np.full((F, T), -1, np.int8)
        self.window_source = np.full((F, T), WINDOW_SOURCE_NONE, np.int8)
        self.reacquired = np.zeros((F, T), bool)
        self.n_windows = np.zeros(T, np.int8)
        self.collapsed = np.zeros(T, bool)
        self.no_window = np.zeros(T, bool)
        self.n_collapsed = [0] * F
        self.tstate = TrackedPlacementState(
            F,
            merge_dist_units=merge_dist_units,
            max_reuse_frames=max_reuse_frames,
            no_merge=no_merge,
            init_centroid=spec.init_centroid,
        )
        self.t = 0  # next frame to PLAN
        self.pending = []  # frames planned, not yet read (oldest first)
        self.warned_drop = False

    @property
    def done_planning(self):
        return self.t >= self.T

    def needs_flush(self):
        """True when this bout cannot plan its next frame until the batch is
        read: its oldest in-flight frame is already `placement_lag` behind."""
        return bool(self.pending) and self.pending[0][0] <= self.t - self.placement_lag

    def plan_next(self):
        """Plan frame `self.t` and return its windows dict, or `None` when
        this frame produced no window at all (the row stays NaN)."""
        t = self.t
        imgs, present = self.reader(self.spec.start_frame + t)
        imgs = np.asarray(imgs)
        peaks, scores = self.detector.peaks(imgs)
        centres = cluster_centres(
            lift_peaks_to_centres(
                peaks,
                scores,
                self.runner.rig,
                min_views=self.min_views,
                max_resid_px=self.max_resid_px,
                max_animals=self.n_flies,
            )
        )
        plan = self.tstate.plan(centres)
        wc, assign_local, win_src, was_tracked = (
            plan.centres,
            plan.assignment,
            plan.source,
            plan.was_tracked,
        )
        if wc.shape[0] == 0:
            self.no_window[t] = True
            self.t += 1
            return None
        cap = min(int(self.runner.batch), MAX_WINDOWS_PER_FRAME)
        if wc.shape[0] > cap:
            if not self.warned_drop:
                warnings.warn(
                    f"fine_track bout {self.spec.idx}: frame {t} planned {wc.shape[0]} "
                    f"windows but at most {cap} fit one frame's reservation; dropping "
                    f"{wc.shape[0] - cap} window(s) (first offender only)",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.warned_drop = True
            win_src = win_src[:cap]
            assign_local = np.where(assign_local < cap, assign_local, -1)
            wc = wc[:cap]
        self.n_windows[t] = int(wc.shape[0])
        self.t += 1
        w = self.runner.windows(imgs, present, wc)
        self.pending.append((t, int(wc.shape[0]), assign_local, win_src, was_tracked))
        return w

    def record(self, out, off, t, nb, assign_local, win_src, was_tracked):
        """One pending frame's read, from a finished forward."""
        if self.n_flies == 2:
            picks, is_collapsed, drop, own = pick_typed_pair(
                out,
                off,
                nb,
                self.runner,
                self.collapse_dist_units,
                own_windows=assign_local,
                window_pref="own",
                min_vis=self.min_vis,
            )
        else:
            picks, is_collapsed, drop, own = _pick_single_fly(
                out, off, nb, self.runner, assign_local, min_vis=self.min_vis
            )
        self.collapsed[t] = is_collapsed
        if drop is not None:
            self.n_collapsed[drop] += 1
        for fi, o in own.items():
            self.own_window[fi, t] = -1 if int(assign_local[fi]) < 0 else (1 if o else 0)
        for fi, (best, _best_b) in picks.items():
            self.kp3d[fi, t] = best["kp3d"]
            self.kp2d[fi, t] = best["kp2d"]
            self.vis[fi, t] = best["vis"]
            self.conf_raw[fi, t] = best["conf_raw"]
            self.exist[fi, t] = best["exist"]
            self.sex_prob[fi, t] = best["sex_prob"]
            self.slot[fi, t] = best["slot"]
        for fi in range(self.n_flies):
            got = fi in picks and np.isfinite(self.kp3d[fi, t]).any()
            if got:
                local = int(picks[fi][1]) - off
                if 0 <= local < len(win_src):
                    self.window_source[fi, t] = WINDOW_SOURCE_CODE[win_src[local]]
                self.reacquired[fi, t] = bool(not was_tracked[fi])
                pts = self.kp3d[fi, t]
                ok = np.isfinite(pts).all(axis=-1)
                self.tstate.note_read(fi, pts[ok].mean(axis=0))
            else:
                if int(assign_local[fi]) >= 0:
                    self.window_source[fi, t] = WINDOW_SOURCE_CODE[win_src[int(assign_local[fi])]]
                self.tstate.note_miss(fi)


def _finish_bout(run, *, kp_order, max_gap_frames, gate_signature, gate_extra, out_dir, dry):
    """One finished `_BoutRun` -> a result dict; writes the pipeline
    artifacts + meta + sex.json unless `dry`."""
    runner, spec, T = run.runner, run.spec, run.T
    n_flies = run.n_flies
    per_fly = {}
    n_missing, runs = {}, {}
    for fly in range(n_flies):
        kp3d_p, kp2d_p, vis_p, conf_raw_p = runner.to_pipeline(
            run.kp3d[fly], run.kp2d[fly], run.vis[fly], run.conf_raw[fly]
        )
        _check_eye_invariant(run.kp3d[fly], kp3d_p, runner.model_order, kp_order)
        conf3d = vis_p.mean(axis=-2)  # over CAMERAS, not time
        per_fly[fly] = {
            "kp3d": kp3d_p,
            "kp2d": kp2d_p,
            "conf": vis_p,
            "conf3d": conf3d,
            "conf3d_mvq_raw": conf_raw_p,
        }
        missing = run.slot[fly] < 0
        n_missing[f"fly{fly}"] = int(missing.sum())
        runs[f"fly{fly}"] = gap_runs(missing)

    if n_flies == 2:
        fly_slots = {"fly0": int(SLOT_FEMALE), "fly1": int(SLOT_MALE)}
        fly_sex = {"fly0": "female", "fly1": "male"}
        typed_slot = (int(SLOT_FEMALE), int(SLOT_MALE))
        disagree = {}
        for fly in range(n_flies):
            wrote = run.slot[fly] >= 0
            disagree[f"fly{fly}"] = (
                float((run.slot[fly][wrote] != typed_slot[fly]).mean()) if wrote.any() else None
            )
    else:
        fly_slots = {"fly0": None}
        fly_sex = {"fly0": "unknown"}
        disagree = {"fly0": None}
    sex = _mvq_sex_json(run.sex_prob, run.exist, n_missing, T, sex_head_disagree_frac=disagree)

    long_gaps = {f: [g for g in runs[f] if g[1] > int(max_gap_frames)] for f in runs}
    meta = {
        "front_end": "mvq",
        "lifter": "fine_track",
        "checkpoint": runner.checkpoint,
        "step": runner.step_label,
        "gates": json.loads(gate_signature),
        "exist_thresh": float(runner.exist_thresh),
        "exist_temperature": float(runner.exist_temperature),
        "vis_temperature": float(runner.vis_temperature),
        "cameras": list(runner.cameras),
        "keypoint_names_mvq": list(runner.model_order.names),
        "keypoint_names_written": list(kp_order.names),
        "identity": "sex" if n_flies == 2 else "unknown",
        "identity_resolved": "sex" if n_flies == 2 else "unknown",
        "window_pref": runner.window_pref,
        "fly_slots": fly_slots,
        "fly_sex": fly_sex,
        "bout_idx": int(spec.idx),
        "bout_start_frame": int(spec.start_frame),
        "n_frames": int(T),
        "n_missing": n_missing,
        "n_no_window": int(run.no_window.sum()),
        "collapse_dist_units": float(run.collapse_dist_units),
        "n_collapsed": {f"fly{f}": int(run.n_collapsed[f]) for f in range(n_flies)},
        "collapsed_frac": float(run.collapsed.mean()) if T else 0.0,
        "max_gap_frames": int(max_gap_frames),
        # Every run of consecutive missed frames, and the subset longer than
        # `max_gap_frames` -- the frames this bout genuinely cannot solve.
        "gap_runs": {f: [[a, n] for a, n in runs[f]] for f in runs},
        "n_long_gaps": {f: len(v) for f, v in long_gaps.items()},
        "n_long_gap_frames": {f: int(sum(n for _a, n in v)) for f, v in long_gaps.items()},
        "sex_head_disagree_frac": disagree,
        "init_centroid": (
            None
            if spec.init_centroid is None
            else np.asarray(spec.init_centroid, float).round(3).tolist()
        ),
        "per_frame": {
            "n_windows": run.n_windows.astype(int).tolist(),
            "no_window": run.no_window.astype(int).tolist(),
            "collapsed": run.collapsed.astype(int).tolist(),
            "slot": run.slot.astype(int).tolist(),
            "own_window": run.own_window.astype(int).tolist(),
            "window_source": run.window_source.astype(int).tolist(),
            "reacquired": run.reacquired.astype(int).tolist(),
            "exist": np.round(np.nan_to_num(run.exist, nan=-1.0), 4).tolist(),
            "sex_prob": np.round(np.nan_to_num(run.sex_prob, nan=-1.0), 4).tolist(),
        },
    }
    meta.update(gate_extra)

    if dry:
        return {
            "idx": int(spec.idx),
            "skipped": False,
            "meta": meta,
            "kp3d": [per_fly[f]["kp3d"] for f in range(n_flies)],
            "kp2d": [per_fly[f]["kp2d"] for f in range(n_flies)],
            "conf3d": [per_fly[f]["conf3d"] for f in range(n_flies)],
        }

    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    cams = Order(runner.cameras)
    for fly in range(n_flies):
        d = os.path.join(out_dir, f"fly{fly}")
        save_npz(
            os.path.join(d, "kp2d.npz"),
            arrays={"kp2d": per_fly[fly]["kp2d"], "conf": per_fly[fly]["conf"]},
            kp=kp_order,
            cams=cams,
        )
        save_npz(
            os.path.join(d, "kp3d.npz"),
            arrays={
                "kp3d": per_fly[fly]["kp3d"],
                "conf3d": per_fly[fly]["conf3d"],
                "conf3d_mvq_raw": per_fly[fly]["conf3d_mvq_raw"],
            },
            kp=kp_order,
            cams=None,
            extra={"gates": gate_signature},
        )
    _atomic_write_json(os.path.join(out_dir, "sex.json"), sex)
    _atomic_write_json(os.path.join(out_dir, "mvq_meta.json"), meta)
    return {"idx": int(spec.idx), "skipped": False, "meta": meta}


def fine_track_bouts(
    runner,
    specs,
    *,
    detector,
    reader_factory,
    kp_order: Order,
    out_root,
    num_animals: int = 2,
    placement_lag=None,
    bouts_in_flight=4,
    min_vis="auto",
    no_merge=DEFAULT_NO_MERGE,
    max_gap_frames=DEFAULT_MAX_GAP_FRAMES,
    force=False,
    dry=False,
    progress_every=0,
):
    """Lift a LIST of bouts mask-free, batching their windows together.

    Args:
        runner: `tracking.detector.mvq.runner.MVQRunner` with
            `window_pref="own"` (refused otherwise -- the gate string this
            writes promises it).
        specs: iterable of `tracking.io.bouts.BoutSpec`.
        detector: `tracking.detector.centerdetect.detector.CenterDetector`
            (or a test double with the same `.peaks(frames)` protocol), used
            ONLY to acquire a fly with no live track.
        reader_factory: `spec -> reader`, where `reader(absolute_slot)`
            returns `(frames (C,H,W,3) uint8 RGB, present (C,))` in
            `runner.cameras` order and is called with the STRICTLY
            INCREASING sequence `spec.start_frame .. spec.start_frame +
            spec.n_frames - 1` (`tracking.io.video.SlotReader`). A reader
            with a `.close()` is closed when its bout finishes.
        kp_order: the pipeline's canonical keypoint order -- the written
            axis (`runner.to_pipeline`'s target).
        out_root: bout directories are written to
            `<out_root>/bouts/bout_<idx:05d>/`. May be `None` only when
            `dry=True` (nothing is written).
        num_animals: 1 (free-running) or 2 (courtship) flies -- matching
            `coarse.py`'s `coarse_pass`. `num_animals=1` reads via the
            `SEX_UNKNOWN` single-fly rule (`MVQRunner.read_typed`) and
            writes only `fly0/`; `sex.json` then reports identity
            "unknown" rather than assuming a sex (CLAUDE.md: don't present
            a guess as an observation). Any other value is refused by name.
        placement_lag: how many frames behind frame t a tracked window's
            centre may be (`resolved_placement_lag`; 8 by default).
        bouts_in_flight: how many bouts are advanced in lockstep at once --
            caps both the batch's fan-out and how many `reader_factory`
            decoder threads are open simultaneously.
        force: re-lift a bout whose `kp3d.npz` already carries this run's
            gate string.
        dry: compute and return each bout's arrays without writing
            anything; also bypasses the staleness check (a caller that asked
            for the arrays wants them computed, not skipped).
        progress_every: print a line every this many TOTAL frames read
            across all bouts.

    Returns one result dict per spec, in the order given. Every dict has
    `idx`, `skipped`, `meta`; a `dry=True` call also carries `kp3d`, `kp2d`,
    `conf3d`, each a `num_animals`-list indexed by fly.
    """
    if str(getattr(runner, "window_pref", "own")) != "own":
        raise ValueError(
            f"fine_track_bouts reads each fly out of its OWN tracked window, but this "
            f"runner is window_pref={runner.window_pref!r}; the gate string would then "
            f"promise a rule the pass did not run"
        )
    if not dry and out_root is None:
        raise ValueError("out_root is required unless dry=True")
    n_flies = int(num_animals)
    if n_flies not in (1, 2):
        raise ValueError(f"num_animals must be 1 or 2, got {num_animals}")

    specs = list(specs)
    lag = resolved_placement_lag(placement_lag)
    mv = resolve_min_vis(min_vis)
    gate_signature = fine_gate_string(runner, placement_lag=lag, no_merge=no_merge, min_vis=min_vis)
    gate_extra = {
        "placement": "tracked",
        "placement_lag": int(lag),
        "no_merge": bool(no_merge),
        "min_vis": mv,
        "max_reuse_frames": int(DEFAULT_MAX_REUSE_FRAMES),
        "merge_dist_units": float(MERGE_DIST_UNITS),
    }

    results: list = [None] * len(specs)
    pending_specs = []
    for i, spec in enumerate(specs):
        out_dir = (
            None
            if out_root is None
            else os.path.join(str(out_root), "bouts", f"bout_{int(spec.idx):05d}")
        )
        if (
            not dry
            and not force
            and out_dir is not None
            and bout_is_current(out_dir, gate_signature, n_flies=n_flies)
        ):
            results[i] = {
                "idx": int(spec.idx),
                "skipped": True,
                "meta": {"gates": gate_signature, "out_dir": out_dir},
            }
            continue
        pending_specs.append((i, spec, out_dir))

    active: list = []  # (i, spec, out_dir, _BoutRun)
    readers: list = []

    def _activate_next():
        while len(active) < int(bouts_in_flight) and pending_specs:
            i, spec, out_dir = pending_specs.pop(0)
            reader = reader_factory(spec)
            readers.append(reader)
            run = _BoutRun(
                spec,
                runner,
                reader,
                detector,
                num_animals=n_flies,
                merge_dist_units=MERGE_DIST_UNITS,
                max_reuse_frames=DEFAULT_MAX_REUSE_FRAMES,
                no_merge=no_merge,
                placement_lag=lag,
                min_vis=mv,
                min_views=MIN_VIEWS,
                max_resid_px=MAX_RESID_PX,
                collapse_dist_units=COLLAPSE_DIST_UNITS,
            )
            active.append((i, spec, out_dir, run))

    _activate_next()

    batch, pend, rows = [], [], 0
    n_read = n_forwards = 0
    t_start = time.time()

    def flush():
        nonlocal batch, pend, rows, n_forwards
        if not pend:
            return
        out = runner.infer(_concat_windows(batch))
        n_forwards += 1
        for run, off, t, nb, assign_local, win_src, was_tracked in pend:
            run.record(out, off, t, nb, assign_local, win_src, was_tracked)
        for _i, _spec, _out_dir, run in active:
            run.pending = []
        batch, pend, rows = [], [], 0

    try:
        while active:
            progressed = False
            for _i, _spec, _out_dir, run in list(active):
                if run.done_planning:
                    continue
                # Read BEFORE planning: this bout's lag budget may be spent,
                # or the batch may have no room for its reservation.
                if run.needs_flush() or rows + MAX_WINDOWS_PER_FRAME > runner.batch:
                    flush()
                w = run.plan_next()
                progressed = True
                n_read += 1
                if w is not None:
                    t_planned, nb, assign_local, win_src, was_tracked = run.pending[-1]
                    batch.append(w)
                    pend.append((run, rows, t_planned, nb, assign_local, win_src, was_tracked))
                    rows += nb
                if progress_every and n_read % int(progress_every) == 0:
                    el = time.time() - t_start
                    print(
                        f"[fine] {n_read} frames  {n_read / max(el, 1e-9):.1f} frames/s  "
                        f"{n_forwards} forwards  {len(active)} bouts in flight",
                        flush=True,
                    )
            if not progressed:
                break
            finished = [
                (i, spec, out_dir, run) for i, spec, out_dir, run in active if run.done_planning
            ]
            if finished:
                flush()
                for i, _spec, out_dir, run in finished:
                    results[i] = _finish_bout(
                        run,
                        kp_order=kp_order,
                        max_gap_frames=max_gap_frames,
                        gate_signature=gate_signature,
                        gate_extra=gate_extra,
                        out_dir=out_dir,
                        dry=dry,
                    )
                    if hasattr(run.reader, "close"):
                        run.reader.close()
                active = [
                    (i, spec, out_dir, run)
                    for i, spec, out_dir, run in active
                    if not run.done_planning
                ]
                _activate_next()
        flush()
        for i, _spec, out_dir, run in active:
            results[i] = _finish_bout(
                run,
                kp_order=kp_order,
                max_gap_frames=max_gap_frames,
                gate_signature=gate_signature,
                gate_extra=gate_extra,
                out_dir=out_dir,
                dry=dry,
            )
            if hasattr(run.reader, "close"):
                run.reader.close()
    finally:
        for r in readers:
            if hasattr(r, "close"):
                try:
                    r.close()
                except Exception:  # noqa: BLE001 -- teardown only
                    pass

    return results


def fine_track_bout(runner, spec, **kw):
    """One bout, i.e. `fine_track_bouts` with a single spec."""
    return fine_track_bouts(runner, [spec], **kw)[0]
