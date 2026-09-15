"""The coarse pass: a whole recording at a wide stride, mask-free.

Sweeps a recording at `stride` (default 16), locating each fly with
CenterDetect + tracked window placement and lifting each window with an
`MVQRunner` checkpoint, producing the per-fly per-frame arrays
`write_coarse_tracks` turns into `coarse_tracks.npz` -- the file
`tracking.detector.gates` reads to gate frame ranges into bouts.

**THE AXIS CONVENTION IS A HARD CONTRACT (CLAUDE.md).** Every per-fly
per-frame array here -- `exist`, `n_valid_cams`, `wing_angle_deg`, `speed`,
`height`, `trackable`, `slot`, `centre_source`, `sex_prob` -- is `(F, T)`:
FLY axis FIRST, then time. `centroid` is `(F, T, 3)`; `kp3d` is
`(F, T, K, 3)`. `gates` reads the file with `num_animals, n_frames =
exist.shape`, where a transposed write is SILENT -- it would read T animals
and F frames and nothing would raise.

**KEYPOINTS BY NAME.** `coarse_features` takes a `kp_order: Order` and looks
every keypoint up by name (`_kp(kp3d, kp_order, "WingL_base")`) -- never by
integer, the trap that once turned a middle-left leg into a "collapsed right
wing vein" (CLAUDE.md).

**PLACEMENT.** Every window is placed by TRACKED placement
(`tracking_state.TrackedPlacementState`/`plan_windows_tracked`); there is no
CenterDetect-only-clustering fallback. `no_merge=True` and `placement_lag=8`
are this module's defaults.

**BATCHING SPANS FRAMES.** `MVQRunner.infer` always pads to `runner.batch`, so
one window per call would compute `batch` windows and discard all but one.
Windows accumulate across frames and flush with `_concat_windows` when the
next frame's would not fit, or when `placement_lag` frames have elapsed since
the oldest pending frame.

**KEYPOINT ORDER, MODEL VS PIPELINE.** `MVQRunner.infer`/`read_typed` return
`kp3d` in the CHECKPOINT's order (`runner.model_order`); `runner.to_pipeline`
is the one place that permutes it. Every `kp3d` stored here has already been
through it, so `coarse_features`'s by-name lookups are correct.

**CHUNKED RESUME.** `coarse_pass(..., resume_from=...)` continues
tracked-placement state from a previous result's `last_centroid`/
`last_centres` and returns the new frames concatenated onto the old, so a
caller can process a recording in blocks, write each through
`write_coarse_tracks` as a complete gates-readable file, and feed it back via
`load_partial` after a restart. A resumed and a single-shot run cover the same
frames and reuse centres identically, but are NOT byte-identical:
`write_coarse_tracks` stores `kp3d` as float16, so a chunk that round-tripped
through disk computes `wing_angle_deg` from the quantised value -- an inherent
~0.05 mm difference, not a bug.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path
from typing import NamedTuple

import numpy as np

from tracking.detector.centerdetect.detector import cluster_centres, lift_peaks_to_centres
from tracking.detector.mvq.runner import (
    HALLUCINATION_VIS_THRESH,
    SEX_FEMALE,
    SEX_MALE,
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
    plan_windows_tracked,
    resolved_placement_lag,
)
from tracking.geometry.rig import CameraRig
from tracking.io.artifacts import load_npz, save_npz
from tracking.io.names import Order

# `plan_windows_tracked` is re-exported, not called directly (only via
# `TrackedPlacementState.plan`), so callers can reason about window placement
# without knowing it lives in `tracking_state.py`.
__all__ = [
    "TrackedPlacementState",
    "plan_windows_tracked",
    "resolved_placement_lag",
    "FloorPlane",
    "fit_floor",
    "coarse_features",
    "coarse_pass",
    "concat_tracks",
    "write_coarse_tracks",
    "load_partial",
]

# Frame-level (not per-fly) bookkeeping of where THIS frame's centres came
# from. Purely informational: the placement decision itself is
# `plan_windows_tracked`'s business, not this value's.
CENTRE_DETECTED, CENTRE_REUSED, CENTRE_NONE = 0, 1, 2

TRACKABLE_EXIST = 0.5  # typed-slot existence threshold for "this fly is trackable"
FLOOR_FIT_N = 2000  # coarse frames; ~20s of recording at stride 16, 800 fps
FLOOR_SKEW_MARGINAL = 0.1  # see fit_floor's orientation-heuristic docs

# Keypoints the features are built from, BY NAME (never by index).
KP_HEAD, KP_TAIL = "Scutellum", "Abd_tip"
WING_PAIRS = (("WingL_base", "WingL_V13"), ("WingR_base", "WingR_V13"))


class FloorPlane(NamedTuple):
    """`normal . x + offset` = height above the floor, in world units.

    `normal` is a unit vector oriented so a fly's height is POSITIVE (see
    `fit_floor`). `orientation` records which mechanism picked that sign
    (`"hint"` when `fit_floor`'s `up_hint` decided it directly, `"skew"`
    when the bottom-heaviness heuristic did); `skew` is that heuristic's own
    statistic, recorded even when `up_hint` made the actual decision.
    """

    normal: np.ndarray
    offset: float
    orientation: str = "skew"
    skew: float = float("nan")

    def height_of(self, points):
        """(...,3) world points -> (...,) height above the plane, units."""
        return np.asarray(points, np.float64) @ np.asarray(self.normal, np.float64) + self.offset

    def as_dict(self) -> dict:
        return {
            "normal": [float(v) for v in self.normal],
            "offset": float(self.offset),
            "orientation": str(self.orientation),
            "skew": float(self.skew),
        }


def _svd_normal(pts: np.ndarray) -> np.ndarray:
    """Unit normal of the least-variance direction of `(N,3)` points, via SVD
    about their mean. Not sign-resolved -- callers pick "up" themselves."""
    _, _, vh = np.linalg.svd(pts - pts.mean(axis=0), full_matrices=False)
    return vh[-1] / np.linalg.norm(vh[-1])


def fit_floor(
    centroids,
    *,
    exist=None,
    n_fit: int = FLOOR_FIT_N,
    floor_pct: float = 3.0,
    up_hint=None,
    skew_marginal: float = FLOOR_SKEW_MARGINAL,
) -> FloorPlane:
    """Least-squares floor plane from the first `n_fit` finite, trackable
    coarse centroids.

    Args:
        centroids: (F, N, 3) per-fly 3D centroids in frame order, or any
            (..., 3) array.
        exist: optional (F, N) existence array (`coarse_pass`'s own
            `exist`); a centroid is used only if `exist >= TRACKABLE_EXIST`.
            `None` fits every finite point.
        n_fit: how many finite, trackable points to fit on -- the plane is a
            property of the ARENA, so the first ~20s of a recording is as
            good as all of it and far cheaper.
        floor_pct: the percentile of the along-normal coordinate the plane
            is placed at (low single digits: just under the lowest observed
            centroids).
        up_hint: optional (3,) known-up direction. When given, the fitted
            normal is oriented by `dot(normal, up_hint) > 0` and the
            bottom-heaviness skew heuristic below is skipped entirely
            (`FloorPlane.orientation == "hint"`).
        skew_marginal: only used when `up_hint` is None. The heuristic
            assumes floor points are the majority of the trackable sample;
            `|skew| < skew_marginal` warns that the sign is close to a coin
            flip and recommends `up_hint` instead.

    Returns:
        `FloorPlane` with a unit `normal` pointing UP and an `offset` at the
        bottom of the fly cloud (heights ~0 on the floor, positive up a wall).

    Raises:
        ValueError: fewer than 3 finite, trackable coarse centroids.
    """
    a = np.asarray(centroids, np.float64)
    # Frame-major order: `centroids` is (F,N,3), so reshaping fly-major would
    # take "the first n_fit" all from fly 0. Transpose to (N,F,3) first.
    pts_all = np.transpose(a, (1, 0, 2)).reshape(-1, 3) if a.ndim == 3 else a.reshape(-1, 3)
    ok = np.isfinite(pts_all).all(axis=1)
    if exist is not None:
        e = np.asarray(exist, np.float64)
        e = np.transpose(e, (1, 0)).reshape(-1) if a.ndim == 3 else e.reshape(-1)
        if e.shape[0] != pts_all.shape[0]:
            raise ValueError(
                f"exist has {e.shape[0]} entries but centroids reshape to "
                f"{pts_all.shape[0]}; exist must carry the SAME (fly, frame) axes as centroids"
            )
        ok = ok & (e >= TRACKABLE_EXIST)
    pts = pts_all[ok]
    if pts.shape[0] < 3:
        raise ValueError(
            f"floor fit needs >= 3 finite, trackable (exist >= {TRACKABLE_EXIST}) coarse "
            f"centroids, got {pts.shape[0]}"
        )
    pts = pts[: int(n_fit)]

    hint = None if up_hint is None else np.asarray(up_hint, np.float64)

    def _orient(normal, s):
        if hint is not None:
            return (normal, s) if np.dot(normal, hint) >= 0 else (-normal, -s)
        return (normal, s) if s.mean() >= np.median(s) else (-normal, -s)

    normal = _svd_normal(pts)
    s = pts @ normal
    normal, s = _orient(normal, s)

    # Refit on the BOTTOM half only (by this first pass's height), which is
    # dominated by the true floor regardless of the wall fraction.
    low = s <= np.percentile(s, 50.0)
    if low.sum() >= 3:
        normal2 = _svd_normal(pts[low])
        if np.dot(normal2, normal) < 0:
            normal2 = -normal2
        normal = normal2
        s = pts @ normal
        normal, s = _orient(normal, s)

    skew = float((s.mean() - np.median(s)) / (np.std(s) + 1e-9))
    if hint is not None:
        orientation = "hint"
    else:
        orientation = "skew"
        if abs(skew) < float(skew_marginal):
            warnings.warn(
                f"fit_floor: orientation skew={skew:.4f} is below the marginal threshold "
                f"{skew_marginal} -- the bottom-heaviness heuristic is close to a coin flip "
                f"here; the normal's sign is UNCERTAIN. Pass up_hint=<a known up direction> "
                f"to resolve it directly instead of trusting this heuristic.",
                RuntimeWarning,
                stacklevel=2,
            )

    offset = -float(np.percentile(s, float(floor_pct)))
    return FloorPlane(normal.astype(np.float64), offset, orientation, skew)


def _angle_deg(a, b) -> np.ndarray:
    """Angle between two (...,3) vector fields, degrees, NaN-propagating."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (a * b).sum(axis=-1) / (na * nb)
        cos = np.where((na > 0) & (nb > 0), cos, np.nan)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _kp(kp3d: np.ndarray, kp_order: Order, name: str) -> np.ndarray:
    """One keypoint's (F,N,3) track, BY NAME. Raises (`OrderMismatch`, via
    `Order.index`) if the model does not have it, rather than silently
    indexing whatever sits at that position."""
    k = kp_order.index(name)
    return kp3d[:, :, k]


def coarse_features(tracks: dict, kp_order: Order, *, floor: FloorPlane) -> dict:
    """Per-frame behavioural features for bout detection.

    Args:
        tracks: needs `kp3d` (F,N,K,3), `centroid` (F,N,3), `exist` (F,N).
        kp_order: the keypoint order of `tracks["kp3d"]` (the pipeline's
            canonical order -- `kp3d` must already have been permuted into
            it, e.g. via `MVQRunner.to_pipeline`).
        floor: a `FloorPlane` (see `fit_floor`).

    Returns dict of:
        dist (N,)             inter-fly 3D centroid distance, units (NaN if F < 2)
        heading_deg (N,)      angle between the MALE's anterior direction
            (Abd_tip -> Scutellum) and the male->female vector; 0 = pointed
            straight at her.
        speed (F,N)           ||centroid[t] - centroid[t-1]||, units per
            coarse frame; [:,0] is NaN.
        wing_angle_deg (F,N)  max over L/R of the angle between the body
            axis (Scutellum -> Abd_tip) and the wing vector
            (WingX_base -> WingX_V13); NaN-aware max, so one dropped wing
            does not erase the other.
        height (F,N)          centroid height above `floor`, units.
        trackable (F,N)       exist >= TRACKABLE_EXIST AND a finite centroid.
    """
    kp3d = np.asarray(tracks["kp3d"], np.float64)
    centroid = np.asarray(tracks["centroid"], np.float64)
    exist = np.asarray(tracks["exist"], np.float64)
    if kp3d.shape[2] != len(kp_order):
        raise ValueError(
            f"kp3d has {kp3d.shape[2]} keypoints but kp_order has {len(kp_order)}; the "
            f"feature lookup is BY NAME and needs the model's own (permuted) order"
        )
    n_flies, n_frames = kp3d.shape[0], kp3d.shape[1]

    head = _kp(kp3d, kp_order, KP_HEAD)
    tail = _kp(kp3d, kp_order, KP_TAIL)
    body_axis = tail - head  # Scutellum -> Abd_tip
    wing = [
        _angle_deg(body_axis, _kp(kp3d, kp_order, tip) - _kp(kp3d, kp_order, base))
        for base, tip in WING_PAIRS
    ]
    wing_angle_deg = np.fmax(wing[0], wing[1]).astype(np.float32)

    height = floor.height_of(centroid).astype(np.float32)

    speed = np.full((n_flies, n_frames), np.nan, np.float32)
    if n_frames > 1:
        speed[:, 1:] = np.linalg.norm(centroid[:, 1:] - centroid[:, :-1], axis=-1)

    dist = np.full(n_frames, np.nan, np.float32)
    heading_deg = np.full(n_frames, np.nan, np.float32)
    if n_flies >= 2:
        dist = np.linalg.norm(centroid[0] - centroid[1], axis=-1).astype(np.float32)
        forward_male = (head - tail)[1]  # anterior direction, fly 1 = male
        male_to_female = centroid[0] - centroid[1]
        heading_deg = _angle_deg(forward_male, male_to_female).astype(np.float32)

    with np.errstate(invalid="ignore"):
        trackable = (exist >= TRACKABLE_EXIST) & np.isfinite(centroid).all(axis=-1)
    return {
        "dist": dist,
        "heading_deg": heading_deg,
        "speed": speed,
        "wing_angle_deg": wing_angle_deg,
        "height": height,
        "trackable": np.asarray(trackable, bool),
    }


def _concat_windows(batch: list[dict]) -> dict:
    """Concatenate several frames' `MVQRunner.windows(...)` dicts along the
    leading (window) axis into one dict for a single `runner.infer` call."""
    keys = batch[0].keys()
    return {k: np.concatenate([b[k] for b in batch], axis=0) for k in keys}


def coarse_pass(
    reader,
    runner,
    detector,
    *,
    frames,
    rig: CameraRig,
    kp_order: Order,
    num_animals: int = 2,
    stride: int,
    placement_lag=None,
    merge_dist_units: float = 30.0,
    no_merge: bool = True,
    max_reuse_frames=None,
    min_views: int = 3,
    max_resid_px: float = 25.0,
    partial_every=None,
    resume_from: dict | None = None,
    progress=None,
) -> dict:
    """Run the coarse pass over `frames` (absolute video frame indices).

    Args:
        reader: callable `(frame_idx) -> (frames (C,H,W,3) uint8 RGB,
            present (C,) bool)` in `rig.cameras` order (`tracking.io.video.
            SlotReader`).
        runner: `tracking.detector.mvq.runner.MVQRunner`.
        detector: `tracking.detector.centerdetect.detector.CenterDetector`
            (or a test double with the same `.peaks(frames)` protocol).
        frames: iterable of absolute frame indices (the stride sample).
        rig: this recording's `CameraRig` -- must be `runner.rig` (checked
            indirectly: `lift_peaks_to_centres` requires the same camera
            count and order as `runner.cam_mats`).
        kp_order: the pipeline's canonical keypoint order (`runner.kp_order`).
        num_animals: 1 or 2 flies.
        stride: the video-frame spacing of `frames` -- carried through only
            for `write_coarse_tracks`'s meta json, not used for indexing here.
        placement_lag: `tracking_state.resolved_placement_lag`'s knob (default
            8 when `None`).
        merge_dist_units / no_merge: `tracking_state.plan_windows_tracked`'s
            knobs (defaults 30.0 / `True`).
        max_reuse_frames: `TrackedPlacementState`'s track-expiry knob
            (default `tracking_state.DEFAULT_MAX_REUSE_FRAMES` when `None`).
        min_views / max_resid_px: `lift_peaks_to_centres`'s knobs.
        partial_every: if given, `progress` (see below) is additionally
            invoked at every `partial_every`th frame with a 4th argument: a
            *complete*, gates-readable snapshot of the tracks-so-far dict
            (concatenated onto `resume_from` if given). This is what lets a
            caller checkpoint a long pass to disk (via `write_coarse_tracks`)
            without `coarse_pass` itself doing any file IO.
        resume_from: a previous `coarse_pass` (or `load_partial`) result to
            continue tracked-placement state from; the returned dict is this
            call's frames concatenated onto it (`concat_tracks`).
        progress: callable `(done, total, frames_per_s[, checkpoint])` after
            every frame; `checkpoint` is `None` except at `partial_every`
            boundaries (see above).

    Returns a dict (F = num_animals, N = len(frames) [+ resume_from's own N]):
        frame (N,) i64, kp3d (F,N,K,3) f32 -- already through
        `runner.to_pipeline`, so keypoint-axis lookups against `kp_order`
        are correct -- centroid (F,N,3) f32, exist (F,N) f32, sex_prob (F,N)
        f32, slot (F,N) i8, centre_source (F,N) i8, n_windows (N,) i8,
        collapsed (F,N) bool, window_source (F,N) i8, reacquired (F,N) bool,
        W, H, last_centres (F,3) f32, last_centroid (F,3) f32.
    """
    frame_list = [int(f) for f in frames]
    n_flies = int(num_animals)
    if n_flies not in (1, 2):
        raise ValueError(f"num_animals must be 1 or 2, got {num_animals}")
    n_frames, n_kp = len(frame_list), runner.K
    lag = resolved_placement_lag(placement_lag)
    max_reuse = DEFAULT_MAX_REUSE_FRAMES if max_reuse_frames is None else int(max_reuse_frames)

    kp3d = np.full((n_flies, n_frames, n_kp, 3), np.nan, np.float32)
    centroid = np.full((n_flies, n_frames, 3), np.nan, np.float32)
    exist = np.full((n_flies, n_frames), np.nan, np.float32)
    sex_prob = np.full((n_flies, n_frames), np.nan, np.float32)
    slot = np.full((n_flies, n_frames), -1, np.int8)
    centre_source = np.full((n_flies, n_frames), CENTRE_NONE, np.int8)
    n_windows = np.zeros(n_frames, np.int8)
    collapsed = np.zeros((n_flies, n_frames), bool)
    window_source = np.full((n_flies, n_frames), WINDOW_SOURCE_NONE, np.int8)
    reacquired = np.zeros((n_flies, n_frames), bool)

    init_centroid = None if resume_from is None else resume_from.get("last_centroid")
    prev_centres = None if resume_from is None else resume_from.get("last_centres")
    tstate = TrackedPlacementState(
        n_flies,
        merge_dist_units=merge_dist_units,
        max_reuse_frames=max_reuse,
        no_merge=no_merge,
        init_centroid=init_centroid,
    )
    want = [SEX_UNKNOWN] if n_flies == 1 else [SEX_FEMALE, SEX_MALE]

    pend: list = []  # (t, off, nb, assign_local, win_src, was_tracked)
    batch: list = []
    rows = 0
    W = H = None
    t_start = time.time()
    warned_drop = False

    def _read_batch():
        nonlocal pend, batch, rows
        if not pend:
            return
        out = runner.infer(_concat_windows(batch))
        for t, off, nb, assign_local, win_src, was_tracked in pend:
            if n_flies == 2:
                picks, is_collapsed, _dropped, _own = pick_typed_pair(
                    out,
                    off,
                    nb,
                    runner,
                    own_windows=assign_local,
                    window_pref="own",
                    min_vis=HALLUCINATION_VIS_THRESH,
                )
                collapsed[:, t] = is_collapsed
            else:
                picks = {}
                for fi, want_sex in enumerate(want):
                    cands = []
                    for b in range(off, off + nb):
                        r = runner.read_typed(out, b, want_sex=want_sex)
                        if r is not None and _passes_vis_guard(r, min_vis=HALLUCINATION_VIS_THRESH):
                            cands.append((-float(r["exist"]), int(b), r))
                    if not cands:
                        continue
                    own_row = -1
                    if assign_local is not None and int(assign_local[fi]) >= 0:
                        own_row = int(off) + int(assign_local[fi])
                    _ne, best_b, best = prefer_own_window(cands, own_row, "own", lambda c: c[1])
                    picks[fi] = (best, best_b)

            for fi, (best, _best_b) in picks.items():
                pts3d, _, _, _ = runner.to_pipeline(best["kp3d"], None, None, None)
                pts = np.asarray(pts3d, np.float32)
                kp3d[fi, t] = pts
                ok = np.isfinite(pts).all(axis=-1)
                if ok.any():
                    centroid[fi, t] = pts[ok].mean(axis=0)
                exist[fi, t] = best["exist"]
                sex_prob[fi, t] = best["sex_prob"]
                slot[fi, t] = best["slot"]

            for fi in range(n_flies):
                got = fi in picks and np.isfinite(centroid[fi, t]).all()
                if got:
                    local = int(picks[fi][1]) - off
                    if 0 <= local < len(win_src):
                        window_source[fi, t] = WINDOW_SOURCE_CODE[win_src[local]]
                    reacquired[fi, t] = bool(not was_tracked[fi])
                    tstate.note_read(fi, centroid[fi, t])
                else:
                    if assign_local is not None and int(assign_local[fi]) >= 0:
                        window_source[fi, t] = WINDOW_SOURCE_CODE[win_src[int(assign_local[fi])]]
                    tstate.note_miss(fi)
        pend, batch, rows = [], [], 0

    def _plan(t, fidx):
        nonlocal prev_centres, rows, W, H, warned_drop
        if pend and pend[0][0] <= t - lag:
            # THE LAG: every pending frame at or before t-lag must be read
            # NOW, since frame t's tracked window may be centred on state
            # that is only final through t-lag (see the module docstring).
            _read_batch()
        imgs, present = reader(fidx)
        imgs = np.asarray(imgs)
        if W is None:
            H, W = int(imgs.shape[1]), int(imgs.shape[2])

        peaks, scores = detector.peaks(imgs)
        cd_centres = lift_peaks_to_centres(
            peaks, scores, rig, min_views=min_views, max_resid_px=max_resid_px, max_animals=n_flies
        )
        cd_centres = cluster_centres(cd_centres)

        if cd_centres.shape[0] and np.isfinite(cd_centres).any():
            src_frame = CENTRE_DETECTED
            prev_centres = cd_centres
        elif prev_centres is not None and prev_centres.shape[0]:
            src_frame = CENTRE_REUSED
        else:
            src_frame = None

        was_tracked = tstate.was_tracked()
        plan = tstate.plan(cd_centres)
        wc, assign_local, win_src = plan.centres, plan.assignment, plan.source
        if wc.shape[0] == 0:
            # Nothing tracked and nothing detected/reused: the row stays NaN.
            return
        if src_frame is None:
            src_frame = CENTRE_NONE  # windows exist (pure carry-forward), but
            # CenterDetect said nothing new this frame

        if wc.shape[0] > runner.batch:
            if not warned_drop:
                warnings.warn(
                    f"coarse_pass: frame {fidx} planned {wc.shape[0]} windows but "
                    f"runner.batch={runner.batch}; dropping {wc.shape[0] - runner.batch} "
                    f"window(s) (only the first offending frame is reported)",
                    RuntimeWarning,
                    stacklevel=2,
                )
                warned_drop = True
            win_src = win_src[: runner.batch]
            assign_local = np.where(assign_local < runner.batch, assign_local, -1)
            wc = wc[: runner.batch]
        if rows + wc.shape[0] > runner.batch:
            _read_batch()

        batch.append(runner.windows(imgs, present, wc))
        pend.append((t, rows, int(wc.shape[0]), assign_local, win_src, was_tracked))
        rows += int(wc.shape[0])
        centre_source[:, t] = src_frame
        n_windows[t] = int(wc.shape[0])

    def _snapshot(done: int) -> dict:
        snap = {
            "frame": np.asarray(frame_list[:done], np.int64),
            "kp3d": kp3d[:, :done],
            "centroid": centroid[:, :done],
            "exist": exist[:, :done],
            "sex_prob": sex_prob[:, :done],
            "slot": slot[:, :done],
            "centre_source": centre_source[:, :done],
            "n_windows": n_windows[:done],
            "collapsed": collapsed[:, :done],
            "window_source": window_source[:, :done],
            "reacquired": reacquired[:, :done],
            "W": W,
            "H": H,
            "last_centres": prev_centres,
            "last_centroid": tstate.centroids().astype(np.float32),
        }
        return concat_tracks([resume_from, snap]) if resume_from is not None else snap

    for t, fidx in enumerate(frame_list):
        _plan(t, fidx)
        done = t + 1
        if progress is not None:
            checkpoint = None
            if partial_every and done % int(partial_every) == 0:
                _read_batch()  # flush so the checkpoint's rows are final
                checkpoint = _snapshot(done)
            progress(done, n_frames, done / max(time.time() - t_start, 1e-9), checkpoint)
    _read_batch()

    return _snapshot(n_frames)


def concat_tracks(chunks: list[dict | None]) -> dict:
    """Glue consecutive `coarse_pass`/`load_partial` results (the
    resume/chunk path). Frame-axis arrays are concatenated in order; `None`
    entries (no `resume_from`) and empty chunks are skipped.
    """
    chunks = [c for c in chunks if c is not None and np.asarray(c["frame"]).shape[0]]
    if not chunks:
        raise ValueError("nothing to concatenate")
    first = chunks[0]
    out: dict = {"frame": np.concatenate([c["frame"] for c in chunks], axis=0)}
    for k in (
        "kp3d",
        "centroid",
        "exist",
        "sex_prob",
        "slot",
        "centre_source",
        "collapsed",
        "window_source",
        "reacquired",
    ):
        out[k] = np.concatenate([np.asarray(c[k]) for c in chunks], axis=1)
    out["n_windows"] = np.concatenate([np.asarray(c["n_windows"]) for c in chunks], axis=0)
    out["W"] = first.get("W")
    out["H"] = first.get("H")
    out["last_centres"] = chunks[-1].get("last_centres")
    out["last_centroid"] = chunks[-1].get("last_centroid")
    return out


def _meta_path(path: Path) -> Path:
    return path.with_suffix("").with_suffix(".meta.json")


def write_coarse_tracks(
    path,
    tracks: dict,
    features: dict,
    *,
    rig: CameraRig,
    kp_order: Order,
    session_dir,
    stride: int,
    meta_extra: dict | None = None,
) -> dict:
    """`coarse_tracks.npz` + `.meta.json`, in the `(F, T, ...)` axis
    convention `tracking.detector.gates` reads.

    Args:
        path: output `.../coarse_tracks.npz` (the meta json is written
            beside it as `.../coarse_tracks.meta.json`).
        tracks: `coarse_pass`'s output (or an equivalent dict); must carry
            `coarse_frame` (`coarse_pass` calls this `frame`; both spellings
            are accepted, `coarse_frame` taking precedence), `exist`,
            `centroid`, `kp3d`. `W`/`H` are optional -- when absent (as in a
            purely synthetic tracks dict), `n_valid_cams`/`sep2d_med` are
            computed from finite reprojections only, without an image-bounds
            check.
        features: `coarse_features`'s output.
        rig: this recording's `CameraRig` -- `rig.cameras` becomes the
            file's camera-order stamp, `rig.project` reprojects `centroid`
            to compute `n_valid_cams`/`sep2d_med`.
        kp_order: stamped via `tracking.io.artifacts.save_npz`.
        session_dir / stride: recorded in the meta json.
        meta_extra: extra keys folded into the meta json (checkpoint,
            timings, ...).

    The meta json is written FIRST (tmp-name + `os.replace`), then the npz
    (via `save_npz`, itself atomic) -- a kill between the two leaves a meta
    describing a T one write ahead of the npz still on disk, which
    `load_partial` tolerates (it trusts the npz's own arrays for shape and
    the meta only for run-identity fields); the reverse order could leave a
    complete npz with no meta at all, which `load_partial` cannot resume
    from (no W/H).
    """
    path = Path(path)
    centroid = np.asarray(tracks["centroid"], np.float32)  # (F,T,3)
    n_flies, n_frames = centroid.shape[0], centroid.shape[1]
    coarse_frame = np.asarray(tracks.get("coarse_frame", tracks.get("frame")), np.int64)
    exist = np.asarray(tracks["exist"], np.float32)
    kp3d = np.asarray(tracks["kp3d"], np.float16)
    w, h = tracks.get("W"), tracks.get("H")

    if n_frames:
        with np.errstate(invalid="ignore", divide="ignore"):
            uv = rig.project(centroid)  # (F,T,C,2) float64
    else:
        uv = np.zeros((n_flies, 0, rig.n_cameras, 2))
    finite = np.isfinite(uv).all(axis=-1)  # (F,T,C)
    if w is not None and h is not None:
        with np.errstate(invalid="ignore"):
            inside = (
                (uv[..., 0] >= 0)
                & (uv[..., 0] <= int(w) - 1)
                & (uv[..., 1] >= 0)
                & (uv[..., 1] <= int(h) - 1)
                & finite
            )
    else:
        # No image bounds to check against: "valid" degrades to "finite" --
        # see the docstring's note on tracks dicts with no W/H.
        inside = finite
    valid = finite & inside
    n_valid_cams = valid.sum(axis=-1).astype(np.int16)  # (F,T)

    sep2d_med = np.full(n_frames, np.nan, np.float32)
    if n_flies >= 2 and n_frames:
        both = valid[0] & valid[1]
        d2 = np.linalg.norm(uv[0] - uv[1], axis=-1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices are normal
            sep2d_med = np.nanmedian(np.where(both, d2, np.nan), axis=-1).astype(np.float32)

    # `features` is `coarse_features`'s output in ordinary use, but a caller
    # (or a unit test) may supply only the subset it cares about -- everything
    # here has a well-defined "unknown" default (matching `coarse_features`'s
    # own all-NaN F<2/N<=1 degenerate cases) rather than requiring the whole set.
    dist = np.asarray(features.get("dist", np.full(n_frames, np.nan)), np.float32)
    arrays: dict[str, np.ndarray] = {
        "coarse_frame": coarse_frame,
        "exist": exist,
        "centroid": centroid,
        "kp3d": kp3d,
        "n_valid_cams": n_valid_cams,
        "sep2d_med": sep2d_med,
        "wing_angle_deg": np.asarray(
            features.get("wing_angle_deg", np.full((n_flies, n_frames), np.nan)), np.float32
        ),
        "heading_deg": np.asarray(
            features.get("heading_deg", np.full(n_frames, np.nan)), np.float32
        ),
        "speed": np.asarray(
            features.get("speed", np.full((n_flies, n_frames), np.nan)), np.float32
        ),
        "height": np.asarray(
            features.get("height", np.full((n_flies, n_frames), np.nan)), np.float32
        ),
        "dist": dist,
        "sep3d": dist,  # == dist, under the SAM3-schema name `gates.py` reads
        "trackable": np.asarray(features.get("trackable", np.zeros((n_flies, n_frames))), bool),
    }
    for k in (
        "sex_prob",
        "slot",
        "centre_source",
        "n_windows",
        "collapsed",
        "window_source",
        "reacquired",
        "last_centres",
    ):
        if tracks.get(k) is not None:
            arrays[k] = np.asarray(tracks[k])

    os.makedirs(path.parent, exist_ok=True)
    floor = tracks.get("floor")
    meta = {
        "session_dir": str(session_dir),
        "stride": int(stride),
        "cameras": list(rig.cameras.names),
        "W": int(w) if w is not None else None,
        "H": int(h) if h is not None else None,
        "num_animals": int(n_flies),
        "n_coarse": int(n_frames),
        "coarse0": int(coarse_frame[0] // int(stride)) if n_frames else 0,
        "source": "mvq",
        "floor": floor.as_dict() if isinstance(floor, FloorPlane) else None,
        "frac_trackable": (
            [float(np.mean(arrays["trackable"][f])) for f in range(n_flies)] if n_frames else []
        ),
    }
    meta.update(meta_extra or {})
    meta_path = _meta_path(path)
    tmp_meta = meta_path.with_suffix(meta_path.suffix + ".tmp")
    tmp_meta.write_text(json.dumps(meta, indent=2, default=str))
    os.replace(tmp_meta, meta_path)

    save_npz(path, arrays=arrays, kp=kp_order, cams=rig.cameras)
    return meta


def load_partial(path, num_animals: int, *, rig: CameraRig, kp_order: Order) -> dict | None:
    """Read a previously written `coarse_tracks(.partial).npz` back into a
    `coarse_pass`-shaped dict so a run can `resume_from` it.

    Returns `None` if `path` does not exist (nothing to resume from). Raises
    (via `tracking.io.artifacts.load_npz`'s order-stamp check) if the file
    was written under a different keypoint or camera order -- resuming
    across a geometry change would silently splice two different runs into
    one file. `kp3d` widens from the on-disk float16 back to float32 (see
    the module docstring's note on why a resumed chunk is not byte-identical
    to a single-shot run).
    """
    path = Path(path)
    if not path.exists():
        return None
    z = load_npz(path, kp=kp_order, cams=rig.cameras)
    n_frames = int(np.asarray(z["coarse_frame"]).shape[0])
    n_flies = int(np.asarray(z["exist"]).shape[0])
    if n_flies != int(num_animals):
        raise ValueError(
            f"{path} has {n_flies} flies, num_animals={num_animals}; refusing to resume "
            f"with a different fly count"
        )

    def _get(key, default):
        return np.asarray(z[key]) if key in z else default

    tr: dict = {
        "frame": np.asarray(z["coarse_frame"], np.int64),
        "kp3d": np.asarray(z["kp3d"], np.float32),
        "centroid": np.asarray(z["centroid"], np.float32),
        "exist": np.asarray(z["exist"], np.float32),
        "sex_prob": _get("sex_prob", np.full((n_flies, n_frames), np.nan, np.float32)),
        "slot": _get("slot", np.full((n_flies, n_frames), -1, np.int8)),
        "centre_source": _get("centre_source", np.full((n_flies, n_frames), CENTRE_NONE, np.int8)),
        "n_windows": _get("n_windows", np.zeros(n_frames, np.int8)),
        "collapsed": _get("collapsed", np.zeros((n_flies, n_frames), bool)),
        "window_source": _get(
            "window_source", np.full((n_flies, n_frames), WINDOW_SOURCE_NONE, np.int8)
        ),
        "reacquired": _get("reacquired", np.zeros((n_flies, n_frames), bool)),
    }
    tr["last_centres"] = z["last_centres"] if "last_centres" in z else None
    # The own-window/tracked-placement predecessor across the resume
    # boundary: the last written frame's per-fly centroid (NaN where that
    # fly was not read, which simply leaves it untracked on the first frame
    # back -- same as a fresh recording start).
    tr["last_centroid"] = tr["centroid"][:, -1].copy() if n_frames else None

    meta = json.loads(_meta_path(path).read_text()) if _meta_path(path).exists() else {}
    tr["W"] = meta.get("W")
    tr["H"] = meta.get("H")
    tr["floor"] = None
    return tr
