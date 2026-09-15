"""The recording-scoped stages: `fine`, `kpvideo`, `sidebyside`, `collect`,
`fit_run_floor`.

Spec section 5's other rows (`coarse`, `gates`, `bouts`) are `"recording"`
scope too but are not this module's job; this one wires the rows whose heavy
collaborators are an `MVQRunner`/`CenterDetector` forward pass (`fine`), a
render (`kpvideo`), and a session-wide aggregate (`collect`).

**Traps, each already shipped once (fix round 1, 2026-09-13, folded in).**

1. `fine` resumes on its GATE STRING, never on file presence. A different
   checkpoint or threshold writes the exact same filenames
   (`kp2d.npz`/`kp3d.npz`/`mvq_meta.json`/`sex.json`), so an existence check
   alone reports "done" while mixing two models' outputs into one run with no
   error anywhere. There is exactly ONE place this gate is persisted -- the
   `gates` field embedded in each fly's `kp3d.npz`, written by
   `fine_track_bouts` and read back by `detector.fine.bout_is_current`.
   `fine_bout_is_current` below is a thin adapter onto that SAME mechanism,
   not a second sidecar store: an earlier draft of this module wrote and read
   its own `.fine.signature` file, which was exactly the "invent a second
   scheme" the brief warns against, just one layer up.

2. `kpvideo` is BOUT-scoped, not bout-fly: `render_kpvideo` draws every fly
   discovered under one `bout_dir` into ONE combined video, so there is no
   way to render "just one fly's" reprojection and no `fly` parameter to
   receive. `kpvideo_bout` (renamed from an earlier `kpvideo_bout_fly`) drops
   it; `Stage("kpvideo", ...)`'s scope in `pipeline.stages` is `"bout"`.

3. `kpvideo` must REFUSE a bout with no recorded `bout_start_frame`.
   Defaulting it to 0 renders this bout's keypoints over footage from
   hundreds of thousands of frames earlier -- and the result still looks like
   a skeleton on a fly. That shipped once (fixed in commit 969f728).
   `render_kpvideo` itself carries this guard, but only AFTER checking the
   rig's camera order against `video_dir` -- so `kpvideo_bout` checks
   `bout_start_frame` first, before either `rig` or `video_dir` is touched,
   so the refusal fires for the right reason even when those are not yet
   available (as in a dry wiring check).

4. The arena floor is fitted ONCE per recording from every bout-fly pooled,
   never per bout (spec 5.3): a per-bout floor is not merely noisier, it can
   come out perpendicular to the truth (measured: a vertical normal, pitch
   median -74.3 deg, fit from 60 frames of one fly; the same bout's own 550
   frames give a horizontal floor, -15.2 deg). Just as importantly, that pool
   must happen BEFORE any bout-fly's posture is scored against it --
   `postprocess_bout_fly` is what scores posture, and it needs the finished
   plane, not one fit after the fact. So the fit lives in its OWN function,
   `fit_run_floor`, which runs after every `preprocess` (whose `kp3d_filt.npz`
   it pools) and before any `postprocess` (which reads its `floor.json`).
   `collect_session` -- which runs LAST, once every bout-fly's `qc.json`
   already exists -- only READS that file; it never fits its own, which would
   silently be a different (and differently timed) plane than the one
   postprocess actually scored against.

5. `sidebyside` -- unlike `kpvideo` above -- is BOUT-FLY scoped, not bout:
   each fly's camera is built from THAT fly's own model->world fit, so a
   combined render would need two similarity transforms live in one MuJoCo
   scene at once. It must also SKIP, not raise, when `outputs.h5` is absent
   (a bout-fly the IK already refused, e.g. Session0 bout 8's female with
   0.000 finite root keypoints) -- see `sidebyside_bout_fly`'s docstring.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from tracking.detector.coarse import FloorPlane
from tracking.detector.fine import (
    DEFAULT_MAX_GAP_FRAMES,
    DEFAULT_NO_MERGE,
    bout_is_current,
    fine_gate_string,
    fine_track_bouts,
)
from tracking.io.names import as_order
from tracking.postprocess.combine import collect_bout_entries, combine_session
from tracking.qc.posture import fit_recording_floor
from tracking.qc.session import collect_rows, write_scorecard
from tracking.viz.kpvideo import render_kpvideo
from tracking.viz.sidebyside import render_sidebyside

__all__ = [
    "fine_bout_is_current",
    "fine_recording",
    "kpvideo_bout",
    "sidebyside_bout_fly",
    "fit_run_floor",
    "collect_session",
]


def fine_bout_is_current(bout_dir, gate_string, *, n_flies: int = 2) -> bool:
    """Is bout `bout_dir`'s fine lift current at `gate_string`?

    A direct adapter onto `detector.fine.bout_is_current` -- the ONE place a
    fine gate is persisted (the `gates` field embedded in each fly's own
    `kp3d.npz`, written by `fine_track_bouts`). This function exists only so
    callers in `pipeline` reach the check through this module rather than
    reaching into `tracking.detector.fine` directly; it adds no store of its
    own and no different notion of "current".
    """
    return bout_is_current(bout_dir, gate_string, n_flies=n_flies)


def fine_recording(
    runner,
    detector,
    *,
    specs,
    reader_factory,
    kp_order,
    out_root,
    num_animals: int = 2,
    placement_lag=None,
    no_merge=DEFAULT_NO_MERGE,
    min_vis="auto",
    bouts_in_flight: int = 4,
    max_gap_frames: int = DEFAULT_MAX_GAP_FRAMES,
    force: bool = False,
    dry: bool = False,
    progress_every: int = 0,
):
    """Fine-lift every bout in `specs` for one recording.

    `num_animals` MUST come from the recording (`RecordingSpec.num_animals`):
    passing the default 2 for a free-running (single-fly) recording writes a
    `fly1/` full of NaN that every downstream per-fly stage then scales, fits
    offsets for, and solves -- a silent, expensive mistake with no exception
    anywhere.

    This function does no resume filtering of its own: `fine_track_bouts`
    already resumes per bout internally (`detector.fine.bout_is_current`, the
    gate embedded in `kp3d.npz`), and there is exactly one place that gate is
    stored -- re-deciding the same question here from a second store would
    only risk the two disagreeing. `fine_gate_string` is still computed
    up front, purely to name (in the log line below) exactly which run this
    call is -- the same string `fine_track_bouts` derives internally from
    `runner`/`placement_lag`/`no_merge`/`min_vis`, recomputed here only for
    that label, never written anywhere.

    Returns `fine_track_bouts`'s own per-spec result list, unmodified.
    """
    from tracking.conventions import announce

    specs = list(specs)
    gate = fine_gate_string(runner, placement_lag=placement_lag, no_merge=no_merge, min_vis=min_vis)
    announce(
        "pipeline.recording_stages",
        "fine_recording",
        f"gate={gate!r} num_animals={num_animals} n_specs={len(specs)}",
    )

    return fine_track_bouts(
        runner,
        specs,
        detector=detector,
        reader_factory=reader_factory,
        kp_order=kp_order,
        out_root=out_root,
        num_animals=num_animals,
        placement_lag=placement_lag,
        no_merge=no_merge,
        min_vis=min_vis,
        bouts_in_flight=bouts_in_flight,
        max_gap_frames=max_gap_frames,
        force=force,
        dry=dry,
        progress_every=progress_every,
    )


def kpvideo_bout(
    bout_dir,
    *,
    rig,
    kp_order,
    video_dir=None,
    recording=None,
    out_path=None,
    source: str = "kp3d",
    compare=None,
    colour_by: str = "fly",
    fps: float = 30,
) -> str:
    """Render the standard kp-video for one BOUT (`Stage("kpvideo", "bout", ...)`).

    `render_kpvideo` draws every fly discovered under `bout_dir` into ONE
    combined video -- there is no per-fly output to select, which is why this
    function (unlike the bout-fly stages) takes no `fly` parameter.

    REFUSES a bout with no recorded `bout_start_frame` in `mvq_meta.json`,
    checked BEFORE `rig`, `video_dir` or `render_kpvideo` itself are ever
    touched. Defaulting that value to 0 does not degrade the render -- it
    silently plays this bout's keypoints back over a different, unrelated
    part of the recording, and the overlay still looks like a working
    skeleton doing it (fixed once, in commit 969f728). `render_kpvideo`
    itself carries the same guard, but only after first checking the rig's
    camera order against `video_dir`; checking here first means the refusal
    fires for the right reason even when `rig`/`video_dir` are not yet
    available to check.
    """
    bout_dir = Path(bout_dir)
    meta_path = bout_dir / "mvq_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if "bout_start_frame" not in meta:
        raise FileNotFoundError(
            f"{meta_path} is missing or has no `bout_start_frame`, so bout "
            f"{bout_dir.name}'s kp-video cannot know which video frames belong "
            f"to this bout. Refusing rather than defaulting to 0, which would "
            f"draw this bout's keypoints over a different part of the recording "
            f"and look plausible doing it."
        )

    return render_kpvideo(
        bout_dir,
        recording=recording,
        rig=rig,
        kp_order=kp_order,
        video_dir=video_dir,
        source=source,
        out_path=out_path if out_path is not None else bout_dir / "kpvideo.mp4",
        compare=compare,
        colour_by=colour_by,
        fps=fps,
    )


def sidebyside_bout_fly(
    fly_dir,
    *,
    fly: int,
    anatomy,
    rig,
    video_dir,
    cameras=None,
    frames=None,
    n_preview_frames: int = 8,
    geom_size: float = 0.004,
    pad: int = 170,
    fps: float = 4.0,
):
    """Render the side-by-side check for ONE bout-fly (`Stage("sidebyside",
    "bout_fly", ...)`, Task 16).

    Per BOUT-FLY, not per bout, unlike `kpvideo_bout` above: each fly's
    camera is built from THAT fly's own `umeyama` fit (`render_sidebyside`'s
    docstring), so one combined render per bout would need two similarity
    transforms live in the same MuJoCo scene at once.

    SKIPS -- returns `None` -- rather than raising when this bout-fly has no
    `outputs.h5`. Measured on Session0: bout 8's female has 0.000 finite root
    keypoints, the IK correctly refuses to solve it, and no `outputs.h5` is
    ever written. Raising here would report a SECOND, misleading failure for
    a bout-fly the pipeline already knows failed upstream, at a stage that by
    definition has nothing to render.

    `bout_start_frame` comes from the BOUT's own `mvq_meta.json`
    (`fly_dir.parent`, mirroring `kpvideo_bout`'s read of the same file) --
    never passed in or defaulted. Defaulting to 0 would still produce a
    playable video, with the real-camera panel seeked to a different part of
    the recording than the pose it is drawn beside: two panels of two
    different moments, looking like one (`kpvideo_bout`'s docstring measured
    the concrete cost of exactly this default, fixed once in 969f728).
    """
    fly_dir = Path(fly_dir)
    bout_dir = fly_dir.parent

    if not (fly_dir / "outputs.h5").exists():
        print(
            f"skip  sidebyside          {bout_dir.name}/fly{fly}  "
            f"(no outputs.h5 -- IK did not solve this bout-fly)"
        )
        return None

    meta_path = bout_dir / "mvq_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if "bout_start_frame" not in meta:
        raise FileNotFoundError(
            f"{meta_path} is missing or has no `bout_start_frame`, so "
            f"{bout_dir.name}/fly{fly}'s side-by-side render cannot know which "
            f"video frames belong to this bout. Refusing rather than defaulting "
            f"to 0 -- see this function's docstring for what that default costs."
        )

    return render_sidebyside(
        fly_dir,
        fly=fly,
        anatomy=anatomy,
        rig=rig,
        video_dir=video_dir,
        bout_start_frame=int(meta["bout_start_frame"]),
        video_path=fly_dir / "sidebyside.mp4",
        still_path=fly_dir / "sidebyside_still.png",
        pose_source_path=fly_dir / "sidebyside.pose_source.json",
        cameras=cameras,
        frames=frames,
        n_preview_frames=n_preview_frames,
        geom_size=geom_size,
        pad=pad,
        fps=fps,
    )


def _pool_kp3d_filt(run_root: Path, kp_order) -> list[np.ndarray]:
    """Every bout-fly's `kp3d_filt.npz` `kp3d` array under `run_root`, as a list."""
    kp = as_order(kp_order)
    filt_paths = sorted(run_root.glob("bouts/bout_*/fly*/kp3d_filt.npz"))
    if not filt_paths:
        raise FileNotFoundError(
            f"no bout-fly kp3d_filt.npz under {run_root}/bouts; the arena floor "
            f"is fit from every bout-fly's filtered keypoints and there is "
            f"nothing here to pool"
        )
    pooled = []
    for p in filt_paths:
        with np.load(p, allow_pickle=True) as z:
            arr = np.asarray(z["kp3d"], dtype=np.float64)
        if arr.shape[-2] != len(kp):
            raise ValueError(f"{p}: kp3d has {arr.shape[-2]} keypoints, kp_order names {len(kp)}")
        pooled.append(arr)
    return pooled


def fit_run_floor(run_root, *, kp_order, n_fit: int = 2000, up_hint=None) -> dict:
    """Fit the arena floor ONCE, pooled over every bout-fly under `run_root`.

    Runs AFTER every `preprocess` (which writes the `kp3d_filt.npz` this pools)
    and BEFORE any `postprocess` (which is what scores posture against it).
    Persisted as `floor.json` at the RUN ROOT (never inside a bout directory)
    so a resumed or array-scheduled run reads the same plane every other bout
    was scored on -- a floor recomputed per job is a different plane per job.

    Spec 5.3: the floor is a property of the ARENA. A per-bout fit is not
    merely noisier; measured on one real bout it came out perpendicular to
    the truth (a vertical normal, pitch median -74.3 deg, from 60 frames of
    one fly, where the same bout's own 550 frames give a horizontal floor,
    -15.2 deg) -- and every posture number computed against it is then
    meaningless.
    """
    run_root = Path(run_root)
    pooled_kp3d = _pool_kp3d_filt(run_root, kp_order)
    floor = fit_recording_floor(pooled_kp3d, n_fit=n_fit, up_hint=up_hint)

    payload = floor.as_dict()
    floor_path = run_root / "floor.json"
    tmp_path = floor_path.with_suffix(floor_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2))
    os.replace(tmp_path, floor_path)
    return payload


def collect_session(
    run_root,
    *,
    excluded=(),
    json_path=None,
    md_path=None,
    anatomy=None,
    source_hz=None,
    git_sha=None,
    run_name=None,
    fly_id=None,
    bucket=None,
) -> dict:
    """Aggregate one recording's bout-fly QC into a session scorecard, then
    combine every bout-fly's solve into the downstream dataset h5.

    `fly_id`/`bucket` pass straight through to `combine_session` (from
    `cfg.recording.name`/`cfg.recording.assay` at the real call site in
    `run.py`) so `info/fly_ids`/`info/buckets` are correct even when
    `run_root` doesn't follow the path shape `combine.py`'s fallback
    inference requires -- exactly the shape task 18's verification run root
    (`Outfiles/2026-09-14-session0-full`) does not have.

    Reads the pooled arena floor from `floor.json` at the run root -- written
    by `fit_run_floor`, which must already have run (after `preprocess`,
    before `postprocess`) so every bout-fly's `qc.json` was scored against
    THIS plane. Raises a clear error if `floor.json` is absent rather than
    silently fitting a new one here: `collect_session` runs LAST, after every
    `qc.json` already exists, so a floor fit at this point could only ever be
    a DIFFERENT plane than the one postprocess actually used -- exactly the
    per-bout-timing defect spec 5.3 warns about, moved to a different stage
    rather than fixed.

    `collect_rows`/`write_scorecard` (`tracking.qc.session`) do the actual
    per-bout-fly row gathering and file writing; this function's job is only
    to read the floor `fit_run_floor` already committed and hand the rest to
    them unchanged.

    COMBINING is opt-in on `anatomy` (`None` by default): callers that only
    want the scorecard -- and the tests above, which predate `combine.py`
    being wired in and pin the floor-reading behaviour in isolation -- pass
    none of `anatomy`/`source_hz`/`git_sha`/`run_name` and get exactly the old
    two files. The real driver (`run.py`) always supplies all four; supplying
    `anatomy` without the other three is a caller bug, so that combination
    raises rather than silently skipping.

    `out_path` is built HERE, as `<run_root>/ik_output_combined_<anatomy.name>
    _<run_name>.h5` (spec section 4; `pipeline/stages.py`'s `Stage("collect",
    ...)` comment) -- not accepted as a parameter -- because `anatomy.name`
    and `run_name` are exactly the two pieces of information the caller holds
    and this function needs anyway to name the file correctly; taking a
    pre-built `out_path` instead would let a caller pass one missing either
    piece with nothing here to catch it.

    A session with NO usable bout-fly (every solve NaN'd, or none ran) SKIPS
    the combined h5 with a clear stderr line rather than writing one. Writing
    an empty-N file was the other option; skipping was chosen because
    `combine_session`/`_combine` already treat zero entries as an error
    (`ValueError`, tested in `test_combine.py`) -- a second, N=0-shaped
    "successful" file would be a second way to spell the same fact the
    ValueError already spells, and a consumer would need to know to check for
    it. This function still must not let that ValueError surface here: the
    scorecard above is already on disk, and raising past that point would
    make a half-finished collect (scorecard yes, dataset exception) look like
    a crashed one when the run itself did nothing wrong.
    """
    run_root = Path(run_root)
    floor_path = run_root / "floor.json"
    if not floor_path.exists():
        raise FileNotFoundError(
            f"{floor_path} does not exist. collect_session reads the pooled arena "
            f"floor `fit_run_floor` writes (it must run after preprocess and before "
            f"postprocess) rather than fitting its own here -- by the time collect "
            f"runs, every bout-fly's qc.json has already been scored against SOME "
            f"floor, and fitting a fresh one now could only disagree with it."
        )
    floor_dict = json.loads(floor_path.read_text())
    floor = FloorPlane(
        np.asarray(floor_dict["normal"], dtype=np.float64),
        float(floor_dict["offset"]),
        str(floor_dict.get("orientation", "skew")),
        float(floor_dict.get("skew", float("nan"))),
    )

    rows = collect_rows(run_root, excluded=excluded)
    json_path = Path(json_path) if json_path is not None else run_root / "session_qc.json"
    md_path = Path(md_path) if md_path is not None else run_root / "session_qc.md"
    report = write_scorecard(run_root, rows, json_path=json_path, md_path=md_path)
    result = {**report, "floor": floor.as_dict()}

    if anatomy is None:
        return result
    if source_hz is None or git_sha is None or run_name is None:
        raise ValueError(
            "collect_session got `anatomy` but not all of source_hz/git_sha/"
            "run_name -- combining is all-or-nothing, not a partial default"
        )
    out_path = run_root / f"ik_output_combined_{anatomy.name}_{run_name}.h5"

    entries, combine_excluded = collect_bout_entries(run_root)
    if not entries:
        print(
            f"collect_session: no usable bout-fly solve under {run_root} "
            f"({len(combine_excluded)} excluded) -- skipping {out_path}; "
            f"the scorecard above is still written",
            file=sys.stderr,
        )
        result["combined_h5"] = None
        return result

    result["combined_h5"] = combine_session(
        run_root,
        anatomy=anatomy,
        source_hz=source_hz,
        git_sha=git_sha,
        out_path=out_path,
        fly_id=fly_id,
        bucket=bucket,
    )
    return result
