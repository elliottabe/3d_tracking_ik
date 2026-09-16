"""The recording-scoped stages: `fine`, `kpvideo`, `sidebyside`, `collect`,
`fit_run_floor`.
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
    """Is bout `bout_dir`'s fine lift current at `gate_string`?"""
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
    """Fine-lift every bout in `specs` for one recording."""
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
    """Render the standard kp-video for one BOUT (`Stage("kpvideo", "bout", ...)`)."""
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
    """Fit the arena floor ONCE, pooled over every bout-fly under `run_root`."""
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
