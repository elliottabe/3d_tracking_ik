"""The stage registry: spec section 5's table, as data."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Stage", "STAGES", "stage_by_name", "ordered"]


@dataclass(frozen=True)
class Stage:
    """One pipeline stage. `scope` decides what the driver loops over."""

    name: str
    scope: str  # "recording" | "bout" | "bout_fly" | "session"
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    resume: str  # "exists" | "signature" | "partial" | "always"


# Declaration order IS the dependency order; `ordered` sorts into it.
STAGES: tuple[Stage, ...] = (
    Stage("coarse", "recording", ("video", "rig"), ("coarse_tracks.npz",), "partial"),
    Stage("gates", "recording", ("coarse_tracks.npz",), ("bouts.csv",), "exists"),
    # The alternative entry: a hand-supplied bout_summary.csv, no coarse pass.
    Stage("bouts", "recording", ("bout_summary.csv",), ("bouts.csv",), "exists"),
    Stage(
        "fine",
        "recording",
        ("video", "rig", "bouts.csv"),
        ("kp2d.npz", "kp3d.npz", "mvq_meta.json", "sex.json"),
        "signature",
    ),
    Stage("kpvideo", "bout", ("kp3d.npz", "video"), ("kpvideo.mp4",), "exists"),
    Stage(
        "preprocess",
        "bout_fly",
        ("kp3d.npz",),
        ("kp3d_filt.npz",),
        "exists",
    ),
    Stage(
        "fit_fly_constants",
        "recording",
        ("kp3d_filt.npz",),
        ("scale.json", "offsets_fly.h5"),
        "exists",
    ),
    Stage("fit_run_floor", "recording", ("kp3d_filt.npz",), ("floor.json",), "exists"),
    Stage("ik", "bout_fly", ("kp3d_filt.npz", "offsets_fly.h5"), ("stac_ik.h5",), "exists"),
    Stage(
        "postprocess",
        "bout_fly",
        ("stac_ik.h5", "kp2d.npz", "floor.json"),
        ("outputs.h5", "fitted.npz", "qc.json"),
        "exists",
    ),
    Stage(
        "sidebyside",
        "bout_fly",
        ("outputs.h5", "kp2d.npz", "video"),
        ("sidebyside.mp4", "sidebyside_still.png", "sidebyside.pose_source.json"),
        "exists",
    ),
    Stage(
        "collect",
        "session",
        ("bouts",),
        ("session_qc.json", "session_qc.md", "ik_output_combined.h5"),
        "always",
    ),
)

_BY_NAME = {s.name: s for s in STAGES}
_ORDER = {s.name: i for i, s in enumerate(STAGES)}


def stage_by_name(name: str) -> Stage:
    """The stage called `name`, or a `KeyError` naming the valid stages."""
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError(f"unknown stage {name!r}; valid stages are {sorted(_BY_NAME)}") from None


def ordered(names) -> tuple[str, ...]:
    """`names` sorted into dependency order, de-duplicated."""
    seen = {stage_by_name(n).name for n in names}
    return tuple(sorted(seen, key=lambda n: _ORDER[n]))
