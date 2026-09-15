"""The stage registry: spec section 5's table, as data.

One row per stage, declared once. The driver reads nothing about stages from
anywhere else -- a stage's scope, artifacts and resume rule live here or they
do not exist.

`resume` is not decoration:
  "exists"    -- the artifact's presence is proof the work was done.
  "signature" -- the artifact's presence is NOT proof: a different checkpoint,
                 threshold or placement is a different computation writing the
                 same filename (spec section 5: "a different checkpoint,
                 threshold or placement is not the same computation and must
                 not be silently reused").
  "partial"   -- the artifact is written in chunks and is resumable
                 mid-recording, so its presence proves nothing about
                 completeness. The stage owns its own chunk-level check;
                 `resume.stage_is_current` REFUSES to answer for it rather
                 than guessing (spec section 5 gives `coarse` this key).
  "always"    -- a session summary over a changed set of bouts is a different
                 summary, so it is never skipped.
"""

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
    # "partial", not "exists": spec section 5 gives this stage the resume key
    # "chunked partial, resumable mid-recording". A half-written
    # `coarse_tracks.npz` is NOT the same state as a complete one, and calling
    # it "exists" would resume a truncated recording as though it were done --
    # precisely the class of silent reuse this registry exists to prevent.
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
    # "bout", not "bout_fly": `render_kpvideo` draws every fly in the bout into
    # ONE combined video, so there is no per-fly output a "bout_fly" scope
    # could distinguish -- a driver looping this stage per fly would render
    # the same file twice.
    Stage("kpvideo", "bout", ("kp3d.npz", "video"), ("kpvideo.mp4",), "exists"),
    Stage(
        "preprocess",
        "bout_fly",
        ("kp3d.npz",),
        ("kp3d_filt.npz",),
        "exists",
    ),
    # The two POOLED steps. Both read `kp3d_filt.npz`, so they sit after every
    # `preprocess`; `ik` reads `offsets_fly.h5` and `postprocess` reads
    # `floor.json`, so they sit before both. Declaring them here rather than in
    # the driver is what makes `test_the_declared_order_satisfies_every_reads_
    # writes_dependency` enforce those barriers -- otherwise the ordering holds
    # only because `plan.py` happens to implement it.
    #
    # `scale.json`/`offsets_fly.h5` are per FLY and `floor.json` per RECORDING,
    # all at the run root. Fitting either per bout is the defect spec 5.1
    # records as a 38x body-scale error, and the per-bout floor measured
    # vertical on a real bout.
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
    # Reads `outputs.h5` (the fitted pose), so it sits AFTER `postprocess` --
    # `test_the_declared_order_satisfies_every_reads_writes_dependency` checks
    # this table's declaration order against `reads`/`writes` for every row,
    # so this row's own position is what places it, not a driver convention.
    # "bout_fly", not "bout" like `kpvideo`: each fly gets its OWN
    # model->world camera build (`umeyama` fit from THAT fly's own fitted
    # sites), so one combined render per bout would need two similarity
    # transforms live in the same MuJoCo scene at once.
    Stage(
        "sidebyside",
        "bout_fly",
        ("outputs.h5", "kp2d.npz", "video"),
        ("sidebyside.mp4", "sidebyside_still.png", "sidebyside.pose_source.json"),
        "exists",
    ),
    # `ik_output_combined.h5` is a PLACEHOLDER, like `offsets_fly.h5` above: the
    # real name is `ik_output_combined_<anatomy>_<name>.h5` (spec section 4) and
    # only the caller knows the anatomy and run name. Spec section 5 lists it on
    # this row and an earlier draft of this plan dropped it.
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
    """The stage called `name`, or a `KeyError` naming the valid stages.

    Raising with the full list matters because the failure mode is a plausible
    misspelling on a CLI (`kp_video` for `kpvideo`), and a driver that silently
    dropped an unknown stage would run a SUBSET of what was asked for and exit 0.
    """
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError(f"unknown stage {name!r}; valid stages are {sorted(_BY_NAME)}") from None


def ordered(names) -> tuple[str, ...]:
    """`names` sorted into dependency order, de-duplicated.

    The caller's order is DISCARDED, deliberately. `stages=[ik,preprocess]` is a
    reasonable thing to type and a catastrophic thing to obey: the IK would read
    a `kp3d_filt.npz` preprocess has not written -- missing on a clean run, and
    STALE on a rerun, which fails silently and looks like a bad fit.
    """
    seen = {stage_by_name(n).name for n in names}
    return tuple(sorted(seen, key=lambda n: _ORDER[n]))
