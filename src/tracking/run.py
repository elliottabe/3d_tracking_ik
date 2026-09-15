"""`python -m tracking.run` -- the Hydra entry point.

Reads exactly one config (`configs/pipeline.yaml`, spec section 6), resolves
it through `pipeline.plan.resolve` into an ordered, expanded plan (the two
pooled-fit barriers already in place -- see `plan.py`'s module docstring for
the guarantee this whole task exists for), and either PRINTS that plan
(`dry_run=true`, spec section 7) or executes each unresolved item, wrapping
every stage actually run in `pipeline.timing.timed`.

    python -m tracking.run recording=session0 mvq=v2 run.name=pose_v2
    python -m tracking.run recording=session0 stages=[fine,preprocess,ik] bout_ids=[28]
    python -m tracking.run recording=stairs dry_run=true

**`dry_run` runs NOTHING.** It never creates `run.root`, never reads
`bouts.csv`, never constructs an anatomy/rig/detector -- `plan.resolve` only
reads plain config values and checks file EXISTENCE (never contents), so a
dry run against a session directory that does not exist at all still
succeeds and only prints.

**Execution wiring in this entry point is partial, and says so as it runs.**
`coarse` and `gates` have no cfg-driven constructor for the detector/runner
objects their real callables need (`MVQRunner`, `CenterDetector`) anywhere in
this entry point yet -- that remains future work, not something this entry
point fakes. Every OTHER stage is wired to its real callable: `bouts` to
`tracking.io.bouts.read_bout_summary`/`write_bouts_csv` (the hand-supplied-
summary alternative to `gates`, spec section 5's "(alt)" row); `fine` to
`tracking.detector.fine.fine_track_bouts`, fed the `MVQRunner`/
`CenterDetector` this entry point now DOES build, from the `mvq`/
`centerdetect` config groups; and `kpvideo`, `preprocess`, the two pooled
fits, `ik`, `postprocess`, `sidebyside`, `collect` to `pipeline.bout_stages` /
`pipeline.recording_stages`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

import tracking.utils.path_utils  # noqa: F401 -- registers ${repo_root:} etc. before compose
from tracking.conventions import NotFit
from tracking.pipeline import plan as P
from tracking.pipeline import stages as S
from tracking.pipeline.stages import stage_by_name
from tracking.pipeline.timing import timed

__all__ = ["main"]


def _print_plan(cfg, works: tuple[P.Work, ...]) -> None:
    ordered_names = list(S.ordered(cfg.stages)) if list(cfg.stages) else []
    print(f"requested stages (dependency order, before the pooled-fit barriers): {ordered_names}")
    print(f"recording = {cfg.recording.name!r}  run.root = {cfg.run.root}")
    for w in works:
        status = f"SKIP ({w.skip_reason})" if w.skip_reason else "RUN "
        print(f"  [{status}] {w.stage:<20} {w.target:<24} {w.out_dir}")
    print(f"{len(works)} work item(s); dry run -- nothing executed")


def _discover_bout_ids(run_root: Path) -> list[int]:
    """Every bout id in `run_root/bouts.csv`, written by `gates`/`bouts`."""
    from tracking.io.bouts import read_bouts_csv

    bouts_csv = run_root / "bouts.csv"
    if not bouts_csv.exists():
        raise FileNotFoundError(
            f"{bouts_csv} does not exist and no bout_ids= was given on the "
            f"CLI -- run 'gates' or 'bouts' first (they write it), or pass "
            f"bout_ids=[...] explicitly"
        )
    return [b.idx for b in read_bouts_csv(bouts_csv)]


def _load_anatomy(cfg: DictConfig):
    """`inverse_kinematics.anatomy.load_anatomy`, fed a DETACHED anatomy cfg.

    `configs/anatomy/v1.yaml` self-references its own `root` key
    (`mjcf_path: "${root}/..."`) as a BARE, unprefixed interpolation, which
    OmegaConf resolves ABSOLUTELY from the config's own root node. That is
    `cfg.anatomy` itself only when `cfg.anatomy` IS the root -- true for a
    config loaded standalone, false for the live child node Hydra composes
    here (whose absolute root is the whole pipeline config, which has no
    top-level `root` key). Round-tripping through
    `to_container(resolve=False)` -> `OmegaConf.create` detaches it,
    reproducing the standalone shape `load_anatomy` expects (see
    `load_anatomy`'s own docstring for the same fact from the callee side).
    """
    from tracking.inverse_kinematics.anatomy import load_anatomy

    detached = OmegaConf.create(OmegaConf.to_container(cfg.anatomy, resolve=False))
    return load_anatomy(detached)


def _recording_spec(cfg: DictConfig):
    from tracking.io.recording import RecordingSpec

    return RecordingSpec.from_config(cfg.recording)


def _rig(spec):
    from tracking.geometry.rig import CameraRig

    return CameraRig.from_calib_dir(spec.calib_dir)


def _plain(value):
    """A config subtree as a PLAIN `dict`/`list`/scalar, resolved.

    The numeric modules (`filter_kp3d`, `solver_settings_from_cfg`,
    `solve_bout`, `write_stac_h5`, ...) deliberately REFUSE an OmegaConf
    object (`conventions.refuse_config_object`/`plain_mapping`): a `DictConfig`
    is itself a `Mapping`, so nothing stops it walking in, and it then fails
    deep inside with a confusing error, or silently defaults an unread key
    OFF. Conversion happens at the boundary that owns the config -- `run.py`
    is that boundary for every stage-config subtree handed to a
    `tracking.*` callable -- so this repo has exactly one place per subtree
    where "still Hydra-coupled or not" is visible, rather than a silent
    conversion hiding it.

    `OmegaConf.is_config` is `False` for `None` and plain scalars, so this is
    also safe to call on a leaf value (e.g. `postprocess.floor.up_hint`,
    `None` by default) that is a container only when a caller overrides it
    to one.

    The ONE deliberate exception is the anatomy config: `load_anatomy`
    itself documents that it ACCEPTS (and requires) an OmegaConf node, so
    `_load_anatomy` above does its own, different conversion
    (`resolve=False`, then re-wrapped rather than left plain) and must not
    route through this helper.
    """
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _git_sha() -> str:
    """This repo's HEAD commit, or `"unknown"` if it cannot be read.

    No existing helper does this (`grep -rn "git_sha\\|rev-parse" src/` finds
    only `combine.py`'s parameter and `scripts/dev/gen_solver_reference.py`'s
    one-off `subprocess.run` against a hardcoded path) -- this is the first
    caller that needs it from a live run, so it is added here rather than a
    second copy of that script's inline call. Best-effort, not a hard
    failure: the combined h5 is the pipeline's terminal deliverable, and a
    `.git`-less checkout (a tarball deploy, a stripped worktree) must not
    block writing the dataset that is the whole point of the run -- it would
    just be a run nobody could trace back to code, which `git_sha=="unknown"`
    already says plainly.
    """
    import subprocess

    repo_root = Path(__file__).resolve().parents[2]
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except OSError:
        return "unknown"
    sha = proc.stdout.strip()
    return sha if proc.returncode == 0 and sha else "unknown"


def _fly_from_target(target: str) -> int:
    """`"bout3/fly1"` -> `1`; `"fly1"` -> `1`."""
    return int(target.rsplit("fly", 1)[1])


def _bout_from_target(target: str) -> int:
    """`"bout3/fly1"` -> `3`; `"bout3"` -> `3`."""
    return int(target.split("/", 1)[0].removeprefix("bout"))


def _execute(work: P.Work, cfg: DictConfig, ctx: dict[str, Any], bout_ids: list[int]) -> None:
    """Run one unresolved `Work` item, timed."""
    stage, out_dir = work.stage, work.out_dir
    anatomy, spec, rig = ctx["anatomy"], ctx["spec"], ctx["rig"]
    run_root = Path(cfg.run.root)
    timing_path = run_root / "timing.json"

    if stage in ("coarse", "gates"):
        raise NotImplementedError(
            f"stage {stage!r} has no cfg-driven driver wiring in this entry point yet "
            f"(it needs a constructed MVQRunner/CenterDetector this pure-Python driver "
            f"does not build) -- simply not implemented in this entry point yet, unlike "
            f"'bouts'/'fine' and every other stage in the chain, which ARE wired to their "
            f"real callables"
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    if stage == "bouts":
        from tracking.io.bouts import read_bout_summary, write_bouts_csv

        # The hand-supplied-summary ALTERNATIVE to `gates` (spec section 5's
        # "(alt)" row; `plan.resolve` refuses a stage list naming both). No
        # coarse pass, no detector/runner needed -- this is plain CSV I/O,
        # in `run_root/bouts.csv`, the exact shape `_discover_bout_ids`
        # reads back (`read_bouts_csv`'s columns): a writer whose own reader
        # cannot parse it is this project's most-repeated defect, so this
        # goes through `write_bouts_csv` rather than a bespoke `csv.writer`
        # call here that could drift from it. `write_bouts_csv` itself is
        # already atomic (tmp file + `os.replace`), so a run killed mid-write
        # never leaves `_discover_bout_ids` a half-written file to choke on.
        bouts = read_bout_summary(spec.bouts_csv, session_tag=spec.name)
        with timed(timing_path, stage, n_items=len(bouts)):
            write_bouts_csv(run_root / "bouts.csv", bouts)
        return

    if stage == "fine":
        from tracking.detector.centerdetect.detector import CenterDetector
        from tracking.detector.fine import fine_track_bouts
        from tracking.detector.mvq.runner import MVQRunner
        from tracking.io.bouts import read_bout_summary, select
        from tracking.io.video import SlotReader

        # The mask-free lift: `fine_track_bouts` batches every bout's
        # windows through ONE `MVQRunner`, so both it and the `CenterDetector`
        # (used only to (re)acquire a fly with no live track -- this route
        # carries no coarse-pass `init_centroid`, see `BoutSpec`'s docstring)
        # are built once here, from the `mvq`/`centerdetect` config groups,
        # rather than per bout.
        runner = MVQRunner(
            str(cfg.mvq.checkpoint),
            step=cfg.mvq.step,
            attn_impl=cfg.mvq.attn_impl,
            rig=rig,
            kp_order=anatomy.kp_order,
            batch=int(cfg.mvq.batch),
            exist_thresh=float(cfg.mvq.exist_thresh),
            # `fine_track_bouts` REFUSES a runner that isn't `window_pref=
            # "own"` -- the gate string it writes into every bout's
            # `kp3d.npz` promises exactly that rule was run, so this is read
            # off `cfg.mvq.window_pref` (currently always "own",
            # `configs/mvq/v2.yaml`) rather than hardcoded, so a config that
            # ever tried a different value would be caught by that refusal
            # instead of silently mislabelling its output.
            window_pref=str(cfg.mvq.window_pref),
        )
        detector = CenterDetector(
            str(cfg.centerdetect.checkpoint), min_score=float(cfg.centerdetect.min_score)
        )

        def _reader_factory(bout_spec):
            return SlotReader(
                spec.session_dir, rig.cameras, start_slot=bout_spec.start_frame, stride=1
            )

        # The SAME hand-supplied summary `bouts` reads (fact verified
        # 2026-09-14: 30 BoutSpecs, idx 1..30) -- not `run_root/bouts.csv`,
        # which only the BOUT-SCOPED stages (`kpvideo`/`preprocess`/`ik`/
        # `postprocess`) need discovered up front, so `fine` can run without
        # depending on `bouts` having already written that file in this same
        # invocation. `select` narrows to the caller's `bout_ids` (CLI
        # `bout_ids=[...]`) when given, else keeps every bout in the summary.
        specs = select(read_bout_summary(spec.bouts_csv, session_tag=spec.name), bout_ids)

        with timed(timing_path, stage, n_items=len(specs)):
            fine_track_bouts(
                runner,
                specs,
                detector=detector,
                reader_factory=_reader_factory,
                kp_order=anatomy.kp_order,
                out_root=run_root,
                num_animals=spec.num_animals,
                dry=False,
            )
        return

    if stage == "kpvideo":
        from tracking.pipeline.recording_stages import kpvideo_bout

        with timed(timing_path, stage, n_items=1):
            kpvideo_bout(out_dir, rig=rig, kp_order=anatomy.kp_order, video_dir=spec.session_dir)
        return

    if stage == "sidebyside":
        from tracking.pipeline.recording_stages import sidebyside_bout_fly

        fly = _fly_from_target(work.target)
        with timed(timing_path, stage, n_items=1):
            sidebyside_bout_fly(
                out_dir,
                fly=fly,
                anatomy=anatomy,
                rig=rig,
                video_dir=spec.session_dir,
                cameras=_plain(cfg.viz.sidebyside.cameras),
                frames=_plain(cfg.viz.sidebyside.frames),
                n_preview_frames=int(cfg.viz.sidebyside.n_preview_frames),
                geom_size=float(cfg.viz.sidebyside.geom_size),
                pad=int(cfg.viz.sidebyside.pad),
                fps=float(cfg.viz.sidebyside.fps),
            )
        return

    if stage == "preprocess":
        from tracking.pipeline.bout_stages import preprocess_bout_fly

        with timed(timing_path, stage, n_items=1):
            preprocess_bout_fly(
                anatomy,
                bout_dir=out_dir,
                kp_order=anatomy.kp_order,
                cfg=_plain(cfg.preprocess.filter),
            )
        return

    if stage == "fit_fly_constants":
        from tracking.pipeline.bout_stages import fit_fly_constants, load_bout_kp3d

        fly = _fly_from_target(work.target)
        kp3d_by_bout, conf_by_bout = {}, {}
        for b in bout_ids:
            fly_dir = run_root / "bouts" / f"bout_{int(b):05d}" / f"fly{fly}"
            filt = fly_dir / "kp3d_filt.npz"
            if not filt.exists():
                continue
            kp3d, conf = load_bout_kp3d(filt, kp_order=anatomy.kp_order)
            kp3d_by_bout[b] = kp3d
            conf_by_bout[b] = conf
        with timed(timing_path, stage, n_items=len(kp3d_by_bout)):
            fit_fly_constants(
                anatomy,
                fly=fly,
                kp3d_by_bout=kp3d_by_bout,
                conf_by_bout=conf_by_bout,
                run_root=run_root,
                model_xml=anatomy.model_xml,
                n_frames=int(cfg.preprocess.offsets.n_frames),
                min_conf=float(cfg.preprocess.offsets.min_conf),
                mad_k=float(cfg.preprocess.offsets.mad_k),
            )
        return

    if stage == "fit_run_floor":
        from tracking.pipeline.recording_stages import fit_run_floor

        with timed(timing_path, stage, n_items=len(bout_ids)):
            fit_run_floor(
                run_root,
                kp_order=anatomy.kp_order,
                n_fit=int(cfg.postprocess.floor.n_fit),
                up_hint=_plain(cfg.postprocess.floor.up_hint),
            )
        return

    if stage == "ik":
        import json

        from tracking.inverse_kinematics.offsets_fit import load_offsets
        from tracking.pipeline.bout_stages import ik_bout_fly, load_bout_kp3d

        fly = _fly_from_target(work.target)
        kp3d_filt, _ = load_bout_kp3d(out_dir / "kp3d_filt.npz", kp_order=anatomy.kp_order)
        offsets = load_offsets(
            run_root / f"offsets_fly{fly}.h5",
            kp_order=anatomy.kp_order,
        )
        scale = float(
            json.loads((run_root / "scale.json").read_text())["scale_by_fly"][str(fly)]["scale"]
        )
        # `dt` is a property of the RECORDING, not a solver setting --
        # `configs/ik/default.yaml`'s `per_frame:` block deliberately has no
        # `dt` key (`solver_settings_from_cfg` reads exactly that block).
        # `solve_bout` requires it in seconds and refuses to infer it from
        # `mj_model.opt.timestep` (the XML's 0.0001 -- see perframe.py's `dt`
        # guard): that would scale every qvel by 12.5x with nothing NaN and
        # no residual moved. `recording.fps` is the sole source of truth
        # (`configs/recording/template.yaml`), already validated `> 0` by
        # `RecordingSpec.from_config`, so it is added here rather than
        # re-checked.
        with timed(timing_path, stage, n_items=int(kp3d_filt.shape[0])):
            ik_bout_fly(
                anatomy,
                bout_dir=out_dir,
                kp3d_filt=kp3d_filt,
                offsets=offsets,
                scale=scale,
                per_frame_cfg={**_plain(cfg.ik.per_frame), "dt": 1.0 / float(spec.fps)},
            )
        return

    if stage == "postprocess":
        import json

        from tracking.detector.coarse import FloorPlane
        from tracking.pipeline.bout_stages import postprocess_bout_fly
        from tracking.qc.posture import fly_sex_label

        fly = _fly_from_target(work.target)
        floor_dict = json.loads((run_root / "floor.json").read_text())
        floor = FloorPlane(
            __import__("numpy").asarray(floor_dict["normal"], dtype=float),
            float(floor_dict["offset"]),
            str(floor_dict.get("orientation", "skew")),
            float(floor_dict.get("skew", float("nan"))),
        )
        sex_json = out_dir.parent / "sex.json"
        sex = (
            fly_sex_label(json.loads(sex_json.read_text()), fly) if sex_json.exists() else "unknown"
        )
        kp_scale = float(
            json.loads((run_root / "scale.json").read_text())["scale_by_fly"][str(fly)]["scale"]
        )
        with timed(timing_path, stage, n_items=1):
            postprocess_bout_fly(
                anatomy,
                bout_dir=out_dir,
                kp_order=anatomy.kp_order,
                cameras=spec.cameras,
                rig=rig,
                floor=floor,
                sex=sex,
                kp_scale=kp_scale,
                source=str(cfg.postprocess.source),
                conf_thresh=float(cfg.postprocess.conf_thresh),
            )
        return

    if stage == "collect":
        from tracking.pipeline.recording_stages import collect_session

        # `anatomy`, `source_hz` and `run_name` pass through so
        # `collect_session` can name the combined h5
        # `ik_output_combined_<anatomy.name>_<run.name>.h5` (spec section 4;
        # `pipeline/stages.py`'s `Stage("collect", ...)` comment): the
        # PLACEHOLDER `ik_output_combined.h5` it declares cannot carry
        # either piece, because only THIS call site knows both -- naming it
        # here (rather than handing `collect_session` a pre-built path)
        # would let a caller build one with either piece missing and nothing
        # would catch it.
        with timed(timing_path, stage, n_items=len(bout_ids)):
            collect_session(
                run_root,
                anatomy=anatomy,
                # `recording.fps` is the SOLE source of `source_hz`
                # (`configs/recording/template.yaml`); the IK's own `dt` is
                # `1/fps` (see the `ik` branch above), so a wrong value here
                # doesn't just mislabel an attr, it labels every velocity in
                # the file with a rate that was never actually run.
                source_hz=float(_plain(cfg.recording.fps)),
                git_sha=_git_sha(),
                run_name=_plain(cfg.run.name),
                # Passed rather than left to `combine.py`'s path inference:
                # a foreign run root (`run.allow_foreign_root=true`, e.g.
                # this task's verification root `Outfiles/2026-09-14-
                # session0-full`) doesn't match `<assay>/<Session>/
                # <recording>/<pose dir>`, and this call site is the one
                # place that already holds both values from the config.
                fly_id=_plain(cfg.recording.name),
                bucket=_plain(cfg.recording.assay),
            )
        return

    # pragma: no cover -- plan.resolve() already validated `stage`
    raise AssertionError(f"unhandled stage {stage!r}")


RUN_ROOT_MARKER = ".tracking_run"


def _claim_run_root(run_root: Path, *, allow_foreign: bool) -> None:
    """Create `run_root`, refusing a directory this pipeline did not create.

    `run.root` lands inside the SHARED processed tree
    (`processed/<assay>/<recording>/<run.name>`), beside reference runs like
    `pose_maskfree_v2` that cannot be regenerated. Before 2026-09-14 it lived
    in a private `processed/ik_port` namespace where a wrong `run.name` landed
    somewhere empty and harmless; now the same typo lands on top of data
    someone else depends on.

    So a run root is CLAIMED: the first write drops a `.tracking_run` marker,
    and a directory that already exists WITHOUT one belongs to something else.
    Refusing is the whole point -- `exist_ok=True` would happily merge this
    run's artifacts into a reference run, and the result would look like a
    successful run rather than like damage.

    An EMPTY existing directory is claimed rather than refused: that is what a
    `mkdir -p` or a cancelled run leaves behind, and treating it as foreign
    would make the guard fire on its own debris.
    """
    marker = run_root / RUN_ROOT_MARKER
    if not run_root.exists():
        run_root.mkdir(parents=True)
        marker.write_text("written by tracking.run; see run.allow_foreign_root\n")
        return
    if marker.exists() or not any(run_root.iterdir()):
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
        return
    if allow_foreign:
        marker.write_text("adopted via run.allow_foreign_root=true\n")
        return
    raise SystemExit(
        f"refusing to write into {run_root}: it exists, is not empty, and carries no "
        f"{RUN_ROOT_MARKER} marker, so this pipeline did not create it. It may be a "
        f"reference run or another tool's output, and merging into it would look like "
        f"success rather than damage.\n"
        f"  If you meant a NEW run:      run.name=<something-new>\n"
        f"  If you meant to adopt this:  run.allow_foreign_root=true"
    )


def _context(cfg: DictConfig, works: list[P.Work]) -> dict[str, Any]:
    """The shared per-run objects, built once and only if something will run."""
    spec = _recording_spec(cfg)
    # Real execution only -- NEVER under `dry_run`, which must work without the
    # recording present at all. `validate()` checks `session_dir`/`calib_dir`
    # exist and enforces `cameras` against the calibration glob order; a
    # permuted rig plots one camera's keypoints on another camera's image while
    # still looking almost plausible, so this must raise before any stage reads
    # it rather than let a whole recording run against it.
    spec.validate()
    ctx: dict[str, Any] = {"spec": spec}
    if any(w.skip_reason is None for w in works):
        ctx["anatomy"] = _load_anatomy(cfg)
        ctx["rig"] = _rig(spec)
    return ctx


# Stages whose failure is CONFINED to one bout-fly. A pooled fit
# (`fit_fly_constants`, `fit_run_floor`) or a session stage is deliberately NOT
# here: those produce constants every later bout consumes, so continuing past
# one would run the rest of the campaign against a missing or half-written
# constant -- the failure would spread instead of being contained.
_ISOLATED_SCOPES = ("bout", "bout_fly")


def _run_works(works, cfg, ctx, bout_ids, run_root) -> list[tuple[str, str, str]]:
    """Execute `works` in order, isolating per-bout-fly failures.

    A 30-bout campaign must not lose 29 good bouts to one bad one. Measured on
    Session0: the female's root keypoints are entirely absent in bout 8
    (0.000 finite), and the IK correctly REFUSES to solve it -- which, before
    this, aborted the whole run at the first such bout and left bouts 19, 26
    and 28 unattempted even though bout 28 solves cleanly.

    Only bout-scoped stages are isolated (see `_ISOLATED_SCOPES`).

    **Two outcomes, not one.** `NotFit` means the data cannot support a fit
    (Session0's bouts 8 and 19 have a female who is 0.000 finite throughout);
    anything else means the code broke. Only the second sets the exit code.
    Conflating them cost a whole campaign: 58 of 60 bout-flies solved, the run
    exited 1 for the other two, SLURM marked the array FAILED, and `afterok`
    never let `postprocess` or `collect` run at all. Both lists are printed by
    target and reason, so "finished with failures" still cannot be mistaken
    for "finished".
    """
    failures: list[tuple[str, str, str]] = []
    not_fit: list[tuple[str, str, str]] = []
    for w in works:
        if w.skip_reason:
            print(f"skip  {w.stage:<20} {w.target:<24} ({w.skip_reason})")
            continue
        print(f"run   {w.stage:<20} {w.target}")
        scope = stage_by_name(w.stage).scope
        try:
            _execute(w, cfg, ctx, bout_ids)
        except Exception as exc:  # noqa: BLE001 -- reported, not swallowed
            if scope not in _ISOLATED_SCOPES:
                raise
            bucket, label = (not_fit, "NOFIT") if isinstance(exc, NotFit) else (failures, "FAIL ")
            print(f"{label} {w.stage:<20} {w.target:<24} {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            bucket.append((w.target, w.stage, f"{type(exc).__name__}: {exc}"))
    return failures, not_fit


def main(cfg: DictConfig) -> int:
    """Resolve `cfg` into a plan and either print it or run it. Returns an exit code."""
    stage_names = list(cfg.stages)
    bout_ids = list(cfg.bout_ids) if cfg.get("bout_ids") is not None else None

    if bool(cfg.dry_run):
        works = P.resolve(cfg, stages=stage_names, bout_ids=bout_ids)
        _print_plan(cfg, works)
        return 0

    run_root = Path(cfg.run.root)
    _claim_run_root(run_root, allow_foreign=bool(cfg.run.get("allow_foreign_root", False)))

    # Derived from the registry, never a hand-kept list: a bout-scoped stage
    # added to STAGES but forgotten here would run with an EMPTY bout list and
    # report success having done nothing.
    needs_bouts = any(stage_by_name(s).scope in ("bout", "bout_fly") for s in stage_names)
    produces_bouts = [s for s in ("bouts", "gates") if s in stage_names]

    # `bouts.csv` may not exist YET. When the same invocation both produces it
    # (`bouts`/`gates`) and consumes it, discovery has to happen BETWEEN the
    # two -- not once, up front, as it did until 2026-09-14. The one-shot
    # command this entry point exists for,
    #   stages=[bouts,fine,preprocess,ik,postprocess,collect]
    # crashed on a fresh run root for exactly this reason: the stage that
    # writes the file had not run when the file was read.
    #
    # The producer is run in its own pass first; `resume` then makes it SKIP in
    # the main pass, so it is scheduled once and executed once.
    if bout_ids is None and needs_bouts and produces_bouts:
        pre = P.resolve(cfg, stages=produces_bouts, bout_ids=[])
        _run_works(pre, cfg, _context(cfg, pre), [], run_root)

    if bout_ids is None and needs_bouts:
        bout_ids = _discover_bout_ids(run_root)
    bout_ids = bout_ids or []

    works = P.resolve(cfg, stages=stage_names, bout_ids=bout_ids)

    # Real execution only -- NEVER under `dry_run`, which must work without
    # the recording present at all (that is how the dry-run test proves
    # nothing ran). `validate()` checks `session_dir`/`calib_dir` exist and
    # enforces the `cameras` list against the calibration glob order; a
    # permuted rig plots one camera's keypoints on another camera's image
    # while still looking almost plausible, so this must raise here, before
    # any stage reads it, rather than let the whole recording run against it.
    failures, not_fit = _run_works(works, cfg, _context(cfg, works), bout_ids, run_root)
    if not_fit:
        print(f"\n{len(not_fit)} bout-fly item(s) NOT FIT (no usable data):", file=sys.stderr)
        for target, stage, err in not_fit:
            print(f"  {stage:<12} {target:<18} {err}", file=sys.stderr)
    if failures:
        print(f"\n{len(failures)} bout-fly item(s) FAILED and were skipped:", file=sys.stderr)
        for target, stage, err in failures:
            print(f"  {stage:<12} {target:<18} {err}", file=sys.stderr)
        return 1
    return 0


@hydra.main(config_path="../../configs", config_name="pipeline", version_base=None)
def _hydra_entry(cfg: DictConfig) -> None:
    sys.exit(main(cfg))


if __name__ == "__main__":
    _hydra_entry()
