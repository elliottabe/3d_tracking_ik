"""`python -m tracking.run` -- the Hydra entry point."""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import hydra  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

import tracking.utils.path_utils  # noqa: F401,E402 -- registers ${repo_root:} before compose
from tracking.conventions import NotFit  # noqa: E402
from tracking.pipeline import plan as P  # noqa: E402
from tracking.pipeline import stages as S  # noqa: E402
from tracking.pipeline.stages import stage_by_name  # noqa: E402
from tracking.pipeline.timing import timed  # noqa: E402

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
    """`inverse_kinematics.anatomy.load_anatomy`, fed a DETACHED anatomy cfg."""
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
    """A config subtree as a PLAIN `dict`/`list`/scalar, resolved."""
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _git_sha() -> str:
    """This repo's HEAD commit, or `"unknown"` if it cannot be read."""
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

        runner = MVQRunner(
            str(cfg.mvq.checkpoint),
            step=cfg.mvq.step,
            attn_impl=cfg.mvq.attn_impl,
            rig=rig,
            kp_order=anatomy.kp_order,
            batch=int(cfg.mvq.batch),
            exist_thresh=float(cfg.mvq.exist_thresh),
            window_pref=str(cfg.mvq.window_pref),
        )
        detector = CenterDetector(
            str(cfg.centerdetect.checkpoint), min_score=float(cfg.centerdetect.min_score)
        )

        def _reader_factory(bout_spec):
            return SlotReader(
                spec.session_dir, rig.cameras, start_slot=bout_spec.start_frame, stride=1
            )

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

        with timed(timing_path, stage, n_items=len(bout_ids)):
            collect_session(
                run_root,
                anatomy=anatomy,
                source_hz=float(_plain(cfg.recording.fps)),
                git_sha=_git_sha(),
                run_name=_plain(cfg.run.name),
                fly_id=_plain(cfg.recording.name),
                bucket=_plain(cfg.recording.assay),
            )
        return

    # pragma: no cover -- plan.resolve() already validated `stage`
    raise AssertionError(f"unhandled stage {stage!r}")


RUN_ROOT_MARKER = ".tracking_run"


def _claim_run_root(run_root: Path, *, allow_foreign: bool) -> None:
    """Create `run_root`, refusing a directory this pipeline did not create."""
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
    spec.validate()
    ctx: dict[str, Any] = {"spec": spec}
    if any(w.skip_reason is None for w in works):
        ctx["anatomy"] = _load_anatomy(cfg)
        ctx["rig"] = _rig(spec)
    return ctx


_ISOLATED_SCOPES = ("bout", "bout_fly")


def _run_works(works, cfg, ctx, bout_ids, run_root) -> list[tuple[str, str, str]]:
    """Execute `works` in order, isolating per-bout-fly failures."""
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

    needs_bouts = any(stage_by_name(s).scope in ("bout", "bout_fly") for s in stage_names)
    produces_bouts = [s for s in ("bouts", "gates") if s in stage_names]

    if bout_ids is None and needs_bouts and produces_bouts:
        pre = P.resolve(cfg, stages=produces_bouts, bout_ids=[])
        _run_works(pre, cfg, _context(cfg, pre), [], run_root)

    if bout_ids is None and needs_bouts:
        bout_ids = _discover_bout_ids(run_root)
    bout_ids = bout_ids or []

    works = P.resolve(cfg, stages=stage_names, bout_ids=bout_ids)

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
