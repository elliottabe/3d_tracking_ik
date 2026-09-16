"""The SLURM job graph, built as DATA, so its shape is a unit test."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from tracking.pipeline.stages import STAGES, Stage, ordered, stage_by_name

__all__ = ["Job", "build_recording_chain", "build_session_graph"]


@dataclass(frozen=True)
class Job:
    """One `sbatch` submission, described but not yet sent anywhere."""

    name: str
    stage: str
    recording: str
    array: str | None
    depends_on: tuple[str, ...]
    command: tuple[str, ...]


_DEFAULT_RECORDING_STAGES: tuple[str, ...] = (
    "coarse",
    "gates",
    "fine",
    "kpvideo",
    "preprocess",
    "ik",
    "postprocess",
    "sidebyside",
)


def _with_pooled_fits(stage_names: Sequence[str]) -> tuple[str, ...]:
    """`stage_names`, plus the pooled fits `ik`/`postprocess` need, in order."""
    names = set(ordered(stage_names))
    if {"ik", "postprocess"} & names:
        names.add("fit_fly_constants")
    if "postprocess" in names:
        names.add("fit_run_floor")
    return ordered(names)


def _artifact_producers() -> dict[str, tuple[str, ...]]:
    """Every artifact filename in the registry -> EVERY stage that writes it."""
    producers: dict[str, list[str]] = {}
    for stage in STAGES:
        for artifact in stage.writes:
            producers.setdefault(artifact, []).append(stage.name)
    return {artifact: tuple(names) for artifact, names in producers.items()}


_PRODUCERS_OF = _artifact_producers()


def _depends_on(stage: Stage, *, present: set[str], job_name_of: dict[str, str]) -> tuple[str, ...]:
    """Job names `stage` must wait on, derived from `reads` against `writes`."""
    deps = set()
    for artifact in stage.reads:
        for producer in _PRODUCERS_OF.get(artifact, ()):
            if producer != stage.name and producer in present:
                deps.add(job_name_of[producer])
    return tuple(sorted(deps))


def _command(
    stage: str,
    *,
    recording: str,
    run_root: str,
    bout_ids: Sequence[int] | None,
    run_name: str | None = None,
) -> tuple[str, ...]:
    """The `python -m tracking.run` argv one job body runs."""
    argv = [sys.executable, "-m", "tracking.run"]
    if recording != "*":
        group, _, rec_id = recording.partition(":")
        argv.append(f"recording={group}")
        if rec_id:
            argv.append(f"recording.id='{rec_id}'")
    argv += [f"run.root={run_root}", f"stages=[{stage}]"]
    if run_name is not None:
        argv.append(f"run.name={run_name}")
    if bout_ids:
        argv.append(f"bout_ids=[{','.join(str(b) for b in bout_ids)}]")
    return tuple(argv)


def build_recording_chain(
    recording: str,
    *,
    bout_ids: Sequence[int],
    run_root: str,
    stages: Sequence[str] | None = None,
    run_name: str | None = None,
) -> list[Job]:
    """The full job chain for ONE recording, in registry (dependency) order."""
    requested = _DEFAULT_RECORDING_STAGES if stages is None else stages
    if "collect" in requested:
        raise ValueError(
            "'collect' cannot be requested via stages=(...); build_session_graph "
            "adds one collect job per recording itself, after that recording's "
            "chain. Naming it here builds a SECOND collect job with the same "
            "name and the same run root, and the two race to write one "
            "recording's scorecard and combined h5 (task 19 defect 2). "
            "Request the recording stages you want without it, e.g. "
            "stages=[coarse,gates,fine,kpvideo,preprocess,ik,postprocess]."
        )
    stage_names = _with_pooled_fits(requested)
    present = set(stage_names)
    job_name_of = {name: f"{name}:{recording}" for name in stage_names}
    bout_ids = list(bout_ids)

    jobs = []
    for name in stage_names:
        stage = stage_by_name(name)
        array = (
            f"0-{len(bout_ids) - 1}" if stage.scope in ("bout", "bout_fly") and bout_ids else None
        )
        jobs.append(
            Job(
                name=job_name_of[name],
                stage=name,
                recording=recording,
                array=array,
                depends_on=_depends_on(stage, present=present, job_name_of=job_name_of),
                command=_command(
                    name,
                    recording=recording,
                    run_root=run_root,
                    bout_ids=bout_ids,
                    run_name=run_name,
                ),
            )
        )
    return jobs


def _collect_gate(chain: Sequence[Job]) -> str | None:
    """The job in `chain` that `collect` must wait for, or `None` for no collect."""
    for job in chain:
        if job.stage == "postprocess":
            return job.name
    return None


def build_session_graph(
    recordings: Sequence[str],
    *,
    run_roots: Mapping[str, str] | str,
    bout_ids: Mapping[str, Sequence[int]] | Sequence[int],
    stages: Sequence[str] | None = None,
    run_name: str | None = None,
) -> list[Job]:
    """Every recording's chain, plus ONE `collect` job gating all of them."""
    if isinstance(run_roots, str):
        if len(recordings) > 1:
            raise ValueError(
                f"run_roots must be a {{recording: root}} mapping for "
                f"{len(recordings)} recordings -- a single string would give "
                "every recording the SAME run root, silently merging their "
                "bouts (task 19 defect 1)"
            )
        run_roots = {recordings[0]: run_roots} if recordings else {}

    bout_ids_of: Mapping[str, Sequence[int]] = (
        bout_ids if isinstance(bout_ids, Mapping) else {r: bout_ids for r in recordings}
    )

    jobs: list[Job] = []
    sinks: list[str] = []
    for recording in recordings:
        chain = build_recording_chain(
            recording,
            bout_ids=bout_ids_of[recording],
            run_root=run_roots[recording],
            stages=stages,
            run_name=run_name,
        )
        jobs.extend(chain)
        sinks.append(_collect_gate(chain))

    for recording, sink in zip(recordings, sinks, strict=True):
        if sink is None:
            continue
        jobs.append(
            Job(
                name=f"collect:{recording}",
                stage="collect",
                recording=recording,
                array=None,
                depends_on=(sink,),
                command=_command(
                    "collect",
                    recording=recording,
                    run_root=run_roots[recording],
                    bout_ids=None,
                    run_name=run_name,
                ),
            )
        )
    return jobs
