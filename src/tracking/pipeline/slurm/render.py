"""One `graph.Job` -> the `sbatch` invocation that would submit it."""

from __future__ import annotations

import shlex
from collections.abc import Mapping, Sequence

from tracking.pipeline.slurm.graph import Job

__all__ = ["render_sbatch", "dry_run_ids"]


def dry_run_ids(jobs: Sequence[Job]) -> dict[str, str]:
    """`{job.name: "<name_JOBID>"}` for every job -- a dry run's `dep_ids`."""
    return {job.name: f"<{job.name}_JOBID>" for job in jobs}


def render_sbatch(job: Job, *, dep_ids: Mapping[str, str]) -> list[str]:
    """`job` -> the `sbatch` argv that submits it (nothing is executed here)."""
    argv = ["sbatch", "--parsable", f"--job-name={job.name}"]
    if job.array is not None:
        argv.append(f"--array={job.array}")
    if job.depends_on:
        try:
            ids = [dep_ids[name] for name in job.depends_on]
        except KeyError as exc:
            raise KeyError(
                f"{job.name} depends on {exc.args[0]!r}, which has no id in dep_ids -- "
                f"it must be submitted (or dry-run placeholder-ed) before {job.name}"
            ) from None
        argv.append("--dependency=afterok:" + ":".join(ids))
    argv.append("--wrap=" + shlex.join(job.command))
    return argv
