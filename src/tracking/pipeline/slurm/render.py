"""One `graph.Job` -> the `sbatch` invocation that would submit it.

Deliberately thin: everything cluster-specific (partition, account, GPU
resource request, the `module load cuda` / `LD_LIBRARY_PATH` / backend-assert
job body from VERIFIED FACT 1 in this task's brief) is the SHELL wrapper's
job (`scripts/slurm/session_pipeline.sh`), not this module's -- a job's
NAME/ARRAY/DEPENDENCY shape (what `graph.py` computes and what
`test_slurm_graph.py` checks) does not change with which partition or GPU
setup a campaign happens to use, so keeping that decision out of here means
neither this module nor its tests need to know which cluster they run on.

The one piece of cluster-shaped behaviour that DOES belong here is the
dry-run placeholder convention: a real
`--dependency=afterok:<id>` needs ids `sbatch` has not handed out yet at
graph-build time, so a caller either passes the REAL ids it collected while
submitting earlier jobs in the chain, or -- under `--dry-run`, where nothing
is ever submitted -- passes `dry_run_ids(jobs)`'s placeholders instead. Both
are the same `Mapping[str, str]` shape, so `render_sbatch` itself does not
need to know which one it was handed.
"""

from __future__ import annotations

import shlex
from collections.abc import Mapping, Sequence

from tracking.pipeline.slurm.graph import Job

__all__ = ["render_sbatch", "dry_run_ids"]


def dry_run_ids(jobs: Sequence[Job]) -> dict[str, str]:
    """`{job.name: "<name_JOBID>"}` for every job -- a dry run's `dep_ids`.

    The placeholder NAMES the job it stands in for (`<fine:rec_JOBID>`, not a
    bare `<JOBID>`), because a dry run's whole point is a human reading the
    printed chain before anything is submitted -- an unlabelled placeholder
    would force them to count `--dependency=` positions against a job list to
    find out which id is missing, exactly the kind of by-position reasoning
    CLAUDE.md's naming rule exists to rule out.
    """
    return {job.name: f"<{job.name}_JOBID>" for job in jobs}


def render_sbatch(job: Job, *, dep_ids: Mapping[str, str]) -> list[str]:
    """`job` -> the `sbatch` argv that submits it (nothing is executed here).

    `--parsable` always: it is what makes the real (non-dry-run) caller able
    to read a bare job id off stdout and feed it into `dep_ids` for the next
    job in the chain.

    `--array` is only added when `job.array` is set (`Stage.scope` in
    `("bout", "bout_fly")`, per `graph.py`) -- an unconditional
    `--array=None` would be a literal, wrong argument string, not an absent
    flag.

    `--dependency=afterok:<id1>:<id2>...` is built by looking up EVERY name
    in `job.depends_on` against `dep_ids`; a missing entry raises immediately
    naming the missing job, rather than silently rendering `sbatch` a
    dependency string missing one id -- `sbatch` accepts a malformed
    dependency spec without complaint and the job then waits forever, the
    exact failure mode `test_every_dependency_names_a_job_that_exists`
    guards against one layer up (a name that does not exist in the graph at
    all); this is the companion check for a name that exists in the graph but
    was never assigned an id because the caller submitted jobs out of order.

    The job body itself (`job.command`, plus whatever GPU setup and resource
    flags the shell wrapper adds) is passed via `--wrap`: `shlex.join` quotes
    it as ONE shell word so `sbatch` runs it with `/bin/sh -c`, rather than
    splitting `job.command`'s own spaces into separate `sbatch` arguments.
    """
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
