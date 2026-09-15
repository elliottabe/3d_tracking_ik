#!/bin/bash
# Submit a multi-recording campaign of `python -m tracking.run` as one SLURM
# job graph: every recording gets its own
#
#   coarse -> gates -> fine[array] -> preprocess[array] -> BARRIER
#       -> fit_fly_constants + fit_run_floor -> BARRIER
#       -> ik[array], postprocess[array]
#
# (`kpvideo[array]` also runs off `fine`, independent of the barrier), and
# ONE session-level `collect` job gates on every recording's last stage.
#
# This script is a THIN wrapper: `tracking.pipeline.slurm.graph` builds the
# whole dependency graph as data (see that module's docstring for why -- the
# short version: `--dependency=afterok:` strings are unverifiable except by
# submitting and waiting, so the graph SHAPE lives in a unit-tested module,
# not here) and `tracking.pipeline.slurm.render` turns one `Job` into an
# `sbatch` argv. This script's OWN job is everything that render.py
# deliberately does not know: which SLURM partition/account/resources to
# request, the GPU environment setup + backend assertion every job body
# needs, the local-worker concurrency cap, and the training-process guard --
# see `tracking/pipeline/slurm/render.py`'s module docstring for why that
# split exists.
#
# Usage:
#   scripts/slurm/session_pipeline.sh --recordings <r1,r2,...> \
#       [--bout-ids <b1,b2,...>] [--run-root PATH] [--run-name NAME] \
#       [--stages s1,s2,...] [--slurm gpu_l40s|ckpt_all] \
#       [--local-gpus N [--max-local-gpus N]] [--guard-pattern PATTERN] \
#       [--dry-run]
#
#   scripts/slurm/session_pipeline.sh --dry-run --recordings session0 --bout-ids 28
#   scripts/slurm/session_pipeline.sh --dry-run --recordings 'session1:*'
#
# A `--recordings` token is a config-group name (`session0`), a group plus
# an explicit recording id (`session1:2026_04_02_12_11_50`), or a group plus
# `*` to GLOB every recording under that group's session dir that has a
# `courtship_bout_summary.csv` (Task 20, `tracking.pipeline.slurm.discover`)
# -- one whose summary is missing or empty is SKIPPED with a named line on
# stderr, never silently, since "N chains submitted" vs. "M recordings
# exist" is exactly the discrepancy an operator needs told. `--bout-ids` is
# then optional: omitted, each discovered recording's ids come from its OWN
# `bouts_csv` (Session1's runnable recordings range 2..30 bouts, so one
# shared list would size one recording's `--array=` from another's count);
# given, it applies to exactly one recording and is refused for more.
#
# `--dry-run` prints every `sbatch` invocation this campaign would run, with
# dependency ids as `<name_JOBID>` placeholders (nothing has a real id yet --
# the source repo's `mvq_session_pipeline.sh` uses the same convention), and
# submits NOTHING.
#
# Exit status: non-zero if ANY recording's chain failed to submit -- the
# other recordings are still attempted (a queue slot is worth more than
# symmetry) and `collect` then gates only on the recordings that actually
# submitted, never on a job id that will never exist.
#
# Each recording's run root is resolved from `configs/pipeline.yaml` (the
# SAME composition a local `python -m tracking.run` resolves), one root per
# recording -- never a single hardcoded/shared directory. A shared root used
# to merge every recording's `bouts/bout_00001/` into the same path (task
# 19 defect 1); `--run-root PATH` still overrides it, but only for a single
# recording, since one path cannot be right for more than one.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

RECORDINGS=""
BOUT_IDS=""
RUN_NAME=pose_v2
RUN_ROOT=""
STAGES=""
SLURM_CFG=gpu_l40s
LOCAL_GPUS=0
MAX_LOCAL_GPUS=4
# The training run this port's own campaigns have collided with before
# (project MEMORY "Compute-node-only heavy work" / "GPU-node parallel
# limits"). Overridable because the guarded process's name changes between
# training campaigns -- a stale default here would either guard nothing
# (silently unsafe) or guard everything (permanently refuses to start).
GUARD_PATTERN='tracking.run'
DRY=0

while [ $# -gt 0 ]; do
    case "$1" in
        --recordings)      RECORDINGS="$2"; shift 2 ;;
        --bout-ids)        BOUT_IDS="$2"; shift 2 ;;
        --run-name)        RUN_NAME="$2"; shift 2 ;;
        --run-root)        RUN_ROOT="$2"; shift 2 ;;
        --stages)          STAGES="$2"; shift 2 ;;
        --slurm)           SLURM_CFG="$2"; shift 2 ;;
        --local-gpus)      LOCAL_GPUS="$2"; shift 2 ;;
        --max-local-gpus)  MAX_LOCAL_GPUS="$2"; shift 2 ;;
        --guard-pattern)   GUARD_PATTERN="$2"; shift 2 ;;
        --dry-run)         DRY=1; shift ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

[ -n "$RECORDINGS" ] || {
    echo "usage: $0 --recordings <r1,r2,...> [--bout-ids <b1,b2,...>]" >&2
    echo "          [--run-root path] [--run-name name] [--stages s1,s2,...]" >&2
    echo "          [--slurm gpu_l40s|ckpt_all] [--local-gpus N [--max-local-gpus N]]" >&2
    echo "          [--guard-pattern pattern] [--dry-run]" >&2
    echo "  recording tokens: session0 | session1:<id> | session1:*  (glob," >&2
    echo "  Task 20 -- every recording under the group's session dir with a" >&2
    echo "  courtship_bout_summary.csv). --bout-ids is required only when it" >&2
    echo "  is not given: omitted, each discovered recording's ids come from" >&2
    echo "  its own bouts_csv; given, it is refused for more than one" >&2
    echo "  recording (one list cannot size every recording's --array=)." >&2
    exit 2
}

# --------------------------------------------------------------------------
# `--run-root` overrides ONE recording's root. `build_session_graph` (task
# 19 defect 1) resolves a run root PER RECORDING from the Hydra config --
# `${paths.base_dir}/${recording.assay}/${recording.name}/${run.name}` --
# because a single shared root merges every recording's `bouts/bout_00001/`
# into the same path, one recording silently overwriting another. Accepting
# `--run-root` for N>1 recordings would apply that one path to all of them
# and reintroduce exactly that defect, so it is refused here rather than
# applied to the first recording or quietly ignored.
# --------------------------------------------------------------------------
N_RECORDINGS="$(tr ',' '\n' <<< "$RECORDINGS" | grep -c .)"
if [ -n "$RUN_ROOT" ] && [ "$N_RECORDINGS" -gt 1 ]; then
    echo "refusing --run-root with $N_RECORDINGS recordings: one path cannot be" >&2
    echo "right for more than one recording. Omit --run-root and let each" >&2
    echo "recording's run root resolve from the Hydra config, or submit" >&2
    echo "recordings one at a time, each with its own --run-root." >&2
    exit 2
fi

SLURM_YAML="$REPO/configs/slurm/${SLURM_CFG}.yaml"
[ -f "$SLURM_YAML" ] || { echo "no such slurm config: $SLURM_YAML" >&2; exit 2; }

# --------------------------------------------------------------------------
# VERIFIED FACT 2: the concurrency cap is a REFUSAL, not a clamp.
#
# 8 concurrent JAX pipelines took a 128 GB-cgroup node to CUDA_ERROR_UNKNOWN
# (measured 2026-09-13, and the same number project MEMORY "GPU-node
# parallel limits" already carries). Silently clamping --local-gpus to the
# cap would hide an operator mistake -- "I meant 8" -- and reintroduce that
# crash the next time this script runs on an identically-sized node. This
# check runs BEFORE the --dry-run branch below: a dry run that silently
# accepted an over-cap request would never warn the operator before the
# real (non-dry) invocation actually burns the node.
# --------------------------------------------------------------------------
if [ "$LOCAL_GPUS" -gt "$MAX_LOCAL_GPUS" ]; then
    echo "refusing --local-gpus $LOCAL_GPUS > $MAX_LOCAL_GPUS: 8 concurrent JAX" >&2
    echo "pipelines took a 128 GB-cgroup node to CUDA_ERROR_UNKNOWN (measured" >&2
    echo "2026-09-13). Raise --max-local-gpus deliberately if this node is bigger." >&2
    exit 2
fi

# --------------------------------------------------------------------------
# VERIFIED FACT 3: refuse to start local GPU workers while the user's own
# training run is alive on this node -- trampling it costs far more than a
# queued job waiting its turn.
#
# `pgrep -f "$GUARD_PATTERN"` self-matches: the pattern text is typically
# also present in THIS script's own argv (and in the `bash`/interactive
# shell that launched it), so a naive guard finds itself and its own
# ancestors and refuses forever. The fix (ported from the source repo's
# `mvq_local_lift.sh::mvq_guard_pids`, which hit exactly this bug measured
# 2026-09-04): walk this process's OWN ancestor chain first, then exclude
# any `pgrep` hit whose pid is in that chain, keeping only survivors whose
# command actually looks like a python training process.
# --------------------------------------------------------------------------
_guard_pids() {   # $1 = pattern -> live python pids matching it, excluding US
    local pattern="$1" self_tree=" " p=$$ pid
    while [ -n "$p" ] && [ "$p" -gt 1 ] 2>/dev/null; do
        self_tree="$self_tree$p "
        p="$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')"
    done
    for pid in $(pgrep -f "$pattern" 2>/dev/null); do
        case "$self_tree" in *" $pid "*) continue ;; esac
        case "$(ps -o comm= -p "$pid" 2>/dev/null)" in *python*) echo "$pid" ;; esac
    done
}

if [ "$LOCAL_GPUS" -gt 0 ] && [ "$DRY" -eq 0 ]; then
    HITS="$(_guard_pids "$GUARD_PATTERN")"
    if [ -n "$HITS" ]; then
        echo "refusing to start local GPU work: python process(es) $HITS match" >&2
        echo "  $GUARD_PATTERN" >&2
        echo "and are still alive (this node's GPUs may be held by the user's training run)." >&2
        exit 3
    fi
fi

echo "Recordings : $RECORDINGS"
echo "Bout ids   : ${BOUT_IDS:-(auto, per recording -- read from its own bouts_csv)}"
echo "Slurm cfg  : $SLURM_CFG  ($SLURM_YAML)"
# "Run root" is printed by the Python step below, once it has actually
# resolved each recording's root from the Hydra config -- restating it here
# from $RUN_ROOT/$RUN_NAME would be the exact second construction task 19
# defect 1 removed.
echo

cd "$REPO"

# --------------------------------------------------------------------------
# Build the graph (tracking.pipeline.slurm.graph) and either print every `sbatch`
# invocation (--dry-run) or actually submit them in order, feeding each
# job's real id into the next job's --dependency= as it becomes available.
#
# The graph's own job order is already topological (`build_session_graph`
# appends each recording's chain -- itself in registry/dependency order --
# then appends `collect` last), so this loop never needs its own
# topological sort: by the time job N is reached, everything job N could
# possibly depend on has already been submitted (or has already failed and
# been recorded in `failed_recordings`).
#
# VERIFIED FACT 1: every GPU job body carries BOTH `module load cuda` (this
# cluster's convention; harmless but insufficient alone -- it still leaves
# JAX on cpu, `Unable to load cuSPARSE`) AND the env's bundled CUDA wheels
# on LD_LIBRARY_PATH (the load-bearing part, measured on g3120: wheels alone
# already gives 8 gpu devices), THEN asserts the backend before running the
# real command. A silent CPU fallback does not crash -- it writes a
# complete, wrong `timing.json` and every other artifact looks normal
# (Phase C1 lost the GPU on a real run and it was caught only by an
# unrelated failure downstream), so this assertion is not optional and is
# applied to every stage except `collect` (pure-Python JSON aggregation,
# no JAX import at all -- wrapping it in a GPU assertion would refuse to
# run it on the very CPU-only partition it belongs on).
# --------------------------------------------------------------------------
python3 - "$RECORDINGS" "$BOUT_IDS" "$RUN_ROOT" "$STAGES" "$DRY" "$SLURM_YAML" "$RUN_NAME" "$REPO" <<'PYEOF'
import dataclasses
import shlex
import subprocess
import sys
from pathlib import Path

import yaml
from hydra import compose, initialize_config_dir

import tracking.utils.path_utils  # noqa: F401 -- registers ${repo_root:} etc. before compose
from tracking.pipeline.slurm.discover import expand_recordings, resolve_bout_ids
from tracking.pipeline.slurm.graph import build_session_graph
from tracking.pipeline.slurm.render import dry_run_ids, render_sbatch

(
    recordings_arg,
    bout_ids_arg,
    run_root_override,
    stages_arg,
    dry_arg,
    slurm_yaml_path,
    run_name,
    repo,
) = sys.argv[1:9]
_raw_tokens = [r for r in recordings_arg.split(",") if r]
explicit_bout_ids = [int(b) for b in bout_ids_arg.split(",") if b] if bout_ids_arg else None
stages = [s for s in stages_arg.split(",") if s] or None
dry = dry_arg == "1"

with open(slurm_yaml_path) as f:
    slurm_cfg = yaml.safe_load(f)


def _recording_overrides(recording: str) -> list[str]:
    """`recording={group}` (+ `recording.id='{id}'` for a `group:id` token).

    The one place both `_resolve_run_root` and `_bouts_csv_and_tag` below
    turn a Task 20 recording token into Hydra overrides -- kept as a single
    function so the two never drift into building this differently (see
    `graph.py::_command`'s identical split, which a real submitted job's
    argv must also match byte-for-byte, or a dry run's printed `recording=`
    would say one thing while `run.root=`/`bout_ids=[...]` said another).

    `recording.id` is quoted for the same reason `_command` quotes it
    (measured 2026-09-14): unquoted, Hydra's override grammar reads an
    all-digits-and-underscores value as a Python int literal and silently
    drops every underscore (`2026_04_02_11_52_43` -> id `20260402115243`),
    which then 404s reading that (nonexistent) recording's `bouts_csv`.
    """
    group, _, rec_id = recording.partition(":")
    overrides = [f"recording={group}"]
    if rec_id:
        overrides.append(f"recording.id='{rec_id}'")
    return overrides


def _compose(recording: str, *extra: str):
    with initialize_config_dir(config_dir=str(Path(repo) / "configs"), version_base=None):
        return compose(config_name="pipeline", overrides=[*_recording_overrides(recording), *extra])


def _resolve_run_root(recording: str) -> str:
    """`recording`'s run root, resolved exactly like a local run resolves it.

    Composes `configs/pipeline.yaml` (task 19 defect 1) instead of
    rebuilding `<base_dir>/<assay>/<recording.name>/<run.name>` here in
    Python: a second construction is a second thing that can disagree with
    `pipeline.yaml`'s own `run.root` interpolation, which is exactly how a
    stale hardcoded directory reached `sbatch` in the first place.
    """
    return _compose(recording, f"run.name={run_name}").run.root


def _session_dir_parent(group: str) -> Path:
    """`group`'s session dir, one level up -- what a `group:*` token globs.

    Composes the group's config with its DEFAULT id rather than rebuilding
    `<data_dir>/Video_recordings/.../<group>` in shell: `session_dir` is
    `configs/recording/<group>.yaml`'s own interpolation, and a second,
    independent construction of that path is exactly task 19 defect 1 one
    level up the directory tree.
    """
    return Path(_compose(group).recording.session_dir).parent


def _bouts_csv_and_tag(recording: str) -> tuple[Path, str]:
    """`(bouts_csv, session_tag)` for `read_bout_summary`, per Task 20."""
    cfg = _compose(recording)
    return Path(cfg.recording.bouts_csv), cfg.recording.name


recordings, _skip_lines = expand_recordings(_raw_tokens, session_dir_parent_of=_session_dir_parent)
for line in _skip_lines:
    print(line, file=sys.stderr)
if not recordings:
    print(f"refusing: no recording in {_raw_tokens!r} has a usable bout summary", file=sys.stderr)
    sys.exit(2)

# The bash-level `--run-root` refusal (above) counts RAW comma-separated
# tokens, before a `group:*` token is expanded here -- a single token can
# still expand to N>1 recordings, which that count never sees. Re-checked
# here, after expansion, against the actual chain count this run root would
# be applied to.
if run_root_override and len(recordings) > 1:
    print(
        f"refusing --run-root with {len(recordings)} recordings (expanded from "
        f"{_raw_tokens!r}): one path cannot be right for more than one recording.",
        file=sys.stderr,
    )
    sys.exit(2)

try:
    bout_ids = resolve_bout_ids(
        recordings, explicit=explicit_bout_ids, bouts_csv_and_tag_of=_bouts_csv_and_tag
    )
except ValueError as exc:
    print(f"refusing: {exc}", file=sys.stderr)
    sys.exit(2)

# `--run-root` (already refused above for >1 recordings) overrides the one
# recording it applies to; otherwise every recording's root comes from the
# Hydra config, per recording -- never one shared value (see
# `build_session_graph`'s docstring for why that merges recordings' bouts).
run_roots = (
    {recordings[0]: run_root_override}
    if run_root_override
    else {r: _resolve_run_root(r) for r in recordings}
)
if len(recordings) == 1:
    print(f"Run root   : {run_roots[recordings[0]]}")
else:
    for r in recordings:
        print(f"Run root   : {r} -> {run_roots[r]}")

# VERIFIED FACT 1's exact job body (this task's brief, "VERIFIED FACTS"
# section) -- `module load cuda` alone measurably leaves JAX on cpu on this
# cluster (`Unable to load cuSPARSE`); the env's own bundled wheels on
# LD_LIBRARY_PATH are what actually put 8 devices on jax.devices(). Both are
# kept: `module load cuda` is harmless (measured identical to wheels-alone)
# and is this cluster's convention.
_GPU_SETUP = (
    "module load cuda; "
    'SP=$(python -c "import nvidia, pathlib; print(pathlib.Path(nvidia.__file__).parent)"); '
    'export LD_LIBRARY_PATH="$(ls -d "$SP"/*/lib | tr \'\\n\' \':\')$LD_LIBRARY_PATH"; '
    "unset JAX_PLATFORMS; "
    "python -c \"import jax,sys; sys.exit(0 if jax.default_backend()=='gpu' else 1)\" "
    '|| { echo "FATAL: JAX backend is not gpu; refusing to burn a GPU allocation on CPU" >&2; exit 1; }; '
)


def _array_bout_selection(command: tuple[str, ...]) -> str:
    """`command`'s `bout_ids=[...]` element, replaced by a shell expression
    that picks ONE id via `$SLURM_ARRAY_TASK_ID` at RUN time (task 21,
    defect 1: `--array=0-29` was decorative -- every task ran the SAME
    `bout_ids=[...]` argv, i.e. all 30 bouts, 30 times over).

    `graph.py::_command`'s docstring is explicit that baking a single index
    into `Job.command` at RENDER time would break "one `Job.command` is the
    whole truth of what this stage runs" -- every array task shares ONE
    `sbatch --wrap=`, so a render-time index would be wrong for 29 of 30
    tasks. This function instead emits shell code that is IDENTICAL for
    every task (so the invariant holds) and resolves which bout to run only
    when SLURM actually sets `$SLURM_ARRAY_TASK_ID` for that task.

    The ids come from `command` itself, not re-derived: a bash array built
    from the SAME `bout_ids=[...]` string `_command` already rendered, so
    there is exactly one place a bout-id list is spelled out. Indexing is
    `_BOUT_IDS[$SLURM_ARRAY_TASK_ID]` -- the ACTUAL id list, never
    `task_id + 1` -- because Session1's recordings have non-contiguous ids
    and Session0's start at 1, so arithmetic on the index picks the wrong
    bout the moment one is skipped (measured: ids `[3, 7, 11]`, task `1`
    must resolve to bout `7`, not `2` or `11`).

    The replaced argv element is double-quoted (`"bout_ids=[$_BOUT_ID]"`),
    not single-quoted like every other (static) element `shlex.quote`
    handles below -- single quotes would suppress `$_BOUT_ID` expansion and
    hand `tracking.run` the literal text `$_BOUT_ID` as a Hydra override.
    """
    idx = next(i for i, arg in enumerate(command) if arg.startswith("bout_ids=["))
    ids_csv = command[idx].removeprefix("bout_ids=[").removesuffix("]")
    select = (
        f"_BOUT_IDS=({ids_csv.replace(',', ' ')}); "
        '_BOUT_ID="${_BOUT_IDS[$SLURM_ARRAY_TASK_ID]}"; '
    )
    rendered = [
        '"bout_ids=[$_BOUT_ID]"' if i == idx else shlex.quote(arg) for i, arg in enumerate(command)
    ]
    return select + " ".join(rendered)


def _gpu_wrapped(job):
    """`job`, with its command run through VERIFIED FACT 1's setup+assert.

    `collect` is the one stage this whole campaign runs that never imports
    JAX (`recording_stages.collect_session` reads and writes plain JSON/HDF5
    summaries) -- wrapping it in the same GPU backend assertion would refuse
    to run it on the CPU-only partition it belongs on, for a device it never
    touches.

    `job.array is not None` (`Stage.scope` in `("bout", "bout_fly")`) routes
    through `_array_bout_selection` instead of a plain `shlex.join`: task
    21 defect 1, above. A recording- or session-scoped job (`job.array is
    None`) never references `$SLURM_ARRAY_TASK_ID` at all -- there is no
    array task id for `sbatch` to set, and a stray reference there would be
    an empty-string index into a one-element list, silently "working" for
    the wrong reason.
    """
    if job.stage == "collect":
        return job
    body = _array_bout_selection(job.command) if job.array is not None else shlex.join(job.command)
    inner = _GPU_SETUP + body
    return dataclasses.replace(job, command=("bash", "-lc", inner))


try:
    # `run_name` travels with `run_roots`: both are resolved from $RUN_NAME
    # above, so `collect` cannot name its combined h5 for a different run than
    # the one whose root it reads.
    jobs = build_session_graph(
        recordings,
        run_roots=run_roots,
        bout_ids=bout_ids,
        stages=stages,
        run_name=run_name,
    )
except ValueError as exc:
    # `collect` requested via `--stages` (task 19 defect 2) or a run_roots
    # mismatch surfaces here as a plain, readable refusal -- not a
    # traceback -- matching every other early-refusal in this script.
    print(f"refusing: {exc}", file=sys.stderr)
    sys.exit(2)

# `--partition`/`--account`/... are cluster resource choices `render.py`
# deliberately does not know (see its module docstring) -- prepended here,
# read from the SAME `configs/slurm/<profile>.yaml` the operator picked with
# `--slurm`, so there is one source of truth for which queue/allocation a
# campaign lands on.
resource_flags = [
    f"--partition={slurm_cfg['partition']}",
    f"--account={slurm_cfg['account']}",
    f"--time={slurm_cfg['time']}",
    f"--cpus-per-task={slurm_cfg['cpus_per_task']}",
    f"--mem={slurm_cfg['mem']}",
]
if job_gres := slurm_cfg.get("gres"):
    resource_flags.append(f"--gres={job_gres}")
# `constraint` EXCLUDES GPUs too small for this model, which a wide
# checkpoint partition otherwise includes. Optional: a profile pinned to one
# GPU type via `gres` does not need it.
if constraint := slurm_cfg.get("constraint"):
    resource_flags.append(f"--constraint={constraint}")
# Nodes slurm still considers healthy but whose GPUs this stack cannot use.
if exclude := slurm_cfg.get("exclude"):
    resource_flags.append(f"--exclude={exclude}")
if slurm_cfg.get("requeue"):
    resource_flags.append("--requeue")
# `collect` is CPU-only aggregation (source repo precedent: its own
# session-collect job requests `--gpus=0` and none of the GPU constraint) --
# requesting a GPU node for it would hold a scarce L40S for a job that never
# touches one.
collect_resource_flags = [
    f"--partition={slurm_cfg['partition']}",
    f"--account={slurm_cfg['account']}",
    "--time=1:00:00",
    "--cpus-per-task=2",
    "--mem=16G",
    "--gpus=0",
]

# No job set `--output`/`--error` before this: `sbatch` then wrote
# `slurm-<jobid>.out` into the SUBMITTER's cwd (the repo root), one file per
# job -- a 30-task `ik` array alone leaves 30 of them, and a cancelled
# campaign already left one. An ABSOLUTE path is required: the submitter's
# cwd is not guaranteed to be the repo (this script itself `cd`s there, but
# nothing stops it being sourced or symlinked from elsewhere). `%x-%j` (job
# name, job id) covers scalar and array jobs alike -- for an array task
# `%j` is THAT TASK's own id, so array tasks never collide on one filename;
# `%x` groups a failed task's log under a name a human already knows
# (`ik:session1:2026_04_02_15_25_51`), rather than requiring a job-id lookup
# to find it. `slurm_logs/` is gitignored (as is `slurm-*.out`, the bare
# default this replaces).
_SLURM_LOGS_DIR = f"{repo}/slurm_logs"
_LOG_FLAGS = [f"--output={_SLURM_LOGS_DIR}/%x-%j.out", f"--error={_SLURM_LOGS_DIR}/%x-%j.out"]
resource_flags += _LOG_FLAGS
collect_resource_flags += _LOG_FLAGS

if dry:
    dep_ids = dry_run_ids(jobs)
    for job in jobs:
        flags = collect_resource_flags if job.stage == "collect" else resource_flags
        argv = render_sbatch(_gpu_wrapped(job), dep_ids=dep_ids)
        print("+ " + " ".join(argv[:2] + flags + argv[2:]))
    print(f"(dry-run) {len(jobs)} job(s) described above; nothing submitted")
    sys.exit(0)

# `sbatch` creates the log FILE but never the directory -- a missing
# `slurm_logs/` fails the job at launch with an error that does not
# obviously name this as the cause. Only reached past the `dry` branch
# above, so a dry run stays side-effect free (creates nothing on disk).
Path(_SLURM_LOGS_DIR).mkdir(parents=True, exist_ok=True)

dep_ids: dict[str, str] = {}
failed_recordings: set[str] = set()
any_failed = False

for job in jobs:
    if job.stage == "collect":
        # Gate only on the recordings that actually got a job id -- a
        # dependency naming a job that will NEVER be submitted is accepted
        # by `sbatch` without complaint and then waits forever (the same
        # failure mode `test_every_dependency_names_a_job_that_exists`
        # guards against at the graph level).
        present = tuple(d for d in job.depends_on if d in dep_ids)
        missing = [d for d in job.depends_on if d not in dep_ids]
        if missing:
            print(f"collect: gating only on {present} -- {missing} failed upstream", file=sys.stderr)
            any_failed = True
        job = dataclasses.replace(job, depends_on=present)
        if not present:
            print("collect: skipped -- no recording chain was submitted", file=sys.stderr)
            continue
    elif job.recording in failed_recordings:
        print(f"skip {job.name}: recording {job.recording} failed upstream", file=sys.stderr)
        continue

    flags = collect_resource_flags if job.stage == "collect" else resource_flags
    argv = render_sbatch(_gpu_wrapped(job), dep_ids=dep_ids)
    argv = argv[:2] + flags + argv[2:]
    print("+ " + " ".join(argv))
    try:
        out = subprocess.run(argv, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: sbatch failed for {job.name} (exit {exc.returncode}): {exc.stderr}", file=sys.stderr)
        failed_recordings.add(job.recording)
        any_failed = True
        continue
    job_id = out.stdout.strip().split(";")[0]
    dep_ids[job.name] = job_id
    print(f"Submitted {job.stage}: {job_id}  ({job.name})")

print(f"submitted {len(dep_ids)}/{len(jobs)} job(s)")
print("Monitor : squeue -u $USER")
sys.exit(1 if any_failed else 0)
PYEOF
