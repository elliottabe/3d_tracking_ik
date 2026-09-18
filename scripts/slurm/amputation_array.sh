#!/bin/bash
# Submit the amputation cohort as ONE SLURM array: one task per recording,
# each running the whole chain end to end.
#
#   scripts/slurm/amputation_array.sh --run-name ik_v1 [--dry-run]
#
# Per-recording dependency chains (scripts/slurm/session_pipeline.sh's shape)
# were rejected for this cohort: bouts average ~500 frames, so 103 chains of
# ~6 jobs each is scheduler churn for work that fits in one task.
#
# THE MANIFEST IS FROZEN AT SUBMIT TIME. If each task re-globbed the data
# root, a directory appearing or being removed mid-campaign would shift every
# index after it and tasks would silently process the wrong recording, or one
# twice. The glob happens once, here; each task reads its own line.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DATA_ROOT="/gscratch/portia/eabe/data/Johnson_lab/processed/amputation"
MANIFEST_DIR="$REPO/slurm_logs"
RUN_NAME=""
SLURM_CFG=gpu_l40s
CONCURRENCY=10
RECORDINGS=""
DRY=0

while [ $# -gt 0 ]; do
    case "$1" in
        --run-name)      RUN_NAME="$2"; shift 2 ;;
        --data-root)     DATA_ROOT="$2"; shift 2 ;;
        --manifest-dir)  MANIFEST_DIR="$2"; shift 2 ;;
        --slurm)         SLURM_CFG="$2"; shift 2 ;;
        --concurrency)   CONCURRENCY="$2"; shift 2 ;;
        --recordings)    RECORDINGS="$2"; shift 2 ;;
        --dry-run)       DRY=1; shift ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

[ -n "$RUN_NAME" ] || {
    echo "usage: $0 --run-name NAME [--data-root DIR] [--manifest-dir DIR]" >&2
    echo "          [--slurm gpu_l40s|ckpt_all] [--concurrency N]" >&2
    echo "          [--recordings id1,id2,...] [--dry-run]" >&2
    echo "  --run-name is REQUIRED: the pipeline's own default is 'debug'," >&2
    echo "  which must never name a real campaign's outputs." >&2
    exit 2
}

# A missing data root is a different operator error from an empty one, and the
# generic "no usable recording" refusal below would hide which it was.
[ -d "$DATA_ROOT" ] || {
    echo "refusing: --data-root does not exist: $DATA_ROOT" >&2
    exit 2
}

# -- Discovery. A recording is usable only if it has BOTH the 3D table and a
#    bout table with at least one data row. A skip is always named on stderr:
#    "N tasks submitted" vs "103 directories exist" is exactly the discrepancy
#    an operator needs told.
if [ -n "$RECORDINGS" ]; then
    CANDIDATES="$(tr ',' '\n' <<< "$RECORDINGS" | grep . || true)"
else
    CANDIDATES="$(find "$DATA_ROOT" -mindepth 1 -maxdepth 1 -type d -name '2026_*' \
                  -printf '%f\n' 2>/dev/null | sort || true)"
fi

USABLE=""
while IFS= read -r rec; do
    [ -n "$rec" ] || continue
    d="$DATA_ROOT/$rec"
    if [ ! -f "$d/data3D.csv" ]; then
        echo "skip $rec: no data3D.csv" >&2; continue
    fi
    if [ ! -f "$d/running_bouts_summary.csv" ]; then
        echo "skip $rec: no running_bouts_summary.csv" >&2; continue
    fi
    if [ "$(wc -l < "$d/running_bouts_summary.csv")" -lt 2 ]; then
        echo "skip $rec: running_bouts_summary.csv has no data rows" >&2; continue
    fi
    USABLE="$USABLE$rec"$'\n'
done <<< "$CANDIDATES"

USABLE="$(printf '%s' "$USABLE" | grep . || true)"
N="$(printf '%s\n' "$USABLE" | grep -c . || true)"
if [ "$N" -eq 0 ]; then
    echo "refusing: no recording under $DATA_ROOT has both a data3D.csv and a" >&2
    echo "non-empty running_bouts_summary.csv" >&2
    exit 2
fi

# -- Freeze the list. Tasks index THIS file, never a fresh glob.
mkdir -p "$MANIFEST_DIR"
# ABSOLUTE, always: the job body does `cd "$REPO"` before reading the manifest,
# so a relative --manifest-dir would resolve against $REPO on the compute node
# and silently find nothing -- defeating the one guarantee this script exists
# to provide.
MANIFEST_DIR="$(cd "$MANIFEST_DIR" && pwd)"
MANIFEST="$MANIFEST_DIR/amputation_$(date +%Y%m%d-%H%M%S).manifest"
printf '%s\n' "$USABLE" > "$MANIFEST"

echo "Data root  : $DATA_ROOT"
echo "Manifest   : $MANIFEST ($N recording(s))"
echo "Run name   : $RUN_NAME"
echo "Slurm cfg  : $SLURM_CFG"
echo

SLURM_YAML="$REPO/configs/slurm/${SLURM_CFG}.yaml"
[ -f "$SLURM_YAML" ] || { echo "no such slurm config: $SLURM_YAML" >&2; exit 2; }
_y() { grep -E "^$1:" "$SLURM_YAML" | head -1 | sed -E "s/^$1:[[:space:]]*//; s/^['\"]//; s/['\"]$//"; }

RESOURCE_FLAGS=(
    "--partition=$(_y partition)" "--account=$(_y account)"
    "--time=$(_y time)" "--cpus-per-task=$(_y cpus_per_task)" "--mem=$(_y mem)"
)
# `|| true`: under `set -euo pipefail`, each of these lines is safe today only
# because more commands follow it -- a failing `[ -n ... ]` test on the LAST
# command of the script would exit the script with status 1. That safety is
# accidental, not structural: a future reorder that moved one of these to the
# end would turn a missing/false key into a silent non-submission instead of
# just skipping the optional flag. `|| true` makes each line's exit status
# independent of position.
[ -n "$(_y gres)" ]       && RESOURCE_FLAGS+=("--gres=$(_y gres)") || true
[ -n "$(_y constraint)" ] && RESOURCE_FLAGS+=("--constraint=$(_y constraint)") || true
[ -n "$(_y exclude)" ]    && RESOURCE_FLAGS+=("--exclude=$(_y exclude)") || true
# ckpt-all is preemptible and its profile sets `requeue: true`; without this a
# preempted array task is simply lost. session_pipeline.sh does the same
# (`if slurm_cfg.get("requeue")`). Compared against "true" rather than tested
# for non-emptiness so that `requeue: false` does NOT add the flag.
[ "$(_y requeue)" = "true" ] && RESOURCE_FLAGS+=("--requeue") || true

# -- VERIFIED FACT 1, copied from session_pipeline.sh: `module load cuda` alone
#    measurably leaves JAX on cpu on this cluster; the env's own bundled wheels
#    on LD_LIBRARY_PATH are what actually put devices on jax.devices(). The
#    backend assertion is not optional -- a silent CPU fallback does not crash,
#    it writes a complete and wrong set of artifacts.
_GPU_SETUP='module load cuda; export MUJOCO_GL=egl; '
_GPU_SETUP+='SP=$(python -c "import nvidia, pathlib; print(pathlib.Path(nvidia.__file__).parent)"); '
_GPU_SETUP+='export LD_LIBRARY_PATH="$(ls -d "$SP"/*/lib | tr '"'"'\n'"'"' '"'"':'"'"')$LD_LIBRARY_PATH"; '
_GPU_SETUP+='unset JAX_PLATFORMS; '
_GPU_SETUP+='python -c "import jax,sys; sys.exit(0 if jax.default_backend()==\"gpu\" else 1)" '
_GPU_SETUP+='|| { echo "FATAL: JAX backend is not gpu; refusing to burn a GPU allocation on CPU" >&2; exit 1; }; '

# `recording.id` stays SINGLE-QUOTED inside the body: unquoted, Hydra's
# override grammar reads an all-digits-and-underscores value as a Python int
# literal and silently drops every underscore (2026_07_06_16_55_07 ->
# 20260706165507), which then 404s on a directory that never existed.
BODY="$_GPU_SETUP"
BODY+="cd $(printf '%q' "$REPO"); "
BODY+="_REC=\"\$(sed -n \"\$((SLURM_ARRAY_TASK_ID + 1))p\" $(printf '%q' "$MANIFEST"))\"; "
BODY+='[ -n "$_REC" ] || { echo "no manifest line for task $SLURM_ARRAY_TASK_ID" >&2; exit 1; }; '
BODY+="echo \"recording: \$_REC\"; "
BODY+="python -m tracking.run recording=amputation \"recording.id='\$_REC'\" "
BODY+="ik=amputation anatomy=v1 run.name=$(printf '%q' "$RUN_NAME") "
BODY+="'stages=[bouts,ingest3d,preprocess,ik,postprocess,collect]'"

FLAGS=("--job-name=amputation-$RUN_NAME"
       "--array=0-$((N - 1))%$CONCURRENCY"
       "${RESOURCE_FLAGS[@]}"
       "--output=$REPO/slurm_logs/%x-%A_%a.out"
       "--error=$REPO/slurm_logs/%x-%A_%a.out")

# The body is printed VERBATIM, not %q-escaped: an operator reading a dry run
# needs to see the `recording.id='...'` quoting exactly as the job will get it,
# and %q would rewrite those quotes into something that no longer reads as the
# thing being checked.
if [ "$DRY" -eq 1 ]; then
    printf '+ sbatch'
    printf ' %s' "${FLAGS[@]}"
    printf ' --wrap %s\n' "$BODY"
    echo "(dry-run) $N task(s) described above; nothing submitted"
    exit 0
fi

# Only past the dry branch: sbatch creates the log FILE but never its
# directory, and a dry run must leave nothing behind but its manifest.
mkdir -p "$REPO/slurm_logs"
sbatch "${FLAGS[@]}" --wrap "$BODY"
echo "Monitor : squeue -u \$USER"
