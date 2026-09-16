"""The SLURM job graph, built as DATA, so its shape is a unit test.

A multi-recording campaign is normally only verifiable by submitting it and
waiting -- a queue slot and up to an hour per mistake in the dependency
chain. That is exactly backwards for the one property that actually matters
here: whether `fit_fly_constants`/`fit_run_floor` sit as BARRIERS between
every `preprocess` and every `ik`/`postprocess`. Those two pooled fits are
the reason `pipeline.plan` exists at all -- pooling a fly's body
scale and marker offsets across every bout it appears in, rather than fitting
them from whichever bout happens to run first, is what fixed the measured
38x body-scale error (spec 5.1). A SLURM chain that let one `ik` array task
start before its fly's `fit_fly_constants` job finished would reintroduce
that exact defect, ONE LEVEL DOWN, where no Python-side test would ever see
it -- the bug would live entirely in `--dependency=` strings a human typed
into a submission script.

So this module never calls `sbatch`. It builds a `Job` per stage (`Job` is a
plain frozen dataclass -- nothing here holds a live subprocess or a real job
id), wires `depends_on` by asking the SAME registry `pipeline.plan` already
depends on (`tracking.pipeline.stages`) which artifact each stage reads and
writes, and returns the whole thing as a list `render.py` (or a human) can
inspect before anything is submitted. Two consequences of building it this
way:

  - The stage ORDER and which stages are array jobs are never hand-restated
    here. `stages.ordered()` sorts stage names into the registry's declared
    dependency order (already proven, by `test_stages.py`'s own
    reads/writes-consistency test, to place `fit_fly_constants`/
    `fit_run_floor` after every `preprocess` and before `ik`/`postprocess`),
    and `Stage.scope` says whether a stage is `"bout"`/`"bout_fly"` (array
    over bouts) or `"recording"`/`"session"` (one job). A second, hand-kept
    ordering or a second "which stages are arrays" list would be a second
    place this chain's shape could drift from the registry built
    specifically to prevent that.
  - `depends_on` is computed from `Stage.reads`/`Stage.writes`, not from
    stage ADJACENCY in the ordered list. Adjacency would make `ik` depend
    only on the stage immediately before it in registry order
    (`fit_run_floor`), never directly on `fit_fly_constants` two stages
    earlier -- true today only because both pooled fits happen to sit next
    to each other, and silently wrong the moment a future stage is inserted
    between them. Matching `ik`'s `reads=("kp3d_filt.npz", "offsets_fly.h5")`
    against every earlier stage's `writes` finds BOTH real producers
    (`preprocess` and `fit_fly_constants`) regardless of what sits between
    them.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from tracking.pipeline.stages import STAGES, Stage, ordered, stage_by_name

__all__ = ["Job", "build_recording_chain", "build_session_graph"]


@dataclass(frozen=True)
class Job:
    """One `sbatch` submission, described but not yet sent anywhere.

    `name` is `"<stage>:<recording>"` (the session-level `collect` job uses
    `"*"` for `recording`, spelled out in `build_session_graph`) -- prefixed
    with the stage first, not last, so a caller can find "the fine job for
    this recording" with a plain `.startswith("fine:")`, the pattern every
    test in `test_slurm_graph.py` relies on rather than parsing the name
    apart.

    `array` is `None` for a single job or `"0-<n-1>"` for an array covering
    `n` bouts (never a bare stage-index or bout-index integer -- CLAUDE.md's
    "never index a keypoint or camera axis by integer" rule applies equally
    here: an array string is meaningless without knowing which bout list it
    was sized against, so it is always built next to that list, never passed
    around as a lone number).

    `depends_on` names other `Job.name` values (never job IDS -- those do
    not exist until something calls `sbatch`, which this module never does).
    `render.render_sbatch` is what turns a name into an `afterok:<id>`
    dependency string, once submission has actually started handing out ids.
    """

    name: str
    stage: str
    recording: str
    array: str | None
    depends_on: tuple[str, ...]
    command: tuple[str, ...]


# `pipeline.yaml`'s own default stage list (spec section 6), minus
# `collect`, which `build_session_graph` adds per recording after that
# recording's chain -- building it here too would give one recording two
# collect jobs racing to write its scorecard and combined h5.
#
# Kept in sync with `configs/pipeline.yaml` BY HAND, and that is how
# `sidebyside` went missing: task 16 wired the stage end to end (registry
# row, plan expansion, run.py branch, tests) and added only `viz: default`
# to pipeline.yaml's Hydra `defaults:` block, so the stage itself joined
# neither list. It ran only when named explicitly, and a full Session0
# campaign produced 30 kpvideos and zero side-by-sides with nothing
# reporting a thing. `test_the_slurm_defaults_match_the_config` now pins
# the two lists against each other.
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
    """`stage_names`, plus the pooled fits `ik`/`postprocess` need, in order.

    Mirrors `pipeline.plan.resolve`'s own auto-insertion (see that module's
    docstring): a caller reasonably asks for `stages=[ik]` alone (say,
    `preprocess` already ran in an earlier campaign) without ever naming
    `fit_fly_constants` -- and MUST still get the barrier `ik` depends on,
    rather than a graph where `ik` has nothing to wait on and starts as soon
    as it is submitted. Unlike `plan.resolve`, this does not need to track
    an insertion INDEX: `ordered()` re-sorts by the registry's own
    declaration order regardless of what order names are added in, so
    adding the pooled names to the set and re-sorting is the whole
    operation.
    """
    names = set(ordered(stage_names))
    if {"ik", "postprocess"} & names:
        names.add("fit_fly_constants")
    if "postprocess" in names:
        names.add("fit_run_floor")
    return ordered(names)


def _artifact_producers() -> dict[str, tuple[str, ...]]:
    """Every artifact filename in the registry -> EVERY stage that writes it.

    Built from the WHOLE registry (`STAGES`), not just the stages a given
    chain requested, so a producer stage that a caller left out of
    `stages=` is simply absent from the returned dependency (see
    `_depends_on`'s `present` filter) rather than crashing on a lookup miss.

    This used to map an artifact to the ONE stage that writes it -- true for
    most artifacts, but false for `bouts.csv`: `gates` (the coarse-pass
    route) and `bouts` (the hand-supplied `bout_summary.csv` route) BOTH
    write it, as two mutually-exclusive ways to produce the same file. A
    dict comprehension over `STAGES` in registry order let `bouts` (which
    sorts after `gates`) silently overwrite `gates` as "the" producer; a
    chain built from `_DEFAULT_RECORDING_STAGES` contains `gates`, not
    `bouts`, so that producer was never `present` and `_depends_on` dropped
    the edge -- not a crash, not a wrong answer on this artifact's own test,
    just `fine` submitted with no dependency at all, free to start beside
    `coarse` and read whatever `bouts.csv` a previous run left on disk.
    Tuple-valued so `_depends_on` can wait on every producer that is
    actually present, not guess which one is "the real" one.
    """
    producers: dict[str, list[str]] = {}
    for stage in STAGES:
        for artifact in stage.writes:
            producers.setdefault(artifact, []).append(stage.name)
    return {artifact: tuple(names) for artifact, names in producers.items()}


_PRODUCERS_OF = _artifact_producers()


def _depends_on(stage: Stage, *, present: set[str], job_name_of: dict[str, str]) -> tuple[str, ...]:
    """Job names `stage` must wait on, derived from `reads` against `writes`.

    For each artifact `stage` reads, look up EVERY registry stage that
    writes it and keep the ones that are part of THIS chain (`present`) and
    are not `stage` itself -- ordinarily exactly one, but `bouts.csv` has
    two mutually-exclusive producers (`gates`, `bouts`), and a chain that
    for some reason requested both (`stages=[gates, bouts, fine]`) must wait
    on both rather than silently picking one. `set` -> `sorted` -> `tuple`
    keeps the result deterministic regardless of `stage.reads`' own order or
    Python's dict/set iteration, which matters under `pytest -n 8`: a test
    comparing `depends_on` must not flake because two runs enumerated a set
    in different orders.
    """
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
    """The `python -m tracking.run` argv one job body runs.

    `recording=<recording>` doubles as BOTH this job's identity (for
    `Job.name`/logging) and the Hydra config-group override
    (`configs/recording/<name>.yaml`) that supplies `session_dir`/
    `calib_dir`/`cameras` -- the same convention the
    `mvq_session_pipeline.sh` uses (`recording=$RECORDING_CFG`, its own
    header comment on why NOT `recording.timestamp=...`). `bout_ids` is
    included in full (every bout this chain covers), not narrowed to one
    bout per array task: which bout an `--array` task index maps to is a
    render-time decision (`render.py`, reading `$SLURM_ARRAY_TASK_ID`), and
    baking a single index in here would make this argv depend on which
    array slot renders it, breaking the "one `Job.command` is the whole
    truth of what this stage runs" invariant every other stage in this
    module relies on.

    `recording="*"` (`build_session_graph`'s marker for "not one recording")
    skips the override rather than handing Hydra a literal `*`, since there
    is no `configs/recording/*.yaml`. It is now used ONLY when there are no
    recordings at all: this module used to pass it for the session `collect`
    job on the claim that `collect_session` reads only `run.root`, and that
    claim was false. `_execute`'s collect branch reads `cfg.recording.fps`
    as the SOLE source of the combined h5's `source_hz` -- a placeholder
    there labels every velocity in the dataset with a rate that was never
    run. Measured 2026-09-15: skipping the override left `recording=template`
    in place and `spec.validate()` killed the job on
    `/CHANGEME/path/to/session`, which is the loud version of that outcome.
    The `collect` job now carries the recording whose run root it actually
    runs against (see `build_session_graph`).

    `recording` may also be `"<group>:<id>"` (task 20's `session1:<id>` /
    `session1:*`-expanded tokens, `tracking.pipeline.slurm.discover`) -- a
    recording GROUP shared by several recordings that differ only in which
    one `recording.id` selects. Split on `:` and add `recording.id=<id>` as
    a second override rather than trying to fold it into one override
    string: `recording=group:id` is not valid Hydra syntax, and a caller
    composing this chain's `run_root` (`session_pipeline.sh`) must resolve
    the SAME two overrides, so the split has to happen the same way in both
    places.

    `recording.id` is quoted (`recording.id='<id>'`), not bare -- measured
    2026-09-14: Hydra's override grammar treats an unquoted, all-digits-and-
    underscores value as a Python-style int literal and silently EATS every
    underscore (`2026_04_02_11_52_43` composes to the id `20260402115243`,
    a real recording directory that number is not), which then 404s reading
    `bouts_csv`. The quotes force string type and are themselves ordinary
    characters once this whole argv element is `shlex`-quoted as one shell
    word (`_gpu_wrapped`) -- the job body's `python -m tracking.run` still
    receives literal `recording.id='2026_04_02_11_52_43'` and Hydra's
    grammar honours the inner quotes as it does on an interactive CLI.
    """
    argv = [sys.executable, "-m", "tracking.run"]
    if recording != "*":
        group, _, rec_id = recording.partition(":")
        argv.append(f"recording={group}")
        if rec_id:
            argv.append(f"recording.id='{rec_id}'")
    argv += [f"run.root={run_root}", f"stages=[{stage}]"]
    # `run.name` travels WITH `run.root`, always from the same caller. It does
    # not pick the root (that is passed explicitly and wins), but `collect`
    # builds `ik_output_combined_<anatomy>_<run.name>.h5` from it, and a job
    # that omitted it inherited `pipeline.yaml`'s `run.name: debug` -- which
    # put `ik_output_combined_v1_debug.h5` inside a `pose_v2_20260914` root on
    # 2026-09-15, the exact confusion that placeholder name exists to prevent.
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
    """The full job chain for ONE recording, in registry (dependency) order.

    `coarse -> gates -> fine[array] -> preprocess[array] -> BARRIER
        -> fit_fly_constants + fit_run_floor -> BARRIER
        -> ik[array], postprocess[array]`

    (`kpvideo[array]` also appears, reading `fine`'s output independently of
    the barrier -- it is a rendering side output nothing downstream reads,
    so it neither blocks nor is blocked by the pooled fits.) `collect` is
    NOT part of this chain -- see `_DEFAULT_RECORDING_STAGES`'s docstring;
    `build_session_graph` adds one `collect:<recording>` job per recording,
    after that recording's sink.

    Raises `ValueError` if `stages` names `collect` explicitly (task 19
    defect 2): a live campaign passed `collect` in `--stages`, which built a
    SECOND collect job here, duplicating the one `build_session_graph` adds
    for this recording and making the campaign's only aggregation depend on
    it -- when the duplicate failed, nothing aggregated at all. The refusal
    still stands now that collect is per-recording: the duplicate would
    carry the same name and the same root, so the two would race to write
    one recording's scorecard and combined h5. Refusing up front means an
    operator who typed `collect` finds out before submission, not from a
    missing summary afterward.
    """
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
    """The job in `chain` that `collect` must wait for, or `None` for no collect.

    `postprocess`, not `chain[-1]`. `collect_session` reads each bout-fly's
    `qc.json`/`outputs.h5` and the run's `floor.json` -- all written by
    `postprocess` (which already gates on `fit_run_floor`). Nothing it reads
    comes from a render.

    Taking the chain's tail worked only by accident: `postprocess` is last in
    `_DEFAULT_RECORDING_STAGES` today. `sidebyside` sits AFTER it in registry
    order, so adding that stage to the defaults would have made an `afterok`
    on video encoding gate the dataset h5 -- one corrupt mp4, one node whose
    EGL is broken, and `collect` never runs, exactly the way a failed `ik`
    array task used to strand it. Renders are side outputs nothing downstream
    reads; they must not be able to withhold the data.

    Returns `None` -- meaning "build no collect job for this recording" --
    when the chain has no `postprocess`. This used to fall back to the
    chain's tail, and that fallback put the coupling straight back: a
    `--stages sidebyside` catch-up pass built a collect gating `afterok` on
    the RENDER, so one failed render left a permanently stranded
    `DependencyNeverSatisfied` collect (measured 2026-09-15, job 40200105).

    No collect is the right answer, not an unconditional one. A chain
    without `postprocess` is not producing the per-bout-fly `outputs.h5`
    collect aggregates; those either already exist, in which case the
    recording was already collected, or they do not, in which case
    collecting now would describe a run that has not happened. A render-only
    pass should render.
    """
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
    """Every recording's chain, plus ONE `collect` job gating all of them.

    `collect` (spec section 5: `scope="session"`, `resume="always"` -- a
    session summary over a changed bout set is always a different summary)
    aggregates the whole session, so it is built exactly once here rather
    than once per recording -- a driver that emitted one `collect` per
    recording would produce N partial session summaries and no single job a
    downstream consumer could depend on to mean "the session is done".

    It depends on each recording's LAST stage in registry order (normally
    `postprocess`, the sink every other stage in that chain feeds -- see
    `build_recording_chain`'s docstring for the full order), not on every
    job in every chain: depending on the sink is sufficient because
    `depends_on` inside `build_recording_chain` already chains everything
    upstream of it, and `afterok` on the sink alone keeps this job's own
    `--dependency=` string from growing quadratically with chain length.
    A recording contributing NOTHING to that list (a typo'd name, an empty
    chain) would make `collect` gate on fewer recordings than asked for --
    exactly what `test_one_collect_gates_on_every_recording` checks by name.

    `run_roots` maps each recording name to ITS OWN run root, resolved by
    the caller from the SAME Hydra config a local run resolves (`compose(
    config_name="pipeline", overrides=[f"recording={r}", ...]).run.root`) --
    never rebuilt as `<base>/<run_name>` here or in the shell wrapper (task
    19 defect 1). A single shared root merges every recording's bouts into
    one directory: `bouts/bout_00001/` from recording A and recording B is
    the SAME path, so the second recording silently overwrites the first --
    the exact failure a real 8-job campaign was cancelled for. A plain
    string is still accepted as shorthand for a single recording (mirrors
    `build_recording_chain`'s own `run_root: str` parameter) and is REFUSED
    for more than one, rather than silently reused for every one of them,
    which would reintroduce that same defect one call signature later.

    `bout_ids` mirrors `run_roots`'s two shapes for the same reason (task
    20): a `{recording: ids}` mapping gives each recording ITS OWN bout
    list -- required the moment recordings differ, as Session1's do (2..30
    bouts across its runnable recordings) -- while a plain `Sequence[int]`
    is shorthand applied to every recording alike, kept for callers (and
    every existing test in this module) that already pass one shared list
    for recordings known to share a bout range. Unlike `run_roots`, this
    shorthand is not refused for N>1 here: the CLI-facing refusal (`--bout-
    ids` given for more than one discovered recording,
    `tracking.pipeline.slurm.discover.resolve_bout_ids`) is what an operator
    actually hits, ahead of this call; refusing again here would only be
    reachable by a caller that already resolved a mapping and downgraded it
    to a list, which is not a mistake this signature needs to guard against.

    The `collect` job itself runs against `run_roots[recordings[0]]`, and
    carries `recording=recordings[0]` to match -- one root and its own
    recording, never the config default, whose placeholder `fps` would be
    written into the combined h5 as `source_hz`. The job is still NAMED
    `collect:<recording>`, one per recording, each gating on that recording's
    own sink and reading that recording's own root.

    **This used to be ONE session-level `collect:*` job against
    `run_roots[recordings[0]]`.** `collect_session` aggregates a SINGLE run
    root's `bouts/` directory, and once task 19 gave every recording its own
    root, one job could only ever describe the first: Session1's 12
    recordings would have produced 12 complete sets of bout artifacts, one
    scorecard, one combined h5, and no error anywhere. The docstring called
    that a known gap; this is the follow-up.

    The trade that argued for a single job -- "one job a downstream consumer
    can depend on to mean the session is done" -- is worth less than N-1
    recordings' results. Nothing in this repo gates on such a sentinel; an
    operator reads `squeue`, and the wrapper prints every submitted id. If a
    single pooled artifact is wanted, `postprocess.combine.combine_many`
    takes several run roots and writes one file, with `info/fly_ids` and
    `info/buckets` keeping the recordings separable -- it is deliberately
    explicit rather than the default, because pooling mixes DLT calibration
    frames.
    """
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

    # ONE collect per recording, each against ITS OWN root -- never one job
    # against `run_roots[recordings[0]]`, which aggregated recording 1 and
    # left the other N-1 with no scorecard and no combined h5 at all.
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
