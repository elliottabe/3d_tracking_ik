"""Resolving a Hydra config + a requested stage list into an ordered plan.

**The one thing that matters most: the two pooled-fit barriers.** Both
`fit_fly_constants` (a fly's body scale + marker offsets, pooled across every
bout that fly appears in) and `fit_run_floor` (the arena floor, pooled across
every bout-fly in the recording) read `kp3d_filt.npz`, so EVERY `preprocess`
bout-fly must finish before either pooled step runs, and both pooled steps
must finish before any `ik` (needs the offsets) or `postprocess` (needs the
floor AND the scale) starts:

    fine -> preprocess (every bout-fly) -> BARRIER
         -> fit_fly_constants (once per fly) + fit_run_floor (once per run) -> BARRIER
         -> ik, postprocess (every bout-fly) -> collect

A driver that ran `preprocess -> ik` bout by bout would fit each fly's marker
offsets from whichever bout happened to run first -- the historic 38x
body-scale error and the male fitted with the female's offsets (spec 5.1) --
and would score posture against a per-bout floor that has been measured
coming out VERTICAL on a real bout (spec 5.3 / Ruling 19). Neither failure
raises; both produce confident, wrong numbers.

**The barrier is enforced structurally, not just by this module.**
`fit_fly_constants`/`fit_run_floor` are ROWS in `pipeline.stages.STAGES`
(Ruling 15/21 of this plan's ledger split them out of the registry's single
`preprocess`/`postprocess` rows, and a later fix folded them back in as their
own rows rather than leaving them only to this driver): `fit_fly_constants`
declares `reads=("kp3d_filt.npz",)` (what `preprocess` writes) and
`writes=("scale.json", "offsets_fly.h5")` (what `ik` reads), and
`fit_run_floor` declares `writes=("floor.json",)` (what `postprocess` reads).
The WHOLE table's declaration order is therefore checkable against every
stage's `reads`/`writes`: a registry edit that put a pooled fit before
`preprocess`, or `ik` before a pooled fit, contradicts the declarations on its
own -- a future driver rewrite cannot quietly reintroduce a per-bout offsets
fit by getting `plan.py` wrong, because the registry itself would refuse to
declare that order.

This module's OWN job is narrower: `resolve` discards the caller's stage
order (`stages.ordered`, backed by the registry's declaration order) rather
than trusting a caller to ask for the pooled steps in the right place, and --
because a caller can reasonably ask for `ik` alone with `preprocess` already
done in an earlier run, without ever naming the pooled steps -- inserts
`fit_fly_constants`/`fit_run_floor` itself whenever `ik`/`postprocess` need
them and the caller didn't already ask for them explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tracking.pipeline.resume import stage_is_current
from tracking.pipeline.stages import Stage, ordered, stage_by_name
from tracking.preprocess.offsets import per_fly_offsets_name

__all__ = ["Work", "resolve"]


@dataclass(frozen=True)
class Work:
    """One unit of scheduled work: run (or skip) `stage` for `target`.

    `target` names WHAT this Work item is for (a fly, a bout, a bout-fly, or
    the whole recording) -- never a bare index, per CLAUDE.md's naming rule.
    A skipped item is KEPT, not dropped: `skip_reason` carries why, so a
    fully-resumed plan is still visibly a plan (spec: dropping it would make
    `dry_run` claim there is nothing to do, indistinguishable from a
    misconfigured run that matched no bouts).
    """

    stage: str
    target: str
    out_dir: Path
    skip_reason: str | None


# Stages this module never asks `resume.stage_is_current` about, so they are
# always scheduled (`skip_reason=None`) rather than resolved here:
#   - "coarse" resumes on a CHUNKED PARTIAL artifact; `stage_is_current`
#     deliberately RAISES `NotImplementedError` for it (spec section 5: its
#     presence proves nothing about completeness) rather than guessing, and
#     the stage owns its own chunk-level resume check internally.
#   - "fine" resumes on a GATE STRING computed from the constructed
#     runner/checkpoint (`detector.fine.fine_gate_string`); building that
#     object is a real (GPU-adjacent) side effect this pure planning step
#     must not perform, and `fine_track_bouts` already resumes per bout
#     internally against the ONE store that gate lives in
#     (`recording_stages` module docstring, trap 1).
_NEVER_SKIP_AT_PLAN_LEVEL = frozenset({"coarse", "fine"})


def _bout_dir(run_root: Path, bout_id: int) -> Path:
    return run_root / "bouts" / f"bout_{int(bout_id):05d}"


def _fly_dir(run_root: Path, bout_id: int, fly: int) -> Path:
    return _bout_dir(run_root, bout_id) / f"fly{int(fly)}"


def _skip_reason(stage: Stage, *, out_dir: Path, artifacts, signature: str | None = None):
    """`None` if `stage` must run for `out_dir`; else why it can be skipped."""
    if stage_is_current(stage, out_dir=out_dir, artifacts=artifacts, signature=signature):
        return f"{stage.name} artifacts already present at {out_dir}"
    return None


def _expand(
    name: str, *, run_root: Path, recording_name: str, num_animals: int, bout_ids: list[int]
) -> list[Work]:
    """Every `Work` item for one stage NAME, at its own scope."""
    if name in _NEVER_SKIP_AT_PLAN_LEVEL:
        return [Work(name, recording_name, run_root, None)]

    # `fit_fly_constants` is checked BEFORE the scope branches below: its scope
    # is "recording" but its ARTIFACTS are per fly (`offsets_fly<N>.h5`), so the
    # generic recording branch would emit one work item where two are needed.
    if name == "fit_fly_constants":
        stage = stage_by_name(name)
        out = []
        for f in range(num_animals):
            artifacts = ("scale.json", per_fly_offsets_name(f))
            reason = _skip_reason(stage, out_dir=run_root, artifacts=artifacts)
            out.append(Work(name, f"fly{f}", run_root, reason))
        return out

    # Everything else expands by its registry SCOPE, not by name. These were
    # three hand-kept name lists until 2026-09-14, and a stage added to the
    # registry but forgotten here does not fail loudly -- it falls through to
    # the recording branch and silently plans ONE work item where a bout_fly
    # stage needs one per fly per bout. `sidebyside` had to be added to two
    # such lists to work at all, which is what prompted this.
    stage = stage_by_name(name)
    if stage.scope == "bout":
        out = []
        for b in bout_ids:
            d = _bout_dir(run_root, b)
            out.append(
                Work(name, f"bout{b}", d, _skip_reason(stage, out_dir=d, artifacts=stage.writes))
            )
        return out

    if stage.scope == "bout_fly":
        out = []
        for b in bout_ids:
            for f in range(num_animals):
                d = _fly_dir(run_root, b, f)
                out.append(
                    Work(
                        name,
                        f"bout{b}/fly{f}",
                        d,
                        _skip_reason(stage, out_dir=d, artifacts=stage.writes),
                    )
                )
        return out

    if stage.scope in ("recording", "session"):
        reason = _skip_reason(stage, out_dir=run_root, artifacts=stage.writes)
        return [Work(name, recording_name, run_root, reason)]

    if name == "fit_run_floor":
        stage = stage_by_name(name)
        reason = _skip_reason(stage, out_dir=run_root, artifacts=stage.writes)
        return [Work(name, recording_name, run_root, reason)]

    # pragma: no cover -- ordered() already validated `name` against the registry
    raise AssertionError(f"unhandled stage {name!r}")


def resolve(cfg, *, stages, bout_ids=None) -> tuple[Work, ...]:
    """`cfg` + a requested stage list -> the full, ordered, expanded plan.

    `stages.ordered` sorts the requested names into the registry's declared
    dependency order -- the caller's own order is discarded on purpose (a
    reasonable thing to type, e.g. `stages=[ik, preprocess]`, is a
    catastrophic thing to obey). This function then inserts the two pooled
    barriers: `fit_fly_constants` whenever `ik` or `postprocess` is
    requested (both need the fitted offsets -- `postprocess_bout_fly` reads
    `kp_scale` off `scale.json` even though it does not read the offsets
    file directly), and `fit_run_floor` whenever `postprocess` is requested
    (it scores posture against the pooled floor). Both are placed
    immediately after `preprocess` if present, else immediately before the
    earlier of `ik`/`postprocess` -- so a request for `ik` alone (preprocess
    already done in an earlier run) still gets the barrier it depends on.

    `gates` and `bouts` are ALTERNATIVES (both write `bouts.csv`; spec
    section 5 marks `bouts` "(alt)"); requesting both is refused rather than
    letting the second silently overwrite the first's output.
    """
    stage_names = list(stages)
    if {"gates", "bouts"} <= set(stage_names):
        raise ValueError(
            "stages contains both 'gates' and 'bouts' -- they are ALTERNATIVE "
            "entries and both write bouts.csv (spec section 5 marks 'bouts' "
            "'(alt)'); running both means the second silently overwrites the "
            "first. Pick one."
        )

    registry_order = list(ordered(stage_names))  # raises KeyError naming the valid stages

    # Both pooled steps are now real registry rows (see this module's
    # docstring), so `ordered` already places one correctly if the caller
    # named it explicitly -- `pooled` below only adds the ones the caller
    # did NOT ask for but `ik`/`postprocess` still need, so a stage never
    # ends up scheduled (and its `Work` items duplicated) twice.
    needs_offsets = "ik" in registry_order or "postprocess" in registry_order
    needs_floor = "postprocess" in registry_order
    pooled = [
        n
        for n, needed in (("fit_fly_constants", needs_offsets), ("fit_run_floor", needs_floor))
        if needed and n not in registry_order
    ]

    if pooled:
        if "preprocess" in registry_order:
            insert_at = registry_order.index("preprocess") + 1
        else:
            insert_at = min(i for i, n in enumerate(registry_order) if n in ("ik", "postprocess"))
        final_names = registry_order[:insert_at] + pooled + registry_order[insert_at:]
    else:
        final_names = registry_order

    run_root = Path(cfg.run.root)
    recording_name = str(cfg.recording.name)
    num_animals = int(cfg.recording.num_animals)
    resolved_bout_ids = list(bout_ids) if bout_ids is not None else []

    works: list[Work] = []
    for name in final_names:
        works.extend(
            _expand(
                name,
                run_root=run_root,
                recording_name=recording_name,
                num_animals=num_animals,
                bout_ids=resolved_bout_ids,
            )
        )
    return tuple(works)
