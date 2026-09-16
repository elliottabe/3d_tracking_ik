"""Resolving a Hydra config + a requested stage list into an ordered plan."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tracking.pipeline.resume import stage_is_current
from tracking.pipeline.stages import Stage, ordered, stage_by_name
from tracking.preprocess.offsets import per_fly_offsets_name

__all__ = ["Work", "resolve"]


@dataclass(frozen=True)
class Work:
    """One unit of scheduled work: run (or skip) `stage` for `target`."""

    stage: str
    target: str
    out_dir: Path
    skip_reason: str | None


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

    if name == "fit_fly_constants":
        stage = stage_by_name(name)
        out = []
        for f in range(num_animals):
            artifacts = ("scale.json", per_fly_offsets_name(f))
            reason = _skip_reason(stage, out_dir=run_root, artifacts=artifacts)
            out.append(Work(name, f"fly{f}", run_root, reason))
        return out

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
    """`cfg` + a requested stage list -> the full, ordered, expanded plan."""
    stage_names = list(stages)
    if {"gates", "bouts"} <= set(stage_names):
        raise ValueError(
            "stages contains both 'gates' and 'bouts' -- they are ALTERNATIVE "
            "entries and both write bouts.csv (spec section 5 marks 'bouts' "
            "'(alt)'); running both means the second silently overwrites the "
            "first. Pick one."
        )

    registry_order = list(ordered(stage_names))  # raises KeyError naming the valid stages

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
