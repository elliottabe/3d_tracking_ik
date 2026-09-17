"""The curation stage registry. Declaration order is dependency order."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Stage", "STAGES", "stage_by_name", "ordered"]


@dataclass(frozen=True)
class Stage:
    name: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]


STAGES: tuple[Stage, ...] = (
    Stage("merge", ("tiers",), ("annotations/", "images/", "calibrations/", "manifest.json")),
    Stage("import_masks", ("sam_dir",), ("<root>_masks/",)),
    Stage("validate", ("manifest.json",), ("report",)),
    Stage("package", ("manifest.json",), ("zip",)),
)

_BY_NAME = {s.name: s for s in STAGES}


def stage_by_name(name: str) -> Stage:
    """>>> stage_by_name("merge").name
    'merge'
    """
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError(f"{name!r} is not a curate stage; known: {list(_BY_NAME)}") from None


def ordered(names) -> list[str]:
    """Requested stage names, sorted into declaration order.

    >>> ordered(["package", "merge"])
    ['merge', 'package']
    """
    wanted = {stage_by_name(n).name for n in names}
    return [s.name for s in STAGES if s.name in wanted]
