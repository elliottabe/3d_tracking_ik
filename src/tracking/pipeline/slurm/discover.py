"""Recording discovery and per-recording bout ids for a SLURM campaign."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from tracking.io.bouts import read_bout_summary

__all__ = ["list_recordings_with_summary", "expand_recordings", "resolve_bout_ids"]

_DEFAULT_SUMMARY_NAME = "courtship_bout_summary.csv"


def list_recordings_with_summary(
    session_dir_parent: Path, *, csv_name: str = _DEFAULT_SUMMARY_NAME
) -> tuple[list[str], list[tuple[str, str]]]:
    """Every recording id directly under `session_dir_parent`, split by summary."""
    included: list[str] = []
    skipped: list[tuple[str, str]] = []
    for entry in sorted(p.name for p in session_dir_parent.iterdir() if p.is_dir()):
        csv_path = session_dir_parent / entry / csv_name
        if not csv_path.exists():
            skipped.append((entry, f"no {csv_name}"))
            continue
        if not read_bout_summary(csv_path):
            skipped.append((entry, f"{csv_name} has no bouts"))
            continue
        included.append(entry)
    return included, skipped


def expand_recordings(
    tokens: Sequence[str],
    *,
    session_dir_parent_of: Callable[[str], Path],
    csv_name: str = _DEFAULT_SUMMARY_NAME,
) -> tuple[list[str], list[str]]:
    """Each `--recordings` token -> concrete `group` / `group:id` tokens."""
    resolved: list[str] = []
    skip_lines: list[str] = []
    for token in tokens:
        group, sep, rest = token.partition(":")
        if not sep or rest != "*":
            resolved.append(token)
            continue
        parent = session_dir_parent_of(group)
        included, skipped = list_recordings_with_summary(parent, csv_name=csv_name)
        resolved += [f"{group}:{rid}" for rid in included]
        skip_lines += [f"skip {group}:{rid}: {reason}" for rid, reason in skipped]
    return resolved, skip_lines


def resolve_bout_ids(
    recordings: Sequence[str],
    *,
    explicit: Sequence[int] | None,
    bouts_csv_and_tag_of: Callable[[str], tuple[Path, str]],
) -> dict[str, list[int]]:
    """Per-recording bout ids for `build_session_graph`, never one shared list."""
    if explicit is not None:
        if len(recordings) > 1:
            raise ValueError(
                f"--bout-ids given with {len(recordings)} recordings: one list "
                "cannot be right for recordings with different bout counts "
                "(Session1 alone ranges 2..30). Omit --bout-ids and let each "
                "recording's ids come from its own bouts_csv, or submit "
                "recordings one at a time, each with its own --bout-ids."
            )
        return {recordings[0]: list(explicit)} if recordings else {}

    resolved: dict[str, list[int]] = {}
    for recording in recordings:
        bouts_csv, session_tag = bouts_csv_and_tag_of(recording)
        resolved[recording] = sorted(
            b.idx for b in read_bout_summary(bouts_csv, session_tag=session_tag)
        )
    return resolved
