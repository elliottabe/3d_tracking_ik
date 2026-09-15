"""Recording discovery and per-recording bout ids for a SLURM campaign.

Session1 has fourteen recordings and one config file; Session0 has one of
each. `--recordings session0` already names one config group directly, but
Session1 would otherwise need fourteen hand-made groups or fourteen
hand-typed CLI invocations -- and even then, `--bout-ids` took ONE list
shared by every recording, silently wrong the moment recordings differ (this
task's brief: Session1's runnable recordings have 2..30 bouts each).

This module is deliberately pure Python with no Hydra/`configs/` import: the
two things it does -- glob a session directory for recordings that have a
parseable bout summary, and refuse (or resolve) per-recording bout ids -- are
both testable against a plain directory tree or a fake lookup callback,
independent of which cluster or which real recording configs exist. The
caller (`scripts/slurm/session_pipeline.sh`) is the only place that ever
composes a real Hydra config; it supplies that as a callback
(`session_dir_parent_of` / `bouts_csv_and_tag_of`) rather than this module
rebuilding `<data_dir>/Video_recordings/.../<id>` itself, which would be a
second construction of a path `configs/recording/*.yaml` already owns (the
exact failure task 19 defect 1 was: a second construction that can disagree
with the config's own interpolation).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from tracking.io.bouts import read_bout_summary

__all__ = ["list_recordings_with_summary", "expand_recordings", "resolve_bout_ids"]

_DEFAULT_SUMMARY_NAME = "courtship_bout_summary.csv"


def list_recordings_with_summary(
    session_dir_parent: Path, *, csv_name: str = _DEFAULT_SUMMARY_NAME
) -> tuple[list[str], list[tuple[str, str]]]:
    """Every recording id directly under `session_dir_parent`, split by summary.

    Included: `<id>/<csv_name>` exists and `read_bout_summary` parses it to
    at least one bout. Two of Session1's fourteen recordings have no
    `courtship_bout_summary.csv` at all (verified 2026-09-14) -- silently
    dropping them from a glob would make "12 chains submitted" and "14
    recordings exist" indistinguishable, exactly the discrepancy an operator
    needs told. Skipped entries carry a reason (not just an id) so the
    caller can print one named line per skip rather than a bare count.
    """
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
    """Each `--recordings` token -> concrete `group` / `group:id` tokens.

    Three token shapes (this task's brief): a bare config-group name
    (`session0`) and an explicit `group:id` both pass through unchanged --
    an explicit id is composed later, once, by the caller, so this function
    never needs to know that recording exists on disk. `group:*` GLOBS: every
    directory under that group's session dir with a parseable bout summary
    (`list_recordings_with_summary`) becomes one `group:id` token; anything
    else is named in `skip_lines`, never dropped silently.

    `session_dir_parent_of(group)` composes that group's own Hydra config
    and returns `recording.session_dir`'s PARENT. Injecting it as a callback
    keeps this function testable against a tmp_path directory tree with no
    Hydra/`configs/` involved, while `session_pipeline.sh` remains the one
    place that actually resolves the path.
    """
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
    """Per-recording bout ids for `build_session_graph`, never one shared list.

    SLURM needs each recording's array SIZE at submit time (`--array=0-N`),
    so this cannot be deferred to the driver's own runtime
    `_discover_bout_ids` -- and Session1's runnable recordings have 2..30
    bouts each (this task's brief), so a single list sizes every array
    wrong except by coincidence for exactly one recording.

    `explicit` (the CLI's `--bout-ids`) is REFUSED for more than one
    recording -- same shape as task 19's `--run-root` refusal, and for the
    same reason: silently reusing one list for N>1 recordings is a worse
    failure than making the operator submit them one at a time. With
    `explicit` omitted, each recording's ids come from its own `bouts_csv`
    via `read_bout_summary`'s `idx` values, NOT `range(n)` -- a curated
    summary that has dropped a bout makes `idx` and row position disagree.
    """
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
