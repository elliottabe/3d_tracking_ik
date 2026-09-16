"""Bout tables: a shared `BoutSpec` for hand-supplied and coarse-pass-derived bouts."""

from __future__ import annotations

import csv
import os
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np

_BOUT_ALIASES = {"bout", "bout_idx"}

_CSV_FIELDS = ("idx", "start_frame", "end_frame", "n_frames", "source")


@dataclass(frozen=True)
class BoutSpec:
    """One bout for a pass to process."""

    idx: int
    start_frame: int
    end_frame: int
    n_frames: int
    source: str
    init_centroid: np.ndarray | None = field(default=None, compare=False)


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _lower_lookup(row: dict[str, str]) -> dict[str, str]:
    """Map lower-cased column name -> original key, so lookups are case-insensitive."""
    return {k.lower(): k for k in row}


def _get_bout_idx(row: dict[str, str], lower: dict[str, str]) -> int:
    for alias in _BOUT_ALIASES:
        if alias in lower:
            return int(row[lower[alias]])
    raise ValueError(f"no bout/bout_idx column found; columns were {list(row)}")


def _fly_id_matches(fly_id: str, session_tag: str) -> bool:
    """`session_tag` at a `/` component boundary of `fly_id`, boundary-terminated."""
    parts = fly_id.split("/")
    for i in range(len(parts)):
        candidate = "/".join(parts[i:])
        if (
            candidate == session_tag
            or candidate.startswith(session_tag + "_")
            or candidate.startswith(session_tag + "/")
        ):
            return True
    return False


def read_bout_summary(path: str | Path, *, session_tag: str | None = None) -> list[BoutSpec]:
    """Read a hand-supplied bout_summary CSV, stamping ``source="summary"``."""
    rows = _read_rows(path)
    bouts: list[BoutSpec] = []
    for row in rows:
        lower = _lower_lookup(row)

        if "fly_id" in lower and session_tag is not None:
            if not _fly_id_matches(row[lower["fly_id"]], session_tag):
                continue

        idx = _get_bout_idx(row, lower)
        start_frame = int(row[lower["start_frame"]])
        end_frame = int(row[lower["end_frame"]])
        derived_n_frames = end_frame - start_frame + 1

        if "n_frames" in lower:
            n_frames = int(row[lower["n_frames"]])
            if n_frames != derived_n_frames:
                raise ValueError(
                    f"bout {idx}: n_frames {n_frames} disagrees with "
                    f"end_frame - start_frame + 1 = {derived_n_frames}"
                )
        else:
            n_frames = derived_n_frames

        bouts.append(
            BoutSpec(
                idx=idx,
                start_frame=start_frame,
                end_frame=end_frame,
                n_frames=n_frames,
                source="summary",
            )
        )
    return bouts


def read_bouts_csv(path: str | Path) -> list[BoutSpec]:
    """Read a bouts CSV previously written by `write_bouts_csv` (any `source`)."""
    rows = _read_rows(path)
    bouts: list[BoutSpec] = []
    for row in rows:
        lower = _lower_lookup(row)
        idx = _get_bout_idx(row, lower)
        bouts.append(
            BoutSpec(
                idx=idx,
                start_frame=int(row[lower["start_frame"]]),
                end_frame=int(row[lower["end_frame"]]),
                n_frames=int(row[lower["n_frames"]]),
                source=row[lower["source"]],
            )
        )
    return bouts


def write_bouts_csv(path: str | Path, bouts: Sequence[BoutSpec]) -> None:
    """Write `bouts` to `path` atomically as `bout,start_frame,end_frame,n_frames,source`."""
    path = Path(path)
    columns = [
        f.name if f.name != "idx" else "bout" for f in fields(BoutSpec) if f.name in _CSV_FIELDS
    ]
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for b in bouts:
            writer.writerow([b.idx, b.start_frame, b.end_frame, b.n_frames, b.source])
    os.replace(tmp_path, path)


def select(bouts: Sequence[BoutSpec], ids: Sequence[int] | None) -> list[BoutSpec]:
    """Return the bouts whose `idx` is in `ids`; `None`/empty `ids` returns all."""
    if not ids:
        return list(bouts)
    by_idx = {b.idx: b for b in bouts}
    missing = [i for i in ids if i not in by_idx]
    if missing:
        raise ValueError(f"bout ids not in the table: {missing}")
    return [by_idx[i] for i in ids]
