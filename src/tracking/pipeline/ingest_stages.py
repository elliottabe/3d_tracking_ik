"""The 3D-given ingest: a whole-recording keypoint CSV into per-bout artifacts."""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from tracking.conventions import announce
from tracking.io.artifacts import save_npz
from tracking.io.kp3d_csv import read_bout_kp3d
from tracking.io.names import Order, as_order

__all__ = ["ingest3d_recording", "sex_json_for"]

_SEX_LABELS = {"f": "female", "m": "male", "female": "female", "male": "male"}


def sex_json_for(index_csv, recording: str) -> dict[str, Any] | None:
    """This recording's `sex.json` payload from the cohort index, or None.

    The recording's own column is unnamed in the index header, so it is found
    by value. The sex column IS named, so it is read by header position -- a
    left-to-right scan for the first sex-looking cell would return a flyID of
    "F" or "M" before ever reaching the real sex column.

    A recording absent from the index yields None, which the caller writes as
    no file at all -- `fly_sex_label` then reports "unknown".
    """
    with open(index_csv, newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return None

    header = [c.strip().lower() for c in rows[0]]
    sex_col = header.index("sex") if "sex" in header else None

    for row in rows[1:]:
        cells = [c.strip() for c in row]
        if recording not in cells:
            continue
        if sex_col is not None and sex_col < len(cells):
            cells = [cells[sex_col]]
        for cell in cells:
            label = _SEX_LABELS.get(cell.lower())
            if label:
                return {
                    "identity": "sex",
                    "sex_by_fly": {"0": label},
                    "method": "cohort_index",
                    "authority": "cohort_index",
                    "source": str(index_csv),
                }
        return None
    return None


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def ingest3d_recording(
    run_root,
    *,
    spec,
    kp_order: Order | Sequence[str],
    bouts: Sequence,
    force: bool = False,
) -> dict[str, Any]:
    """Slice `spec.kp3d_csv` into one `fly0/kp3d.npz` per bout in `bouts`."""
    run_root = Path(run_root)
    order = as_order(kp_order)
    if spec.kp3d_csv is None:
        raise ValueError(
            "ingest3d needs recording.kp3d_csv; this recording declares none, so "
            "there is no 3D table to slice"
        )

    def _bout_dir(idx: int) -> Path:
        return run_root / "bouts" / f"bout_{int(idx):05d}"

    todo = [b for b in bouts if force or not (_bout_dir(b.idx) / "fly0" / "kp3d.npz").exists()]
    n_skipped = len(bouts) - len(todo)
    announce(
        "pipeline.ingest_stages",
        "ingest3d_recording",
        f"{len(todo)} bout(s) to write, {n_skipped} already present at {run_root}",
    )

    sex = sex_json_for(spec.index_csv, spec.name) if spec.index_csv is not None else None

    by_bout = read_bout_kp3d(spec.kp3d_csv, todo, kp_order=order) if todo else {}
    for bout in todo:
        kp3d, conf3d = by_bout[bout.idx]
        bout_dir = _bout_dir(bout.idx)
        # sex.json before kp3d.npz: `todo` above keys the resume check on
        # fly0/kp3d.npz, so it must be the LAST artifact written. A crash or
        # preemption between the two must leave a bout that still looks
        # incomplete (no kp3d.npz yet), not one that resume treats as done
        # and skips forever with no sex.
        if sex is not None:
            _write_json(bout_dir / "sex.json", sex)
        save_npz(
            bout_dir / "fly0" / "kp3d.npz",
            arrays={"kp3d": np.asarray(kp3d), "conf3d": np.asarray(conf3d)},
            kp=order,
            cams=None,
        )

    return {
        "run_root": str(run_root),
        "n_written": len(todo),
        "n_skipped": n_skipped,
        "sex": None if sex is None else sex["sex_by_fly"].get("0"),
    }
