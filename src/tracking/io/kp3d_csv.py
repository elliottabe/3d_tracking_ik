"""Reading a whole-recording 3D keypoint CSV into per-bout arrays.

The amputation cohort's `data3D.csv` carries two header rows -- keypoint name
repeated four times, then `x,y,z,confidence` -- and one row per frame of the
entire recording. Values are already in the pipeline's world units
(`preprocess.scale.WORLD_UNITS_TO_MM`); nothing here rescales them.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from tracking.io.names import Order, as_order

__all__ = ["read_kp3d_header", "read_bout_kp3d"]

_COORDS = ("x", "y", "z", "confidence")


def read_kp3d_header(path) -> list[str]:
    """The keypoint names of a two-row-header 3D CSV, in column order."""
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        try:
            names = next(reader)
            coords = next(reader)
        except StopIteration:
            raise ValueError(f"{path}: fewer than two header rows") from None

    if len(names) != len(coords):
        raise ValueError(
            f"{path}: header rows disagree -- {len(names)} names, {len(coords)} coordinates"
        )
    if not names or len(names) % 4:
        raise ValueError(f"{path}: {len(names)} columns is not a multiple of 4 (x,y,z,confidence)")

    out: list[str] = []
    for i in range(0, len(names), 4):
        block = tuple(c.strip() for c in coords[i : i + 4])
        if block != _COORDS:
            raise ValueError(f"{path}: columns {i}..{i + 3} are {block}, expected {_COORDS}")
        group = {n.strip() for n in names[i : i + 4]}
        if len(group) != 1:
            raise ValueError(
                f"{path}: columns {i}..{i + 3} must all carry one keypoint name, "
                f"got {sorted(group)}"
            )
        out.append(names[i].strip())
    return out


def read_bout_kp3d(
    path,
    bouts: Sequence,
    *,
    kp_order: Order | Sequence[str],
    chunksize: int = 200_000,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """`{bout.idx: (kp3d (T,K,3), conf3d (T,K))}`, one chunked pass over `path`.

    Rows are frame-indexed from 0. Only rows inside a requested bout are kept,
    so a 760k-frame file costs one pass and a few thousand retained rows.
    """
    path = Path(path)
    order = as_order(kp_order)
    file_order = Order(read_kp3d_header(path))
    perm = file_order.permutation_to(order)
    n_file_kp = len(file_order)

    wanted = {int(b.idx): (int(b.start_frame), int(b.end_frame)) for b in bouts}
    if not wanted:
        return {}
    lo = min(s for s, _ in wanted.values())
    hi = max(e for _, e in wanted.values())

    parts: dict[int, list[np.ndarray]] = {idx: [] for idx in wanted}
    n_rows = 0
    for chunk in pd.read_csv(path, skiprows=2, header=None, chunksize=chunksize, dtype=np.float64):
        first, last = int(chunk.index[0]), int(chunk.index[-1])
        n_rows += len(chunk)
        if last < lo or first > hi:
            continue
        for idx, (start, end) in wanted.items():
            if end < first or start > last:
                continue
            parts[idx].append(chunk.loc[max(start, first) : min(end, last)].to_numpy())

    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for idx, (start, end) in wanted.items():
        want = end - start + 1
        got = np.concatenate(parts[idx]) if parts[idx] else np.empty((0, 4 * n_file_kp))
        if got.shape[0] != want:
            raise ValueError(
                f"{path}: bout {idx} wants frames {start}..{end} ({want} rows) but the "
                f"file yielded {got.shape[0]}; the file has {n_rows} frames"
            )
        block = got.reshape(want, n_file_kp, 4)
        out[idx] = (block[:, perm, :3].copy(), block[:, perm, 3].copy())
    return out
