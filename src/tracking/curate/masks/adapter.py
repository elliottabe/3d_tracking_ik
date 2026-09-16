"""Import a SAM/SAM2/SAM3 output tree into the mask sidecar format."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from tracking.curate.masks.store import MaskStore


def import_masks(src_dir: str | Path, out_root: str | Path, *, tool: str = "sam3") -> dict:
    """Copy `<src>/<rec>/<cam>/Frame_<n>.npz` into the sidecar, checking each file.

    Each npz must hold `ann_ids` (N,) and `masks` (N, H, W).

    >>> import_masks("sam3_out", "dataset_masks", tool="sam3")   # doctest: +SKIP
    {'n_files': 19019, 'tool': 'sam3', ...}
    """
    src_dir, store = Path(src_dir), MaskStore(out_root)
    n = 0
    for p in sorted(src_dir.rglob("Frame_*.npz")):
        rel = p.relative_to(src_dir)
        if len(rel.parts) != 3:
            raise ValueError(f"{p}: expected <recording>/<camera>/Frame_<n>.npz")
        with np.load(p) as z:
            if "ann_ids" not in z or "masks" not in z:
                raise ValueError(f"{p}: needs 'ann_ids' and 'masks' arrays")
            if len(z["ann_ids"]) != len(z["masks"]):
                raise ValueError(f"{p}: ann_ids and masks disagree in length")
        dst = store.root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dst)
        n += 1
    return store.write_index(tool=tool, source=str(src_dir)) | {"n_files": n}
