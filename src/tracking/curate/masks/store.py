"""The `<root>_masks/` sidecar: per-(recording, camera, frame) instance masks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def sidecar_path(root: str | Path) -> Path:
    """The mask sidecar that belongs to `root`.

    >>> sidecar_path("/data/dataset")
    PosixPath('/data/dataset_masks')
    """
    root = Path(root)
    return root.with_name(root.name + "_masks")


class MaskStore:
    """Read/write instance masks keyed by annotation id.

    >>> store = MaskStore("dataset_masks")                       # doctest: +SKIP
    >>> store.save("rec", "Cam01", 7, {11: mask})                # doctest: +SKIP
    >>> store.load("rec", "Cam01", 7, 11).shape                  # doctest: +SKIP
    (448, 1936)
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def path(self, recording: str, camera: str, frame: int) -> Path:
        return self.root / recording / camera / f"Frame_{int(frame)}.npz"

    def has(self, recording: str, camera: str, frame: int) -> bool:
        return self.path(recording, camera, frame).exists()

    def load(self, recording: str, camera: str, frame: int, ann_id: int):
        """Mask for one annotation, or None when absent."""
        p = self.path(recording, camera, frame)
        if not p.exists():
            return None
        with np.load(p) as z:
            ids = z["ann_ids"].tolist()
            if int(ann_id) not in ids:
                return None
            return z["masks"][ids.index(int(ann_id))].astype(bool)

    def save(self, recording: str, camera: str, frame: int, masks: dict[int, np.ndarray]) -> None:
        p = self.path(recording, camera, frame)
        p.parent.mkdir(parents=True, exist_ok=True)
        ids = sorted(masks)
        np.savez_compressed(
            p,
            ann_ids=np.array(ids, np.int64),
            masks=np.stack([np.asarray(masks[i], bool) for i in ids]),
        )

    def files(self):
        return sorted(self.root.rglob("Frame_*.npz"))

    def coverage(self, n_expected: int) -> float:
        """Fraction of `n_expected` images that have a mask file."""
        return len(self.files()) / n_expected if n_expected else 0.0

    def write_index(self, *, tool: str, source: str | None = None) -> dict:
        """Write `index.json` describing the layer, and return it."""
        files = self.files()
        index = {
            "n_files": len(files),
            "tool": tool,
            "source": source,
            "recordings": sorted({p.relative_to(self.root).parts[0] for p in files}),
        }
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "index.json").write_text(json.dumps(index, indent=2))
        return index
