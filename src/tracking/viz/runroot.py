"""Reading a finished run root without the config that produced it.

A run root is self-describing: its artifacts stamp the keypoint order they
were solved with and the flies they cover. A consumer -- a renderer, a
notebook -- can therefore recover the anatomy a run used from the run alone,
with no Hydra config, no camera rig, and no notion of which assay it came
from. Courtship, single-animal and amputation runs differ only in what these
functions report: two flies or one, 50 keypoints or 44.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np

__all__ = [
    "discover_bouts",
    "discover_flies",
    "load_bout_observed",
    "load_bout_qpos",
    "load_run_anatomy",
    "outputs_path",
    "stac_path",
    "run_kp_names",
]

_FLY_RE = re.compile(r"^fly(\d+)$")
_BOUT_RE = re.compile(r"^bout_(\d+)$")


def _decode(names) -> list[str]:
    return [n.decode() if isinstance(n, bytes) else str(n) for n in names]


def outputs_path(run_root, *, bout: int, fly: int) -> Path:
    """Where one bout-fly's `outputs.h5` lives."""
    return Path(run_root) / "bouts" / f"bout_{int(bout):05d}" / f"fly{int(fly)}" / "outputs.h5"


def discover_flies(run_root) -> list[int]:
    """Every fly index this run solved, ascending. One entry means single-animal."""
    run_root = Path(run_root)
    flies = {
        int(m.group(1))
        for d in run_root.glob("bouts/bout_*/fly*")
        if d.is_dir() and (m := _FLY_RE.match(d.name))
    }
    if not flies:
        raise FileNotFoundError(
            f"{run_root} has no bouts/bout_*/fly* directories; it is not a finished run root"
        )
    return sorted(flies)


def discover_bouts(run_root, *, fly: int | None = None) -> list[int]:
    """Every bout index, ascending; with `fly`, only the bouts that fly solved."""
    run_root = Path(run_root)
    bouts = set()
    for d in run_root.glob("bouts/bout_*"):
        m = _BOUT_RE.match(d.name)
        if not (m and d.is_dir()):
            continue
        idx = int(m.group(1))
        if fly is None or outputs_path(run_root, bout=idx, fly=fly).exists():
            bouts.add(idx)
    return sorted(bouts)


def run_kp_names(run_root, *, fly: int) -> list[str]:
    """The keypoint order this run was solved with, read from its own artifacts.

    Taken from `offsets_fly{fly}.h5`, which is written once per run and carries
    the order every downstream artifact shares. That file is also what supplies
    the marker offsets, so a run that cannot answer this cannot be rendered
    anyway.
    """
    path = Path(run_root) / f"offsets_fly{int(fly)}.h5"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; this run has no fitted fly{fly}")
    with h5py.File(path, "r") as f:
        return _decode(f["kp_names"][:])


def load_bout_qpos(run_root, *, bout: int, fly: int) -> np.ndarray:
    """`(T, nq)` solved pose for one bout-fly."""
    path = outputs_path(run_root, bout=bout, fly=fly)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist; bout_{int(bout):05d} has no fly{fly} solve"
        )
    with h5py.File(path, "r") as f:
        return np.asarray(f["qpos"][:], dtype=np.float64)


def stac_path(run_root, *, bout: int, fly: int) -> Path:
    """Where one bout-fly's `stac_ik.h5` lives."""
    return Path(run_root) / "bouts" / f"bout_{int(bout):05d}" / f"fly{int(fly)}" / "stac_ik.h5"


def load_bout_observed(run_root, *, bout: int, fly: int) -> np.ndarray | None:
    """`(T, K, 3)` observed keypoints IN MODEL SPACE, or None if absent.

    Read from `stac_ik.h5:kp_data`, NOT from `outputs.h5:kp3d_mm`. The latter is
    in millimetres and runs ~90x larger than the model, so drawing it into a
    MuJoCo scene scatters the cloud far outside the body instead of onto it --
    a wrong picture rather than an error.
    """
    path = stac_path(run_root, bout=bout, fly=fly)
    if not path.exists():
        return None
    with h5py.File(path, "r") as f:
        if "kp_data" not in f:
            return None
        kp = np.asarray(f["kp_data"][:], dtype=np.float64)
    return kp.reshape(kp.shape[0], -1, 3)


def load_run_anatomy(
    run_root, *, fly: int, anatomy_cfg, tracked_kp_names: Sequence[str] | None = None
):
    """The anatomy this run used, with that fly's fitted marker offsets applied.

    `tracked_kp_names` defaults to the run's own order, so a 44-keypoint
    amputation run gets a 44-keypoint anatomy rather than the config's 50. A
    mismatch here is not loud: `site_idxs` would be the wrong length and the
    offsets would land on the wrong markers.
    """
    from omegaconf import OmegaConf

    from tracking.inverse_kinematics.anatomy import load_anatomy
    from tracking.inverse_kinematics.offsets_fit import load_offsets

    run_root = Path(run_root)
    names = (
        list(tracked_kp_names) if tracked_kp_names is not None else run_kp_names(run_root, fly=fly)
    )

    cfg = anatomy_cfg
    if isinstance(cfg, (str, Path)):
        cfg = OmegaConf.load(str(cfg))
    anatomy = load_anatomy(cfg, tracked_kp_names=names, strict=False)

    offsets = load_offsets(run_root / f"offsets_fly{int(fly)}.h5", kp_order=anatomy.kp_order)
    anatomy.mj_model.site_pos[anatomy.site_idxs] = offsets
    return anatomy
