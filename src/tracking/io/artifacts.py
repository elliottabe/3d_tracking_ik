"""Self-describing per-bout artifacts, indexed by name."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tracking.io.names import Order, OrderMismatch, order_sha, require_same

STAMP_KEYS = ("kp_names", "cameras", "order_sha")


class MissingOrderStamp(OrderMismatch):
    """An artifact carries no ordering metadata, so its axes cannot be named."""


class ArrayNameCollision(ValueError):
    """A caller's array name collides with a reserved ordering-stamp key."""


class NamedArray:
    """An ndarray with one or more named axes."""

    __slots__ = ("_values", "_axes")

    def __init__(self, values: np.ndarray, axes: Mapping[int, Order]) -> None:
        values = np.asarray(values)
        for axis, order in axes.items():
            if values.shape[axis] != len(order):
                raise OrderMismatch(
                    f"axis {axis} has length {values.shape[axis]} but its order "
                    f"names {len(order)} entries"
                )
        self._values = values
        self._axes = dict(sorted(axes.items()))

    def __getitem__(self, key: str | tuple[str, ...]) -> np.ndarray:
        names = (key,) if isinstance(key, str) else key
        if not isinstance(names, tuple) or not all(isinstance(n, str) for n in names):
            raise TypeError(
                "index a named axis by name, not position; "
                f"got {key!r}. Use `.values` for raw positional access."
            )
        if len(names) != len(self._axes):
            raise OrderMismatch(
                f"this array has {len(self._axes)} named axes, got {len(names)} name(s)"
            )
        idx: list[Any] = [slice(None)] * self._values.ndim
        for (axis, order), name in zip(self._axes.items(), names, strict=False):
            idx[axis] = order.index(name)
        return self._values[tuple(idx)]

    @property
    def values(self) -> np.ndarray:
        """The raw array. Explicit so a positional access is greppable."""
        return self._values

    @property
    def shape(self) -> tuple[int, ...]:
        return self._values.shape

    def names(self, axis: int) -> tuple[str, ...]:
        return self._axes[axis].names

    def __repr__(self) -> str:
        axes = ", ".join(f"{a}:{len(o)}" for a, o in self._axes.items())
        return f"NamedArray(shape={self._values.shape}, named_axes={{{axes}}})"


def _atomic_savez(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    np.savez_compressed(tmp, **payload)
    produced = tmp if tmp.exists() else tmp.with_suffix(tmp.suffix + ".npz")
    os.replace(produced, path)


def save_npz(
    path,
    *,
    arrays: Mapping[str, np.ndarray],
    kp: Order,
    cams: Order | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Write arrays plus the ordering stamp, atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {k: np.asarray(v) for k, v in arrays.items()}
    overlap = set(payload) & set(STAMP_KEYS)
    if overlap:
        raise ArrayNameCollision(f"array names collide with the ordering stamp: {sorted(overlap)}")
    payload["kp_names"] = np.array(kp.names, dtype=object)
    payload["cameras"] = np.array(cams.names if cams else (), dtype=object)
    payload["order_sha"] = np.array(order_sha(kp, cams))
    for k, v in (extra or {}).items():
        payload[k] = np.array(v) if not isinstance(v, (str, bytes)) else np.array(v)
    _atomic_savez(path, payload)


def load_npz(path, *, kp: Order, cams: Order | None = None) -> dict[str, Any]:
    """Load an artifact, verifying its ordering stamp against the run's orders."""
    z = np.load(str(path), allow_pickle=True)
    if not all(k in z.files for k in STAMP_KEYS):
        missing = [k for k in STAMP_KEYS if k not in z.files]
        raise MissingOrderStamp(
            f"{path} has no order stamp (missing {missing}); its keypoint and camera "
            f"axes cannot be named and must not be guessed"
        )
    require_same(Order(list(z["kp_names"])), kp, what="keypoint")
    stored_cams = [str(c) for c in z["cameras"]]
    stored_order = Order(stored_cams) if stored_cams else None
    if cams is not None and stored_order is not None:
        require_same(stored_order, cams, what="camera")
    want = order_sha(kp, stored_order)
    got = str(z["order_sha"])
    if got != want:
        raise OrderMismatch(f"{path} order_sha {got} != expected {want}")
    out: dict[str, Any] = {}
    for k in z.files:
        if k in STAMP_KEYS:
            continue
        arr = z[k]
        out[k] = arr.item() if arr.ndim == 0 else arr
    return out


@dataclass(frozen=True)
class BoutArtifact:
    """One bout-fly's on-disk arrays, with named axes."""

    bout_dir: Path
    fly: int
    kp: Order
    cams: Order
    _kp3d: dict
    _kp2d: dict | None
    meta: dict

    @classmethod
    def open(cls, bout_dir, fly: int, *, kp: Order, cams: Order) -> BoutArtifact:
        d = Path(bout_dir) / f"fly{fly}"
        kp3d = load_npz(d / "kp3d.npz", kp=kp, cams=None)
        kp2d_path = d / "kp2d.npz"
        # None (not {}) marks "absent" distinctly from "present but empty", so
        # .kp2d/.conf can name the missing file instead of raising a bare KeyError.
        kp2d = load_npz(kp2d_path, kp=kp, cams=cams) if kp2d_path.exists() else None
        # mvq_meta.json and sex.json describe the BOUT (both flies), not one fly.
        meta_path = Path(bout_dir) / "mvq_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return cls(Path(bout_dir), fly, kp, cams, kp3d, kp2d, meta)

    def _require_kp2d(self, field: str) -> dict:
        if self._kp2d is None:
            raise FileNotFoundError(
                f"{self.bout_dir}/fly{self.fly}/kp2d.npz not found; "
                f"'{field}' is unavailable for this bout-fly"
            )
        return self._kp2d

    @property
    def kp3d(self) -> NamedArray:
        return NamedArray(self._kp3d["kp3d"], {1: self.kp})

    @property
    def conf3d(self) -> NamedArray:
        return NamedArray(self._kp3d["conf3d"], {1: self.kp})

    @property
    def kp2d(self) -> NamedArray:
        return NamedArray(self._require_kp2d("kp2d")["kp2d"], {1: self.cams, 2: self.kp})

    @property
    def conf(self) -> NamedArray:
        return NamedArray(self._require_kp2d("conf")["conf"], {1: self.cams, 2: self.kp})

    @property
    def gates(self) -> str:
        return str(self._kp3d.get("gates", ""))
