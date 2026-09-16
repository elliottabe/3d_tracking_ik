"""The ordering contract: one canonical keypoint order, one canonical camera order."""

from __future__ import annotations

import glob
import hashlib
import os
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np


class OrderMismatch(Exception):
    """A name is not in an order, or two orders disagree when they must not."""


@dataclass(frozen=True)
class Order:
    """An immutable, duplicate-free sequence of names for one array axis."""

    names: tuple[str, ...]

    def __init__(self, names: Iterable[str]) -> None:
        tup = tuple(str(n) for n in names)
        if len(set(tup)) != len(tup):
            dupes = sorted({n for n in tup if tup.count(n) > 1})
            raise OrderMismatch(f"duplicate names in order: {dupes}")
        object.__setattr__(self, "names", tup)
        object.__setattr__(self, "_index", {n: i for i, n in enumerate(tup)})

    def index(self, name: str) -> int:
        try:
            return self._index[name]  # type: ignore[attr-defined]
        except KeyError:
            raise OrderMismatch(
                f"{name!r} is not in this order; it has {len(self)} names "
                f"starting {list(self.names[:4])}"
            ) from None

    def __len__(self) -> int:
        return len(self.names)

    def __contains__(self, name: object) -> bool:
        return name in self._index  # type: ignore[attr-defined]

    def __iter__(self) -> Iterator[str]:
        return iter(self.names)

    def permutation_to(self, other: Order) -> np.ndarray:
        """Indices `p` such that `[self.names[i] for i in p] == list(other.names)`."""
        if set(self.names) != set(other.names):
            missing = sorted(set(other.names) - set(self.names))
            extra = sorted(set(self.names) - set(other.names))
            raise OrderMismatch(
                f"permutation needs the same set of names; missing {missing}, extra {extra}"
            )
        return np.array([self.index(n) for n in other.names], dtype=np.int64)

    @property
    def sha(self) -> str:
        h = hashlib.sha256("\n".join(self.names).encode("utf-8"))
        return h.hexdigest()[:16]


def load_keypoint_order(anatomy_cfg) -> Order:
    """Canonical keypoint order from an anatomy config's `model.KP_NAMES`."""
    model = anatomy_cfg["model"] if "model" in anatomy_cfg else None
    if model is None or "KP_NAMES" not in model:
        raise OrderMismatch(
            "anatomy config has no model.KP_NAMES; the canonical keypoint order "
            "cannot be inferred and must not be guessed"
        )
    return Order(model["KP_NAMES"])


def load_camera_order(calib_dir) -> Order:
    """Canonical camera order = the calibration glob order, `sorted(Cam*.yaml)`."""
    paths = sorted(glob.glob(os.path.join(str(calib_dir), "Cam*.yaml")))
    if not paths:
        raise OrderMismatch(f"no Cam*.yaml calibration files in {calib_dir}")
    return Order(os.path.splitext(os.path.basename(p))[0] for p in paths)


def order_sha(kp: Order, cams: Order | None = None) -> str:
    """16-hex digest of a keypoint order and (optionally) a camera order."""
    payload = kp.sha + ("|" + cams.sha if cams is not None else "")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def require_same(order: Order, expected: Order, *, what: str) -> None:
    """Raise unless two orders are identical, naming the first disagreement."""
    if order.names == expected.names:
        return
    for i, (a, b) in enumerate(zip(order.names, expected.names, strict=False)):
        if a != b:
            raise OrderMismatch(f"{what} order differs at position {i}: {a!r} != {b!r}")
    raise OrderMismatch(f"{what} order has {len(order)} names, expected {len(expected)}")


def as_order(value: Sequence[str] | Order) -> Order:
    """Accept either an `Order` or a bare name sequence."""
    return value if isinstance(value, Order) else Order(value)
