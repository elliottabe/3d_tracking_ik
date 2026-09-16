"""The unified training root's on-disk JSON: manifest, sources, instances."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from tracking.io.names import Order, OrderMismatch

SOURCE_KINDS = ("human", "pseudo", "negative", "singlefly")
SPLITS = ("train", "val")


@dataclass(frozen=True)
class SourceEntry:
    """One row of `manifest.sources`: where a set of framesets came from."""

    id: str
    kind: str
    weight: float = 1.0
    origin: str | None = None
    checkpoint: str | None = None
    gates: dict = field(default_factory=dict)
    review: dict = field(default_factory=dict)
    n_framesets: int = 0
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in SOURCE_KINDS:
            raise ValueError(f"source kind {self.kind!r} not in {SOURCE_KINDS}")

    def to_dict(self) -> dict:
        d = {"kind": self.kind, "weight": self.weight, "n_framesets": self.n_framesets}
        for k in ("origin", "checkpoint"):
            if getattr(self, k) is not None:
                d[k] = getattr(self, k)
        for k in ("gates", "review", "extra"):
            if getattr(self, k):
                d[k] = getattr(self, k)
        return d

    @classmethod
    def from_dict(cls, sid: str, d: dict) -> SourceEntry:
        return cls(
            id=sid,
            kind=d["kind"],
            weight=float(d.get("weight", 1.0)),
            origin=d.get("origin"),
            checkpoint=d.get("checkpoint"),
            gates=d.get("gates", {}),
            review=d.get("review", {}),
            n_framesets=int(d.get("n_framesets", 0)),
            extra=d.get("extra", {}),
        )


def instances_path(root: str | Path, split: str) -> Path:
    if split not in SPLITS:
        raise ValueError(f"split {split!r} not in {SPLITS}")
    return Path(root) / "annotations" / f"instances_{split}.json"


def load_instances(root: str | Path, split: str) -> dict:
    """Read `annotations/instances_<split>.json`.

    >>> coco = load_instances("dataset", "train")      # doctest: +SKIP
    >>> sorted(coco)[:3]                               # doctest: +SKIP
    ['annotations', 'categories', 'framesets']
    """
    return json.loads(instances_path(root, split).read_text())


def save_instances(root: str | Path, split: str, coco: dict) -> None:
    p = instances_path(root, split)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(coco))


def load_manifest(root: str | Path) -> dict:
    return json.loads((Path(root) / "manifest.json").read_text())


def save_manifest(root: str | Path, manifest: dict) -> None:
    (Path(root) / "manifest.json").write_text(json.dumps(manifest, indent=2))


def keypoint_order(root: str | Path) -> Order:
    """Canonical keypoint order, cross-checked against every split's copy.

    >>> keypoint_order("dataset")                      # doctest: +SKIP
    Order(names=('Antenna_Base', 'EyeL', ...))
    """
    root = Path(root)
    canon = json.loads((root / "annotations" / "keypoint_names.json").read_text())
    for split in SPLITS:
        p = instances_path(root, split)
        if not p.exists():
            continue
        names = json.loads(p.read_text())["keypoint_names"]
        if names != canon:
            raise OrderMismatch(
                f"instances_{split}.json keypoint_names != annotations/keypoint_names.json"
            )
    return Order(canon)


def frameset_field(manifest: dict, frameset: dict, recording: str, key: str, default):
    """Resolve `key`: frameset, then the recording's manifest entry, then root, then default.

    >>> frameset_field({"weight": 0.3}, {}, "r", "weight", 1.0)
    0.3
    """
    v = frameset.get(key)
    if v is not None:
        return v
    rec = manifest.get("recordings", {}).get(recording, {})
    v = rec.get(key)
    if v is not None:
        return v
    v = manifest.get(key)
    return default if v is None else v


def sources(manifest: dict) -> dict[str, SourceEntry]:
    """`manifest.sources` as `SourceEntry` objects, keyed by id."""
    return {k: SourceEntry.from_dict(k, v) for k, v in manifest.get("sources", {}).items()}
