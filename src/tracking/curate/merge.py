"""Merge training tiers into one unified root with per-frameset provenance."""

from __future__ import annotations

import filecmp
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from tracking.curate import schema
from tracking.curate.calib import calib_fingerprint


@dataclass(frozen=True)
class TierSpec:
    """One input tier and the provenance it contributes."""

    path: str | Path
    source_id: str
    kind: str = "human"
    weight: float = 1.0
    checkpoint: str | None = None
    gates: dict = field(default_factory=dict)
    review: dict = field(default_factory=dict)


def _copy_image(src: Path, dst: Path) -> None:
    """Copy following symlinks; refuse a byte-level conflict at the same path."""
    if dst.exists():
        if not filecmp.cmp(src, dst, shallow=False):
            raise ValueError(f"{dst.name} differs between tiers at the same path: {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)  # follows symlinks


def _merge_calibrations(tiers, out_root):
    """Union calibration groups by fingerprint; return {(tier, old_group): new_group}."""
    out_calib = Path(out_root) / "calibrations"
    out_calib.mkdir(parents=True, exist_ok=True)
    by_fp: dict[str, str] = {}
    mapping: dict[tuple[str, str], str] = {}
    for tier in tiers:
        for d in sorted((Path(tier.path) / "calibrations").glob("*")):
            if not d.is_dir():
                continue
            fp = calib_fingerprint(d)
            if fp not in by_fp:
                label = chr(ord("A") + len(by_fp))
                by_fp[fp] = label
                shutil.copytree(d, out_calib / label, dirs_exist_ok=True)
            mapping[(tier.source_id, d.name)] = by_fp[fp]
    return mapping


def merge_tiers(tiers, out_root, *, splits=schema.SPLITS, copy_images=True) -> dict:
    """Merge `tiers` into `out_root`; return the written manifest.

    >>> merge_tiers([TierSpec("human_root", "human_v12", "human")], "unified")  # doctest: +SKIP
    {'version': 'unified', 'sources': {...}, ...}
    """
    tiers = list(tiers)
    ids = [t.source_id for t in tiers]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate source_id among tiers: {ids}")

    canon = None
    for tier in tiers:
        names = schema.keypoint_order(tier.path).names
        if canon is None:
            canon = names
        elif names != canon:
            raise ValueError(
                f"tier {tier.source_id!r} keypoint_names differ from {tiers[0].source_id!r}"
            )

    out_root = Path(out_root)
    (out_root / "annotations").mkdir(parents=True, exist_ok=True)
    calib_map = _merge_calibrations(tiers, out_root)

    sources: dict[str, dict] = {}
    recordings: dict[str, dict] = {}
    counts = dict.fromkeys(ids, 0)

    for split in splits:
        images, annotations, framesets = [], [], {}
        next_img = next_ann = 1
        for tier in tiers:
            try:
                coco = schema.load_instances(tier.path, split)
            except FileNotFoundError:
                continue
            man = schema.load_manifest(tier.path)
            img_map, ann_map = {}, {}
            for im in coco["images"]:
                img_map[im["id"]] = next_img
                images.append({**im, "id": next_img})
                if copy_images:
                    _copy_image(
                        Path(tier.path) / "images" / im["file_name"],
                        out_root / "images" / im["file_name"],
                    )
                next_img += 1
            for an in coco["annotations"]:
                ann_map[an["id"]] = next_ann
                annotations.append({**an, "id": next_ann, "image_id": img_map[an["image_id"]]})
                next_ann += 1
            for key, fs in coco["framesets"].items():
                rec = fs["recording"]
                old_group = man.get("recordings", {}).get(rec, {}).get("calib_group", "A")
                recordings.setdefault(rec, {})["calib_group"] = calib_map[
                    (tier.source_id, old_group)
                ]
                framesets[f"{tier.source_id}/{key}"] = {
                    **fs,
                    "frames": [img_map[i] for i in fs["frames"]],
                    "ann_ids": [None if i is None else ann_map[i] for i in fs["ann_ids"]],
                    "source_id": tier.source_id,
                    "source": fs.get("source", tier.kind),
                    "weight": fs.get("weight", tier.weight),
                }
                counts[tier.source_id] += 1
        schema.save_instances(
            out_root,
            split,
            {
                "keypoint_names": list(canon),
                "skeleton": [],
                "categories": [],
                "images": images,
                "annotations": annotations,
                "framesets": framesets,
            },
        )

    (out_root / "annotations" / "keypoint_names.json").write_text(json.dumps(list(canon)))
    for tier in tiers:
        sources[tier.source_id] = schema.SourceEntry(
            id=tier.source_id,
            kind=tier.kind,
            weight=tier.weight,
            origin=str(tier.path),
            checkpoint=tier.checkpoint,
            gates=tier.gates,
            review=tier.review,
            n_framesets=counts[tier.source_id],
        ).to_dict()
    manifest = {
        "version": out_root.name,
        "sources": sources,
        "calib_groups": sorted({v for v in calib_map.values()}),
        "recordings": recordings,
    }
    schema.save_manifest(out_root, manifest)
    return manifest
