"""Merge training tiers into one unified root with per-frameset provenance."""

from __future__ import annotations

import filecmp
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tracking.curate import schema
from tracking.curate.calib import calib_fingerprint
from tracking.curate.masks.store import MaskStore, sidecar_path


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


def _preflight_calib_groups(tiers, calib_map):
    """Per-tier calib_group for every recording, from manifests alone; no images touched.

    >>> _preflight_calib_groups([], {})
    {}
    """
    resolved: dict[tuple[str, str], str] = {}
    for tier in tiers:
        man = schema.load_manifest(tier.path)
        for rec, rec_manifest in man.get("recordings", {}).items():
            group = rec_manifest.get("calib_group") if isinstance(rec_manifest, dict) else None
            if group is None:
                raise ValueError(
                    f"tier {tier.source_id!r} recording {rec!r} has no calib_group in its manifest"
                )
            resolved[(tier.source_id, rec)] = calib_map[(tier.source_id, group)]
    return resolved


def _merge_masks(tiers, ann_maps, out_root) -> dict:
    """Remap each tier's mask sidecar into `out_root`'s id space, merging on path collision.

    >>> _merge_masks([], {}, "unified")  # doctest: +SKIP
    {'files_written': 0, 'rows_remapped': 0, 'rows_dropped': 0, 'collisions_merged': 0}
    """
    dst_store = MaskStore(sidecar_path(out_root))
    stats = {"files_written": 0, "rows_remapped": 0, "rows_dropped": 0, "collisions_merged": 0}
    written: set[Path] = set()
    for tier in tiers:
        src_dir = Path(tier.path) / "masks"
        if not src_dir.is_dir():
            continue
        ann_map = ann_maps[tier.source_id]
        for p in sorted(src_dir.rglob("Frame_*.npz")):
            rel = p.relative_to(src_dir)
            if len(rel.parts) != 3:
                raise ValueError(f"{p}: expected <recording>/<camera>/Frame_<n>.npz")
            recording, camera, fname = rel.parts
            frame = int(fname.removeprefix("Frame_").removesuffix(".npz"))
            with np.load(p) as z:
                old_ids = z["ann_ids"]
                keep = np.array([int(a) in ann_map for a in old_ids], dtype=bool)
                stats["rows_dropped"] += int((~keep).sum())
                if not keep.any():
                    continue
                new_ids = np.array([ann_map[int(a)] for a in old_ids[keep]], dtype=np.int64)
                kept_masks = z["masks"][keep]
                kept_matched = (
                    z["matched"][keep] if "matched" in z.files else np.ones(len(new_ids), bool)
                )
            stats["rows_remapped"] += len(new_ids)
            dst = dst_store.path(recording, camera, frame)
            if dst.exists():
                stats["collisions_merged"] += 1
                with np.load(dst) as prior:
                    new_ids = np.concatenate([prior["ann_ids"], new_ids])
                    kept_masks = np.concatenate([prior["masks"], kept_masks])
                    kept_matched = np.concatenate([prior["matched"], kept_matched])
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dst, ann_ids=new_ids, masks=kept_masks, matched=kept_matched)
            written.add(dst)
    stats["files_written"] = len(written)
    return stats


def merge_tiers(tiers, out_root, *, splits=schema.SPLITS, copy_images=True, masks=True) -> dict:
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
    resolved_calib = _preflight_calib_groups(tiers, calib_map)

    sources: dict[str, dict] = {}
    recording_groups: dict[str, set[str]] = {}
    counts = dict.fromkeys(ids, 0)
    tier_ann_maps: dict[str, dict[int, int]] = {i: {} for i in ids}

    for split in splits:
        images, annotations, framesets = [], [], {}
        next_img = next_ann = 1
        for tier in tiers:
            try:
                coco = schema.load_instances(tier.path, split)
            except FileNotFoundError:
                continue
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
                try:
                    new_group = resolved_calib[(tier.source_id, rec)]
                except KeyError:
                    raise ValueError(
                        f"tier {tier.source_id!r} recording {rec!r} is not in its "
                        f"manifest's recordings"
                    ) from None
                recording_groups.setdefault(rec, set()).add(new_group)
                framesets[f"{tier.source_id}/{key}"] = {
                    **fs,
                    "frames": [img_map[i] for i in fs["frames"]],
                    "ann_ids": [None if i is None else ann_map[i] for i in fs["ann_ids"]],
                    "source_id": tier.source_id,
                    "source": fs.get("source", tier.kind),
                    "weight": fs.get("weight", tier.weight),
                    "calib_group": new_group,
                }
                counts[tier.source_id] += 1
            tier_ann_maps[tier.source_id].update(ann_map)
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
    recordings: dict[str, dict] = {}
    for rec, groups in recording_groups.items():
        if len(groups) == 1:
            recordings[rec] = {"calib_group": next(iter(groups))}
        else:
            recordings[rec] = {"calib_groups": sorted(groups)}
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
    if masks:
        manifest["masks"] = _merge_masks(tiers, tier_ann_maps, out_root)
    schema.save_manifest(out_root, manifest)
    return manifest
