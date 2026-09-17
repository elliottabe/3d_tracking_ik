"""Merge training tiers into one unified root with per-frameset provenance."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
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
    """Copy `src` to `dst` following symlinks; a no-op if `dst` already exists.

    >>> import tempfile
    >>> d = Path(tempfile.mkdtemp())
    >>> (d / "src.jpg").write_bytes(b"x")
    1
    >>> _copy_image(d / "src.jpg", d / "out" / "src.jpg")
    >>> (d / "out" / "src.jpg").read_bytes()
    b'x'
    """
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=dst.parent, prefix=f".{dst.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out_f, open(src, "rb") as in_f:
            shutil.copyfileobj(in_f, out_f)
        os.replace(tmp_path, dst)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _copy_images(jobs: Iterable[tuple[Path, Path]], workers: int) -> None:
    """Copy each (src, dst) pair through a thread pool, once per unique destination.

    >>> _copy_images([], 4)
    """
    by_dst: dict[Path, Path] = {}
    for src, dst in jobs:
        by_dst.setdefault(dst, src)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda item: _copy_image(item[1], item[0]), by_dst.items()))


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


def _image_stem(rel: str | Path) -> str:
    """A frame's identity across file types: its relative path without the suffix.

    >>> _image_stem("rec/Cam01/Frame_7.jpg"), _image_stem(Path("rec/Cam01/Frame_7.npz"))
    ('rec/Cam01/Frame_7', 'rec/Cam01/Frame_7')
    """
    return str(Path(rel).with_suffix(""))


def _merge_mask_group(
    dst: Path, sources: list[tuple[Path, dict, str | None]]
) -> tuple[int, int, int, bool]:
    """Fold one destination's per-tier npz sources in order; return counts and whether it wrote.

    Each source is (npz, the tier's {(split, old id): new id} map, the split its
    frame belongs to). A frame in none of the tier's split files carries
    `split=None`, which drops every row.

    >>> _merge_mask_group(Path("x"), [])
    (0, 0, 0, False)
    """
    rows_remapped = rows_dropped = collisions = 0
    merged_ids = merged_masks = merged_matched = None
    for p, ann_map, split in sources:
        with np.load(p) as z:
            old_ids = z["ann_ids"]
            keep = np.array([(split, int(a)) in ann_map for a in old_ids], dtype=bool)
            rows_dropped += int((~keep).sum())
            if not keep.any():
                continue
            new_ids = np.array([ann_map[(split, int(a))] for a in old_ids[keep]], dtype=np.int64)
            kept_masks = z["masks"][keep]
            kept_matched = (
                z["matched"][keep] if "matched" in z.files else np.ones(len(new_ids), bool)
            )
        rows_remapped += len(new_ids)
        if merged_ids is None:
            merged_ids, merged_masks, merged_matched = new_ids, kept_masks, kept_matched
        else:
            collisions += 1
            merged_ids = np.concatenate([merged_ids, new_ids])
            merged_masks = np.concatenate([merged_masks, kept_masks])
            merged_matched = np.concatenate([merged_matched, kept_matched])
    if merged_ids is None:
        return rows_remapped, rows_dropped, collisions, False
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, ann_ids=merged_ids, masks=merged_masks, matched=merged_matched)
    return rows_remapped, rows_dropped, collisions, True


def _merge_masks(tiers, ann_maps, img_splits, out_root, workers=64) -> dict:
    """Remap each tier's mask sidecar into `out_root`'s id space, merging on path collision.

    A mask file's annotation ids mean whatever the split of its own frame says
    they mean, so each file is resolved through `img_splits[tier][stem]` -- the
    split that owns `<recording>/<camera>/Frame_<n>` in that tier.

    >>> _merge_masks([], {}, {}, "unified", 4)
    {'files_written': 0, 'rows_remapped': 0, 'rows_dropped': 0, 'collisions_merged': 0}
    """
    dst_store = MaskStore(sidecar_path(out_root))
    groups: dict[Path, list[tuple[Path, dict, str]]] = {}
    for tier in tiers:
        src_dir = Path(tier.path) / "masks"
        if not src_dir.is_dir():
            continue
        ann_map = ann_maps[tier.source_id]
        splits = img_splits[tier.source_id]
        for p in sorted(src_dir.rglob("Frame_*.npz")):
            rel = p.relative_to(src_dir)
            if len(rel.parts) != 3:
                raise ValueError(f"{p}: expected <recording>/<camera>/Frame_<n>.npz")
            recording, camera, fname = rel.parts
            frame = int(fname.removeprefix("Frame_").removesuffix(".npz"))
            dst = dst_store.path(recording, camera, frame)
            groups.setdefault(dst, []).append((p, ann_map, splits.get(_image_stem(rel))))

    stats = {"files_written": 0, "rows_remapped": 0, "rows_dropped": 0, "collisions_merged": 0}
    if workers == 1:
        results = (_merge_mask_group(*item) for item in groups.items())
        for rows_remapped, rows_dropped, collisions, wrote in results:
            stats["rows_remapped"] += rows_remapped
            stats["rows_dropped"] += rows_dropped
            stats["collisions_merged"] += collisions
            stats["files_written"] += int(wrote)
        return stats
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for rows_remapped, rows_dropped, collisions, wrote in pool.map(
            lambda item: _merge_mask_group(*item), groups.items()
        ):
            stats["rows_remapped"] += rows_remapped
            stats["rows_dropped"] += rows_dropped
            stats["collisions_merged"] += collisions
            stats["files_written"] += int(wrote)
    return stats


def merge_tiers(
    tiers, out_root, *, splits=schema.SPLITS, copy_images=True, masks=True, workers=32
) -> dict:
    """Merge `tiers` into `out_root`; return the written manifest.

    Image and annotation ids are unique across the root, not just within one
    split file -- the mask sidecar keys rows by annotation id alone. A tier's
    own ids are only unique within its split file, so a mask row is remapped
    through the (split, old id) its own frame belongs to; an image that appears
    in two of a tier's split files raises.

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
    tier_ann_maps: dict[str, dict[tuple[str, int], int]] = {i: {} for i in ids}
    tier_img_splits: dict[str, dict[str, str]] = {i: {} for i in ids}

    next_img = next_ann = 1
    for split in splits:
        images, annotations, framesets = [], [], {}
        for tier in tiers:
            try:
                coco = schema.load_instances(tier.path, split)
            except FileNotFoundError:
                continue
            img_map, ann_map = {}, {}
            owner = tier_img_splits[tier.source_id]
            for im in coco["images"]:
                img_map[im["id"]] = next_img
                images.append({**im, "id": next_img})
                next_img += 1
                name = im.get("file_name")
                if name is None:
                    continue
                stem = _image_stem(name)
                if owner.setdefault(stem, split) != split:
                    raise ValueError(
                        f"tier {tier.source_id!r} image {name!r} is in both the "
                        f"{owner[stem]!r} and {split!r} split files; its mask rows cannot "
                        f"be resolved to one annotation id space"
                    )
            if copy_images:
                _copy_images(
                    (
                        (
                            Path(tier.path) / "images" / im["file_name"],
                            out_root / "images" / im["file_name"],
                        )
                        for im in coco["images"]
                    ),
                    workers,
                )
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
            tier_ann_maps[tier.source_id].update(
                {(split, old): new for old, new in ann_map.items()}
            )
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
        manifest["masks"] = _merge_masks(tiers, tier_ann_maps, tier_img_splits, out_root, workers)
    schema.save_manifest(out_root, manifest)
    return manifest
