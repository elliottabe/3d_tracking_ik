"""Check a unified training root. Returns findings; never raises on bad data."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tracking.curate import schema
from tracking.curate.masks.store import MaskStore
from tracking.geometry.rig import CameraRig
from tracking.io.names import OrderMismatch

MIN_CAMS = 3
LEVELS = ("error", "warning", "info")
MASK_SAMPLE = 200


@dataclass(frozen=True)
class Finding:
    level: str
    code: str
    message: str


def _err(code, msg):
    return Finding("error", code, msg)


def _check_keypoints(root, out):
    try:
        return schema.keypoint_order(root)
    except OrderMismatch as exc:
        out.append(_err("keypoint_names_mismatch", str(exc)))
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        out.append(_err("keypoint_names_unreadable", str(exc)))
    return None


def _check_calibrations(root, out):
    """Affine row, one Cam*.yaml per camera, camera names equal across groups."""
    names_seen = {}
    for group_dir in sorted((Path(root) / "calibrations").glob("*")):
        if not group_dir.is_dir():
            continue
        try:
            rig = CameraRig.from_calib_dir(group_dir)
        except (OrderMismatch, ValueError, OSError, SystemError) as exc:
            out.append(_err("calibration_unreadable", f"{group_dir.name}: {exc}"))
            continue
        names_seen[group_dir.name] = rig.cameras.names
        for cam, m in zip(rig.cameras.names, rig.matrices_f64, strict=True):
            if not np.allclose(m[2], [0.0, 0.0, 0.0, 1.0]):
                out.append(
                    _err(
                        "calibration_not_affine",
                        f"{group_dir.name}/{cam}: projection row 2 is {m[2].tolist()}, "
                        f"expected [0, 0, 0, 1]",
                    )
                )
    if len(set(names_seen.values())) > 1:
        out.append(_err("camera_names_differ", f"calibration groups disagree: {names_seen}"))
    return names_seen


def _check_split(root, split, kp_order, sources, recordings, out):
    """Check one split's instances file; corrupt JSON is already reported by `_check_keypoints`."""
    try:
        coco = schema.load_instances(root, split)
    except (OSError, json.JSONDecodeError):
        return set()
    missing_keys = [k for k in ("images", "annotations", "framesets") if k not in coco]
    if missing_keys:
        out.append(_err("split_missing_keys", f"{split}: missing key(s) {missing_keys}"))
    n_kp = len(kp_order) if kp_order is not None else None

    raw_images = coco.get("images", [])
    good_images = [i for i in raw_images if "id" in i and "file_name" in i]
    if len(good_images) != len(raw_images):
        out.append(
            _err(
                "malformed_row",
                f"{split}: {len(raw_images) - len(good_images)} image row(s) missing id/file_name",
            )
        )
    raw_anns = coco.get("annotations", [])
    good_anns = [a for a in raw_anns if "id" in a]
    if len(good_anns) != len(raw_anns):
        out.append(
            _err(
                "malformed_row",
                f"{split}: {len(raw_anns) - len(good_anns)} annotation row(s) missing id",
            )
        )
    images = {i["id"]: i for i in good_images}
    anns = {a["id"]: a for a in good_anns}

    for a in good_anns:
        if n_kp is not None and len(a.get("keypoints", [])) != 3 * n_kp:
            out.append(
                _err(
                    "keypoints_length",
                    f"{split} annotation {a['id']}: {len(a.get('keypoints', []))} values, "
                    f"expected {3 * n_kp}",
                )
            )
            break

    n_symlink, first_symlink = 0, None
    for rel, img_id in ((i["file_name"], i["id"]) for i in good_images):
        p = Path(root) / "images" / rel
        if p.is_symlink():
            n_symlink += 1
            first_symlink = first_symlink or (rel, img_id)
    if n_symlink:
        rel, img_id = first_symlink
        out.append(
            _err(
                "image_is_symlink",
                f"{split} image {img_id}: {rel} is a symlink (first offender, {n_symlink} total)",
            )
        )

    n_missing, first_missing = 0, None
    for rel, img_id in ((i["file_name"], i["id"]) for i in good_images):
        p = Path(root) / "images" / rel
        if not p.is_symlink() and not p.exists():
            n_missing += 1
            first_missing = first_missing or (rel, img_id)
    if n_missing:
        rel, img_id = first_missing
        out.append(
            _err(
                "image_missing",
                f"{split} image {img_id}: {rel} not on disk (first offender, {n_missing} total)",
            )
        )

    framesets = coco.get("framesets", {})
    if not isinstance(framesets, dict):
        out.append(
            _err(
                "malformed_row",
                f"{split}: framesets is a {type(framesets).__name__}, expected a mapping",
            )
        )
        framesets = {}

    referenced_groups: set[str] = set()
    for key, fs in framesets.items():
        frames, ann_ids = fs.get("frames", []), fs.get("ann_ids", [])
        if len(frames) != len(ann_ids):
            out.append(
                _err("ann_ids_length", f"{key}: {len(frames)} frames vs {len(ann_ids)} ann_ids")
            )
            continue
        missing = [i for i in frames if i not in images]
        if missing:
            out.append(_err("frameset_image_missing", f"{key}: unknown image ids {missing[:4]}"))
        unknown = [i for i in ann_ids if i is not None and i not in anns]
        if unknown:
            out.append(_err("frameset_ann_missing", f"{key}: unknown ann ids {unknown[:4]}"))
        if sum(i is not None for i in ann_ids) < MIN_CAMS:
            out.append(
                _err(
                    "too_few_cameras",
                    f"{key}: {sum(i is not None for i in ann_ids)} resolved "
                    f"cameras, need >= {MIN_CAMS}",
                )
            )
        if int(fs.get("fly_id", 0)) < 0 and fs.get("center3D") is None:
            out.append(_err("negative_without_center3d", f"{key}: negative carries no center3D"))
        sid = fs.get("source_id")
        if sid is not None and sid not in sources:
            out.append(_err("unknown_source_id", f"{key}: source_id {sid!r} not in manifest"))
        fs_group = fs.get("calib_group")
        if fs_group is not None:
            referenced_groups.add(fs_group)
        else:
            rec_entry = recordings.get(fs.get("recording"))
            rec_entry = rec_entry if isinstance(rec_entry, dict) else {}
            if rec_entry.get("calib_group") is None:
                out.append(
                    _err(
                        "frameset_calib_unresolvable",
                        f"{key}: no calib_group on the frameset or its recording",
                    )
                )

    per_image = Counter(a["image_id"] for a in good_anns if "image_id" in a)
    n_zero = max(0, len(images) - len(per_image))
    n_two = sum(1 for v in per_image.values() if v >= 2)
    if n_zero:
        out.append(
            Finding(
                "info", "zero_annotation_images", f"{split}: {n_zero} image(s) carry no annotation"
            )
        )
    if images:
        out.append(
            Finding(
                "info",
                "two_fly_fraction",
                f"{split}: {n_two}/{len(images)} images have >= 2 animals "
                f"({100 * n_two / len(images):.1f}%)",
            )
        )

    dupes = {name: n for name, n in Counter(i["file_name"] for i in good_images).items() if n > 1}
    if dupes:
        out.append(
            Finding(
                "warning",
                "duplicate_file_name",
                f"{split}: {len(dupes)} file_name(s) shared by multiple image rows, e.g. "
                f"{next(iter(dupes))!r} x{dupes[next(iter(dupes))]}",
            )
        )

    return referenced_groups


def _check_mask_ann_ids(root, masks_root, split, out):
    """Warn when a frameset's ann_ids have no row in the mask file of their frame.

    Reads an evenly spaced sample of at most `MASK_SAMPLE` framesets of the
    split: a real root carries ~145k sidecar files and opening them all would
    dominate the run. A frame with no mask file at all is skipped, not
    reported -- a tier without masks legitimately has none.
    """
    store = MaskStore(masks_root)
    try:
        coco = schema.load_instances(root, split)
    except (OSError, json.JSONDecodeError):
        return
    names = {
        i["id"]: i["file_name"] for i in coco.get("images", []) if "id" in i and "file_name" in i
    }
    framesets = coco.get("framesets", {})
    if not isinstance(framesets, dict):
        return
    checked = unresolved = 0
    first = None
    keys = list(framesets)
    for key in keys[:: max(1, len(keys) // MASK_SAMPLE)][:MASK_SAMPLE]:
        fs = framesets[key]
        if not isinstance(fs, dict):
            continue
        for img_id, ann_id in zip(fs.get("frames", []), fs.get("ann_ids", []), strict=False):
            rel = names.get(img_id)
            if ann_id is None or rel is None or rel.count("/") != 2:
                continue
            recording, camera, fname = rel.split("/")
            try:
                frame = int(fname.removeprefix("Frame_").rsplit(".", 1)[0])
                if not store.has(recording, camera, frame):
                    continue
                found = store.load(recording, camera, frame, ann_id) is not None
            except Exception:
                continue
            checked += 1
            unresolved += not found
            if not found and first is None:
                first = f"{key}: {rel} has no mask row for ann {ann_id}"
    if unresolved:
        out.append(
            Finding(
                "warning",
                "mask_ann_unresolved",
                f"{split}: {unresolved}/{checked} sampled frameset slot(s) have a mask "
                f"file that does not carry their annotation id ({first})",
            )
        )


def validate_root(root: str | Path, *, masks_root: str | Path | None = None) -> list[Finding]:
    """Every check from the format spec, as a flat finding list.

    >>> [f.code for f in validate_root("dataset") if f.level == "error"]   # doctest: +SKIP
    []
    """
    out: list[Finding] = []
    kp_order = _check_keypoints(root, out)
    names_seen = _check_calibrations(root, out)
    try:
        manifest = schema.load_manifest(root)
    except (OSError, json.JSONDecodeError) as exc:
        out.append(_err("manifest_unreadable", str(exc)))
        return out
    recordings = manifest.get("recordings", {})
    referenced = set(manifest.get("calib_groups", []))
    for rec, r in recordings.items():
        if not isinstance(r, dict):
            continue
        referenced.add(r.get("calib_group"))
        referenced.update(r.get("calib_groups") or [])
        if len(r.get("calib_groups", [])) > 1:
            out.append(
                Finding(
                    "info",
                    "recording_multi_calib",
                    f"{rec}: spans calib_groups {r['calib_groups']}",
                )
            )
    try:
        srcs = schema.sources(manifest)
    except (KeyError, ValueError) as exc:
        out.append(_err("manifest_sources_invalid", str(exc)))
        srcs = {}
    for sid, entry in srcs.items():
        if entry.kind in ("pseudo", "negative", "singlefly") and not entry.checkpoint:
            out.append(
                Finding(
                    "warning",
                    "source_without_checkpoint",
                    f"source {sid!r} ({entry.kind}) records no checkpoint",
                )
            )
    for split in schema.SPLITS:
        referenced |= _check_split(root, split, kp_order, srcs, recordings, out)
    for group in sorted(referenced - names_seen.keys() - {None}):
        out.append(_err("calib_group_missing", f"calib_group {group!r} has no calibrations/ dir"))
    if masks_root is not None:
        if not Path(masks_root).exists():
            out.append(
                Finding(
                    "warning",
                    "masks_root_missing",
                    f"{masks_root} does not exist; copy-paste will be off",
                )
            )
        else:
            for split in schema.SPLITS:
                _check_mask_ann_ids(root, masks_root, split, out)
    return out


def format_findings(findings: list[Finding]) -> str:
    """Human-readable report, errors first.

    >>> print(format_findings([Finding("error", "x", "bad")]))   # doctest: +SKIP
    ERROR   x: bad
    1 error, 0 warnings, 0 info
    """
    order = {lvl: i for i, lvl in enumerate(LEVELS)}
    lines = [
        f"{f.level.upper():<7} {f.code}: {f.message}"
        for f in sorted(findings, key=lambda f: order[f.level])
    ]
    counts = Counter(f.level for f in findings)
    lines.append(f"{counts['error']} error, {counts['warning']} warnings, {counts['info']} info")
    return "\n".join(lines)
