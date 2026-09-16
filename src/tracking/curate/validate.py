"""Check a unified training root. Returns findings; never raises on bad data."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tracking.curate import schema
from tracking.geometry.rig import CameraRig
from tracking.io.names import OrderMismatch

MIN_CAMS = 3
LEVELS = ("error", "warning", "info")


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


def _check_split(root, split, kp_order, sources, out):
    """Check one split's instances file; corrupt JSON is already reported by `_check_keypoints`."""
    try:
        coco = schema.load_instances(root, split)
    except (OSError, json.JSONDecodeError):
        return
    missing_keys = [k for k in ("images", "annotations", "framesets") if k not in coco]
    if missing_keys:
        out.append(_err("split_missing_keys", f"{split}: missing key(s) {missing_keys}"))
    images = {i["id"]: i for i in coco.get("images", [])}
    anns = {a["id"]: a for a in coco.get("annotations", [])}
    n_kp = len(kp_order) if kp_order is not None else None

    for a in coco.get("annotations", []):
        if n_kp is not None and len(a.get("keypoints", [])) != 3 * n_kp:
            out.append(
                _err(
                    "keypoints_length",
                    f"{split} annotation {a['id']}: {len(a.get('keypoints', []))} values, "
                    f"expected {3 * n_kp}",
                )
            )
            break

    for rel, img_id in ((i["file_name"], i["id"]) for i in coco.get("images", [])):
        p = Path(root) / "images" / rel
        if p.is_symlink():
            out.append(_err("image_is_symlink", f"{split} image {img_id}: {rel} is a symlink"))
            break

    for rel, img_id in ((i["file_name"], i["id"]) for i in coco.get("images", [])):
        p = Path(root) / "images" / rel
        if not p.is_symlink() and not p.exists():
            out.append(_err("image_missing", f"{split} image {img_id}: {rel} not on disk"))
            break

    for key, fs in coco.get("framesets", {}).items():
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

    per_image = Counter(a["image_id"] for a in coco.get("annotations", []))
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


def validate_root(root: str | Path, *, masks_root: str | Path | None = None) -> list[Finding]:
    """Every check from the format spec, as a flat finding list.

    >>> [f.code for f in validate_root("dataset") if f.level == "error"]   # doctest: +SKIP
    []
    """
    out: list[Finding] = []
    kp_order = _check_keypoints(root, out)
    _check_calibrations(root, out)
    try:
        manifest = schema.load_manifest(root)
    except (OSError, json.JSONDecodeError) as exc:
        out.append(_err("manifest_unreadable", str(exc)))
        return out
    srcs = schema.sources(manifest)
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
        _check_split(root, split, kp_order, srcs, out)
    if masks_root is not None and not Path(masks_root).exists():
        out.append(
            Finding(
                "warning",
                "masks_root_missing",
                f"{masks_root} does not exist; copy-paste will be off",
            )
        )
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
