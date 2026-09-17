"""Package a validated root as a tiered zip with checksums and a data sheet."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from tracking.curate import schema
from tracking.curate.masks.store import sidecar_path
from tracking.curate.validate import format_findings, validate_root

TIERS = ("human", "all", "masks")


def _subset(coco: dict, keep_ids: set[str]) -> dict:
    """Restrict a split to `keep_ids` source ids, dropping now-unreferenced rows."""
    framesets = {k: v for k, v in coco["framesets"].items() if v.get("source_id") in keep_ids}
    img_ids = {i for fs in framesets.values() for i in fs["frames"]}
    ann_ids = {i for fs in framesets.values() for i in fs["ann_ids"] if i is not None}
    return {
        **coco,
        "images": [i for i in coco["images"] if i["id"] in img_ids],
        "annotations": [a for a in coco["annotations"] if a["id"] in ann_ids],
        "framesets": framesets,
    }


def dataset_markdown(manifest: dict, counts: dict) -> str:
    """The DATASET.md that travels inside the archive."""
    lines = [
        f"# {manifest.get('version', 'training root')}",
        "",
        "## Sources",
        "",
        "| id | kind | weight | framesets | checkpoint |",
        "| --- | --- | --- | --- | --- |",
    ]
    for sid, s in sorted(manifest.get("sources", {}).items()):
        lines.append(
            f"| {sid} | {s['kind']} | {s.get('weight', 1.0)} | "
            f"{s.get('n_framesets', 0)} | {s.get('checkpoint') or '-'} |"
        )
    lines += ["", "## Contents", ""]
    lines += [f"- {k}: {v}" for k, v in sorted(counts.items())]
    lines += ["", "Format: `docs/training.md`.", ""]
    return "\n".join(lines)


def package_root(root, out_zip, *, tier="all", masks_root=None) -> dict:
    """Validate, then write a tiered zip beside `CHECKSUMS` and `DATASET.md`.

    >>> package_root("unified", "human.zip", tier="human")       # doctest: +SKIP
    {'tier': 'human', 'n_framesets': 2814, 'n_images': 19131}
    """
    if tier not in TIERS:
        raise ValueError(f"tier {tier!r} not in {TIERS}")
    root, out_zip = Path(root), Path(out_zip)
    findings = validate_root(root, masks_root=masks_root)
    if any(f.level == "error" for f in findings):
        raise ValueError(f"refusing to package an invalid root:\n{format_findings(findings)}")

    manifest = schema.load_manifest(root)
    if tier == "masks":
        src = Path(masks_root) if masks_root else sidecar_path(root)
        if not src.exists():
            raise FileNotFoundError(f"mask sidecar not found: {src}")
        members = {str(p.relative_to(src)): p for p in sorted(src.rglob("*")) if p.is_file()}
        return _write_zip(out_zip, members, manifest, {"mask_files": len(members)}, tier)

    keep = {
        sid
        for sid, s in manifest.get("sources", {}).items()
        if tier == "all" or s["kind"] == "human"
    }
    if not keep:
        raise ValueError(f"tier {tier!r} matches no sources; refusing to ship an empty archive")
    members: dict[str, Path] = {}
    staged: dict[str, bytes] = {}
    n_fs = n_img = 0
    for split in schema.SPLITS:
        try:
            coco = schema.load_instances(root, split)
        except FileNotFoundError:
            continue
        sub = _subset(coco, keep)
        staged[f"annotations/instances_{split}.json"] = json.dumps(sub).encode()
        n_fs += len(sub["framesets"])
        n_img += len(sub["images"])
        for im in sub["images"]:
            members[f"images/{im['file_name']}"] = root / "images" / im["file_name"]

    if n_fs == 0:
        raise ValueError(f"tier {tier!r} has 0 framesets; refusing to ship an empty archive")

    staged["annotations/keypoint_names.json"] = (
        root / "annotations" / "keypoint_names.json"
    ).read_bytes()
    kept_sources = {k: v for k, v in manifest.get("sources", {}).items() if k in keep}
    staged["manifest.json"] = json.dumps({**manifest, "sources": kept_sources}, indent=2).encode()
    for p in sorted((root / "calibrations").rglob("*")):
        if p.is_file():
            members[str(p.relative_to(root))] = p

    counts = {"framesets": n_fs, "images": n_img, "sources": len(kept_sources)}
    return _write_zip(
        out_zip, members, {**manifest, "sources": kept_sources}, counts, tier, staged=staged
    )


def _write_zip(
    out_zip: Path,
    members: dict[str, Path],
    manifest,
    counts,
    tier,
    *,
    staged: dict[str, bytes] | None = None,
) -> dict:
    staged = staged or {}
    digests = []
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in sorted(staged.items()):
            z.writestr(name, data)
            digests.append((hashlib.sha256(data).hexdigest(), name))
        for name, path in sorted(members.items()):
            data = path.read_bytes()
            z.writestr(name, data)
            digests.append((hashlib.sha256(data).hexdigest(), name))
        checksums = "".join(f"{h}  {n}\n" for h, n in sorted(digests, key=lambda x: x[1]))
        z.writestr("CHECKSUMS", checksums)
        z.writestr("DATASET.md", dataset_markdown(manifest, counts))
    return {
        "tier": tier,
        "zip": str(out_zip),
        "n_members": len(digests),
        **{f"n_{k}": v for k, v in counts.items()},
    }
