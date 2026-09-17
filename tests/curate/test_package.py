import json
import zipfile

import pytest

from tracking.curate.masks.store import sidecar_path
from tracking.curate.merge import TierSpec, merge_tiers
from tracking.curate.package import TIERS, package_root


def build(make_tier, tmp_path):
    a = make_tier("a", recording="rec1")
    b = make_tier("b", recording="rec2")
    out = tmp_path / "unified"
    merge_tiers(
        [
            TierSpec(a, "human_v1", "human", 1.0),
            TierSpec(b, "pseudo_v1", "pseudo", 0.3, checkpoint="/ckpt"),
        ],
        out,
    )
    return out


def test_tiers_are_named(make_tier, tmp_path):
    assert TIERS == ("human", "all", "masks")


def test_all_tier_contains_both_sources(make_tier, tmp_path):
    root = build(make_tier, tmp_path)
    summary = package_root(root, tmp_path / "all.zip", tier="all")
    assert summary["n_framesets"] == 6
    with zipfile.ZipFile(tmp_path / "all.zip") as z:
        assert any(n.endswith("DATASET.md") for n in z.namelist())
        assert any(n.endswith("CHECKSUMS") for n in z.namelist())


def test_human_tier_drops_pseudo_framesets(make_tier, tmp_path):
    root = build(make_tier, tmp_path)
    summary = package_root(root, tmp_path / "human.zip", tier="human")
    assert summary["n_framesets"] == 3
    with zipfile.ZipFile(tmp_path / "human.zip") as z:
        coco = json.loads(z.read("annotations/instances_train.json"))
        assert {fs["source_id"] for fs in coco["framesets"].values()} == {"human_v1"}
        assert all("rec2" not in n for n in z.namelist())


def test_invalid_root_is_refused(make_tier, tmp_path):
    root = build(make_tier, tmp_path)
    (root / "annotations" / "keypoint_names.json").write_text(json.dumps(["Z"]))
    with pytest.raises(ValueError, match="refusing to package"):
        package_root(root, tmp_path / "bad.zip", tier="all")


def test_checksums_cover_every_member(make_tier, tmp_path):
    root = build(make_tier, tmp_path)
    package_root(root, tmp_path / "all.zip", tier="all")
    with zipfile.ZipFile(tmp_path / "all.zip") as z:
        listed = {line.split("  ", 1)[1] for line in z.read("CHECKSUMS").decode().splitlines()}
        members = {n for n in z.namelist() if n not in ("CHECKSUMS", "DATASET.md")}
    assert listed == members


def test_masks_tier_missing_sidecar_raises(make_tier, tmp_path):
    root = build(make_tier, tmp_path)
    missing = sidecar_path(root)
    assert not missing.exists()
    with pytest.raises(FileNotFoundError, match=str(missing)):
        package_root(root, tmp_path / "masks.zip", tier="masks")


def test_human_tier_with_no_human_sources_is_rejected(make_tier, tmp_path):
    a = make_tier("a", recording="rec1")
    out = tmp_path / "unified"
    merge_tiers(
        [TierSpec(a, "pseudo_v1", "pseudo", 0.3, checkpoint="/ckpt")],
        out,
    )
    with pytest.raises(ValueError, match="no sources"):
        package_root(out, tmp_path / "human.zip", tier="human")


def test_masks_tier_counts_files(make_tier, tmp_path):
    root = build(make_tier, tmp_path)
    sidecar = sidecar_path(root)
    (sidecar / "rec1" / "Cam01").mkdir(parents=True)
    (sidecar / "rec1" / "Cam01" / "Frame_100.npz").write_bytes(b"data")
    summary = package_root(root, tmp_path / "masks.zip", tier="masks")
    assert summary["n_mask_files"] == 1
    with zipfile.ZipFile(tmp_path / "masks.zip") as z:
        assert "rec1/Cam01/Frame_100.npz" in z.namelist()
