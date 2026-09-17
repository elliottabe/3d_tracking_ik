import json

import pytest

from tracking.curate import schema
from tracking.curate.merge import TierSpec, merge_tiers


def spec(path, sid, kind="human", weight=1.0, **kw):
    return TierSpec(path=path, source_id=sid, kind=kind, weight=weight, **kw)


def test_ids_are_renumbered_without_collision(make_tier, tmp_path):
    a = make_tier("a", recording="rec1")
    b = make_tier("b", recording="rec2")
    merge_tiers(
        [spec(a, "a"), spec(b, "b", kind="pseudo", weight=0.3, checkpoint="/c")], tmp_path / "out"
    )
    coco = schema.load_instances(tmp_path / "out", "train")
    ids = [i["id"] for i in coco["images"]]
    assert len(ids) == len(set(ids))
    ann_ids = [a_["id"] for a_ in coco["annotations"]]
    assert len(ann_ids) == len(set(ann_ids))


def test_frameset_references_survive_renumbering(make_tier, tmp_path):
    a = make_tier("a", recording="rec1")
    b = make_tier("b", recording="rec2")
    merge_tiers([spec(a, "a"), spec(b, "b")], tmp_path / "out")
    coco = schema.load_instances(tmp_path / "out", "train")
    images = {i["id"] for i in coco["images"]}
    anns = {x["id"] for x in coco["annotations"]}
    for fs in coco["framesets"].values():
        assert set(fs["frames"]) <= images
        assert {i for i in fs["ann_ids"] if i is not None} <= anns


def test_images_are_dereferenced(make_tier, tmp_path):
    a = make_tier("a", recording="rec1")
    coco = json.loads((a / "annotations" / "instances_train.json").read_text())
    rel = coco["images"][0]["file_name"]
    target = tmp_path / "away.jpg"
    (a / "images" / rel).rename(target)
    (a / "images" / rel).symlink_to(target)
    merge_tiers([spec(a, "a")], tmp_path / "out")
    assert not (tmp_path / "out" / "images" / rel).is_symlink()
    assert (tmp_path / "out" / "images" / rel).exists()


def test_every_frameset_gets_its_source_id(make_tier, tmp_path):
    a = make_tier("a", recording="rec1")
    merge_tiers([spec(a, "human_v1")], tmp_path / "out")
    coco = schema.load_instances(tmp_path / "out", "train")
    assert {fs["source_id"] for fs in coco["framesets"].values()} == {"human_v1"}


def test_duplicate_source_id_is_rejected(make_tier, tmp_path):
    a, b = make_tier("a", recording="rec1"), make_tier("b", recording="rec2")
    with pytest.raises(ValueError, match="duplicate source_id"):
        merge_tiers([spec(a, "x"), spec(b, "x")], tmp_path / "out")


def test_conflicting_image_bytes_are_rejected(make_tier, tmp_path):
    a = make_tier("a", recording="rec1")
    b = make_tier("b", recording="rec1", calib_seed=0)
    coco = json.loads((b / "annotations" / "instances_train.json").read_text())
    (b / "images" / coco["images"][0]["file_name"]).write_bytes(b"different")
    with pytest.raises(ValueError, match="differs between tiers"):
        merge_tiers([spec(a, "a"), spec(b, "b")], tmp_path / "out")


def test_keypoint_name_disagreement_is_rejected(make_tier, tmp_path):
    a, b = make_tier("a", recording="rec1"), make_tier("b", recording="rec2")
    (b / "annotations" / "keypoint_names.json").write_text(json.dumps(["X", "Y"]))
    for split in ("train", "val"):
        p = b / "annotations" / f"instances_{split}.json"
        coco = json.loads(p.read_text())
        coco["keypoint_names"] = ["X", "Y"]
        p.write_text(json.dumps(coco))
    with pytest.raises(ValueError, match="keypoint_names"):
        merge_tiers([spec(a, "a"), spec(b, "b")], tmp_path / "out")


def test_cross_tier_calib_conflict_is_rejected(make_tier, tmp_path):
    a = make_tier("a", recording="rec1", calib_seed=0)
    b = make_tier("b", recording="rec1", calib_seed=1)
    with pytest.raises(ValueError, match="calib_group conflict"):
        merge_tiers([spec(a, "a"), spec(b, "b")], tmp_path / "out", copy_images=False)


def test_missing_calib_group_in_manifest_is_rejected(make_tier, tmp_path):
    a = make_tier("a", recording="rec1")
    man_path = a / "manifest.json"
    man = json.loads(man_path.read_text())
    del man["recordings"]["rec1"]["calib_group"]
    man_path.write_text(json.dumps(man))
    with pytest.raises(ValueError, match="no calib_group"):
        merge_tiers([spec(a, "a")], tmp_path / "out")


def test_merged_root_validates_clean(make_tier, tmp_path):
    from tracking.curate.validate import validate_root

    a, b = make_tier("a", recording="rec1"), make_tier("b", recording="rec2")
    merge_tiers(
        [spec(a, "a"), spec(b, "b", kind="pseudo", weight=0.3, checkpoint="/c")], tmp_path / "out"
    )
    assert [f for f in validate_root(tmp_path / "out") if f.level == "error"] == []
