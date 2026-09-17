import json

import numpy as np
import pytest

from tracking.curate import schema
from tracking.curate.masks.store import MaskStore, sidecar_path
from tracking.curate.merge import TierSpec, merge_tiers


def spec(path, sid, kind="human", weight=1.0, **kw):
    return TierSpec(path=path, source_id=sid, kind=kind, weight=weight, **kw)


def write_mask(tier_root, recording, camera, frame, ann_ids, masks, matched=None):
    p = tier_root / "masks" / recording / camera / f"Frame_{frame}.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    matched = np.ones(len(ann_ids), bool) if matched is None else np.asarray(matched)
    np.savez_compressed(
        p, ann_ids=np.array(ann_ids, np.int64), masks=np.stack(masks), matched=matched
    )


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


def test_every_frameset_gets_a_calib_group(make_tier, tmp_path):
    a = make_tier("a", recording="rec1")
    merge_tiers([spec(a, "human_v1")], tmp_path / "out")
    coco = schema.load_instances(tmp_path / "out", "train")
    assert all(fs["calib_group"] == "A" for fs in coco["framesets"].values())


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


def test_cross_tier_calib_conflict_is_recorded_per_frameset(make_tier, tmp_path):
    a = make_tier("a", recording="rec1", calib_seed=0)
    b = make_tier("b", recording="rec1", calib_seed=1)
    manifest = merge_tiers([spec(a, "a"), spec(b, "b")], tmp_path / "out", copy_images=False)
    assert manifest["recordings"]["rec1"]["calib_groups"] == ["A", "B"]
    assert "calib_group" not in manifest["recordings"]["rec1"]
    coco = schema.load_instances(tmp_path / "out", "train")
    by_source = {fs["source_id"]: fs["calib_group"] for fs in coco["framesets"].values()}
    assert by_source == {"a": "A", "b": "B"}


def test_cross_tier_calib_conflict_still_copies_images(make_tier, tmp_path):
    a = make_tier("a", recording="rec1", calib_seed=0)
    b = make_tier("b", recording="rec1", calib_seed=1)
    out = tmp_path / "out"
    merge_tiers([spec(a, "a"), spec(b, "b")], out, copy_images=True)
    assert any((out / "images").iterdir())


def test_agreeing_tiers_get_a_single_recording_calib_group(make_tier, tmp_path):
    a = make_tier("a", recording="rec1", calib_seed=0)
    b = make_tier("b", recording="rec1", calib_seed=0)
    manifest = merge_tiers([spec(a, "a"), spec(b, "b")], tmp_path / "out", copy_images=False)
    assert manifest["recordings"]["rec1"]["calib_group"] == "A"
    assert "calib_groups" not in manifest["recordings"]["rec1"]
    coco = schema.load_instances(tmp_path / "out", "train")
    assert {fs["calib_group"] for fs in coco["framesets"].values()} == {"A"}


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


def test_mask_ids_are_remapped_to_the_merged_id(make_tier, tmp_path):
    a = make_tier("a", recording="rec1", n_frames=1)
    b = make_tier("b", recording="rec2", n_frames=1)
    mask = np.zeros((2, 2), bool)
    mask[0, 1] = True
    write_mask(b, "rec2", "Cam01", 100, [1], [mask])
    out = tmp_path / "out"
    manifest = merge_tiers([spec(a, "a"), spec(b, "b")], out, copy_images=False)
    coco = schema.load_instances(out, "train")
    b_first_new_id = next(
        fs["ann_ids"][0]
        for fs in coco["framesets"].values()
        if fs["source_id"] == "b" and fs["recording"] == "rec2"
    )
    store = MaskStore(sidecar_path(out))
    loaded = store.load("rec2", "Cam01", 100, b_first_new_id)
    assert loaded is not None
    assert np.array_equal(loaded, mask)
    assert manifest["masks"] == {
        "files_written": 1,
        "rows_remapped": 1,
        "rows_dropped": 0,
        "collisions_merged": 0,
    }


def test_mask_rows_that_do_not_survive_are_dropped(make_tier, tmp_path):
    a = make_tier("a", recording="rec1", n_frames=1)
    surviving = np.zeros((2, 2), bool)
    stray = np.ones((2, 2), bool)
    write_mask(a, "rec1", "Cam01", 100, [1, 9999], [surviving, stray], matched=[True, False])
    out = tmp_path / "out"
    manifest = merge_tiers([spec(a, "a")], out, copy_images=False)
    store = MaskStore(sidecar_path(out))
    assert np.array_equal(store.load("rec1", "Cam01", 100, 1), surviving)
    assert store.load("rec1", "Cam01", 100, 9999) is None
    p = store.path("rec1", "Cam01", 100)
    with np.load(p) as z:
        assert z["ann_ids"].tolist() == [1]
    assert manifest["masks"]["rows_dropped"] == 1
    assert manifest["masks"]["rows_remapped"] == 1


def test_mask_file_dropped_entirely_when_no_row_survives(make_tier, tmp_path):
    a = make_tier("a", recording="rec1", n_frames=1)
    write_mask(a, "rec1", "Cam01", 100, [9999], [np.zeros((2, 2), bool)])
    out = tmp_path / "out"
    manifest = merge_tiers([spec(a, "a")], out, copy_images=False)
    assert not MaskStore(sidecar_path(out)).path("rec1", "Cam01", 100).exists()
    assert manifest["masks"]["files_written"] == 0
    assert manifest["masks"]["rows_dropped"] == 1


def test_mask_collision_between_tiers_is_merged_not_clobbered(make_tier, tmp_path):
    a = make_tier("a", recording="rec1", n_frames=1, calib_seed=0)
    b = make_tier("b", recording="rec1", n_frames=1, calib_seed=1)
    mask_a = np.zeros((2, 2), bool)
    mask_b = np.ones((2, 2), bool)
    write_mask(a, "rec1", "Cam01", 100, [1], [mask_a])
    write_mask(b, "rec1", "Cam01", 100, [1], [mask_b])
    out = tmp_path / "out"
    manifest = merge_tiers([spec(a, "a"), spec(b, "b")], out, copy_images=False)
    coco = schema.load_instances(out, "train")
    a_id = next(fs["ann_ids"][0] for fs in coco["framesets"].values() if fs["source_id"] == "a")
    b_id = next(fs["ann_ids"][0] for fs in coco["framesets"].values() if fs["source_id"] == "b")
    store = MaskStore(sidecar_path(out))
    assert np.array_equal(store.load("rec1", "Cam01", 100, a_id), mask_a)
    assert np.array_equal(store.load("rec1", "Cam01", 100, b_id), mask_b)
    assert manifest["masks"]["collisions_merged"] == 1
    assert manifest["masks"]["files_written"] == 1
    assert manifest["masks"]["rows_remapped"] == 2


def test_masks_false_skips_the_whole_step(make_tier, tmp_path):
    a = make_tier("a", recording="rec1", n_frames=1)
    write_mask(a, "rec1", "Cam01", 100, [1], [np.zeros((2, 2), bool)])
    out = tmp_path / "out"
    manifest = merge_tiers([spec(a, "a")], out, copy_images=False, masks=False)
    assert "masks" not in manifest
    assert not sidecar_path(out).exists()
