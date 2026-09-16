import json

import cv2
import numpy as np

from tracking.curate import validate


def codes(findings, level=None):
    return {f.code for f in findings if level is None or f.level == level}


def test_clean_root_has_no_errors(make_tier):
    assert codes(validate.validate_root(make_tier("t")), "error") == set()


def test_reports_two_fly_fraction_as_info(make_tier):
    findings = validate.validate_root(make_tier("t", fly_ids=(0, 1)))
    info = [f for f in findings if f.code == "two_fly_fraction"]
    assert len(info) == 1 and info[0].level == "info"


def test_keypoint_name_mismatch_is_an_error(make_tier):
    root = make_tier("t")
    (root / "annotations" / "keypoint_names.json").write_text(json.dumps(["A", "B"]))
    assert "keypoint_names_mismatch" in codes(validate.validate_root(root), "error")


def test_symlinked_image_is_an_error(make_tier, tmp_path):
    root = make_tier("t")
    coco = json.loads((root / "annotations" / "instances_train.json").read_text())
    rel = coco["images"][0]["file_name"]
    target = tmp_path / "elsewhere.jpg"
    (root / "images" / rel).rename(target)
    (root / "images" / rel).symlink_to(target)
    assert "image_is_symlink" in codes(validate.validate_root(root), "error")


def test_missing_image_is_an_error(make_tier):
    root = make_tier("t")
    coco = json.loads((root / "annotations" / "instances_train.json").read_text())
    (root / "images" / coco["images"][0]["file_name"]).unlink()
    assert "image_missing" in codes(validate.validate_root(root), "error")


def test_non_affine_calibration_is_an_error(make_tier):
    root = make_tier("t")
    p = root / "calibrations" / "A" / "Cam01.yaml"
    m = np.zeros((3, 4))
    m[:2, :3] = 1.0
    m[2] = [0.1, 0.0, 0.0, 1.0]
    fs = cv2.FileStorage(str(p), cv2.FILE_STORAGE_WRITE)
    fs.write("projectionMatrix", m)
    fs.release()
    assert "calibration_not_affine" in codes(validate.validate_root(root), "error")


def test_ann_ids_length_mismatch_is_an_error(make_tier):
    root = make_tier("t")
    coco = json.loads((root / "annotations" / "instances_train.json").read_text())
    key = next(iter(coco["framesets"]))
    coco["framesets"][key]["ann_ids"] = coco["framesets"][key]["ann_ids"][:1]
    (root / "annotations" / "instances_train.json").write_text(json.dumps(coco))
    assert "ann_ids_length" in codes(validate.validate_root(root), "error")


def test_too_few_resolved_slots_is_an_error(make_tier):
    root = make_tier("t")
    coco = json.loads((root / "annotations" / "instances_train.json").read_text())
    key = next(iter(coco["framesets"]))
    coco["framesets"][key]["ann_ids"] = [coco["framesets"][key]["ann_ids"][0], None, None]
    (root / "annotations" / "instances_train.json").write_text(json.dumps(coco))
    assert "too_few_cameras" in codes(validate.validate_root(root), "error")


def test_negative_without_center3d_is_an_error(make_tier):
    root = make_tier("t")
    coco = json.loads((root / "annotations" / "instances_train.json").read_text())
    key = next(iter(coco["framesets"]))
    coco["framesets"][key]["fly_id"] = -1
    (root / "annotations" / "instances_train.json").write_text(json.dumps(coco))
    assert "negative_without_center3d" in codes(validate.validate_root(root), "error")


def test_unresolved_source_id_is_an_error(make_tier):
    root = make_tier("t", source={"source_id": "nope"})
    assert "unknown_source_id" in codes(validate.validate_root(root), "error")


def test_keypoint_array_length_is_checked(make_tier):
    root = make_tier("t")
    coco = json.loads((root / "annotations" / "instances_train.json").read_text())
    coco["annotations"][0]["keypoints"] = [1.0, 2.0]
    (root / "annotations" / "instances_train.json").write_text(json.dumps(coco))
    assert "keypoints_length" in codes(validate.validate_root(root), "error")


def test_format_findings_mentions_every_error(make_tier):
    root = make_tier("t")
    (root / "annotations" / "keypoint_names.json").write_text(json.dumps(["A"]))
    text = validate.format_findings(validate.validate_root(root))
    assert "keypoint_names_mismatch" in text


def test_corrupt_split_json_does_not_crash(make_tier):
    root = make_tier("t")
    (root / "annotations" / "instances_train.json").write_text("{not valid json")
    assert "keypoint_names_unreadable" in codes(validate.validate_root(root), "error")


def test_split_json_missing_keypoint_names_key_is_unreadable(make_tier):
    root = make_tier("t")
    coco = json.loads((root / "annotations" / "instances_train.json").read_text())
    del coco["keypoint_names"]
    (root / "annotations" / "instances_train.json").write_text(json.dumps(coco))
    assert "keypoint_names_unreadable" in codes(validate.validate_root(root), "error")


def test_split_missing_images_key_does_not_crash(make_tier):
    root = make_tier("t")
    coco = json.loads((root / "annotations" / "instances_train.json").read_text())
    del coco["images"]
    (root / "annotations" / "instances_train.json").write_text(json.dumps(coco))
    findings = validate.validate_root(root)
    assert isinstance(findings, list)


def test_corrupt_calibration_file_is_unreadable(make_tier):
    root = make_tier("t")
    (root / "calibrations" / "A" / "Cam01.yaml").write_bytes(b"\x00\x01garbage not yaml at all")
    assert "calibration_unreadable" in codes(validate.validate_root(root), "error")


def test_corrupt_manifest_is_unreadable(make_tier):
    root = make_tier("t")
    (root / "manifest.json").write_text("{not valid json")
    assert "manifest_unreadable" in codes(validate.validate_root(root), "error")


def test_symlink_and_missing_image_are_both_reported(make_tier, tmp_path):
    root = make_tier("t", n_frames=2)
    coco = json.loads((root / "annotations" / "instances_train.json").read_text())
    images = coco["images"]
    rel_symlink = images[0]["file_name"]
    target = tmp_path / "elsewhere.jpg"
    (root / "images" / rel_symlink).rename(target)
    (root / "images" / rel_symlink).symlink_to(target)
    rel_missing = images[3]["file_name"]
    (root / "images" / rel_missing).unlink()
    found = codes(validate.validate_root(root), "error")
    assert "image_is_symlink" in found
    assert "image_missing" in found
