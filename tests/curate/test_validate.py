import json

import cv2
import numpy as np

from tracking.curate import validate
from tracking.curate.masks.store import MaskStore, sidecar_path
from tracking.curate.merge import TierSpec, merge_tiers


def codes(findings, level=None):
    return {f.code for f in findings if level is None or f.level == level}


def test_clean_root_has_no_errors(make_tier):
    assert codes(validate.validate_root(make_tier("t")), "error") == set()


def test_reports_two_fly_fraction_as_info(make_tier):
    findings = validate.validate_root(make_tier("t", fly_ids=(0, 1)))
    info = [f for f in findings if f.code == "two_fly_fraction"]
    assert len(info) == 2 and {f.level for f in info} == {"info"}


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


def test_split_missing_framesets_key_is_an_error(make_tier):
    root = make_tier("t")
    coco = json.loads((root / "annotations" / "instances_train.json").read_text())
    del coco["framesets"]
    (root / "annotations" / "instances_train.json").write_text(json.dumps(coco))
    assert "split_missing_keys" in codes(validate.validate_root(root), "error")


def test_split_missing_all_structural_keys_names_them_all(make_tier):
    root = make_tier("t")
    coco = json.loads((root / "annotations" / "instances_train.json").read_text())
    del coco["images"]
    del coco["annotations"]
    del coco["framesets"]
    (root / "annotations" / "instances_train.json").write_text(json.dumps(coco))
    findings = [f for f in validate.validate_root(root) if f.code == "split_missing_keys"]
    assert len(findings) == 1
    for key in ("images", "annotations", "framesets"):
        assert key in findings[0].message


def test_empty_but_present_split_keys_are_not_missing_keys(make_tier):
    root = make_tier("t")
    assert "split_missing_keys" not in codes(validate.validate_root(root), "error")


def test_image_row_missing_file_name_does_not_crash(make_tier):
    root = make_tier("t")
    coco = json.loads((root / "annotations" / "instances_train.json").read_text())
    del coco["images"][0]["file_name"]
    (root / "annotations" / "instances_train.json").write_text(json.dumps(coco))
    findings = validate.validate_root(root)
    assert "malformed_row" in codes(findings, "error")


def test_image_row_missing_id_does_not_crash(make_tier):
    root = make_tier("t")
    coco = json.loads((root / "annotations" / "instances_train.json").read_text())
    del coco["images"][0]["id"]
    (root / "annotations" / "instances_train.json").write_text(json.dumps(coco))
    findings = validate.validate_root(root)
    assert "malformed_row" in codes(findings, "error")


def test_annotation_row_missing_id_does_not_crash(make_tier):
    root = make_tier("t")
    coco = json.loads((root / "annotations" / "instances_train.json").read_text())
    del coco["annotations"][0]["id"]
    (root / "annotations" / "instances_train.json").write_text(json.dumps(coco))
    findings = validate.validate_root(root)
    assert "malformed_row" in codes(findings, "error")


def test_framesets_as_list_does_not_crash(make_tier):
    root = make_tier("t")
    coco = json.loads((root / "annotations" / "instances_train.json").read_text())
    coco["framesets"] = list(coco["framesets"].values())
    (root / "annotations" / "instances_train.json").write_text(json.dumps(coco))
    findings = validate.validate_root(root)
    assert "malformed_row" in codes(findings, "error")


def test_manifest_source_with_bad_kind_does_not_crash(make_tier):
    root = make_tier("t")
    man = json.loads((root / "manifest.json").read_text())
    man["sources"] = {"x": {"kind": "pseudolabel"}}
    (root / "manifest.json").write_text(json.dumps(man))
    findings = validate.validate_root(root)
    assert "manifest_sources_invalid" in codes(findings, "error")


def test_manifest_source_missing_kind_does_not_crash(make_tier):
    root = make_tier("t")
    man = json.loads((root / "manifest.json").read_text())
    man["sources"] = {"x": {}}
    (root / "manifest.json").write_text(json.dumps(man))
    findings = validate.validate_root(root)
    assert "manifest_sources_invalid" in codes(findings, "error")


def test_calib_group_missing_from_disk_is_an_error(make_tier):
    root = make_tier("t")
    man = json.loads((root / "manifest.json").read_text())
    man["recordings"]["rec1"]["calib_group"] = "B"
    man["calib_groups"] = ["A", "B"]
    (root / "manifest.json").write_text(json.dumps(man))
    assert "calib_group_missing" in codes(validate.validate_root(root), "error")


def test_recording_spanning_two_calib_groups_is_an_info_finding(make_tier):
    root = make_tier("t")
    man = json.loads((root / "manifest.json").read_text())
    del man["recordings"]["rec1"]["calib_group"]
    man["recordings"]["rec1"]["calib_groups"] = ["A", "B"]
    (root / "manifest.json").write_text(json.dumps(man))
    coco = json.loads((root / "annotations" / "instances_train.json").read_text())
    for fs in coco["framesets"].values():
        fs["calib_group"] = "A"
    (root / "annotations" / "instances_train.json").write_text(json.dumps(coco))
    findings = validate.validate_root(root)
    info = [f for f in findings if f.code == "recording_multi_calib"]
    assert len(info) == 1
    assert "A" in info[0].message and "B" in info[0].message


def test_frameset_without_any_calib_group_is_an_error(make_tier):
    root = make_tier("t")
    man = json.loads((root / "manifest.json").read_text())
    del man["recordings"]["rec1"]["calib_group"]
    (root / "manifest.json").write_text(json.dumps(man))
    assert "frameset_calib_unresolvable" in codes(validate.validate_root(root), "error")


def test_frameset_relying_on_recordings_calib_groups_plural_is_still_an_error(make_tier):
    root = make_tier("t")
    man = json.loads((root / "manifest.json").read_text())
    del man["recordings"]["rec1"]["calib_group"]
    man["recordings"]["rec1"]["calib_groups"] = ["A", "B"]
    (root / "manifest.json").write_text(json.dumps(man))
    assert "frameset_calib_unresolvable" in codes(validate.validate_root(root), "error")


def test_malformed_recording_value_does_not_crash(make_tier):
    root = make_tier("t")
    man = json.loads((root / "manifest.json").read_text())
    man["recordings"]["rec1"] = None
    (root / "manifest.json").write_text(json.dumps(man))
    findings = validate.validate_root(root)
    assert isinstance(findings, list)


def test_duplicate_file_name_across_images_is_a_warning(make_tier):
    root = make_tier("t")
    coco = json.loads((root / "annotations" / "instances_train.json").read_text())
    coco["images"][1]["file_name"] = coco["images"][0]["file_name"]
    (root / "annotations" / "instances_train.json").write_text(json.dumps(coco))
    assert "duplicate_file_name" in codes(validate.validate_root(root), "warning")


def test_symlink_offender_count_is_reported(make_tier, tmp_path):
    root = make_tier("t", n_frames=2)
    coco = json.loads((root / "annotations" / "instances_train.json").read_text())
    for im in coco["images"][:2]:
        rel = im["file_name"]
        target = tmp_path / f"elsewhere_{im['id']}.jpg"
        (root / "images" / rel).rename(target)
        (root / "images" / rel).symlink_to(target)
    findings = [f for f in validate.validate_root(root) if f.code == "image_is_symlink"]
    assert len(findings) == 1
    assert "2 total" in findings[0].message


def merged_with_masks(make_tier, tmp_path, frames=(100,), n_frames=1):
    root = make_tier("a", recording="rec1", n_frames=n_frames)
    for frame in frames:
        p = root / "masks" / "rec1" / "Cam01" / f"Frame_{frame}.npz"
        p.parent.mkdir(parents=True, exist_ok=True)
        ann_id = 1 + 3 * (frame - 100)
        np.savez_compressed(p, ann_ids=np.array([ann_id], np.int64), masks=np.ones((1, 2, 2), bool))
    out = tmp_path / "out"
    merge_tiers([TierSpec(root, "human_v1", "human", 1.0)], out)
    return out, sidecar_path(out)


def break_sidecar_row(sidecar, frame):
    p = MaskStore(sidecar).path("rec1", "Cam01", frame)
    with np.load(p) as z:
        data = {k: z[k] for k in z.files}
    np.savez_compressed(p, **{**data, "ann_ids": np.array([9999], np.int64)})


def test_mask_file_without_the_frameset_ann_id_is_a_warning(make_tier, tmp_path):
    root, sidecar = merged_with_masks(make_tier, tmp_path)
    break_sidecar_row(sidecar, 100)
    findings = validate.validate_root(root, masks_root=sidecar)
    assert "mask_ann_unresolved" in codes(findings, "warning")
    assert codes(findings, "error") == set()


def test_resolvable_mask_rows_are_not_a_finding(make_tier, tmp_path):
    root, sidecar = merged_with_masks(make_tier, tmp_path)
    assert "mask_ann_unresolved" not in codes(validate.validate_root(root, masks_root=sidecar))


def test_frames_without_a_mask_file_are_not_a_finding(make_tier, tmp_path):
    root, sidecar = merged_with_masks(make_tier, tmp_path)
    MaskStore(sidecar).path("rec1", "Cam01", 100).unlink()
    assert "mask_ann_unresolved" not in codes(validate.validate_root(root, masks_root=sidecar))


def test_mask_check_stops_at_the_sample_cap(make_tier, tmp_path, monkeypatch):
    root, sidecar = merged_with_masks(make_tier, tmp_path, frames=(100, 101), n_frames=2)
    break_sidecar_row(sidecar, 101)
    assert "mask_ann_unresolved" in codes(validate.validate_root(root, masks_root=sidecar))
    monkeypatch.setattr(validate, "MASK_SAMPLE", 1)
    assert "mask_ann_unresolved" not in codes(validate.validate_root(root, masks_root=sidecar))
