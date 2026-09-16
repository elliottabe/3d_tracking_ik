import json

import pytest

from tracking.curate import schema
from tracking.io.names import Order, OrderMismatch


def test_keypoint_order_matches_canonical(make_tier):
    root = make_tier("t")
    assert schema.keypoint_order(root) == Order(["Antenna_Base", "EyeL", "EyeR", "Scutellum"])


def test_keypoint_order_rejects_disagreement(make_tier):
    root = make_tier("t")
    (root / "annotations" / "keypoint_names.json").write_text(json.dumps(["A", "B"]))
    with pytest.raises(OrderMismatch, match="keypoint_names"):
        schema.keypoint_order(root)


def test_round_trip_instances(make_tier):
    root = make_tier("t")
    coco = schema.load_instances(root, "train")
    schema.save_instances(root, "train", coco)
    assert schema.load_instances(root, "train") == coco


def test_source_entry_rejects_unknown_kind():
    with pytest.raises(ValueError, match="kind"):
        schema.SourceEntry(id="x", kind="bogus", weight=1.0)


def test_source_entry_round_trips_through_dict():
    e = schema.SourceEntry(
        id="p", kind="pseudo", weight=0.3, checkpoint="/ckpt", gates={"exist_min": 0.8}
    )
    assert schema.SourceEntry.from_dict("p", e.to_dict()) == e


def test_frameset_field_chain():
    manifest = {"weight": 9.0, "recordings": {"r": {"weight": 5.0}}}
    assert schema.frameset_field(manifest, {"weight": 1.0}, "r", "weight", 0.0) == 1.0
    assert schema.frameset_field(manifest, {}, "r", "weight", 0.0) == 5.0
    assert schema.frameset_field(manifest, {}, "other", "weight", 0.0) == 9.0
    assert schema.frameset_field(manifest, {}, "other", "role", "anchor") == "anchor"
