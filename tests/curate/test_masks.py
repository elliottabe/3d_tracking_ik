import json

import numpy as np

from tracking.curate.masks.adapter import import_masks
from tracking.curate.masks.store import MaskStore, sidecar_path


def test_sidecar_path_is_a_sibling(tmp_path):
    assert sidecar_path(tmp_path / "ds") == tmp_path / "ds_masks"


def test_save_and_load_round_trip(tmp_path):
    store = MaskStore(tmp_path / "m")
    m = np.zeros((4, 5), bool)
    m[1, 2] = True
    store.save("rec", "Cam01", 7, {11: m})
    assert np.array_equal(store.load("rec", "Cam01", 7, 11), m)


def test_load_returns_none_for_unknown_ann(tmp_path):
    store = MaskStore(tmp_path / "m")
    store.save("rec", "Cam01", 7, {11: np.zeros((2, 2), bool)})
    assert store.load("rec", "Cam01", 7, 99) is None


def test_has_is_false_when_absent(tmp_path):
    assert not MaskStore(tmp_path / "m").has("rec", "Cam01", 7)


def test_write_index_records_counts(tmp_path):
    store = MaskStore(tmp_path / "m")
    store.save("rec", "Cam01", 7, {11: np.zeros((2, 2), bool)})
    store.save("rec", "Cam02", 7, {12: np.zeros((2, 2), bool)})
    index = store.write_index(tool="sam3")
    assert index["n_files"] == 2 and index["tool"] == "sam3"
    assert json.loads((tmp_path / "m" / "index.json").read_text())["n_files"] == 2


def test_import_masks_copies_npz_tree(tmp_path):
    src = tmp_path / "src" / "rec" / "Cam01"
    src.mkdir(parents=True)
    np.savez_compressed(
        src / "Frame_7.npz", ann_ids=np.array([11]), masks=np.zeros((1, 2, 2), bool)
    )
    summary = import_masks(tmp_path / "src", tmp_path / "out_masks", tool="sam3")
    assert summary["n_files"] == 1
    assert MaskStore(tmp_path / "out_masks").has("rec", "Cam01", 7)
