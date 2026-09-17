import os

import pytest

from tracking.train.data.windows import WindowDataset

ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/unified_v2"
pytestmark = pytest.mark.skipif(not os.path.isdir(ROOT), reason="unified root not present")


@pytest.fixture(scope="module")
def ds():
    return WindowDataset(ROOT, "train", T=1)


def test_every_frameset_becomes_one_window_at_T1(ds):
    assert len(ds) == 36677


def test_negatives_are_identified(ds):
    neg = [i for i in range(len(ds)) if ds.is_negative(i)]
    assert len(neg) == 7302
    assert all(ds.source(i) == "pseudo" for i in neg[:50])
    assert {ds.role(i) for i in neg} == {"negative", "partner"}


def test_window_index_round_trips(ds):
    for i in (0, 100, len(ds) - 1):
        rec, fly, f0 = ds.windows[i]
        assert ds.window_index(rec, fly, f0, ds.delta(i)) == i


def test_calib_group_comes_from_the_frameset(ds):
    groups = {ds.calib_group(i) for i in range(0, len(ds), 997)}
    assert groups <= {"A", "B", "C"}


def test_multi_calib_recording_spans_both_groups(ds):
    seen = {ds.calib_group(i) for i in range(len(ds)) if ds.windows[i][0] == "2026_04_02_17_28_34"}
    assert seen == {"A", "C"}


def test_camera_names_is_an_order_of_seven(ds):
    cams = ds.camera_names(0)
    assert len(cams) == 7
    assert all(n.startswith("Cam") for n in cams)


def test_weight_and_source_follow_the_frameset(ds):
    i = next(i for i in range(len(ds)) if ds.source(i) == "pseudo" and not ds.is_negative(i))
    assert ds.weight(i) == pytest.approx(0.3)
    j = next(i for i in range(len(ds)) if ds.is_negative(i))
    assert ds.weight(j) == pytest.approx(1.0)


def test_pair_deltas_produce_more_windows_at_T2():
    d2 = WindowDataset(ROOT, "train", T=2, pair_deltas=(1, 4, 16))
    assert len(d2) > 0
    assert set(d2.win_delta) <= {1, 4, 16}
