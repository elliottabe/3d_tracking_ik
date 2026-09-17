import dataclasses
import os

import numpy as np
import pytest

from tracking.train.data.copypaste import CopyPasteParams
from tracking.train.data.windows import WINDOW_KEYS, WindowDataset

ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/unified_v2"
MASKS = ROOT + "_masks"
pytestmark = pytest.mark.skipif(not os.path.isdir(ROOT), reason="unified root not present")

# 2026_04_02_17_28_34 spans calibration groups A and C (mid-day recalibration).
MULTI_GROUP_REC = "2026_04_02_17_28_34"
# Small single-group recording, cheap to run several real paste_window() calls on.
SMALL_REC = "2026_01_29_14_09_33"


@pytest.fixture(scope="module")
def multi_group_ds():
    return WindowDataset(
        ROOT, "train", T=1, recordings=[MULTI_GROUP_REC], copy_paste=CopyPasteParams(p=1.0)
    )


@pytest.fixture(scope="module")
def small_ds():
    return WindowDataset(
        ROOT,
        "train",
        T=1,
        recordings=[SMALL_REC],
        masks_root=MASKS,
        copy_paste=CopyPasteParams(p=1.0, max_tries=4),
    )


def test_donor_pool_excludes_negatives(multi_group_ds):
    pooled = [j for v in multi_group_ds._donors.values() for j in v]
    assert pooled
    assert not any(multi_group_ds.is_negative(j) for j in pooled)


def test_donor_pool_is_keyed_by_calib_group_not_recording(multi_group_ds):
    groups = {k[0] for k in multi_group_ds._donors}
    assert groups == {"A", "C"}
    for (grp, _sex, delta), members in multi_group_ds._donors.items():
        assert all(multi_group_ds.calib_group(j) == grp for j in members)
        assert all(multi_group_ds.delta(j) == delta for j in members)


def test_paste_window_is_none_for_a_negative(multi_group_ds):
    i = next(i for i in range(len(multi_group_ds)) if multi_group_ds.is_negative(i))
    assert multi_group_ds.paste_window(i, np.random.default_rng(0)) is None


def test_paste_window_is_none_without_copy_paste_configured(small_ds):
    saved = small_ds.copy_paste
    small_ds.copy_paste = None
    try:
        i = next(i for i in range(len(small_ds)) if not small_ds.is_negative(i))
        assert small_ds.paste_window(i, np.random.default_rng(0)) is None
    finally:
        small_ds.copy_paste = saved


def test_paste_window_adds_a_second_valid_fly(small_ds):
    rng = np.random.default_rng(0)
    elig = [
        i for i in range(len(small_ds)) if not small_ds.is_negative(i) and small_ds.n_flies(i) == 1
    ]
    hits = 0
    for i in elig[:6]:
        out = small_ds.paste_window(i, rng)
        if out is None:
            continue
        hits += 1
        assert bool(out["fly_valid"][0]) and bool(out["fly_valid"][1])
        for k in ("crops", "kp2d", "vis2d", "donor_mask"):
            assert out[k].shape == small_ds[i][k].shape
    assert hits > 0


def _same_sample(a, b):
    return all(np.array_equal(a[k], b[k]) for k in WINDOW_KEYS)


def test_getitem_without_copy_paste_matches_build(small_ds):
    saved = small_ds.copy_paste
    small_ds.copy_paste = None
    try:
        i = next(i for i in range(len(small_ds)) if not small_ds.is_negative(i))
        assert _same_sample(small_ds[i], small_ds._build(i))
    finally:
        small_ds.copy_paste = saved


def test_getitem_p_zero_never_pastes(small_ds):
    saved = small_ds.copy_paste
    small_ds.copy_paste = dataclasses.replace(saved, p=0.0)
    try:
        for i in range(min(len(small_ds), 6)):
            assert _same_sample(small_ds[i], small_ds._build(i))
    finally:
        small_ds.copy_paste = saved


def test_getitem_p_one_pastes_on_an_eligible_window(small_ds):
    elig = [
        i for i in range(len(small_ds)) if not small_ds.is_negative(i) and small_ds.n_flies(i) == 1
    ]
    assert any(not _same_sample(small_ds[i], small_ds._build(i)) for i in elig[:6])


def test_getitem_never_pastes_a_negative(multi_group_ds):
    i = next(i for i in range(len(multi_group_ds)) if multi_group_ds.is_negative(i))
    assert _same_sample(multi_group_ds[i], multi_group_ds._build(i))


def test_getitem_eval_mode_never_pastes(small_ds):
    saved = small_ds.train
    small_ds.train = False
    try:
        for i in range(min(len(small_ds), 6)):
            assert _same_sample(small_ds[i], small_ds._build(i))
    finally:
        small_ds.train = saved


def test_getitem_draw_is_deterministic_given_the_same_seed():
    kwargs = dict(
        recordings=[SMALL_REC], masks_root=MASKS, copy_paste=CopyPasteParams(p=1.0, max_tries=4)
    )
    ds1 = WindowDataset(ROOT, "train", T=1, seed=0, **kwargs)
    ds2 = WindowDataset(ROOT, "train", T=1, seed=0, **kwargs)
    i = next(i for i in range(len(ds1)) if not ds1.is_negative(i) and ds1.n_flies(i) == 1)
    assert _same_sample(ds1[i], ds2[i])
