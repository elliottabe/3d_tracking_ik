import os

import numpy as np
import pytest

from tracking.train.data.augment import MVAugParams, build_lr_swap
from tracking.train.data.windows import WindowDataset

ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/unified_v2"
pytestmark = pytest.mark.skipif(not os.path.isdir(ROOT), reason="unified root not present")


def _ds(**kw):
    return WindowDataset(ROOT, split="train", T=1, train=True, **kw)


def test_augmentation_is_off_by_default():
    assert _ds().aug is None


def test_aug_changes_the_sample_and_keeps_every_key():
    plain = _ds()
    swap = build_lr_swap(plain.kp_order.names)
    aug = _ds(aug=MVAugParams(), lr_swap=swap)
    a, b = plain[0], aug[0]
    assert set(a) == set(b)
    assert not np.array_equal(a["crops"], b["crops"])
    assert a["crops"].shape == b["crops"].shape


def test_aug_preserves_the_reprojection_invariant():
    from tracking.detector.mvq.geometry import project_local
    plain = _ds()
    swap = build_lr_swap(plain.kp_order.names)
    s = _ds(aug=MVAugParams(), lr_swap=swap)[0]
    uv = np.asarray(project_local(s["kp3d_local"][0, 0], s["M"], s["t_local"][0]))  # (K,C,2)
    gt = np.asarray(s["kp2d"][0, 0]).transpose(1, 0, 2)  # (C,K,2) -> (K,C,2)
    m = np.asarray(s["vis2d"][0, 0]).T & np.asarray(s["cam_valid"][0])[None, :]
    assert np.median(np.linalg.norm(uv - gt, axis=-1)[m]) < 1.0


def test_aug_is_deterministic_for_a_fixed_epoch_and_index():
    swap = build_lr_swap(_ds().kp_order.names)
    a = _ds(aug=MVAugParams(), lr_swap=swap)[3]
    b = _ds(aug=MVAugParams(), lr_swap=swap)[3]
    assert np.array_equal(a["crops"], b["crops"])


def test_aug_differs_across_epochs():
    swap = build_lr_swap(_ds().kp_order.names)
    d = _ds(aug=MVAugParams(), lr_swap=swap)
    a = d[3]
    d.epoch = 1
    assert not np.array_equal(a["crops"], d[3]["crops"])


def test_aug_survives_worker_spec_round_trip():
    from tracking.train.data.loaders import build_dataset
    swap = build_lr_swap(_ds().kp_order.names)
    d = _ds(aug=MVAugParams(), lr_swap=swap)
    rebuilt = build_dataset(d.worker_spec())
    assert rebuilt.aug is not None
    assert np.array_equal(d[3]["crops"], rebuilt[3]["crops"])


def test_source_id_is_readable():
    d = _ds()
    assert isinstance(d.source_id(0), str) and d.source_id(0)
