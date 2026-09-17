import os

import numpy as np
import pytest

from tracking.train.data.augment import (
    MVAugParams,
    assert_lr_swap_covers,
    augment_window,
    build_lr_swap,
)

KP = ["Antenna_Base", "EyeL", "EyeR", "Scutellum", "T1L_FeTi", "T1R_FeTi"]


def test_lr_swap_pairs_left_and_right():
    swap = build_lr_swap(KP)
    assert swap[KP.index("EyeL")] == KP.index("EyeR")
    assert swap[KP.index("EyeR")] == KP.index("EyeL")
    assert swap[KP.index("T1L_FeTi")] == KP.index("T1R_FeTi")


def test_lr_swap_is_identity_for_midline_keypoints():
    swap = build_lr_swap(KP)
    assert swap[KP.index("Scutellum")] == KP.index("Scutellum")
    assert swap[KP.index("Antenna_Base")] == KP.index("Antenna_Base")


def test_lr_swap_is_an_involution():
    swap = build_lr_swap(KP)
    assert list(swap[swap]) == list(range(len(KP)))


def test_assert_lr_swap_covers_accepts_a_complete_map():
    assert_lr_swap_covers(KP, build_lr_swap(KP))


def test_assert_lr_swap_covers_rejects_an_unpaired_left():
    bad = ["EyeL", "Scutellum"]
    with pytest.raises(ValueError):
        assert_lr_swap_covers(bad, build_lr_swap(bad))


UNIFIED_ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/unified_v2"


@pytest.mark.skipif(not os.path.isdir(UNIFIED_ROOT), reason="unified root not present")
def test_lr_swap_covers_the_real_50_keypoint_list():
    from tracking.curate.schema import keypoint_order

    names = list(keypoint_order(UNIFIED_ROOT).names)
    assert len(names) == 50
    assert_lr_swap_covers(names, build_lr_swap(names))


N_KP = len(KP)


def _sample(T=1, C=7, K=N_KP, F=2, crop=32):
    rng = np.random.default_rng(0)
    return {
        "crops": rng.integers(0, 255, (T, C, crop, crop, 3), dtype=np.uint8),
        "cam_valid": np.ones((T, C), bool),
        "M": rng.normal(size=(C, 2, 3)).astype(np.float32),
        "t_local": rng.normal(size=(T, C, 2)).astype(np.float32),
        "center3D": np.zeros(3, np.float32),
        "kp3d_local": rng.normal(size=(F, T, K, 3)).astype(np.float32),
        "has3d": np.ones((F, T, K), bool),
        "kp2d": rng.uniform(0, crop, (F, T, C, K, 2)).astype(np.float32),
        "vis2d": np.ones((F, T, C, K), bool),
        "fly_valid": np.array([True, False]),
        "px_scale": np.float32(1.0),
        "is_female": np.bool_(True),
        "donor_mask": np.zeros((T, C, crop, crop), bool),
        "crop_origin": np.zeros((C, 2), np.int32),
        "fly_sex": np.zeros((F,), np.int8),
        "unlabelled_sex": np.int8(-1),
        "sample_weight": np.float32(1.0),
        "is_negative": np.bool_(False),
    }


def test_augment_preserves_keys_and_shapes():
    s = _sample()
    out = augment_window(s, MVAugParams(), np.random.default_rng(1), build_lr_swap(KP))
    assert set(out) == set(s)
    for k in ("crops", "kp3d_local", "kp2d", "vis2d", "M", "t_local"):
        assert out[k].shape == s[k].shape


def test_disabled_augmentation_is_a_passthrough():
    s = _sample()
    out = augment_window(s, MVAugParams(enabled=False), np.random.default_rng(1), build_lr_swap(KP))
    np.testing.assert_array_equal(out["crops"], s["crops"])
    np.testing.assert_allclose(out["kp3d_local"], s["kp3d_local"])


def test_camera_drop_only_ever_clears_cam_valid():
    s = _sample()
    p = MVAugParams(
        cam_drop_p=1.0,
        cam_drop_max=2,
        rot_deg=0.0,
        scale_min=1.0,
        scale_max=1.0,
        translate_frac=0.0,
        world_yaw=False,
        world_tilt_deg=0.0,
        mirror_p=0.0,
    )
    out = augment_window(s, p, np.random.default_rng(3), build_lr_swap(KP))
    assert out["cam_valid"].sum() < s["cam_valid"].sum()
    assert out["cam_valid"].sum() >= s["cam_valid"].sum() - 2
