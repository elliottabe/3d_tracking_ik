import os

import numpy as np
import pytest

from tracking.detector.mvq.slots import SEX_UNKNOWN
from tracking.geometry.rig import CameraRig
from tracking.train.data.windows import CROP, WINDOW_KEYS, WindowDataset

ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/unified_v2"
MASKS = ROOT + "_masks"
pytestmark = pytest.mark.skipif(not os.path.isdir(ROOT), reason="unified root not present")


@pytest.fixture(scope="module")
def ds():
    return WindowDataset(ROOT, "train", T=1, masks_root=MASKS)


def test_sample_has_exactly_the_declared_keys(ds):
    s = ds[0]
    assert set(s) == set(WINDOW_KEYS)


def test_shapes_and_dtypes(ds):
    s = ds[0]
    T, C, K, F = 1, 7, 50, 2
    assert s["crops"].shape == (T, C, CROP, CROP, 3) and s["crops"].dtype == np.uint8
    assert s["cam_valid"].shape == (T, C) and s["cam_valid"].dtype == np.bool_
    assert s["M"].shape == (C, 2, 3) and s["M"].dtype == np.float32
    assert s["t_local"].shape == (T, C, 2)
    assert s["center3D"].shape == (3,)
    assert s["kp3d_local"].shape == (F, T, K, 3)
    assert s["has3d"].shape == (F, T, K)
    assert s["kp2d"].shape == (F, T, C, K, 2)
    assert s["vis2d"].shape == (F, T, C, K)
    assert s["fly_valid"].shape == (F,)
    assert s["donor_mask"].shape == (T, C, CROP, CROP)


def test_host_fly_is_always_instance_zero_and_valid(ds):
    i = next(i for i in range(len(ds)) if not ds.is_negative(i))
    assert ds[i]["fly_valid"][0]


def test_labelled_3d_reprojects_onto_the_labelled_2d(ds):
    i = next(i for i in range(len(ds)) if not ds.is_negative(i))
    s = ds[i]
    uv = np.einsum("cij,kj->kci", s["M"], s["kp3d_local"][0, 0]) + s["t_local"][0][None, :, :]
    ok = s["vis2d"][0, 0].T & s["has3d"][0, 0][:, None]
    err = np.linalg.norm(uv - s["kp2d"][0, 0].transpose(1, 0, 2), axis=-1)
    assert np.median(err[ok]) < 5.0


def test_negative_window_is_empty_and_marked(ds):
    i = next(i for i in range(len(ds)) if ds.is_negative(i))
    s = ds[i]
    assert not s["fly_valid"].any()
    assert not s["has3d"].any()
    assert not s["donor_mask"].any()
    assert s["unlabelled_sex"] == SEX_UNKNOWN
    assert bool(s["is_negative"])


def test_crops_are_not_all_black(ds):
    s = ds[0]
    assert s["crops"][0][s["cam_valid"][0]].max() > 0


def test_donor_mask_is_populated_when_a_sidecar_mask_exists(ds):
    hit = 0
    for i in range(0, 4000, 137):
        if ds.is_negative(i):
            continue
        if ds[i]["donor_mask"].any():
            hit += 1
        if hit >= 2:
            break
    assert hit >= 2


def test_without_masks_root_donor_mask_is_empty():
    d = WindowDataset(ROOT, "train", T=1, masks_root=None)
    i = next(i for i in range(len(d)) if not d.is_negative(i))
    assert not d[i]["donor_mask"].any()


def test_sample_weight_matches_the_window_weight(ds):
    i = next(i for i in range(len(ds)) if ds.source(i) == "pseudo")
    assert float(ds[i]["sample_weight"]) == pytest.approx(ds.weight(i))


def test_non_affine_calibration_raises(ds):
    rig = ds._rig(ds.calib_group(0))
    projective = rig.matrices_f64.copy()
    projective[:, 2, :] = [0.001, 0.0, 0.0, 1.0]
    bad_rig = CameraRig(rig.cameras, projective)
    probe = object.__new__(WindowDataset)
    probe._rigs = {"bad": bad_rig}
    probe._affine_mt = {}
    with pytest.raises(ValueError, match="affine"):
        probe._affine("bad")
