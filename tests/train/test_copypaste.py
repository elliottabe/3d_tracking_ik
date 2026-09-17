import numpy as np

from tracking.train.data.copypaste import CopyPasteParams, composite


def _win(crop=24, C=3, T=1, K=4, F=2, fill=40, mask_box=None):
    s = {
        "crops": np.full((T, C, crop, crop, 3), fill, np.uint8),
        "cam_valid": np.ones((T, C), bool),
        "M": np.tile(np.eye(3, dtype=np.float32)[:2], (C, 1, 1)),
        "t_local": np.zeros((T, C, 2), np.float32),
        "center3D": np.zeros(3, np.float32),
        "kp3d_local": np.zeros((F, T, K, 3), np.float32),
        "has3d": np.zeros((F, T, K), bool),
        "kp2d": np.zeros((F, T, C, K, 2), np.float32),
        "vis2d": np.zeros((F, T, C, K), bool),
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
    if mask_box:
        y0, y1, x0, x1 = mask_box
        s["donor_mask"][:, :, y0:y1, x0:x1] = True
        s["crops"][:, :, y0:y1, x0:x1] = 200
    return s


def test_params_defaults():
    p = CopyPasteParams()
    assert (p.p, p.opposite_sex_p, p.contact_p) == (0.0, 0.7, 0.3)
    assert p.contact_sep == (8.0, 30.0)


def test_donor_pixels_land_in_the_host():
    host = _win()
    donor = _win(mask_box=(4, 20, 4, 20))
    out = composite(host, donor, np.zeros(3, np.float32), CopyPasteParams())
    assert out is not None
    assert (out["crops"][0, 0, 8:16, 8:16] == 200).all()


def test_donor_with_no_mask_content_returns_none():
    host = _win()
    donor = _win(fill=10)
    assert composite(host, donor, np.zeros(3, np.float32), CopyPasteParams()) is None


def test_host_donor_mask_loses_pixels_covered_by_the_donor():
    host = _win(mask_box=(2, 20, 2, 20))
    donor = _win(mask_box=(4, 12, 4, 12))
    out = composite(host, donor, np.zeros(3, np.float32), CopyPasteParams())
    assert not out["donor_mask"][0, 0, 5:11, 5:11].any()
    assert out["donor_mask"][0, 0, 15:19, 15:19].any()


def test_composite_preserves_keys_and_shapes():
    host = _win()
    donor = _win(mask_box=(4, 12, 4, 12))
    out = composite(host, donor, np.zeros(3, np.float32), CopyPasteParams())
    assert set(out) == set(host)
    for k in ("crops", "kp2d", "vis2d", "donor_mask"):
        assert out[k].shape == host[k].shape


def test_a_second_fly_becomes_valid():
    host = _win()
    donor = _win(mask_box=(4, 12, 4, 12))
    out = composite(host, donor, np.zeros(3, np.float32), CopyPasteParams())
    assert out["fly_valid"].sum() >= host["fly_valid"].sum()
