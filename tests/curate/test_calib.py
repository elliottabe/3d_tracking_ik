import pytest

from tracking.curate.calib import calib_fingerprint, group_calibrations


def test_fingerprint_is_stable(make_calib_dir):
    a = make_calib_dir("a", seed=1)
    b = make_calib_dir("b", seed=1)
    assert calib_fingerprint(a) == calib_fingerprint(b)


def test_fingerprint_separates_different_calibrations(make_calib_dir):
    assert calib_fingerprint(make_calib_dir("a", seed=1)) != calib_fingerprint(
        make_calib_dir("b", seed=2)
    )


def test_largest_group_is_A(make_calib_dir):
    dirs = {
        "r1": make_calib_dir("c1", seed=1),
        "r2": make_calib_dir("c2", seed=1),
        "r3": make_calib_dir("c3", seed=2),
    }
    groups = group_calibrations(dirs)
    assert groups["r1"] == groups["r2"] == "A"
    assert groups["r3"] == "B"


def test_camera_count_inferred_not_hardcoded(make_calib_dir):
    d = make_calib_dir("two", seed=1, cameras=("Cam01", "Cam02"))
    assert calib_fingerprint(d)


def test_inconsistent_camera_names_raise(make_calib_dir):
    dirs = {
        "r1": make_calib_dir("c1", seed=1),
        "r2": make_calib_dir("c2", seed=1, cameras=("Cam01", "Cam09", "Cam03")),
    }
    with pytest.raises(ValueError, match="camera names"):
        group_calibrations(dirs)
