"""A recording may declare that it has no camera rig, and must then name no cameras."""

import pytest
from omegaconf import OmegaConf

from tracking.io.recording import RecordingSpec


def _cfg(tmp_path, **over):
    base = {
        "name": "2026_07_06_16_55_07",
        "session_dir": str(tmp_path),
        "calib_dir": None,
        "cameras": [],
        "num_animals": 1,
        "fps": 800.0,
        "bouts_csv": None,
        "kp3d_csv": str(tmp_path / "data3D.csv"),
        "index_csv": None,
    }
    base.update(over)
    return OmegaConf.create(base)


def test_rigless_spec_validates(tmp_path):
    spec = RecordingSpec.from_config(_cfg(tmp_path))
    spec.validate()
    assert spec.has_rig is False
    assert spec.calib_dir is None
    assert spec.kp3d_csv == tmp_path / "data3D.csv"


def test_rigless_spec_refuses_named_cameras(tmp_path):
    spec = RecordingSpec.from_config(_cfg(tmp_path, cameras=["Cam2012630"]))
    with pytest.raises(ValueError, match="names cameras"):
        spec.validate()


def test_missing_kp3d_csv_is_none_not_a_path(tmp_path):
    cfg = _cfg(tmp_path)
    del cfg["kp3d_csv"]
    spec = RecordingSpec.from_config(cfg)
    assert spec.kp3d_csv is None
    assert spec.index_csv is None


def test_rigged_spec_still_requires_its_calib_dir(tmp_path):
    spec = RecordingSpec.from_config(
        _cfg(tmp_path, calib_dir=str(tmp_path / "nope"), cameras=["Cam1"])
    )
    assert spec.has_rig is True
    with pytest.raises(ValueError, match="calib_dir does not exist"):
        spec.validate()
