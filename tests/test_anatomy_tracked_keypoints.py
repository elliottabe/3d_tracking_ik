"""The anatomy is intersected with what a recording actually tracked."""

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import tracking.utils.path_utils  # noqa: F401
from tracking.preprocess.scale import rigid_segment_pairs
from tracking.run import _load_anatomy

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
T1L_DISTAL = ["T1L_Tro", "T1L_FeTi", "T1L_TiTa", "T1L_TaT1", "T1L_TaT3", "T1L_TaTip"]


def _compose(*overrides):
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        return compose(config_name="pipeline", overrides=list(overrides))


def _csv(tmp_path, names):
    path = tmp_path / "data3D.csv"
    path.write_text(
        ",".join(n for n in names for _ in range(4))
        + "\n"
        + ",".join(["x", "y", "z", "confidence"] * len(names))
        + "\n"
    )
    return path


def test_no_kp3d_csv_keeps_the_full_strict_anatomy():
    anatomy = _load_anatomy(_compose("recording=session0"))
    assert len(anatomy.kp_order) == 50
    for name in T1L_DISTAL:
        assert name in anatomy.kp_order


def test_tracked_header_drops_the_amputated_keypoints(tmp_path):
    cfg = _compose("recording=amputation")
    full = OmegaConf.to_container(cfg.anatomy, resolve=False)["model"]["KP_NAMES"]
    tracked = [k for k in full if k not in T1L_DISTAL]
    cfg.recording.kp3d_csv = str(_csv(tmp_path, tracked))

    anatomy = _load_anatomy(cfg)
    assert len(anatomy.kp_order) == 44
    for name in T1L_DISTAL:
        assert name not in anatomy.kp_order
    assert "T1L_ThxCx" in anatomy.kp_order


def test_filtered_anatomy_still_measures_five_legs(tmp_path):
    cfg = _compose("recording=amputation")
    full = OmegaConf.to_container(cfg.anatomy, resolve=False)["model"]["KP_NAMES"]
    tracked = [k for k in full if k not in T1L_DISTAL]
    cfg.recording.kp3d_csv = str(_csv(tmp_path, tracked))

    anatomy = _load_anatomy(cfg)
    pairs = rigid_segment_pairs(anatomy.kp_order)
    assert len(pairs) == 16
    assert not any(a.startswith("T1L") or b.startswith("T1L") for a, b in pairs)
