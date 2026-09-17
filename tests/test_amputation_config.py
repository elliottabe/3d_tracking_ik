"""The amputation config group composes and resolves to the cohort's real paths."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

import tracking.utils.path_utils  # noqa: F401 -- registers ${repo_root:}

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _compose(*overrides):
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        return compose(config_name="pipeline", overrides=list(overrides))


def test_amputation_recording_resolves_flat_run_root():
    cfg = _compose("recording=amputation", "run.name=ik_v1")
    assert cfg.recording.name == "2026_07_06_16_55_07"
    assert cfg.recording.assay == "amputation"
    assert cfg.run.root.endswith("/processed/amputation/2026_07_06_16_55_07/ik_v1")
    assert cfg.recording.session_dir.endswith("/processed/amputation/2026_07_06_16_55_07")


def test_amputation_recording_is_rigless_and_single_fly():
    cfg = _compose("recording=amputation")
    assert cfg.recording.calib_dir is None
    assert list(cfg.recording.cameras) == []
    assert cfg.recording.num_animals == 1
    assert cfg.recording.fps == 800.0


def test_amputation_recording_id_override_flows_into_every_path():
    cfg = _compose("recording=amputation", "recording.id=2026_07_09_10_22_32")
    assert cfg.recording.name == "2026_07_09_10_22_32"
    assert cfg.recording.bouts_csv.endswith("/2026_07_09_10_22_32/running_bouts_summary.csv")
    assert cfg.recording.kp3d_csv.endswith("/2026_07_09_10_22_32/data3D.csv")


def test_ik_amputation_inherits_per_frame_and_adds_freeze():
    cfg = _compose("ik=amputation")
    assert list(cfg.ik.freeze_dof_patterns) == ["*_T1_left"]
    assert cfg.ik.per_frame.n_iter == 500
    assert cfg.ik.per_frame.batch == 512


def test_ik_default_freezes_nothing():
    cfg = _compose()
    assert list(cfg.ik.freeze_dof_patterns) == []


@pytest.mark.parametrize("group", ["session0", "session1", "stairs"])
def test_existing_recordings_still_compose(group):
    cfg = _compose(f"recording={group}")
    assert cfg.recording.calib_dir is not None
    assert len(cfg.recording.cameras) == 7
