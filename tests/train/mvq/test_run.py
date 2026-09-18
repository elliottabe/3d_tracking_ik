"""Driver-level guards that need neither a GPU nor the training root."""

import json

import pytest
from omegaconf import OmegaConf

from tracking.train.mvq.losses import LossWeights
from tracking.train.mvq.run import _write_meta, check_finite, loss_weights


def test_check_finite_passes_a_real_loss():
    check_finite(1.25, 7, 1, {"total": 1.25}, 4)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_check_finite_raises_and_names_the_step(bad, capsys):
    with pytest.raises(ValueError, match="non-finite loss"):
        check_finite(bad, 31, 2, {"total": bad, "reproj": 1.0}, 30)
    out = capsys.readouterr().out
    assert "FATAL" in out and "step 31" in out and "reproj=1.0000" in out


def test_a_missing_loss_group_is_loud(capsys):
    """Every LossWeights default but `other_fly_repulsion` coincides with v2,
    so the silent fallback trained v2 with term 7b never traced."""
    assert loss_weights(None) == LossWeights()
    out = capsys.readouterr().out
    assert "WARNING no `loss` config group" in out
    assert "other_fly_repulsion=0" in out


def test_a_present_loss_group_is_silent_and_wins(capsys):
    w = loss_weights(OmegaConf.create({"other_fly_repulsion": 20.0}))
    assert w.other_fly_repulsion == 20.0
    assert capsys.readouterr().out == ""


def test_write_meta_keeps_an_earlier_runs_calibration_and_val(tmp_path):
    """A requeue rewrites the run-root meta BEFORE the model builds; the
    numbered-step loader prefers that file, so a clobber serves T=1.0."""
    _write_meta(tmp_path, {"model": {}, "val": {"mpjpe3d_units": 0.9}})
    with open(tmp_path / "mvq_run.json") as f:
        meta = json.load(f)
    meta["calibration"] = {"exist_temperature": 0.25, "vis_temperature": 0.57, "n_val": 153}
    with open(tmp_path / "mvq_run.json", "w") as f:
        json.dump(meta, f)

    _write_meta(tmp_path, {"model": {}, "val": None})  # the startup write of a relaunch
    kept = json.loads((tmp_path / "mvq_run.json").read_text())
    assert kept["calibration"]["exist_temperature"] == 0.25
    assert kept["val"] == {"mpjpe3d_units": 0.9}


def test_write_meta_lets_a_fresh_val_replace_the_old_one(tmp_path):
    _write_meta(tmp_path, {"val": {"mpjpe3d_units": 0.9}})
    _write_meta(tmp_path, {"val": {"mpjpe3d_units": 0.4}})
    assert json.loads((tmp_path / "mvq_run.json").read_text())["val"]["mpjpe3d_units"] == 0.4
