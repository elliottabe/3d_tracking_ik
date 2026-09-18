import numpy as np

from tracking.train.mvq.calibrate import fit_temperature, write_calibration


def test_already_calibrated_logits_fit_temperature_near_one():
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 2.0, 20000)
    targets = rng.uniform(size=20000) < 1 / (1 + np.exp(-logits))
    assert 0.85 < fit_temperature(logits, targets) < 1.2


def test_overconfident_logits_fit_a_temperature_above_one():
    rng = np.random.default_rng(1)
    logits = rng.normal(0, 2.0, 20000)
    targets = rng.uniform(size=20000) < 1 / (1 + np.exp(-logits))
    assert fit_temperature(logits * 3.0, targets) > 1.5


def test_write_calibration_uses_the_keys_the_runner_reads(tmp_path):
    import json
    write_calibration(tmp_path, 1.5, 2.0, n_val=153)
    meta = json.loads((tmp_path / "mvq_run.json").read_text())
    assert meta["calibration"] == {
        "exist_temperature": 1.5, "vis_temperature": 2.0, "n_val": 153}


def test_a_fit_that_saturates_the_bound_says_so(capsys):
    """Separable logits drive T to the lower bound; the bounded search returns
    a plausible scalar either way, so the saturation has to be printed."""
    logits = np.concatenate([np.full(500, -8.0), np.full(500, 8.0)])
    targets = np.concatenate([np.zeros(500), np.ones(500)])
    T = fit_temperature(logits, targets, "exist")
    out = capsys.readouterr().out
    assert "SATURATED" in out and "exist" in out, out
    assert T < 0.02


def test_an_interior_fit_is_silent(capsys):
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 2.0, 20000)
    targets = rng.uniform(size=20000) < 1 / (1 + np.exp(-logits))
    fit_temperature(logits, targets, "vis")
    assert capsys.readouterr().out == ""
