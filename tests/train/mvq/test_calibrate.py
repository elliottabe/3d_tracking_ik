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
    write_calibration(tmp_path, 1.5, 2.0)
    meta = json.loads((tmp_path / "mvq_run.json").read_text())
    assert meta["calibration"] == {"exist_temperature": 1.5, "vis_temperature": 2.0}
