import json
import os

import pytest

ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/unified_v2"
pytestmark = pytest.mark.skipif(not os.path.isdir(ROOT), reason="unified root not present")


def _skip_if_oom(exc):
    """Skip on a device out-of-memory (the GPU is shared), re-raise anything else."""
    msg = str(exc)
    if "RESOURCE_EXHAUSTED" in msg or "out of memory" in msg.lower():
        pytest.skip(f"device out of memory: {msg.splitlines()[0]}")
    raise


def test_two_step_smoke_run_writes_a_checkpoint_and_a_manifest(tmp_path):
    from hydra import compose, initialize_config_dir

    from tracking.train.mvq.run import run_training

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    with initialize_config_dir(config_dir=os.path.join(repo, "configs"), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "train=mvq_v2",
                f"paths.train_data_root={ROOT}",
                f"paths.runs_root={tmp_path}",
                "run.name=smoke",
                "train.smoke=true",
                "train.total_steps=2",
                "train.batch_size=2",
                "train.window_lengths=[1]",
                "train.num_workers=2",
                "train.loader_workers=threads",
                "train.eval_every=1000000",
            ],
        )
    try:
        meta = run_training(cfg)
    except Exception as exc:
        _skip_if_oom(exc)
    run_dir = tmp_path / "smoke"
    assert (run_dir / "mvq_run.json").is_file()
    assert (run_dir / "ckpt").is_dir()
    assert meta["train"]["total_steps"] == 2
    assert "sources" in meta["train_data"]

    on_disk = json.loads((run_dir / "mvq_run.json").read_text())
    assert on_disk["model"]["backbone"] == "dinov3_b16"
    assert set(on_disk["calibration"]) == {"exist_temperature", "vis_temperature"}
    assert (run_dir / "final" / "mvq_run.json").is_file()
    drawn = on_disk["train_data"]["mix"]["1"]["realised"]
    assert abs(sum(drawn.values()) - 1.0) < 1e-6


def test_the_module_entry_point_exists():
    import importlib

    assert importlib.util.find_spec("tracking.train.mvq.__main__") is not None
