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


def _final_and_ema(run_dir):
    """(final params, checkpoint EMA, ema_updates) read back off disk, keyed by
    path -- the two trees `save_final` must differ by exactly the debias."""
    import jax
    import numpy as np
    import orbax.checkpoint as ocp

    from tracking.detector.mvq.checkpoint import restore_own_tree

    mngr = ocp.CheckpointManager(
        str(run_dir / "ckpt"),
        options=ocp.CheckpointManagerOptions(read_only=True),
        item_names=("model", "opt", "ema", "ema_meta"),
    )
    step = mngr.latest_step()
    meta = mngr.restore(step, args=ocp.args.Composite(ema_meta=ocp.args.JsonRestore()))["ema_meta"]

    def flat(tree):
        return {
            jax.tree_util.keystr(k): np.asarray(v)
            for k, v in jax.tree_util.tree_flatten_with_path(tree)[0]
        }

    return (
        flat(restore_own_tree(str(run_dir / "final"))),
        flat(restore_own_tree(str(run_dir / "ckpt" / str(step) / "ema"))),
        int(meta["ema_updates"]),
    )


def test_two_step_smoke_run_writes_a_checkpoint_and_a_manifest(tmp_path):
    import numpy as np
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
    # smoke skips the val pass, so calibration is the IDENTITY, never a fit.
    assert on_disk["calibration"] == {"exist_temperature": 1.0, "vis_temperature": 1.0}
    assert (run_dir / "final" / "mvq_run.json").is_file()
    drawn = on_disk["train_data"]["mix"]["1"]["realised"]
    assert set(drawn) == set(on_disk["train_data"]["mix"]["1"]["sources"])
    assert abs(sum(drawn.values()) - 1.0) < 1e-6

    # `final/` must hold the DEBIASED EMA: `save_final` cannot debias, so
    # handing it the raw zero-seeded accumulator would ship weights a factor
    # `1 - decay**t` too small (~500x here) with nothing else to show for it.
    final, ema, updates = _final_and_ema(run_dir)
    common = sorted(set(final) & set(ema))
    assert len(common) > 100
    correction = 1.0 - float(on_disk["train"]["ema"]) ** updates
    assert all(
        np.allclose(final[k], ema[k] / correction, rtol=1e-4, atol=1e-5) for k in common
    ), "final/ is not the debiased EMA"
    assert not all(
        np.allclose(final[k], ema[k], rtol=1e-4, atol=1e-5) for k in common
    ), "final/ matches the RAW accumulator -- with_ema's debias was skipped"


class _StubDS:
    """Windows of one source, carrying the per-frameset weights given."""

    def __init__(self, weights):
        self._w = weights

    def __len__(self):
        return len(self._w)

    def source_id(self, i):
        return "pseudo_x"

    def weight(self, i):
        return self._w[i]


def test_the_pseudo_weight_warning_reads_the_framesets_not_the_manifest(capsys):
    from tracking.train.mvq.config import MVQTrainConfig
    from tracking.train.mvq.run import _mix_report

    # the manifest AGREES with the config, so a check against it would say nothing
    sources = {"pseudo_x": {"kind": "pseudo", "manifest_weight": 0.3}}
    tcfg = MVQTrainConfig(pseudo_weight=0.3)
    rows = _mix_report(_StubDS([1.0, 1.0]), tcfg, {"pseudo_x": 1.0}, sources)
    assert rows["pseudo_x"]["mean_sample_weight"] == 1.0
    assert "WARNING" in capsys.readouterr().out
    _mix_report(_StubDS([0.3, 0.3]), tcfg, {"pseudo_x": 1.0}, sources)
    assert "WARNING" not in capsys.readouterr().out


def test_the_module_entry_point_exists():
    import importlib

    assert importlib.util.find_spec("tracking.train.mvq.__main__") is not None
