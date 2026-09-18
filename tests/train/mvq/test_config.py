import dataclasses

import pytest
from hydra import compose, initialize_config_dir

from tracking.train.mvq.config import MVQTrainConfig

REMOVED = ("prompt_p_start", "prompt_p_end", "prompt_anneal_steps",
           "pseudo_root", "singlefly_root", "negatives_root", "allow_calib_mismatch")

V2 = {"lr": 3e-4, "warmup_steps": 1000, "total_steps": 40000, "batch_size": 32,
      "backbone_lr_mult": 0.1, "ema": 0.999, "jitter_units": 10.0,
      "female_host_weight": 4.27, "female_host_target": 0.5, "negatives_frac": 0.05,
      "copy_paste_p": 0.8, "copy_paste_contact_p": 0.7, "wing_kp_mult": 2.0,
      "loader_workers": "processes", "num_workers": 24}


def test_maskless_and_single_root_fields_are_gone():
    names = {f.name for f in dataclasses.fields(MVQTrainConfig)}
    for gone in REMOVED:
        assert gone not in names, f"{gone} must not survive the maskless/one-root port"


def test_pseudo_weight_survives_as_an_expectation():
    assert MVQTrainConfig().pseudo_weight == 0.3


@pytest.mark.parametrize("key,want", sorted(V2.items()))
def test_v2_preset_matches_the_spec(key, want, repo_root):
    with initialize_config_dir(config_dir=str(repo_root / "configs"), version_base=None):
        cfg = compose(config_name="train", overrides=["train=mvq_v2"])
    assert cfg.train[key] == want


def test_v2_tuple_fields(repo_root):
    with initialize_config_dir(config_dir=str(repo_root / "configs"), version_base=None):
        cfg = compose(config_name="train", overrides=["train=mvq_v2"])
    assert list(cfg.train.window_lengths) == [1, 2]
    assert list(cfg.train.pair_deltas) == [1, 4, 16]
    assert list(cfg.train.copy_paste_contact_sep) == [4.0, 25.0]
