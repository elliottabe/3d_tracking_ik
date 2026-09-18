import dataclasses
import json
import os

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

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


REFERENCE_RUN = (
    "/gscratch/portia/eabe/data/Johnson_lab/jax_mvq_runs/mvq_t2_v2_20260906d/mvq_run.json"
)

# `model` keys left unset here on purpose: `None` means "the backbone preset's
# own value", and `model.py` REJECTS an explicit one that disagrees with the
# preset -- so `None` and the reference's 12/12 for dinov3_b16 are the same
# model. Every other key in every group is compared.
EXEMPT = {"train": set(REMOVED), "model": {"backbone_depth", "backbone_heads"}}


@pytest.mark.skipif(not os.path.isfile(REFERENCE_RUN), reason="reference v2 run not present")
def test_v2_preset_reproduces_the_reference_run(repo_root):
    """`train=mvq_v2` must match the run it claims to reproduce, VALUE by value.

    Completeness against the dataclasses cannot see a field whose default
    silently disagrees with the reference (`loss.other_fly_repulsion` did).
    """
    with open(REFERENCE_RUN) as f:
        ref = json.load(f)
    with initialize_config_dir(config_dir=str(repo_root / "configs"), version_base=None):
        cfg = compose(config_name="train", overrides=["train=mvq_v2"])
    groups = ("train", "loss", "aug", "model")
    got = {g: OmegaConf.to_container(cfg[g], resolve=True) for g in groups}
    diffs = []
    for g in groups:
        exempt = EXEMPT.get(g, set())
        missing = set(ref[g]) - set(got[g]) - exempt
        if missing:
            diffs.append(f"{g}: the reference has {sorted(missing)}, the config does not")
        for k in sorted((set(got[g]) & set(ref[g])) - exempt):
            a, b = got[g][k], ref[g][k]
            if isinstance(a, list) or isinstance(b, list):
                a, b = list(a), list(b)
            if a != b:
                diffs.append(f"{g}.{k}: composed {a!r} != reference {b!r}")
    assert not diffs, "train=mvq_v2 does not reproduce the reference run:\n" + "\n".join(diffs)
