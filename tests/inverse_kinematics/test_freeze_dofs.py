"""Unobserved DOFs are held at their warm start rather than optimised."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

import tracking.utils.path_utils  # noqa: F401
from tracking.inverse_kinematics.perframe import _qs_to_opt
from tracking.run import _load_anatomy

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


@pytest.fixture(scope="module")
def anatomy():
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        cfg = compose(config_name="pipeline", overrides=["recording=session0"])
    return _load_anatomy(cfg)


def test_no_patterns_optimises_everything(anatomy):
    mask = _qs_to_opt(anatomy, ())
    assert mask.shape == (anatomy.nq,)
    assert mask.all()


def test_t1l_pattern_freezes_exactly_the_left_front_leg(anatomy):
    # ELEVEN, not the seven configs/anatomy/v1.yaml's curated joint_names lists:
    # names_qpos comes from align_joint_dims on the COMPILED model, which carries
    # tarsus2..tarsus5_T1_left as well. Measured on the v1 model 2026-09-17.
    mask = _qs_to_opt(anatomy, ["*_T1_left"])
    frozen = [n for n, free in zip(anatomy.names_qpos, mask, strict=True) if not free]
    assert len(frozen) == 11
    assert all(n.endswith("_T1_left") for n in frozen)
    # the root freejoint is never frozen by a leg pattern
    assert mask[:7].all()


def test_pattern_matching_nothing_warns_and_freezes_nothing(anatomy, capsys):
    mask = _qs_to_opt(anatomy, ["*_T9_left"])
    assert mask.all()
    assert "matched no qpos slot" in capsys.readouterr().err


def test_pattern_matching_everything_refuses(anatomy):
    with pytest.raises(ValueError, match="nothing left to solve"):
        _qs_to_opt(anatomy, ["*"])
