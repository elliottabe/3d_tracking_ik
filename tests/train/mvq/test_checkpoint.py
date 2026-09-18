import json

import jax
import numpy as np
import optax
from flax import nnx

from tracking.train.mvq.checkpoint import make_manager, restore_latest, save_step, with_ema


class Tiny(nnx.Module):
    def __init__(self, rngs):
        self.w = nnx.Param(nnx.initializers.ones(rngs.params(), (4,)))

    def __call__(self, x):
        return x * self.w


def _fixture():
    model = Tiny(nnx.Rngs(0))
    opt = nnx.Optimizer(model, optax.sgd(0.1), wrt=nnx.Param)
    return model, opt, nnx.state(model, nnx.Param)


def test_save_then_restore_round_trips_step_and_ema_updates(tmp_path):
    model, opt, ema = _fixture()
    mngr = make_manager(tmp_path / "ckpt")
    save_step(mngr, 7, model, opt, ema, ema_updates=7)
    mngr.wait_until_finished()
    m2, o2, e2, updates, start = restore_latest(make_manager(tmp_path / "ckpt"), *_fixture())
    assert start == 7 and updates == 7


def test_restore_with_no_checkpoint_returns_step_zero(tmp_path):
    model, opt, ema = _fixture()
    _, _, _, updates, start = restore_latest(make_manager(tmp_path / "ckpt"), model, opt, ema)
    assert (updates, start) == (0, 0)


def test_with_ema_debias_is_identity_at_large_t():
    model, _, ema = _fixture()
    scaled = jax.tree.map(lambda v: v * 0.5, ema)
    late = with_ema(model, scaled, 0.999, 1_000_000)
    leaf = jax.tree.leaves(nnx.state(late, nnx.Param))[0]
    assert np.allclose(np.asarray(leaf), 0.5, atol=1e-6)


def test_ema_meta_records_the_zero_seed_flag(tmp_path):
    model, opt, ema = _fixture()
    mngr = make_manager(tmp_path / "ckpt")
    save_step(mngr, 1, model, opt, ema, ema_updates=1)
    mngr.wait_until_finished()
    meta = json.loads((tmp_path / "ckpt" / "1" / "ema_meta" / "ema_meta.json").read_text())
    assert meta["ema_zero_seeded"] is True and meta["ema_updates"] == 1


def test_final_layout_matches_what_the_runner_expects(tmp_path):
    from tracking.train.mvq.checkpoint import save_final
    model, _, ema = _fixture()
    save_final(tmp_path, model, ema, {"calibration": {}})
    assert (tmp_path / "final").is_dir()
