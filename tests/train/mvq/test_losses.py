import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tracking.detector.mvq.slots import N_SLOTS, SEX_FEMALE, SEX_MALE, SEX_UNKNOWN
from tracking.train.mvq.losses import LossWeights, mvq_loss, wing_kp_weight

B, F, T, C, K = 2, 2, 1, 3, 5


def _batch(rng, *, two_fly=True):
    kp3d = rng.normal(size=(B, F, T, K, 3)).astype(np.float32)
    M = np.tile(np.eye(3, 3, dtype=np.float32)[:2], (B, C, 1, 1)).astype(np.float32)
    t_local = rng.normal(size=(B, T, C, 2)).astype(np.float32) * 0.1
    return {
        "kp3d_local": jnp.asarray(kp3d),
        "has3d": jnp.ones((B, F, T, K), bool),
        "kp2d": jnp.asarray(rng.normal(size=(B, F, T, C, K, 2)).astype(np.float32)),
        "vis2d": jnp.ones((B, F, T, C, K), bool),
        "cam_valid": jnp.ones((B, T, C), bool),
        "M": jnp.asarray(M), "t_local": jnp.asarray(t_local),
        "px_scale": jnp.ones((B,), jnp.float32),
        "fly_valid": jnp.asarray(np.array([[True, two_fly]] * B)),
        "fly_sex": jnp.asarray(np.array([[SEX_FEMALE, SEX_MALE]] * B, np.int8)),
        "prompt_on": jnp.zeros((B,), bool),
        "unlabelled_sex": jnp.full((B,), SEX_UNKNOWN, jnp.int8),
        "sample_weight": jnp.ones((B,), jnp.float32),
        "is_negative": jnp.zeros((B,), bool),
    }


def _out(rng, batch, *, perfect=False):
    xyz = rng.normal(size=(B, N_SLOTS, T, K, 3)).astype(np.float32)
    if perfect:
        xyz[:, 1] = np.asarray(batch["kp3d_local"])[:, 0]
        xyz[:, 2] = np.asarray(batch["kp3d_local"])[:, 1]
    return {
        "xyz": jnp.asarray(xyz),
        "uv": jnp.asarray(rng.normal(size=(B, N_SLOTS, T, C, K, 2)).astype(np.float32)),
        "conf_logit": jnp.zeros((B, N_SLOTS, T, K), jnp.float32),
        "vis_logit": jnp.zeros((B, N_SLOTS, T, C, K), jnp.float32),
        "exist_logit": jnp.zeros((B, N_SLOTS), jnp.float32),
        "sex_logit": jnp.zeros((B, N_SLOTS), jnp.float32),
    }


def test_wing_weight_is_by_name_not_index():
    w = wing_kp_weight(["Head", "WingL", "Thorax", "WingR"], mult=2.0)
    assert list(w) == [1.0, 2.0, 1.0, 2.0]


def test_loss_is_finite_and_scalar():
    rng = np.random.default_rng(0)
    b = _batch(rng)
    total, m = mvq_loss(_out(rng, b), b, LossWeights(), np.arange(K))
    assert np.isfinite(float(total)) and np.asarray(total).shape == ()
    assert np.isfinite(float(m["reproj"]))


def test_perfect_3d_drives_mpjpe_to_zero():
    rng = np.random.default_rng(1)
    b = _batch(rng)
    _, m = mvq_loss(_out(rng, b, perfect=True), b, LossWeights(), np.arange(K))
    assert float(m["mpjpe3d_units"]) < 1e-4


def test_wrong_slot_count_raises():
    rng = np.random.default_rng(2)
    b = _batch(rng)
    out = _out(rng, b)
    out["xyz"] = out["xyz"][:, : N_SLOTS - 1]
    with pytest.raises(ValueError, match="n_instances"):
        mvq_loss(out, b, LossWeights(), np.arange(K))


def test_sample_weight_scales_a_sample_out_of_the_loss():
    rng = np.random.default_rng(3)
    b = _batch(rng)
    out = _out(rng, b)
    base = float(mvq_loss(out, b, LossWeights(), np.arange(K))[1]["reproj"])
    b2 = dict(b, sample_weight=jnp.asarray([1.0, 0.0], jnp.float32))
    only_first = float(mvq_loss(out, b2, LossWeights(), np.arange(K))[1]["reproj"])
    assert not np.isclose(base, only_first)


def test_persist_is_zero_at_t_equals_one():
    rng = np.random.default_rng(4)
    b = _batch(rng)
    _, m = mvq_loss(_out(rng, b), b, LossWeights(persist=0.5), np.arange(K))
    assert float(m["persist"]) == 0.0


def test_other_fly_repulsion_off_by_default_and_on_when_set():
    rng = np.random.default_rng(5)
    b = _batch(rng)
    out = _out(rng, b)
    _, m0 = mvq_loss(out, b, LossWeights(other_fly_repulsion=0.0), np.arange(K))
    _, m1 = mvq_loss(out, b, LossWeights(other_fly_repulsion=20.0), np.arange(K))
    assert float(m0["other_rep"]) == 0.0
    assert float(m1["other_rep"]) > 0.0


def test_kp_weight_changes_the_loss():
    rng = np.random.default_rng(6)
    b = _batch(rng)
    out = _out(rng, b)
    flat = float(mvq_loss(out, b, LossWeights(), np.arange(K))[0])
    wing = float(mvq_loss(out, b, LossWeights(), np.arange(K),
                          np.array([1, 2, 1, 2, 1], np.float32))[0])
    assert not np.isclose(flat, wing)


def test_loss_jits():
    rng = np.random.default_rng(7)
    b = _batch(rng)
    out = _out(rng, b)
    f = jax.jit(lambda o, bb: mvq_loss(o, bb, LossWeights(), np.arange(K))[0])
    assert np.isfinite(float(f(out, b)))
