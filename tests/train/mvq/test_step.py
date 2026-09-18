import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tracking.train.mvq.config import MVQTrainConfig
from tracking.train.mvq.step import make_optimizer, normalize_crops


def _skip_if_device_oom(exc):
    """Skip ONLY on a device out-of-memory (the GPU is shared); re-raise
    anything else -- a shape or dtype regression must fail, not read as a pass.
    """
    msg = str(exc)
    if "RESOURCE_EXHAUSTED" in msg or "out of memory" in msg.lower() or "OUT_OF_MEMORY" in msg:
        warnings.warn(f"MVQModel train step not exercised: {msg.splitlines()[0]}", stacklevel=2)
        pytest.skip(f"device out of memory: {msg.splitlines()[0]}")
    raise


def test_normalize_crops_matches_imagenet_stats():
    # `runner.py` names these `_IMAGENET_MEAN`/`_IMAGENET_STD`; consolidating
    # the three ImageNet-stats copies under one public name is future work
    # (plan 3, task-9-brief.md), not this task -- import as they exist today.
    from tracking.detector.mvq.runner import _IMAGENET_MEAN as IMAGENET_MEAN
    from tracking.detector.mvq.runner import _IMAGENET_STD as IMAGENET_STD
    u8 = np.full((1, 1, 1, 2, 2, 3), 255, np.uint8)
    got = np.asarray(normalize_crops(jnp.asarray(u8)))
    want = (1.0 - np.asarray(IMAGENET_MEAN)) / np.asarray(IMAGENET_STD)
    assert np.allclose(got[0, 0, 0, 0, 0], want, atol=1e-5)


def test_optimizer_has_two_lr_groups_and_one_global_clip():
    import inspect

    import tracking.train.mvq.step as step
    src = inspect.getsource(step.make_optimizer)
    assert "clip_by_global_norm" in src and "multi_transform" in src
    assert src.index("clip_by_global_norm") < src.index("multi_transform"), \
        "the global clip must precede the group split"


def test_backbone_lr_is_scaled_by_the_multiplier():
    """The backbone group really is stepped at lr*mult and the head at lr."""
    from flax import nnx

    class Tiny(nnx.Module):
        def __init__(self, rngs):
            self.backbone = nnx.Linear(2, 2, rngs=rngs)
            self.head = nnx.Linear(2, 2, rngs=rngs)

    m = Tiny(nnx.Rngs(0))
    cfg = MVQTrainConfig(
        lr=1e-3, backbone_lr_mult=0.1, warmup_steps=0, total_steps=10, grad_clip=1e9)
    opt = make_optimizer(m, cfg)
    before = jax.tree.map(lambda p: np.asarray(p).copy(), nnx.state(m, nnx.Param))
    grads = jax.tree.map(lambda p: jnp.ones_like(p), nnx.state(m, nnx.Param))
    opt.update(m, grads)
    after = nnx.state(m, nnx.Param)
    bb_kernel = float(after["backbone"]["kernel"][...][0, 0])
    hd_kernel = float(after["head"]["kernel"][...][0, 0])
    d_bb = abs(bb_kernel - float(before["backbone"]["kernel"][0, 0]))
    d_hd = abs(hd_kernel - float(before["head"]["kernel"][0, 0]))
    assert np.isclose(d_bb / d_hd, cfg.backbone_lr_mult, rtol=0.05), (d_bb, d_hd)


def test_train_step_is_maskless():
    """Asserted on BEHAVIOUR, not on the module text: the docstring must stay
    free to name the thing it explains."""
    import inspect

    import tracking.train.mvq.step as step
    batch = {"crops": jnp.zeros((1, 1, 1, 2, 2, 3), jnp.uint8),
             "cam_valid": jnp.ones((1, 1, 1), bool), "M": jnp.zeros((1, 1, 3, 4)),
             "t_local": jnp.zeros((1, 1, 3)), "mask": jnp.ones((1, 1))}
    assert set(step._batch_to_model(batch)) == {"crops", "cam_valid", "M", "t_local"}, \
        "the model call must carry no label mask"
    src = inspect.getsource(step.make_train_step)
    assert "prompt_mask" not in src, "the train step must not reference prompt_mask"
    assert "bernoulli" not in src, "the prompt draw must be gone"
    assert 'prompt_on=jnp.zeros((B,), bool)' in src, "prompt_on must be fixed at False"


@pytest.mark.skipif(not jax.devices(), reason="no jax device")
def test_two_steps_do_not_increase_a_fixed_batch_loss():
    """Reuse test_losses.py's `_batch`/`_out` shapes; build the real model once.

    Skips (rather than fails) if the model can't be built or run here -- this
    is a shared GPU and a concurrent session can leave too little free memory
    for even this tiny model.
    """
    from flax import nnx
    from tests.train.mvq.test_losses import _batch

    # The real class is `MVQModel` (there is no bare `MVQ` alias).
    from tracking.detector.mvq.model import MVQConfig, MVQModel
    from tracking.train.mvq.losses import LossWeights
    from tracking.train.mvq.step import make_optimizer, make_train_step

    rng = np.random.default_rng(0)
    batch = _batch(rng)
    batch["crops"] = jnp.zeros((2, 1, 3, 448, 448, 3), jnp.uint8)
    # lr=1e-5, not 1e-4: with warmup_steps=0 the first AdamW step moves every
    # touched parameter by roughly +/-lr regardless of gradient scale, and at
    # 1e-4 that overshoots badly on a freshly random-initialised decoder
    # (verified: 1e-4 raises this fixed batch's loss ~70%, 1e-5 keeps it
    # within tolerance) -- a real-init/real-image run always has
    # warmup_steps>0, so this is a test-scale artifact, not a step.py bug.
    cfg = MVQTrainConfig(lr=1e-5, warmup_steps=0, total_steps=2, batch_size=2)
    try:
        # num_cameras/num_keypoints must match test_losses._batch's C=3, K=5
        # fixture -- the model's own camera/keypoint axes are otherwise fixed
        # (7, 50) and would not broadcast against it.
        model = MVQModel(MVQConfig(n_instances=4, num_cameras=3, num_keypoints=5),
                          rngs=nnx.Rngs(0))
        opt = make_optimizer(model, cfg)
        ema = nnx.state(model, nnx.Param)
        before = np.asarray(jax.tree.leaves(ema)[0]).copy()
        step = make_train_step(np.arange(5), LossWeights(), cfg.ema)
        losses = []
        for _ in range(2):
            loss, _, ema = step(model, opt, ema, batch)
            losses.append(float(loss))
        after = np.asarray(jax.tree.leaves(nnx.state(model, nnx.Param))[0])
    except Exception as e:
        _skip_if_device_oom(e)
    assert losses[1] <= losses[0] * 1.05, f"loss rose: {losses}"
    assert not np.array_equal(before, after)
