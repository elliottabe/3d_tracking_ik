"""Optimizer and the maskless MVQ train step.

Port of `$SRC/train/train_mvq.py:142-205,1160` (jarvis_jax). Maskless
deviations: `_batch_to_model` drops the label mask entirely (the model's
mask kwarg defaults to `None`), and `prompt_on` is fixed at `False` rather
than drawn from that mask + a Bernoulli prompt probability -- augmentation
now lives in the loader (Task 4). With no mask, `model.py` never builds a
prompt token, so `decoder.py` skips its whole prompt-add branch outright --
the same zero contribution v2 got from `on=False` when a mask existed.
`prompt_on` still matters downstream: `mvq_loss` -> `assign_slots`
(`matching.py`) reads it to route the host fly to the prompted slot, so a
stray `True` would corrupt the existence/sex targets.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from tracking.detector.mvq.runner import _IMAGENET_MEAN, _IMAGENET_STD
from tracking.train.mvq.losses import LossWeights, mvq_loss


def normalize_crops(u8):
    return (u8.astype(jnp.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD


def _labels(params):
    return jax.tree_util.tree_map_with_path(
        lambda path, _: "backbone" if "backbone" in jax.tree_util.keystr(path) else "head", params)


def make_optimizer(model, tcfg):
    """AdamW, two LR groups (backbone vs head), ONE global gradient clip applied
    BEFORE the group split -- clipping inside each group's own transform would
    bound the backbone's and the head's gradient norms independently, which is
    not what `grad_clip` is meant to bound."""
    decay = max(tcfg.total_steps, tcfg.warmup_steps + 1)

    def sched(peak):
        return optax.warmup_cosine_decay_schedule(0.0, peak, tcfg.warmup_steps, decay, 0.0)

    def group(peak):
        return optax.adamw(sched(peak), weight_decay=tcfg.weight_decay)
    tx = optax.chain(
        optax.clip_by_global_norm(tcfg.grad_clip),
        optax.multi_transform({"backbone": group(tcfg.lr * tcfg.backbone_lr_mult),
                               "head": group(tcfg.lr)}, _labels))
    return nnx.Optimizer(model, tx, wrt=nnx.Param)


def _batch_to_model(batch):
    return dict(crops=normalize_crops(batch["crops"]), cam_valid=batch["cam_valid"],
                M=batch["M"], t_local=batch["t_local"])


def make_train_step(part_of_k, weights: LossWeights, ema_decay: float, kp_weight=None):
    """`kp_weight`: the (K,) per-keypoint loss multiplier, or None for uniform.

    No `aug`/`lr_swap`/`key`/`prompt_p`: augmentation runs in the loader, and
    the prompt is always off (see module docstring), so nothing here needs
    randomness.
    """
    pok = np.asarray(part_of_k)
    kpw = None if kp_weight is None else jnp.asarray(kp_weight)

    def loss_fn(model, batch):
        out = model(**_batch_to_model(batch), prompt_on=batch["prompt_on"])
        return mvq_loss(out, batch, weights, pok, kpw)

    @nnx.jit
    def step(model, optimizer, ema, batch):
        B = batch["cam_valid"].shape[0]
        batch = dict(batch, prompt_on=jnp.zeros((B,), bool))
        (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, batch)
        optimizer.update(model, grads)
        params = nnx.state(model, nnx.Param)
        ema = jax.tree_util.tree_map(lambda e, p: ema_decay * e + (1 - ema_decay) * p, ema, params)
        return loss, metrics, ema
    return step
