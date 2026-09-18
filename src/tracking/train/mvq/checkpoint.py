"""MVQ training checkpoints: model/optimizer/EMA save-restore, the
EMA-debias helper, and the final EMA snapshot.

Port of `$SRC/train/train_mvq.py:763-847,1160-1179` (jarvis_jax). Layout
per spec 7.3: `<ckpt_dir>/<step>/` (model, opt, ema, ema_meta items) and
`<run_dir>/final/` (a flat model save) -- both are what
`detector/mvq/checkpoint.py::load_mvq_model` already reads.
"""
from __future__ import annotations

import json
import os

import jax
import orbax.checkpoint as ocp
from flax import nnx


def make_manager(ckpt_dir, *, max_to_keep=3):
    """mvq-local checkpoint manager: model + optimizer + EMA + ema_meta.
    `ema_meta` is `{"ema_updates": int, "ema_zero_seeded": True}`, left at
    Orbax's default `JsonCheckpointHandler` filename -- `detector/mvq/
    checkpoint.py`'s read-only manager builds its `ema_meta` handler the
    same bare way, and `JsonCheckpointHandler.restore` does an exact
    filename lookup with no fallback, so a custom filename here would make
    every numbered checkpoint unloadable by that reader. See
    `restore_latest` for why `ema_meta`'s contents must be explicit rather
    than inferred."""
    os.makedirs(ckpt_dir, exist_ok=True)
    opts = ocp.CheckpointManagerOptions(max_to_keep=max_to_keep, save_interval_steps=1)
    return ocp.CheckpointManager(
        os.path.abspath(ckpt_dir), options=opts,
        item_names=("model", "opt", "ema", "ema_meta"))


def save_step(mngr, steps_done, model, optimizer, ema, ema_updates: int):
    mngr.save(steps_done, args=ocp.args.Composite(
        model=ocp.args.StandardSave(nnx.split(model)[1]),
        opt=ocp.args.StandardSave(nnx.split(optimizer)[1]),
        ema=ocp.args.StandardSave(ema),
        ema_meta=ocp.args.JsonSave({"ema_updates": int(ema_updates), "ema_zero_seeded": True})))


def _replicated_abstract(tree):
    """Abstract (ShapeDtypeStruct) twin of `tree`, replicated over every
    currently-visible device, for topology-independent Orbax restores."""
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P
    repl = NamedSharding(Mesh(jax.devices(), axis_names=("data",)), P())
    return jax.tree_util.tree_map(
        lambda v: jax.ShapeDtypeStruct(v.shape, v.dtype, sharding=repl)
        if hasattr(v, "shape") and hasattr(v, "dtype") else v, tree)


def restore_latest(mngr, model, optimizer, ema):
    """(model, optimizer, ema, ema_updates, start_step); unchanged inputs
    with (ema_updates, start_step) = (0, 0) if no checkpoint exists yet.

    A checkpoint missing `ema_meta`, or whose `ema_zero_seeded` is not True,
    predates the zero-seeded debiased EMA: its EMA was seeded from live
    params, not zero, so `with_ema`'s `/(1-decay**t)` debiasing would
    silently rescale it as though it needed the same correction, which it
    does not. Such a checkpoint is refused (ValueError) rather than
    resumed -- start a fresh run instead.
    """
    latest = mngr.latest_step()
    if latest is None:
        return model, optimizer, ema, 0, 0
    gm, am = nnx.split(model)
    go, ao = nnx.split(optimizer)
    am, ao, ema_t = (_replicated_abstract(t) for t in (am, ao, ema))
    try:
        r = mngr.restore(latest, args=ocp.args.Composite(
            model=ocp.args.StandardRestore(am),
            opt=ocp.args.StandardRestore(ao),
            ema=ocp.args.StandardRestore(ema_t),
            ema_meta=ocp.args.JsonRestore()))
    except KeyError as e:
        raise ValueError(
            f"checkpoint at {mngr.directory} (step {latest}) has no 'ema_meta' item -- "
            f"it predates the zero-seeded, debiased EMA and cannot be resumed. Start a "
            f"fresh run instead -- do not attempt to resume or convert this checkpoint."
        ) from e
    meta = r["ema_meta"]
    if not meta.get("ema_zero_seeded", False):
        raise ValueError(
            f"checkpoint at {mngr.directory} (step {latest}) has ema_meta={meta!r} -- "
            f"'ema_zero_seeded' is not True, so its EMA cannot be resumed by this code. "
            f"Start a fresh run instead.")
    return (nnx.merge(gm, r["model"]), nnx.merge(go, r["opt"]), r["ema"],
            int(meta["ema_updates"]), latest)


def with_ema(model, ema, decay: float, t: int):
    """A NEW module holding the DEBIASED EMA weights, independent of `model`'s
    own Variable objects. `nnx.split`/`nnx.merge` on `model` itself would
    return a module aliasing the SAME Variables as `model` (flax 0.12.8), so
    `nnx.update(em, ema)` would silently overwrite the training model's live
    params too -- `nnx.clone` makes a real copy first.

    `ema` is a raw running sum seeded from ZERO, so after `t` updates it
    still carries a `decay**t` shortfall relative to the true average -- the
    same bias Adam corrects for its moment estimates. `t=0` returns the live
    params undebiased (0/0 is undefined and shouldn't arise in normal use).
    """
    em = nnx.clone(model)
    if t > 0:
        correction = 1.0 - decay ** t
        ema = jax.tree_util.tree_map(lambda e: e / correction, ema)
        nnx.update(em, ema)
    em.eval()
    return em


def save_final(run_dir, model, ema, meta):
    """Merge already-debiased `ema` onto a clone of `model` and write it as
    `<run_dir>/final` -- a flat `StandardCheckpointer` save, matching what
    `detector/mvq/checkpoint.py::load_mvq_model` restores for `step=None` --
    plus `<run_dir>/final/mvq_run.json` holding `meta`.
    """
    final_dir = os.path.join(str(run_dir), "final")
    os.makedirs(final_dir, exist_ok=True)
    em = nnx.clone(model)
    nnx.update(em, ema)
    em.eval()
    ckptr = ocp.StandardCheckpointer()
    ckptr.save(os.path.abspath(final_dir), nnx.split(em)[1], force=True)
    ckptr.wait_until_finished()
    with open(os.path.join(final_dir, "mvq_run.json"), "w") as f:
        json.dump(meta, f, indent=1)
