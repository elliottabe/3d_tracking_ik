"""Load an mvq checkpoint into a fresh `MVQModel` -- a TOLERANT restore.

Two properties are load-bearing and both are deliberate:

  * device-count-independent REPLICATED sharding
    (`replicated_abstract_tree`), so a checkpoint saved on N GPUs restores on
    1 GPU or on CPU without Orbax's "Topology mismatch detected";
  * an explicit `_CKPT_ITEM_HANDLERS` per item for the `ckpt/<step>` route --
    without one, Orbax's `item_metadata(step)` returns `None` for every item
    and warns instead of restoring.

The restore target is built from the CHECKPOINT's own stored tree shapes,
then merged onto a freshly built model BY PATH AND SHAPE
(`merge_state_by_path`) -- not the other way around. Building the target from
the fresh model instead raises the moment the checkpoint's architecture
differs even slightly (a leaf added since, an instance-slot count changed),
which is routine across training runs of the same model family.
"""

from __future__ import annotations

import json
import os
import sys

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tracking.detector.mvq.model import MVQConfig, MVQModel

# Orbax needs an explicit handler per item to answer `item_metadata(step)`
# (without one it warns "could not be restored" and returns None for every
# item) -- see `load_mvq_model`'s `ckpt/` branch, which reads the
# checkpoint's OWN tree shapes from that metadata.
_CKPT_ITEM_HANDLERS = {
    "model": ocp.StandardCheckpointHandler(),
    "ema": ocp.StandardCheckpointHandler(),
    "ema_meta": ocp.JsonCheckpointHandler(),
}

# Files whose bytes identify a checkpoint, hashed in this order by
# `checkpoint_sha256`. `_METADATA` pins the parameter TREE (paths and shapes
# -- a different architecture is a different checkpoint) and
# `_CHECKPOINT_METADATA` carries orbax's commit timestamp, so retraining a
# run at the SAME path with the SAME architecture still moves the signature.
# `ckpt/<step>/` keeps its tree metadata one level down, under `model/`.
_DIGEST_FILES = ("_CHECKPOINT_METADATA", "_METADATA", os.path.join("model", "_METADATA"))


def resolved_step(step):
    """`step` as a CONCRETE int, or None for a `final/` dir.

    `"latest"` is refused by name: it is not a checkpoint identity -- it
    names a different step every time the training job saves -- so a gate
    signature built from it would compare EQUAL to an artifact produced by
    different weights, which is the exact failure the signature exists to
    catch. `MVQRunner` resolves it once at construction and stores the
    concrete int.
    """
    if step is None:
        return None
    try:
        return int(step)
    except (TypeError, ValueError):
        raise ValueError(
            f"mvq.step must be a concrete int; resolve {step!r} before writing the config "
            f"-- a signature built from a moving step would accept weights it was never "
            f"built from"
        ) from None


def checkpoint_dir(checkpoint, step=None):
    """The directory a `(checkpoint, step)` pair actually loads from.

    `step is None`: `checkpoint` IS a `final/` dir. Otherwise it is the RUN
    dir and the weights live in `<run>/ckpt/<step>`.
    """
    step = resolved_step(step)
    return str(checkpoint) if step is None else os.path.join(str(checkpoint), "ckpt", str(step))


def checkpoint_sha256(checkpoint, step=None, *, n=16):
    """First `n` hex chars of sha256 over the checkpoint's metadata files.

    Hashing the parameter shards themselves would be gigabytes per call; the
    metadata is a few hundred KB and moves whenever the tree or the save
    does.
    """
    import hashlib

    d = checkpoint_dir(checkpoint, step)
    h = hashlib.sha256()
    found = []
    for rel in _DIGEST_FILES:
        p = os.path.join(d, rel)
        if os.path.exists(p):
            with open(p, "rb") as f:
                h.update(rel.encode())
                h.update(f.read())
            found.append(rel)
    if not found:
        raise FileNotFoundError(
            f"{d} carries none of {list(_DIGEST_FILES)} -- it is not an orbax checkpoint "
            f"directory, so its gate signature cannot name which weights produced a kp3d.npz"
        )
    return h.hexdigest()[:n]


def _keypath_to_name(path_or_key):
    """Normalise a `jax.tree_util.keystr(...)` path (or a raw path tuple) to
    plain `a/b/c` form, e.g. `"['decoder']['e_inst']"` -> `"decoder/e_inst"`.
    Handles both dict keys (`['layer']`, quoted) and list/sequence indices
    (`[0]`, unquoted)."""
    key = path_or_key if isinstance(path_or_key, str) else jax.tree_util.keystr(path_or_key)
    key = key.replace("'", "").replace("][", "/").replace("[", "/").replace("]", "")
    return key.strip("/")


def replicated_abstract_tree(meta_tree):
    """ShapeDtypeStruct twin of an Orbax METADATA tree, replicated over every
    currently-visible device.

    Two properties matter and both are deliberate:
      - the shapes/dtypes come from the CHECKPOINT, not from any live model,
        so a checkpoint whose architecture differs from the current code's
        can be read at all;
      - the sharding is REPLICATED rather than the checkpoint's own, which
        raises "Topology mismatch detected" the moment the current process's
        device count differs from the training run's.
    Non-array metadata entries pass through untouched.
    """
    repl = NamedSharding(Mesh(jax.devices(), axis_names=("data",)), P())

    def is_arr(x):
        return hasattr(x, "shape") and hasattr(x, "dtype")

    return jax.tree_util.tree_map(
        lambda m: jax.ShapeDtypeStruct(tuple(m.shape), m.dtype, sharding=repl) if is_arr(m) else m,
        meta_tree,
        is_leaf=is_arr,
    )


def restore_own_tree(src_dir):
    """Restore `src_dir` (a StandardCheckpointer dir, e.g. a run's `final/`)
    as ITS OWN pytree via `replicated_abstract_tree`, ready to hand to
    `merge_state_by_path`."""
    ck = ocp.StandardCheckpointer()
    meta = ck.metadata(os.path.abspath(src_dir)).item_metadata.tree
    return ck.restore(os.path.abspath(src_dir), target=replicated_abstract_tree(meta))


def merge_state_by_path(model, restored_tree):
    """Merge an already-restored checkpoint tree onto `model`'s state BY PATH
    AND SHAPE: every leaf whose normalised path and shape match is taken from
    the checkpoint, `decoder/e_inst` with fewer source rows contributes its
    leading rows, and everything else keeps `model`'s own (fresh-init, or
    previously-merged) value. Returns the new model and the list of leaves
    NOT fully restored (human-readable paths) -- the caller reports them.

    The two trees being compared flatten to DIFFERENT key spellings: `pure`
    (from `nnx.to_pure_dict` on the live `model`'s split state) has paths
    like `['decoder']['e_inst']`, while the restored State (each leaf a
    `VariableState` with its own `.value`) flattens with a trailing
    `['value']`. Both are normalised to plain `a/b/c` form here, and that
    trailing `/value` segment is stripped before the two are matched by
    name, or every lookup would silently miss and every leaf would report
    itself skipped.
    """
    gdef, state = nnx.split(model)
    pure = nnx.to_pure_dict(state)
    src_flat = {}
    for k, v in jax.tree_util.tree_flatten_with_path(restored_tree)[0]:
        name = _keypath_to_name(k)
        if name.endswith("/value"):
            name = name[: -len("/value")]
        src_flat[name] = v
    skipped = []

    def merge(path, v):
        name = _keypath_to_name(path)
        s = src_flat.get(name)
        if s is None or not hasattr(v, "shape"):
            skipped.append(name)
            return v
        if tuple(s.shape) == tuple(v.shape):
            return jnp.asarray(s, v.dtype)
        if name.endswith("e_inst") and s.shape[1:] == v.shape[1:] and s.shape[0] < v.shape[0]:
            skipped.append(f"{name} (partial rows 0:{s.shape[0]})")
            return v.at[: s.shape[0]].set(jnp.asarray(s, v.dtype))
        skipped.append(name)
        return v

    new_pure = jax.tree_util.tree_map_with_path(merge, pure)
    nnx.replace_by_pure_dict(state, new_pure)
    return nnx.merge(gdef, state), skipped


def _report_unrestored(what, skipped):
    """Print a path-mismatched restore to STDERR, naming the leaves kept at
    fresh init -- a silent skip here would leave initialised weights, which
    looks like a quality regression rather than a loading bug."""
    if skipped:
        print(
            f"[mvq] {what}: {len(skipped)} leaf/leaves NOT restored (unrestored, kept at "
            f"fresh init): {sorted(skipped)}",
            file=sys.stderr,
            flush=True,
        )
    return skipped


def load_mvq_model(run_dir_or_final, *, step=None, attn_impl=None):
    """Load an mvq checkpoint into a fresh `MVQModel` -- returns `(model, meta)`.

    `step is None` (default): `run_dir_or_final` IS a `final/` directory --
    the EMA the training run already debiased before saving it there, so
    this restores it directly, no further debiasing needed.

    `step` given (an int step, or `"latest"`): `run_dir_or_final` is the RUN
    directory instead (the parent of `final/` and `ckpt/`) -- restores the
    RAW EMA running sum plus the live model's non-Param state from
    `<run_dir>/ckpt/<step>` (an `orbax.CheckpointManager` with items
    `model`/`opt`/`ema`/`ema_meta`), then debiases the EMA by
    `1 - decay**ema_updates`. `decay` is read from the run's
    `mvq_run.json`'s `train.ema` when that file and key exist; it defaults
    to 0.999 otherwise.

    `mvq_run.json` is REQUIRED either way for the model architecture
    (`MVQConfig`). When `step` is given, this loader looks in
    `<run_dir>/mvq_run.json` FIRST and falls back to
    `<run_dir>/final/mvq_run.json` for an older run directory that only
    ever wrote the `final/` copy.

    `attn_impl`: override the config's own `attn_impl` after loading (e.g. a
    "cudnn" GPU training run evaluated on a CPU host, which has no cuDNN
    flash-attention kernel -- pass `attn_impl="xla"`).

    TOLERANT restore (both branches): the restore target is built from the
    CHECKPOINT's own stored tree and merged onto a freshly built model by
    path and shape (`merge_state_by_path`) -- so a checkpoint whose leaf
    count or instance-slot count differs slightly from today's model still
    loads, rather than raising outright. Leaves that could not be restored
    keep their fresh init, are printed to stderr (`_report_unrestored`), and
    are returned in `meta["_unrestored_leaves"]` so a caller can refuse to
    report a metric that depends on an unrestored head.
    """
    final_dir = run_dir_or_final if step is None else os.path.join(run_dir_or_final, "final")
    if step is None:
        meta_path = os.path.join(final_dir, "mvq_run.json")
    else:
        run_dir_meta = os.path.join(run_dir_or_final, "mvq_run.json")
        meta_path = (
            run_dir_meta
            if os.path.exists(run_dir_meta)
            else os.path.join(final_dir, "mvq_run.json")
        )
    with open(meta_path) as f:
        meta = json.load(f)
    cfg_kwargs = dict(meta["model"])
    if attn_impl is not None:
        cfg_kwargs["attn_impl"] = attn_impl
    cfg = MVQConfig(**cfg_kwargs)
    # A REAL (not eval_shape) init: `merge_state_by_path` keeps this model's
    # own values for any leaf the checkpoint does not carry, so they have to
    # exist.
    model = MVQModel(cfg, rngs=nnx.Rngs(0))

    if step is None:
        model, skipped = merge_state_by_path(model, restore_own_tree(final_dir))
        meta["_unrestored_leaves"] = _report_unrestored(f"load {final_dir}", skipped)
        model.eval()
        return model, meta

    ckpt_dir = os.path.join(run_dir_or_final, "ckpt")
    mngr = ocp.CheckpointManager(
        os.path.abspath(ckpt_dir),
        options=ocp.CheckpointManagerOptions(read_only=True),
        item_names=("model", "opt", "ema", "ema_meta"),
        item_handlers=_CKPT_ITEM_HANDLERS,
    )
    use_step = mngr.latest_step() if step == "latest" else step

    def tree_of(x):
        return getattr(x, "tree", x)

    im = mngr.item_metadata(use_step)
    r = mngr.restore(
        use_step,
        args=ocp.args.Composite(
            model=ocp.args.StandardRestore(replicated_abstract_tree(tree_of(im["model"]))),
            ema=ocp.args.StandardRestore(replicated_abstract_tree(tree_of(im["ema"]))),
            ema_meta=ocp.args.JsonRestore(),
        ),
    )
    model, skipped = merge_state_by_path(model, r["model"])
    decay = meta.get("train", {}).get("ema", 0.999)
    t = int(r["ema_meta"]["ema_updates"])
    correction = 1.0 - decay**t if t > 0 else 1.0
    ema_params = jax.tree_util.tree_map(lambda e: e / correction, r["ema"])
    # The debiased EMA overwrites the params it carries; every other leaf
    # (non-Param state, and anything the EMA predates) keeps what the
    # `model` item just restored -- the tolerant generalisation of
    # `nnx.update`.
    model, skipped_ema = merge_state_by_path(model, ema_params)
    meta["_unrestored_leaves"] = _report_unrestored(f"load {ckpt_dir}/{use_step} (model)", skipped)
    _report_unrestored(
        f"load {ckpt_dir}/{use_step} (ema, non-Param leaves expected)",
        [s for s in skipped_ema if s not in skipped],
    )
    model.eval()
    return model, meta
