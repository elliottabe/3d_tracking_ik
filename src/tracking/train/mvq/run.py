"""`python -m tracking.train.mvq` -- the MVQ training driver.

Port of `$SRC/train/train_mvq.py:847-1160` (jarvis_jax). Deviations: ONE
unified training root (one `WindowDataset` per window length, not a
`ConcatWindowDataset` over four separate roots; validation is that same
root's val split), and maskless (no prompt annealing, no `prompt_p`).
Augmentation is applied by the dataset, not by the train step.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import math
import os
import sys
import time

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh
from omegaconf import DictConfig, OmegaConf

import tracking.utils.path_utils  # noqa: F401  registers ${repo_root:}
from tracking.curate import schema
from tracking.detector.backbones.dinov3 import (
    HF_REPOS,
    dinov3_snapshot,
    load_dinov3_safetensors,
)
from tracking.detector.mvq.checkpoint import merge_state_by_path, restore_own_tree
from tracking.detector.mvq.model import MVQConfig, MVQModel
from tracking.train.common.prefetch import prefetch
from tracking.train.common.sharding import data_parallel_mesh, replicate
from tracking.train.data.augment import MVAugParams, assert_lr_swap_covers, build_lr_swap
from tracking.train.data.copypaste import CopyPasteParams
from tracking.train.data.windows import WindowDataset, window_batches
from tracking.train.mvq.calibrate import fit_temperature, write_calibration
from tracking.train.mvq.checkpoint import (
    make_manager,
    restore_latest,
    save_final,
    save_step,
    with_ema,
)
from tracking.train.mvq.config import MVQTrainConfig
from tracking.train.mvq.evaluate import _fwd, _padded_batches, cohorts, evaluate
from tracking.train.mvq.losses import LossWeights, build_part_index, wing_kp_weight
from tracking.train.mvq.matching import assign_slots, slot_ignore
from tracking.train.mvq.sampling import MixCounter, mix_weights
from tracking.train.mvq.step import make_optimizer, make_train_step

__all__ = ["main", "run_training"]

RATIO_BATCHES = 200
PSEUDO_KINDS = ("pseudo", "singlefly")


def _dataclass(cls, node):
    """`cls` built from a config mapping; a list becomes a tuple where the
    field is declared one."""
    data = dict(OmegaConf.to_container(node, resolve=True)) if node is not None else {}
    types = {f.name: str(f.type) for f in dataclasses.fields(cls)}
    return cls(
        **{
            k: tuple(v) if isinstance(v, list) and "tuple" in types.get(k, "") else v
            for k, v in data.items()
        }
    )


def _mesh(tcfg):
    """Data-parallel mesh over the visible devices.

    A batch that does not divide over them cannot be sharded; that is an error
    in a real run and a device-subset fallback under `train.smoke`, whose whole
    point is to run the real pipeline at a batch size no cluster shape fits.
    """
    devices = jax.devices()
    if tcfg.batch_size % len(devices) == 0:
        return data_parallel_mesh()
    if not tcfg.smoke:
        raise ValueError(
            f"batch_size {tcfg.batch_size} is not divisible by the {len(devices)} visible "
            f"devices -- pick a batch size that shards evenly"
        )
    n = max(k for k in range(1, len(devices) + 1) if tcfg.batch_size % k == 0)
    print(
        f"[mvq] smoke: batch_size {tcfg.batch_size} shards over {n} of {len(devices)} devices",
        flush=True,
    )
    return Mesh(devices[:n], axis_names=("data",))


def _source_table(manifest) -> dict:
    """`{source_id: {kind, manifest_weight}}` for the root's declared sources."""
    return {
        sid: {
            "kind": str(entry.get("kind", "unknown")),
            "manifest_weight": float(entry.get("weight", 1.0)),
        }
        for sid, entry in sorted((manifest.get("sources") or {}).items())
    }


def _mix_report(ds, tcfg, mass, sources) -> dict:
    """Window count, sampled mass and mean `sample_weight` per source, for one T.

    The weight that multiplies the loss is each FRAMESET's own `weight`
    (`WindowDataset.weight` -> frameset, then recording, then root default --
    never `manifest["sources"]`, which is an independent place on disk), so
    that realised mean is what `train.pseudo_weight` is checked against. A
    disagreement WARNS: the config number describes the data, it cannot
    override it.
    """
    total = collections.defaultdict(float)
    n = collections.Counter()
    for i in range(len(ds)):
        sid = ds.source_id(i)
        total[sid] += ds.weight(i)
        n[sid] += 1
    rows = {}
    for sid in sorted(mass):
        mean_w = total[sid] / max(n[sid], 1)
        rows[sid] = {
            "n_windows": int(n[sid]),
            "mass": float(mass[sid]),
            "mean_sample_weight": float(mean_w),
        }
        kind = sources.get(sid, {}).get("kind", "unknown")
        flag = ""
        if kind in PSEUDO_KINDS and abs(mean_w - float(tcfg.pseudo_weight)) > 1e-6:
            flag = (
                f"  <-- WARNING: train.pseudo_weight={tcfg.pseudo_weight:g} disagrees; "
                f"training at the framesets' {mean_w:.4f}"
            )
        print(
            f"[mvq]   {sid}: kind={kind} windows={n[sid]} mass={mass[sid]:.4f} "
            f"manifest weight={sources.get(sid, {}).get('manifest_weight', float('nan')):g} "
            f"mean sample_weight={mean_w:.4f}{flag}",
            flush=True,
        )
    return rows


def _temperatures(model, ds, tcfg, mesh):
    """`(exist_T, vis_T)` fitted on the val split's own logits.

    One unprompted pass through `evaluate`'s already-jitted forward, so it
    costs no extra compile; `1.0, 1.0` (the identity, what `MVQRunner` divides
    by) when the split yields no scorable logit.
    """
    n = len(ds)
    e_logits, e_targets, v_logits, v_targets = [], [], [], []
    offset = 0
    for jb in prefetch(_padded_batches(ds, tcfg.batch_size, tcfg.num_workers), mesh):
        b0 = min(tcfg.batch_size, n - offset)
        on = jnp.zeros((tcfg.batch_size,), bool)
        out = _fwd(model, jb, on)
        fly_valid = np.asarray(jb["fly_valid"])
        has = np.asarray(jb["has3d"]).astype(np.float32)
        cen = (np.asarray(jb["kp3d_local"]) * has[..., None]).sum((2, 3)) / np.maximum(
            has.sum((2, 3)), 1.0
        )[..., None]
        exist = np.asarray(out["exist_logit"])
        n_slots = exist.shape[1]
        assign, slot_t = assign_slots(
            jnp.asarray(jb["fly_sex"]),
            jnp.asarray(fly_valid),
            on,
            jnp.asarray(np.linalg.norm(cen, axis=-1)),
            n_slots,
        )
        assign, slot_t = np.asarray(assign), np.asarray(slot_t)
        ignore = np.asarray(slot_ignore(jnp.asarray(jb["unlabelled_sex"]), n_slots)) & ~slot_t
        keep = ~ignore[:b0]
        e_logits.append(exist[:b0][keep])
        e_targets.append(slot_t[:b0][keep])
        vis = np.asarray(out["vis_logit"])  # (B,I,T,C,K)
        idx = np.clip(assign, 0, n_slots - 1)
        per_fly = np.take_along_axis(vis, idx.reshape(idx.shape + (1,) * (vis.ndim - 2)), axis=1)
        m = np.asarray(jb["cam_valid"])[:, None, :, :, None] & (fly_valid & (assign >= 0))[
            :, :, None, None, None
        ]
        m = np.broadcast_to(m, per_fly.shape)[:b0]
        v_logits.append(per_fly[:b0][m])
        v_targets.append(np.asarray(jb["vis2d"])[:b0][m])
        offset += b0
    if not e_logits:
        return 1.0, 1.0
    e_logits, e_targets = np.concatenate(e_logits), np.concatenate(e_targets)
    v_logits, v_targets = np.concatenate(v_logits), np.concatenate(v_targets)
    if not e_logits.size or not v_logits.size:
        return 1.0, 1.0
    return (
        fit_temperature(e_logits, e_targets, "exist"),
        fit_temperature(v_logits, v_targets, "vis"),
    )


def loss_weights(node) -> LossWeights:
    """`LossWeights` from a `loss` config node; a MISSING node WARNS.

    Every `LossWeights` default coincides with the v2 run except
    `other_fly_repulsion`, whose 0 turns term 7b off outright -- so a silent
    fallback here trains a plausible-looking run with no cross-fly repulsion.
    """
    weights = _dataclass(LossWeights, node)
    if node is None:
        print(
            "[mvq] WARNING no `loss` config group -- training on LossWeights() defaults: "
            + " ".join(f"{k}={v:g}" for k, v in dataclasses.asdict(weights).items())
            + ". other_fly_repulsion=0 means the cross-fly repulsion term is NOT traced; "
            "compose with `loss=mvq_v2` (or `loss=default`) to choose explicitly.",
            flush=True,
        )
    return weights


def check_finite(loss, step, T, metrics, latest_ckpt) -> None:
    """Raise on a non-finite training loss, naming the term that went bad.

    The EMA is a running sum that is never reset, so one NaN poisons every
    later checkpoint, and `max_to_keep` deletes the clean ones within a few
    saves. `loss` is already on the host, so the check is free.
    """
    if math.isfinite(loss):
        return
    ms = " ".join(f"{k}={float(v):.4f}" for k, v in metrics.items())
    print(f"[mvq] FATAL non-finite loss {loss} at step {step} (T={T}): {ms}", flush=True)
    raise ValueError(
        f"non-finite loss {loss} at step {step} -- stopping before it reaches the EMA and the "
        f"checkpoints; the last clean checkpoint is step {latest_ckpt}"
    )


def _write_meta(run_dir, meta) -> None:
    """Write `<run_dir>/mvq_run.json`, KEEPING an earlier run's `calibration`
    and `val` where this write has none.

    The startup write carries neither, and `load_mvq_model(..., step=<n>)`
    prefers this file over `final/mvq_run.json`, so a plain overwrite on a
    requeue would silently serve the identity temperature.
    """
    path = os.path.join(run_dir, "mvq_run.json")
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        meta = dict(meta)
        for k in ("calibration", "val"):
            if meta.get(k) is None and old.get(k) is not None:
                meta[k] = old[k]
                print(f"[mvq] kept the existing mvq_run.json {k!r} block", flush=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=1)


def run_training(cfg) -> dict:
    """Train MVQ on `cfg.paths.train_data_root`; returns the written `mvq_run.json`.

    Writes `<run_dir>/ckpt/<step>/` (resumable), `<run_dir>/final/` (the
    debiased EMA weights) and `<run_dir>/mvq_run.json`, where `run_dir` is
    `<paths.runs_root>/<run.name>` -- the layout
    `tracking.detector.mvq.checkpoint.load_mvq_model` reads.
    """
    tcfg = _dataclass(MVQTrainConfig, cfg.train)
    mcfg = _dataclass(MVQConfig, cfg.model)
    aug = _dataclass(MVAugParams, cfg.get("aug"))
    weights = loss_weights(cfg.get("loss"))
    if int(cfg.run.get("seed", tcfg.seed)) != int(tcfg.seed):
        print(
            f"[mvq] run.seed={int(cfg.run.seed)} is NOT used: train.seed={tcfg.seed} seeds the "
            f"model, the sampler and the loader",
            flush=True,
        )
    run_dir = os.path.join(str(cfg.paths.runs_root), str(cfg.run.name))
    os.makedirs(run_dir, exist_ok=True)
    mesh = _mesh(tcfg)

    # Dataset build and cohort validation BEFORE any GPU-heavy work: a
    # misconfigured root should fail before the backbone is fetched and built.
    root = str(cfg.paths.train_data_root)
    manifest = schema.load_manifest(root)
    masks_root = str(cfg.paths.masks_root) if cfg.paths.get("masks_root") else None
    if masks_root and not os.path.isdir(masks_root):
        print(f"[mvq] masks_root {masks_root} is absent -- copy-paste donors uncut", flush=True)
        masks_root = None
    copy_paste = (
        CopyPasteParams(
            p=tcfg.copy_paste_p,
            opposite_sex_p=tcfg.copy_paste_opposite_sex_p,
            contact_p=tcfg.copy_paste_contact_p,
            contact_sep=tuple(tcfg.copy_paste_contact_sep),
        )
        if tcfg.copy_paste_p > 0
        else None
    )
    overrides = dict(tcfg.sex_label_overrides or {})
    names = list(schema.keypoint_order(root).names)
    if len(names) != mcfg.num_keypoints:
        raise ValueError(
            f"{root} declares {len(names)} keypoints but model.num_keypoints="
            f"{mcfg.num_keypoints} -- the mismatch would surface inside mvq_loss, after the "
            f"backbone is fetched and built"
        )
    lr_swap = build_lr_swap(names)
    part_of_k, _ = build_part_index(names)
    kp_weight = wing_kp_weight(names, tcfg.wing_kp_mult)
    if tcfg.wing_kp_mult != 1.0:
        wings = [n for n, w in zip(names, kp_weight, strict=True) if w != 1.0]
        print(f"[mvq] wing keypoint weight x{tcfg.wing_kp_mult} on {wings}", flush=True)

    sources = _source_table(manifest)
    train_data = {"root": root, "masks_root": masks_root, "sources": sources, "mix": {}}
    prepared, counters = {}, {}
    for T in tcfg.window_lengths:
        ds = WindowDataset(
            root,
            "train",
            T=int(T),
            train=True,
            seed=tcfg.seed,
            pair_deltas=tuple(int(d) for d in tcfg.pair_deltas),
            jitter_units=tcfg.jitter_units,
            copy_paste=copy_paste,
            sex_overrides=overrides,
            masks_root=masks_root,
            aug=aug,
            lr_swap=lr_swap,
        )
        # BY NAME, before step 0: an unpaired landmark would be mirrored in
        # pixels and not in labels, training the two sides to average.
        assert_lr_swap_covers(ds.kp_order.names, lr_swap)
        w, mass = mix_weights(ds, tcfg, manifest)  # TRAIN-ONLY (see mix_weights)
        is_f = np.array([ds.is_female(i) for i in range(len(ds))], bool)
        knob = (
            f"female_host_target={tcfg.female_host_target}"
            if tcfg.female_host_target is not None
            else f"female_host_weight={tcfg.female_host_weight}"
        )
        print(
            f"[mvq] T={T} sampler: {int(is_f.sum())}/{len(ds)} female-host windows, {knob} -> "
            f"weight mass female {float(w[is_f].sum()):.4f} "
            f"male/other {float(w[~is_f].sum()):.4f}",
            flush=True,
        )
        train_data["mix"][str(int(T))] = {
            "n_windows": len(ds),
            "sources": _mix_report(ds, tcfg, mass, sources),
        }
        # NOT a proxy: the counter hangs off the dataset, which stays a plain
        # WindowDataset so `worker_spec()` (the process loader) still works.
        counters[int(T)] = MixCounter(ds)
        ds.note_drawn = counters[int(T)].note_drawn
        prepared[int(T)] = (ds, w)
    val_ds = WindowDataset(root, "val", T=1, train=False, sex_overrides=overrides)
    cohort_masks = None
    if not tcfg.smoke:
        cohort_masks = cohorts(val_ds)
        for c in tcfg.val_cohorts:
            if c not in cohort_masks:
                raise ValueError(
                    f"val cohort {c!r} is not a cohort of {root} (known: {sorted(cohort_masks)})"
                )
            if not cohort_masks[c].any():
                raise ValueError(f"val cohort {c!r} is empty on {root}")

    # Written BEFORE the model build so a failed build still leaves a usable
    # config on disk; `val` and the realised mix are filled in by the final
    # write. `model` is what `load_mvq_model` rebuilds MVQConfig from.
    meta = {
        "model": dataclasses.asdict(mcfg),
        "train": dataclasses.asdict(tcfg),
        "loss": dataclasses.asdict(weights),
        "aug": dataclasses.asdict(aug),
        "val": None,
        "keypoint_names": names,
        "train_data": train_data,
    }
    _write_meta(run_dir, meta)

    model = MVQModel(mcfg, rngs=nnx.Rngs(tcfg.seed))
    if tcfg.pretrained:
        repo = HF_REPOS.get(mcfg.backbone)
        if repo is None:
            raise ValueError(
                f"train.pretrained=true but model.backbone={mcfg.backbone!r} has no pretrained "
                f"checkpoint (known: {sorted(HF_REPOS)}) -- set train.pretrained=false"
            )
        model.backbone = load_dinov3_safetensors(model.backbone, dinov3_snapshot(repo))
        print(f"[mvq] loaded {repo}", flush=True)
    # `mngr` before the warm start so a requeue can skip it: resume beats warm
    # start, so reading the source checkpoint only to overwrite it is waste.
    mngr = make_manager(os.path.join(run_dir, "ckpt"))
    if tcfg.warm_start and mngr.latest_step() is not None:
        print(
            f"[mvq] warm start from {tcfg.warm_start} SKIPPED: this run's own ckpt/ is at step "
            f"{mngr.latest_step()}, and resume beats warm start",
            flush=True,
        )
    elif tcfg.warm_start:
        model, skipped = merge_state_by_path(model, restore_own_tree(tcfg.warm_start))
        print(f"[mvq] warm start from {tcfg.warm_start}: not restored: {skipped}", flush=True)
    opt = make_optimizer(model, tcfg)
    # EMA seeded at ZERO -- a running sum, debiased by `with_ema` at read time.
    ema = jax.tree_util.tree_map(jnp.zeros_like, nnx.state(model, nnx.Param))
    model, opt, ema, ema_updates, start = restore_latest(mngr, model, opt, ema)
    if start:
        print(f"[mvq] resume @ {start}", flush=True)
    gm, sm = nnx.split(model)
    model = nnx.merge(gm, replicate(sm, mesh))
    go, so = nnx.split(opt)
    opt = nnx.merge(go, replicate(so, mesh))
    ema = replicate(ema, mesh)

    step_fn = make_train_step(part_of_k, weights, tcfg.ema, kp_weight)
    loader_pool = None
    if tcfg.loader_workers == "processes":
        from tracking.train.data.loaders import ProcessSampleLoader, dataset_spec

        # ONE pool for every T stream: each worker holds a copy of both
        # datasets, so two streams cost `num_workers` processes, not 2x.
        loader_pool = ProcessSampleLoader(
            {T: dataset_spec(d) for T, (d, _) in prepared.items()}, tcfg.num_workers
        )
    elif tcfg.loader_workers != "threads":
        raise ValueError(
            f"train.loader_workers must be 'threads' or 'processes', got {tcfg.loader_workers!r}"
        )
    streams = {}
    for T, (ds, w) in prepared.items():

        def epochs(ds=ds, w=w, T=T):
            e = 0
            while True:
                yield from window_batches(
                    ds,
                    tcfg.batch_size,
                    shuffle=True,
                    seed=tcfg.seed + 1000 * e + T,
                    weights=w,
                    num_workers=tcfg.num_workers,
                    workers=tcfg.loader_workers,
                    pool=loader_pool,
                    pool_key=T,
                )
                e += 1

        streams[T] = prefetch(epochs(), mesh, depth=2)

    Ts = sorted(prepared)
    loss = float("nan")
    t0 = time.time()
    # What the sampler ACTUALLY draws over the first `RATIO_BATCHES` batches, or
    # the whole run if shorter. The counter tallies at loader-draw time, so
    # `prefetch`'s run-ahead adds a few; the tally's own count is reported.
    n_ratio = max(1, min(RATIO_BATCHES, tcfg.total_steps - start))
    seen_f = seen_n = seen_neg = 0
    for i in range(start, tcfg.total_steps):
        T = Ts[i % len(Ts)]
        batch = next(streams[T])
        if i - start < n_ratio:
            is_f = np.asarray(batch["is_female"])
            seen_f += int(is_f.sum())
            seen_n += int(is_f.shape[0])
            seen_neg += int(np.asarray(batch["is_negative"]).sum())
            if i - start == n_ratio - 1:
                pos = max(seen_n - seen_neg, 1)
                print(
                    f"[mvq] realised host sex over the first {n_ratio} batches: female "
                    f"{seen_f}/{seen_n} = {seen_f / max(seen_n, 1):.3f} (of the {pos} "
                    f"non-negative windows: {seen_f / pos:.3f}); negatives "
                    f"{seen_neg}/{seen_n} = {seen_neg / max(seen_n, 1):.3f}",
                    flush=True,
                )
                for T_, counter in sorted(counters.items()):
                    counts, drawn = counter.realised()
                    # over the MASS table's sources, so one that dropped out of
                    # the mix is recorded as 0.000 rather than going missing.
                    realised = {
                        sid: counts.get(sid, 0) / drawn
                        for sid in sorted(train_data["mix"][str(T_)]["sources"])
                    }
                    train_data["mix"][str(T_)]["realised"] = realised
                    train_data["mix"][str(T_)]["realised_windows_drawn"] = int(drawn)
                    print(
                        f"[mvq] T={T_} realised mix over {drawn} windows drawn: "
                        + " ".join(f"{sid}={f:.3f}" for sid, f in realised.items()),
                        flush=True,
                    )
        loss, metrics, ema = step_fn(model, opt, ema, batch)
        loss = float(loss)
        check_finite(loss, i + 1, T, metrics, mngr.latest_step())
        ema_updates += 1
        if (i + 1) % tcfg.log_every == 0:
            ms = " ".join(f"{k}={float(v):.4f}" for k, v in metrics.items() if k != "total")
            print(
                f"step {i + 1}/{tcfg.total_steps} T={T} loss {loss:.4f} {ms} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
        if not tcfg.smoke and (i + 1) % tcfg.eval_every == 0 and i + 1 < tcfg.total_steps:
            em = with_ema(model, ema, tcfg.ema, ema_updates)
            val = evaluate(
                em,
                val_ds,
                tcfg.batch_size,
                cohort_masks=cohort_masks,
                part_of_k=part_of_k,
                weights=weights,
                mesh=mesh,
                num_workers=tcfg.num_workers,
                kp_weight=kp_weight,
            )
            print("  val " + " ".join(f"{k}={v:.4f}" for k, v in val.items()), flush=True)
            # `with_ema` clones and rebinds, so `em` holds the only reference to
            # those buffers; the cache clear is what stops eval's executable
            # fragmenting the pool.
            del em
            jax.clear_caches()
        if (i + 1) % tcfg.save_every == 0:
            save_step(mngr, i + 1, model, opt, ema, ema_updates)
    if loader_pool is not None:
        loader_pool.close()
    save_step(mngr, tcfg.total_steps, model, opt, ema, ema_updates)
    mngr.wait_until_finished()

    # `save_final` cannot debias: it takes no decay/step and assumes its `ema`
    # argument already is the debiased average.
    em = with_ema(model, ema, tcfg.ema, ema_updates)
    val, temps = None, (1.0, 1.0)
    if not tcfg.smoke:
        # The WHOLE final pass is guarded, evaluation included: it is first
        # reached after ~2 days of training, and an OOM here must not cost
        # `final/` and the calibration.
        try:
            val = evaluate(
                em,
                val_ds,
                tcfg.batch_size,  # the batch size training used: a smaller one reshards
                cohort_masks=cohort_masks,
                part_of_k=part_of_k,
                weights=weights,
                mesh=mesh,
                num_workers=tcfg.num_workers,
                kp_weight=kp_weight,
            )
            print("  val " + " ".join(f"{k}={v:.4f}" for k, v in val.items()), flush=True)
            temps = _temperatures(em, val_ds, tcfg, mesh)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[mvq] WARNING final evaluation/calibration failed ({exc!r}) -- keeping the "
                f"final checkpoint; val is {'absent' if val is None else 'the completed pass'} "
                f"and the temperatures stay at the identity 1.0/1.0",
                flush=True,
            )
    print(
        f"[mvq] calibration: exist_temperature={temps[0]:.4f} vis_temperature={temps[1]:.4f} "
        f"(n_val={len(val_ds)})",
        flush=True,
    )
    meta["val"] = val
    _write_meta(run_dir, meta)
    write_calibration(run_dir, *temps, n_val=len(val_ds))
    with open(os.path.join(run_dir, "mvq_run.json")) as f:
        meta = json.load(f)
    save_final(run_dir, model, nnx.state(em, nnx.Param), meta)
    print(f"[mvq] wrote {run_dir} ({time.time() - t0:.0f}s, {ema_updates} ema updates)", flush=True)
    return meta


@hydra.main(version_base=None, config_path="../../../../configs", config_name="train")
def main(cfg: DictConfig) -> int:
    run_training(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
