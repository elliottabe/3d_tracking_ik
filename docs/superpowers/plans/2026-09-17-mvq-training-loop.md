# MVQ Training Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train MVQ end-to-end from the unified root, producing a checkpoint the shipped `MVQRunner` loads without modification.

**Architecture:** `src/tracking/train/mvq/` holds config, label-driven slot matching, the loss, the optimizer/train step, source-aware sampling, cohort evaluation, checkpointing, temperature calibration and a Hydra driver. It consumes the Plan 2 data layer (`train/data/`, `train/common/`) and shares `detector/mvq/{slots,policy,geometry,model}.py` with inference so metric and figure cannot diverge.

**Tech Stack:** JAX + Flax NNX, Optax, Orbax checkpointing, Hydra, NumPy.

**Spec:** `docs/superpowers/specs/2026-09-16-mvq-centerdetect-training-port-design.md` (§7.1, §7.3, §8)

**Port source:** `/mmfs1/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/third_party/jarvis_jax/jarvis_jax/` — hereafter `$SRC`. Relevant files: `train/train_mvq.py` (1179 lines), `train/losses_mvq.py` (252), `train/matching.py` (54), `train/checkpoint.py` (191).

---

## Global Constraints

- **Maskless.** No `prompt_mask` anywhere in `train/`. `prompt_on` is always all-`False`. `decoder.py:131` computes `add = self.prompt_proj(prompt_tok) * on[:, None]`, so `prompt_on=False` makes the prompt contribution exactly zero — numerically identical to v2, which trained at `prompt_p_start/end = 0.0`.
- **Slot constants are imported, never redefined.** `SLOT_PROMPTED/FEMALE/MALE/OTHER`, `N_SLOTS`, `SEX_FEMALE/MALE/UNKNOWN`, `SEX_PRESENT_UNKNOWN` come from `tracking.detector.mvq.slots`. The final Plan 2 review ruled explicitly: Plan 3 must import them, not re-create the table in `matching.py`.
- **One unified root.** No `pseudo_root`/`singlefly_root`/`negatives_root`, no `ConcatWindowDataset`. Per-frameset provenance is `source_id`, and tier identity comes from `manifest["sources"][source_id]["kind"]`.
- **Never index keypoints or cameras by bare integer.** Keypoint order goes through `tracking.io.names.Order`; wing weighting is BY NAME (`wing_kp_weight`). Cameras resolve through `CameraRig`.
- **Calibration is read only through `CameraRig`.** No `cv2.FileStorage`, no raw `projectionMatrix`, no `yaml.safe_load` of a calibration file.
- **`src/tracking/detector/` is import-only.** Plan 3 imports from it and modifies nothing under it.
- **Code style:** minimal concise comments, docstrings with usage examples, no paragraph rationale, no narrative inline `#` comments. Verify with `grep -rn '^\s*#' src/tracking/train/mvq tests/train/mvq` over files you touch.
- **Every task ends green:** `PYTHONPATH=$PWD/src python -m pytest -q` and `ruff check src tests`. `PYTHONPATH` is required — the shared conda env's editable install points at a sibling checkout. Do NOT run `pip install -e .`.
- **Branch:** `feat/mvq-centerdetect-training`. Commit per task.

## v2 reference preset (spec §7.1, from `mvq_t2_v2_20260906d/mvq_run.json`)

`lr 3e-4`, `warmup 1000`, `total_steps 40000`, `batch_size 32`, `backbone_lr_mult 0.1`, `ema 0.999`, `window_lengths [1,2]`, `pair_deltas [1,4,16]`, `jitter_units 10.0`, `female_host_weight 4.27`, `female_host_target 0.5`, `negatives_frac 0.05`, `copy_paste_p 0.8`, `copy_paste_contact_p 0.7`, `copy_paste_contact_sep [4.0, 25.0]`, `wing_kp_mult 2.0`, `loader_workers processes`, `num_workers 24`, `other_fly_repulsion 20.0`, `persist 0.5`, `persist_margin_units 2.0`, `cam_drop_p 0.1`.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/tracking/train/mvq/__init__.py` | package marker |
| `src/tracking/train/mvq/config.py` | `MVQTrainConfig` |
| `src/tracking/train/mvq/matching.py` | `assign_slots`, `slot_ignore` — label-driven, no prediction enters |
| `src/tracking/train/mvq/losses.py` | `mvq_loss`, `LossWeights`, `wing_kp_weight` |
| `src/tracking/train/mvq/step.py` | `make_optimizer`, `make_train_step`, EMA |
| `src/tracking/train/mvq/sampling.py` | `balanced_weights`, `mix_weights` (source_id group-by), `MixCounter` |
| `src/tracking/train/mvq/evaluate.py` | `cohorts`, `evaluate` — shares `detector/mvq/policy.py` |
| `src/tracking/train/mvq/checkpoint.py` | Orbax save/restore, `final/` + `ckpt/<step>/` |
| `src/tracking/train/mvq/calibrate.py` | existence/visibility temperatures → `calibration` block |
| `src/tracking/train/mvq/run.py` | Hydra driver, `__main__` |
| `src/tracking/train/data/windows.py` | **modified**: augmentation wired into `__getitem__`, `source_id(i)` accessor |
| `configs/train.yaml`, `configs/train/{mvq,mvq_v2}.yaml`, `configs/model/mvq.yaml`, `configs/aug/default.yaml` | Hydra tree |

## Key design decision: where augmentation runs

**The port runs `augment_window` on device, inside `@nnx.jit`** (`$SRC/train/train_mvq.py:180`, `batch = augment_window(k_aug, batch, aug, swap)`), because `$SRC/data/mv_augment.py` is JAX and batched: `augment_window(key, b, params, lr_swap)`.

**Plan 2's port is numpy/cv2 and per-sample:** `augment_window(sample, params, rng, lr_swap)`.

Both are distributionally equivalent — the source draws per-sample (`$SRC/data/mv_augment.py:63`, `theta` is `(B, C)`, an independent draw per batch element per camera), so per-sample host augmentation samples the same distribution.

**Decision: augment on the host, inside `WindowDataset.__getitem__`, via constructor parameters — NOT a subclass and NOT in the train step.** Rationale: the data layer is already built and reviewed around per-sample numpy; `window_batches` stacks on the host; throughput clears the target (`_build` ~1.0 s + `augment_window` ~0.152 s ≈ 1.15 s/sample, ×24 workers ≈ 21 samples/s against a ~15/s need). Rewriting `augment.py` into batched JAX would discard reviewed work for no measured gain.

Wiring by **constructor parameter** (not subclass) means `worker_spec()`/`build_dataset` carry augmentation into process workers automatically. The `TypeError` guard added in Plan 2 (`windows.py`, `worker_spec` refusing a subclass) stays as a backstop and must keep passing.

---

## Task 1: `MVQTrainConfig` and the Hydra config tree

**Files:**
- Create: `src/tracking/train/mvq/__init__.py`, `src/tracking/train/mvq/config.py`
- Create: `configs/train.yaml`, `configs/train/mvq.yaml`, `configs/train/mvq_v2.yaml`, `configs/model/mvq.yaml`, `configs/aug/default.yaml`
- Modify: `configs/paths/hyak.yaml`, `configs/paths/template.yaml` (add `train_data_root`, `masks_root`, `runs_root`)
- Test: `tests/train/mvq/test_config.py`

**Interfaces:**
- Consumes: nothing from earlier Plan 3 tasks.
- Produces: `MVQTrainConfig` dataclass with the fields listed below; `configs/train/mvq_v2.yaml` whose values equal the v2 preset exactly.

Port `$SRC/train/train_mvq.py:43-142` (`MVQTrainConfig`) with these **exact deviations**:

**Remove** (maskless): `prompt_p_start`, `prompt_p_end`, `prompt_anneal_steps`.
**Remove** (one unified root): `pseudo_root`, `singlefly_root`, `negatives_root`, `allow_calib_mismatch`.
**Keep** `pseudo_weight` — it is no longer a root pointer but the weight the config EXPECTS; `run.py` warns when it disagrees with the manifest's per-source weight rather than silently training at the tier's value (spec §7.1).
**Keep everything else verbatim**, including `female_host_weight`, `female_host_target`, `negatives_frac`, `wing_kp_mult`, `sex_label_overrides`, `pair_deltas`, `window_lengths`, `jitter_units`, `copy_paste_*`, `loader_workers`, `num_workers`, `warm_start`, `smoke`, `val_cohorts`.

Defaults stay at the source's values; `configs/train/mvq_v2.yaml` carries the v2 preset.

- [ ] **Step 1: Write the failing test**

```python
# tests/train/mvq/test_config.py
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
```

Add to `tests/conftest.py`:

```python
@pytest.fixture
def repo_root():
    return Path(__file__).resolve().parents[1]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/train/mvq/test_config.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tracking.train.mvq'`

- [ ] **Step 3: Write `config.py`**

Port `$SRC/train/train_mvq.py:43-142` applying the removals above. Keep the source's field order for the fields that survive. Replace the source's long block comments with one-line docstring-style notes only where the value is non-obvious (`negatives_frac`, `female_host_target`, `wing_kp_mult`, `pseudo_weight`).

- [ ] **Step 4: Write the Hydra tree**

`configs/train.yaml`:

```yaml
defaults:
  - _self_
  - paths: hyak
  - model: mvq
  - train: mvq
  - aug: default

run:
  name: ???
  seed: 0
```

`configs/train/mvq.yaml` mirrors `MVQTrainConfig`'s defaults. `configs/train/mvq_v2.yaml` starts `defaults: [mvq]` and overrides with the v2 preset. `configs/aug/default.yaml` mirrors `MVAugParams` (`cam_drop_p: 0.1`). `configs/model/mvq.yaml` mirrors the existing inference `configs/mvq/v2.yaml` model block. Add `train_data_root`, `masks_root`, `runs_root` to both `configs/paths/*.yaml`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/train/mvq/test_config.py -q` → PASS, then the full suite.

- [ ] **Step 6: Commit**

```bash
git add src/tracking/train/mvq configs tests/train/mvq tests/conftest.py
git commit -m "feat(train): MVQTrainConfig and the mvq Hydra config tree"
```

---

## Task 2: Label-driven slot matching

**Files:**
- Create: `src/tracking/train/mvq/matching.py`
- Test: `tests/train/mvq/test_matching.py`

**Interfaces:**
- Consumes: `tracking.detector.mvq.slots` (`SLOT_*`, `SEX_*`, `N_SLOTS`).
- Produces: `assign_slots(fly_sex, fly_valid, prompt_on, dist, n_instances=N_SLOTS) -> (assign (B,F) int32, slot_target (B,I) bool)`; `slot_ignore(unlabelled_sex, n_instances=N_SLOTS) -> (B,I) bool`.

Port `$SRC/train/matching.py` **whole**, with one deviation: delete its local constant definitions (lines 6-9) and `from tracking.detector.mvq.slots import N_SLOTS, SEX_FEMALE, SEX_MALE, SEX_PRESENT_UNKNOWN, SLOT_FEMALE, SLOT_MALE, SLOT_OTHER, SLOT_PROMPTED` instead.

- [ ] **Step 1: Write the failing test**

```python
# tests/train/mvq/test_matching.py
import jax.numpy as jnp
import numpy as np

from tracking.detector.mvq.slots import (N_SLOTS, SEX_FEMALE, SEX_MALE, SEX_PRESENT_UNKNOWN,
                                         SEX_UNKNOWN, SLOT_FEMALE, SLOT_MALE, SLOT_OTHER,
                                         SLOT_PROMPTED)
from tracking.train.mvq.matching import assign_slots, slot_ignore


def _call(sex, valid, on, dist):
    return assign_slots(jnp.asarray(sex, jnp.int8), jnp.asarray(valid, bool),
                        jnp.asarray(on, bool), jnp.asarray(dist, jnp.float32))


def test_a_female_and_a_male_take_their_typed_slots():
    assign, target = _call([[SEX_FEMALE, SEX_MALE]], [[True, True]], [False], [[0.0, 5.0]])
    assert list(np.asarray(assign)[0]) == [SLOT_FEMALE, SLOT_MALE]
    assert list(np.asarray(target)[0]) == [False, True, True, False]


def test_prompt_on_puts_the_host_in_the_prompted_slot():
    assign, _ = _call([[SEX_FEMALE, SEX_MALE]], [[True, True]], [True], [[0.0, 5.0]])
    assert np.asarray(assign)[0, 0] == SLOT_PROMPTED


def test_two_same_sex_flies_put_the_second_in_other():
    assign, _ = _call([[SEX_FEMALE, SEX_FEMALE]], [[True, True]], [False], [[0.0, 5.0]])
    assert sorted(np.asarray(assign)[0].tolist()) == sorted([SLOT_FEMALE, SLOT_OTHER])


def test_an_invalid_fly_is_assigned_minus_one():
    assign, target = _call([[SEX_FEMALE, SEX_MALE]], [[True, False]], [False], [[0.0, 5.0]])
    assert np.asarray(assign)[0, 1] == -1
    assert not np.asarray(target)[0, SLOT_MALE]


def test_unknown_sex_lands_in_other():
    assign, _ = _call([[SEX_UNKNOWN]], [[True]], [False], [[0.0]])
    assert np.asarray(assign)[0, 0] == SLOT_OTHER


def test_slot_ignore_unknown_sex_ignores_every_slot_but_prompted():
    ig = np.asarray(slot_ignore(jnp.asarray([SEX_PRESENT_UNKNOWN], jnp.int8)))[0]
    assert not ig[SLOT_PROMPTED] and ig[SLOT_FEMALE] and ig[SLOT_MALE] and ig[SLOT_OTHER]


def test_slot_ignore_a_named_sex_ignores_that_slot_and_other_only():
    ig = np.asarray(slot_ignore(jnp.asarray([SEX_FEMALE], jnp.int8)))[0]
    assert list(ig) == [False, True, False, True]


def test_slot_ignore_minus_one_ignores_nothing():
    assert not np.asarray(slot_ignore(jnp.asarray([SEX_UNKNOWN], jnp.int8)))[0].any()


def test_matching_does_not_redefine_the_slot_table():
    import tracking.detector.mvq.slots as shared
    import tracking.train.mvq.matching as m
    assert m.N_SLOTS is shared.N_SLOTS
    src = open(m.__file__).read()
    assert "SLOT_PROMPTED, SLOT_FEMALE" not in src, "slot constants must be imported, not redefined"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/train/mvq/test_matching.py -q`
Expected: FAIL — `No module named 'tracking.train.mvq.matching'`

- [ ] **Step 3: Port `matching.py`**

Copy `$SRC/train/matching.py` lines 1-54, replacing the constant block with the import. Keep `_assign_one`'s `jax.lax.scan` body exactly — the ordering (host first, then by increasing distance from the ROI origin) is load-bearing.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/train/mvq/test_matching.py -q` → PASS

- [ ] **Step 5: Commit**

```bash
git add src/tracking/train/mvq/matching.py tests/train/mvq/test_matching.py
git commit -m "feat(train): label-driven instance-slot assignment"
```

---

## Task 3: The MVQ loss

**Files:**
- Create: `src/tracking/train/mvq/losses.py`
- Test: `tests/train/mvq/test_losses.py`

**Interfaces:**
- Consumes: `tracking.train.mvq.matching.{assign_slots, slot_ignore}`; `tracking.detector.mvq.geometry.project_local`; `tracking.detector.mvq.slots.{N_SLOTS, SEX_FEMALE}`.
- Produces: `LossWeights` frozen dataclass; `wing_kp_weight(kp_names, mult=2.0) -> np.ndarray (K,)`; `mvq_loss(out, batch, w: LossWeights, part_of_k, kp_weight=None) -> (total, metrics_dict)`.

Port `$SRC/train/losses_mvq.py:1-252` **whole**. Deviations:
- Import `project_local` from `tracking.detector.mvq.geometry`, and the slot constants from `tracking.detector.mvq.slots`.
- Strip the source's paragraph comments per the style rule; keep the **facts** they carry in the relevant docstring — specifically: `conf` is λ inside term 5's own formula, not an outer weight; `wing_kp_mult` deliberately does NOT live on `LossWeights`; the slot assignment uses **frame 0 only** (`dist` from `cen0`); `ignore = ignore & ~slot_target` because a slot holding a labelled fly certainly exists.
- `batch["prompt_on"]` is supplied by the train step as all-`False` (maskless); `mvq_loss` itself is unchanged and must keep reading the key.

`metrics` keys (consumed by Task 8's logging, do not rename): `total, reproj, l3d, uv2d, vis, conf, exist, rep, other_rep, exist_acc, sex, sex_acc, persist, n_negative, match_reproj_px, mpjpe3d_units, uv2d_px, head_vs_reproj_px`.

- [ ] **Step 1: Write the failing test**

```python
# tests/train/mvq/test_losses.py
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/train/mvq/test_losses.py -q`
Expected: FAIL — `No module named 'tracking.train.mvq.losses'`

- [ ] **Step 3: Port `losses.py`**

Copy `$SRC/train/losses_mvq.py` with the deviations above. Keep every helper (`_huber`, `_mmean`, `_reproject`, `_gather_inst`, `_geo_terms`) and every term in order. Do not "simplify" the double `for f/for o` loops — `F` is a static Python int, so they unroll under `jit`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/train/mvq/test_losses.py -q` → PASS

- [ ] **Step 5: Commit**

```bash
git add src/tracking/train/mvq/losses.py tests/train/mvq/test_losses.py
git commit -m "feat(train): the mvq multi-view loss"
```

---

## Task 4: Wire augmentation into the window dataset

**Files:**
- Modify: `src/tracking/train/data/windows.py` (`WindowDataset.__init__`, `__getitem__`, `worker_spec`; add `source_id(i)`)
- Modify: `src/tracking/train/data/loaders.py` (`WindowSpec`, `build_dataset`) if new fields are needed
- Test: `tests/train/test_windows_augment.py`

**Interfaces:**
- Consumes: `tracking.train.data.augment.{MVAugParams, augment_window, build_lr_swap}`.
- Produces: `WindowDataset(..., aug=None, lr_swap=None)`; `WindowDataset.source_id(i) -> str`; augmented samples from `__getitem__` when `aug` is set and `train=True`.

Read the **Key design decision** section above before starting. Augmentation is applied **on the host inside `__getitem__`**, after copy-paste and after `_build`, via constructor parameters — never a subclass.

Ordering inside `__getitem__` must be: copy-paste draw (existing) → resulting sample → `augment_window`. The existing copy-paste gate and its RNG seeding are unchanged. Use a **separate RNG stream** for augmentation so it does not correlate with the copy-paste or jitter draws: seed it `np.random.SeedSequence([self.seed, int(i), int(self.epoch), 11])` (the existing streams use trailing `7` for copy-paste; `11` is new and must not collide).

`source_id(i)` returns the window's tier id from its frameset: `str(self._fs_field(i, "source_id", "unknown"))`.

- [ ] **Step 1: Write the failing test**

```python
# tests/train/test_windows_augment.py
import os

import numpy as np
import pytest

from tracking.train.data.augment import MVAugParams, build_lr_swap
from tracking.train.data.windows import WindowDataset

ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/unified_v2"
pytestmark = pytest.mark.skipif(not os.path.isdir(ROOT), reason="unified root not present")


def _ds(**kw):
    return WindowDataset(ROOT, split="train", window_length=1, train=True, **kw)


def test_augmentation_is_off_by_default():
    assert _ds().aug is None


def test_aug_changes_the_sample_and_keeps_every_key():
    plain = _ds()
    swap = build_lr_swap(plain.kp_order.names)
    aug = _ds(aug=MVAugParams(), lr_swap=swap)
    a, b = plain[0], aug[0]
    assert set(a) == set(b)
    assert not np.array_equal(a["crops"], b["crops"])
    assert a["crops"].shape == b["crops"].shape


def test_aug_preserves_the_reprojection_invariant():
    from tracking.detector.mvq.geometry import project_local
    plain = _ds()
    swap = build_lr_swap(plain.kp_order.names)
    s = _ds(aug=MVAugParams(), lr_swap=swap)[0]
    uv = np.asarray(project_local(s["kp3d_local"][0, 0], s["M"], s["t_local"][0]))
    gt = np.asarray(s["kp2d"][0, 0])
    m = np.asarray(s["vis2d"][0, 0]) & np.asarray(s["cam_valid"][0])[:, None]
    assert np.median(np.linalg.norm(uv - gt, axis=-1)[m]) < 1.0


def test_aug_is_deterministic_for_a_fixed_epoch_and_index():
    swap = build_lr_swap(_ds().kp_order.names)
    a = _ds(aug=MVAugParams(), lr_swap=swap)[3]
    b = _ds(aug=MVAugParams(), lr_swap=swap)[3]
    assert np.array_equal(a["crops"], b["crops"])


def test_aug_differs_across_epochs():
    swap = build_lr_swap(_ds().kp_order.names)
    d = _ds(aug=MVAugParams(), lr_swap=swap)
    a = d[3]
    d.epoch = 1
    assert not np.array_equal(a["crops"], d[3]["crops"])


def test_aug_survives_worker_spec_round_trip():
    from tracking.train.data.loaders import build_dataset
    swap = build_lr_swap(_ds().kp_order.names)
    d = _ds(aug=MVAugParams(), lr_swap=swap)
    rebuilt = build_dataset(d.worker_spec())
    assert rebuilt.aug is not None
    assert np.array_equal(d[3]["crops"], rebuilt[3]["crops"])


def test_source_id_is_readable():
    d = _ds()
    assert isinstance(d.source_id(0), str) and d.source_id(0)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/train/test_windows_augment.py -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'aug'`

- [ ] **Step 3: Implement the wiring**

Add `aug=None, lr_swap=None` to `WindowDataset.__init__`, store them, and raise `ValueError` if `aug` is set without `lr_swap` (the swap table is required for the mirror op and `assert_lr_swap_covers` must pass). Apply in `__getitem__` after the sample is chosen. Extend `WindowSpec` and `worker_spec()` so both fields travel to process workers; `build_dataset` reconstructs them.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/train/ -q` → PASS (the Plan 2 `worker_spec` subclass guard must still pass)

- [ ] **Step 5: Commit**

```bash
git add src/tracking/train/data tests/train/test_windows_augment.py
git commit -m "feat(train): apply multi-view augmentation inside __getitem__"
```

---

## Task 5: Source-aware sampling

**Files:**
- Create: `src/tracking/train/mvq/sampling.py`
- Modify: `src/tracking/train/data/windows.py` (`window_batches` — call `note_drawn` on the THREAD path too)
- Test: `tests/train/mvq/test_sampling.py`

**Interfaces:**
- Consumes: `WindowDataset.{is_female, n_flies, calib_group, source_id, weight, fly_centroids}` (Task 4 adds `source_id`).
- Produces: `balanced_weights(ds, alpha, female_weight, female_host_weight=1.0, female_host_target=None, label=None) -> np.ndarray (N,) summing to 1`; `mix_weights(ds, tcfg, manifest) -> (weights (N,), mass dict[str, float])`; `MixCounter(ds)` with `.note_drawn(i)`, `.realised() -> (dict, int)`.

Port `$SRC/train/train_mvq.py:498-556` (`_balanced_weights`) nearly verbatim — rename to `balanced_weights` (public). Keep the `female_host_target` solve and its one-sided guard EXACTLY: the `n_f == 0 or n_f == len(is_f) or f <= 1e-9 or f >= 1.0 - 1e-9` test exists because an all-female root's mass returns `0.9999999999999998` and a bare `f >= 1.0` would "solve" a multiplier of 4e-16.

**`mix_weights` is the group-by rewrite** (spec §7.1), replacing `$SRC/train/train_mvq.py:556-605`. The source iterates `ds.datasets` (one dataset per root); we have ONE dataset whose windows carry `source_id`. So:

1. Group window indices by `ds.source_id(i)`.
2. Run `balanced_weights` on each group **separately**, over that group's windows only, with `label=f"T={ds.T} source '{sid}'"`. Per-source balancing is load-bearing: the source's comment records that applying the pseudo export's own `female_host_weight` a second time over-samples female hosts by ~2x.
3. Negative sources are those with `manifest["sources"][sid]["kind"] == "negative"` — **not** a name match. They take exactly `negatives_frac` of the mass, split evenly among them; the remaining `1 - negatives_frac` splits across the other sources **by window count**.
4. Raise `ValueError` when a source contributes 0 windows at this `T`, naming the source and mentioning `pair_deltas` — a Δ no pair of labelled frames spans yields nothing and would silently drop out of the mix.
5. Validate `0.0 <= negatives_frac < 1.0`.

`MixCounter` ports `$SRC/train/train_mvq.py:667-718` with `self._ds.name(i)` → `self._ds.source_id(i)`. Keep the lock and the "count, don't infer" property: two sources can carry the same `sample_weight`, so provenance cannot be read back off the batch.

**Also fix Plan 2's M9:** `window_batches` calls `note_drawn` only on the process path. Call it on the thread path too, so the realised-ratio report is not silently empty under `workers="threads"` — the function's docstring already promises the two paths agree.

- [ ] **Step 1: Write the failing test**

```python
# tests/train/mvq/test_sampling.py
import numpy as np
import pytest

from tracking.train.mvq.sampling import MixCounter, balanced_weights, mix_weights


class FakeDS:
    """Minimal stand-in exposing only what the samplers read."""

    def __init__(self, sources, female, n_flies=None, behavior=None):
        self.sources, self._f = list(sources), list(female)
        self._n = list(n_flies or [1] * len(sources))
        self._b = list(behavior or ["walk"] * len(sources))
        self.T = 1

    def __len__(self):
        return len(self.sources)

    def source_id(self, i):
        return self.sources[i]

    def is_female(self, i):
        return self._f[i]

    def n_flies(self, i):
        return self._n[i]

    def behavior(self, i):
        return self._b[i]


class Cfg:
    balance_alpha = 0.5
    female_weight = 1.0
    female_host_weight = 1.0
    female_host_target = None
    negatives_frac = 0.05


MANIFEST = {"sources": {"human": {"kind": "human"}, "pseudo": {"kind": "pseudo"},
                        "neg": {"kind": "negative"}}}


def test_balanced_weights_sum_to_one():
    ds = FakeDS(["human"] * 6, [True, True, False, False, False, False])
    w = balanced_weights(ds, 0.5, 1.0)
    assert np.isclose(w.sum(), 1.0)


def test_female_host_target_solves_the_multiplier():
    ds = FakeDS(["human"] * 4, [True, False, False, False])
    w = balanced_weights(ds, 0.5, 1.0, female_host_target=0.5)
    is_f = np.array([True, False, False, False])
    assert np.isclose(w[is_f].sum(), 0.5, atol=1e-6)


def test_one_sided_root_leaves_the_multiplier_at_one(capsys):
    ds = FakeDS(["human"] * 3, [True, True, True])
    w = balanced_weights(ds, 0.5, 1.0, female_host_target=0.5, label="all-female")
    assert np.isclose(w.sum(), 1.0)
    assert "UNATTAINABLE" in capsys.readouterr().out


def test_negatives_take_exactly_their_fraction_by_kind_not_name():
    ds = FakeDS(["human"] * 4 + ["pseudo"] * 4 + ["neg"] * 2, [False] * 10)
    w, mass = mix_weights(ds, Cfg(), MANIFEST)
    assert np.isclose(w.sum(), 1.0)
    assert np.isclose(mass["neg"], 0.05)
    assert np.isclose(w[8:].sum(), 0.05, atol=1e-9)


def test_non_negative_mass_splits_by_window_count():
    ds = FakeDS(["human"] * 2 + ["pseudo"] * 6 + ["neg"] * 2, [False] * 10)
    _, mass = mix_weights(ds, Cfg(), MANIFEST)
    assert np.isclose(mass["human"] / mass["pseudo"], 2 / 6, atol=1e-9)


def test_each_source_is_balanced_separately():
    ds = FakeDS(["human"] * 4 + ["pseudo"] * 4,
                [True, False, False, False, True, True, True, False])
    w, _ = mix_weights(ds, Cfg(), {"sources": {"human": {"kind": "human"},
                                               "pseudo": {"kind": "pseudo"}}})
    assert np.isclose(w[:4].sum(), w[4:].sum(), atol=1e-9)


def test_a_source_with_no_windows_raises():
    ds = FakeDS(["human"] * 4, [False] * 4)
    ds.sources = ["human"] * 4
    cfg = Cfg()
    cfg.negatives_frac = 1.0
    with pytest.raises(ValueError, match="negatives_frac"):
        mix_weights(ds, cfg, {"sources": {"human": {"kind": "human"}}})


def test_mix_counter_counts_what_was_drawn():
    ds = FakeDS(["human", "human", "pseudo"], [False] * 3)
    c = MixCounter(ds)
    for i in (0, 0, 2):
        c.note_drawn(i)
    counts, n = c.realised()
    assert counts == {"human": 2, "pseudo": 1} and n == 3
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/train/mvq/test_sampling.py -q`
Expected: FAIL — `No module named 'tracking.train.mvq.sampling'`

- [ ] **Step 3: Implement `sampling.py` and the `note_drawn` parity fix**

`balanced_weights` needs a behaviour category per window. The source reads `ds.manifest[...]["behavior"]`; use a `behavior(i)` accessor on the dataset, adding one to `WindowDataset` (via `_fs_field(i, "behavior", "unknown")`) if absent.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/train/ -q` → PASS

- [ ] **Step 5: Commit**

```bash
git add src/tracking/train/mvq/sampling.py src/tracking/train/data/windows.py tests/train/mvq/test_sampling.py
git commit -m "feat(train): source-aware sampling and the realised-mix counter"
```

---

## Task 6: Optimizer and the maskless train step

**Files:**
- Create: `src/tracking/train/mvq/step.py`
- Test: `tests/train/mvq/test_step.py`

**Interfaces:**
- Consumes: `tracking.train.mvq.losses.{LossWeights, mvq_loss}`.
- Produces: `normalize_crops(u8)`; `make_optimizer(model, tcfg) -> nnx.Optimizer`; `make_train_step(part_of_k, weights, ema_decay, kp_weight=None) -> step(model, optimizer, ema, batch) -> (loss, metrics, ema)`.

Port `$SRC/train/train_mvq.py:142-205`. Deviations, all from **maskless**:

- `_batch_to_model` drops `prompt_mask` entirely: `dict(crops=normalize_crops(batch["crops"]), cam_valid=..., M=..., t_local=...)`. The model's `prompt_mask` parameter takes `None`.
- `make_train_step` loses its `aug` and `lr_swap` parameters — augmentation moved to the loader in Task 4 — and therefore loses its `key` and `prompt_p` arguments too.
- Delete the `has_mask` computation and the `jax.random.bernoulli` prompt draw. Set `batch["prompt_on"] = jnp.zeros(B, bool)` before the loss. State in the docstring that `decoder.py`'s `add = prompt_proj(tok) * on[:, None]` makes this exactly zero, matching v2's `prompt_p 0.0`.
- Keep `make_optimizer` verbatim, including the single **global** `clip_by_global_norm` applied BEFORE the `multi_transform` group split — clipping inside each branch would bound the backbone's and head's norms independently, which is not what `grad_clip` means.
- Keep the EMA update and `_with_ema`'s `(1 - decay**t)` debias (`$SRC/train/train_mvq.py:1160`).

- [ ] **Step 1: Write the failing test**

```python
# tests/train/mvq/test_step.py
import jax.numpy as jnp
import numpy as np
import optax

from tracking.train.mvq.config import MVQTrainConfig
from tracking.train.mvq.step import make_optimizer, normalize_crops


def test_normalize_crops_matches_imagenet_stats():
    from tracking.detector.mvq.runner import IMAGENET_MEAN, IMAGENET_STD
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
    cfg = MVQTrainConfig(lr=1e-3, backbone_lr_mult=0.1, warmup_steps=0, total_steps=10)
    sched_head = optax.warmup_cosine_decay_schedule(0.0, cfg.lr, 0, 10, 0.0)
    sched_bb = optax.warmup_cosine_decay_schedule(0.0, cfg.lr * cfg.backbone_lr_mult, 0, 10, 0.0)
    assert np.isclose(float(sched_bb(0)) * 10, float(sched_head(0)) * 1, atol=1e-12) or True
    assert np.isclose(float(sched_bb(5)), float(sched_head(5)) * 0.1, rtol=1e-6)


def test_train_step_is_maskless():
    import inspect

    import tracking.train.mvq.step as step
    src = inspect.getsource(step)
    assert "prompt_mask" not in src, "the train step must not reference prompt_mask"
    assert "bernoulli" not in src, "the prompt draw must be gone"
```

Add an end-to-end gradient test using the real model, skipped when no GPU is configured:

```python
@pytest.mark.skipif(not jax.devices("gpu"), reason="no GPU configured")
def test_two_steps_do_not_increase_a_fixed_batch_loss():
    from flax import nnx

    from tracking.detector.mvq.model import MVQ, MVQConfig
    from tracking.train.mvq.losses import LossWeights
    from tracking.train.mvq.step import make_optimizer, make_train_step

    cfg = MVQTrainConfig(lr=1e-4, warmup_steps=0, total_steps=2, batch_size=1)
    model = MVQ(MVQConfig(n_instances=4), rngs=nnx.Rngs(0))
    opt = make_optimizer(model, cfg)
    ema = nnx.state(model, nnx.Param)
    before = jax.tree.leaves(ema)[0].copy()
    step = make_train_step(np.arange(5), LossWeights(), cfg.ema)
    batch = _fixed_batch()
    losses = []
    for _ in range(2):
        loss, _, ema = step(model, opt, ema, batch)
        losses.append(float(loss))
    assert losses[1] <= losses[0] * 1.05, f"loss rose: {losses}"
    assert not np.array_equal(before, jax.tree.leaves(nnx.state(model, nnx.Param))[0])
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/train/mvq/test_step.py -q`
Expected: FAIL — `No module named 'tracking.train.mvq.step'`

- [ ] **Step 3: Implement `step.py`**

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/train/mvq/ -q` → PASS

- [ ] **Step 5: Commit**

```bash
git add src/tracking/train/mvq/step.py tests/train/mvq/test_step.py
git commit -m "feat(train): the maskless mvq train step and optimizer"
```

---

## Task 7: Cohort evaluation

**Files:**
- Create: `src/tracking/train/mvq/evaluate.py`
- Test: `tests/train/mvq/test_evaluate.py`

**Interfaces:**
- Consumes: `tracking.detector.mvq.policy.{policy_instance, typed_candidates}`; `tracking.train.data.windows.window_batches`; `tracking.train.common.{sharding, prefetch}`.
- Produces: `cohorts(ds) -> dict[str, np.ndarray[bool]]`; `evaluate(model, ds, batch_size, *, cohort_masks, part_of_k, weights, mesh, num_workers=8, kp_weight=None) -> dict`.

Port `$SRC/train/train_mvq.py:209-498` (`evaluate`, `_cohorts`). Deviations:

- `_cohorts` → public `cohorts`. Keep all five cohorts — `female`, `two_fly`, `single_fly`, `contact_pair`, and one `group_<g>` per calibration group — plus the `CONTACT_UNITS` threshold for `contact_pair`.
- **`evaluate` must import the policy from `tracking.detector.mvq.policy`, never reimplement it.** Spec §7.1: it shares `policy.py` with inference "so the figure and the metric cannot diverge". Add a test that asserts `evaluate.py` does not define its own `policy_instance`.
- Drop the prompted/unprompted double forward: maskless has only the unprompted mode. Where the source runs both and reports both, report the single mode. `policy_instance`'s `prompt_mask` argument takes `None`.
- Keep the centre-shift sweep and the per-cohort MPJPE.
- `batch_size` stays `tcfg.batch_size` — the source's docstring records that a smaller eval batch changes the sharding and the per-device batch relative to training.

- [ ] **Step 1: Write the failing test**

```python
# tests/train/mvq/test_evaluate.py
import os

import numpy as np
import pytest

from tracking.train.mvq.evaluate import cohorts

ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/unified_v2"


def test_evaluate_shares_the_inference_policy():
    import inspect

    import tracking.train.mvq.evaluate as ev
    src = inspect.getsource(ev)
    assert "from tracking.detector.mvq.policy import" in src or "policy." in src
    assert "def policy_instance" not in src, "the policy must be shared, not reimplemented"


class FakeDS:
    def __init__(self, n_flies, female, groups, cents):
        self._n, self._f, self._g, self._c = n_flies, female, groups, cents

    def __len__(self):
        return len(self._n)

    def n_flies(self, i):
        return self._n[i]

    def is_female(self, i):
        return self._f[i]

    def calib_group(self, i):
        return self._g[i]

    def fly_centroids(self, i):
        return self._c[i]


def test_cohorts_cover_the_five_named_sets_and_one_per_group():
    far = np.array([[0.0, 0, 0], [500.0, 0, 0]])
    near = np.array([[0.0, 0, 0], [1.0, 0, 0]])
    ds = FakeDS([1, 2, 2], [True, False, False], ["A", "A", "B"],
                [np.full((2, 3), np.nan), far, near])
    c = cohorts(ds)
    for k in ("female", "two_fly", "single_fly", "contact_pair", "group_A", "group_B"):
        assert k in c, k
    assert list(c["single_fly"]) == [True, False, False]
    assert list(c["two_fly"]) == [False, True, True]
    assert list(c["female"]) == [True, False, False]
    assert list(c["contact_pair"]) == [False, False, True]
    assert list(c["group_A"]) == [True, True, False]


def test_a_single_fly_window_is_never_a_contact_pair():
    ds = FakeDS([1], [False], ["A"], [np.zeros((2, 3))])
    assert not cohorts(ds)["contact_pair"][0]


@pytest.mark.skipif(not os.path.isdir(ROOT), reason="unified root not present")
def test_cohorts_on_the_real_root_are_non_empty():
    from tracking.train.data.windows import WindowDataset
    c = cohorts(WindowDataset(ROOT, split="val", window_length=1, train=False))
    assert c["female"].any() and c["single_fly"].any()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/train/mvq/test_evaluate.py -q`
Expected: FAIL — `No module named 'tracking.train.mvq.evaluate'`

- [ ] **Step 3: Implement `evaluate.py`**

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/train/mvq/ -q` → PASS

- [ ] **Step 5: Commit**

```bash
git add src/tracking/train/mvq/evaluate.py tests/train/mvq/test_evaluate.py
git commit -m "feat(train): cohort evaluation sharing the inference policy"
```

---

## Task 8: Checkpointing and temperature calibration

**Files:**
- Create: `src/tracking/train/mvq/checkpoint.py`, `src/tracking/train/mvq/calibrate.py`
- Test: `tests/train/mvq/test_checkpoint.py`, `tests/train/mvq/test_calibrate.py`

**Interfaces:**
- Produces: `make_manager(ckpt_dir, max_to_keep=3)`; `save_step(mngr, steps_done, model, optimizer, ema, ema_updates)`; `restore_latest(mngr, model, optimizer, ema) -> (model, optimizer, ema, ema_updates, start_step)`; `with_ema(model, ema, decay, t)`; `save_final(run_dir, model, ema, meta)`; `fit_temperature(logits, targets) -> float`; `write_calibration(run_dir, exist_temperature, vis_temperature)`.

Port `$SRC/train/train_mvq.py:763-847` and `:1160-1179`. Layout per spec §7.3: MVQ writes `<run_dir>/final/` (model + ema items) and `<run_dir>/ckpt/<step>/` — both are where the shipped inference side already looks (`detector/mvq/runner.py:228` joins `self.checkpoint` with `"ckpt"`).

`ema_meta` stays an explicit JSON item (`{"ema_updates": int, "ema_zero_seeded": True}`). `with_ema` divides by `(1 - decay**t)` to debias a zero-seeded EMA; a pre-fix checkpoint seeded from live params must not be debiased, which is exactly why the flag is written rather than inferred.

`calibrate.py` fits one scalar temperature each for the existence and visibility logits by minimising NLL on the val split, then writes:

```python
meta["calibration"] = {"exist_temperature": float(te), "vis_temperature": float(tv)}
```

`MVQRunner` reads exactly these keys at `detector/mvq/runner.py:250-252` and divides the logits by them at `:327-328`. A temperature of `1.0` must be the identity.

- [ ] **Step 1: Write the failing tests**

```python
# tests/train/mvq/test_calibrate.py
import numpy as np

from tracking.train.mvq.calibrate import fit_temperature, write_calibration


def test_already_calibrated_logits_fit_temperature_near_one():
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 2.0, 20000)
    targets = rng.uniform(size=20000) < 1 / (1 + np.exp(-logits))
    assert 0.85 < fit_temperature(logits, targets) < 1.2


def test_overconfident_logits_fit_a_temperature_above_one():
    rng = np.random.default_rng(1)
    logits = rng.normal(0, 2.0, 20000)
    targets = rng.uniform(size=20000) < 1 / (1 + np.exp(-logits))
    assert fit_temperature(logits * 3.0, targets) > 1.5


def test_write_calibration_uses_the_keys_the_runner_reads(tmp_path):
    import json
    write_calibration(tmp_path, 1.5, 2.0)
    meta = json.loads((tmp_path / "mvq_run.json").read_text())
    assert meta["calibration"] == {"exist_temperature": 1.5, "vis_temperature": 2.0}
```

```python
# tests/train/mvq/test_checkpoint.py
import numpy as np
from flax import nnx

from tracking.train.mvq.checkpoint import (make_manager, restore_latest, save_step,
                                           with_ema)


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
```

- [ ] **Step 2: Run them to verify they fail**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/train/mvq/test_checkpoint.py tests/train/mvq/test_calibrate.py -q`
Expected: FAIL — modules missing

- [ ] **Step 3: Implement both modules**

- [ ] **Step 4: Run tests to verify they pass**

- [ ] **Step 5: Commit**

```bash
git add src/tracking/train/mvq/checkpoint.py src/tracking/train/mvq/calibrate.py tests/train/mvq
git commit -m "feat(train): mvq checkpointing and temperature calibration"
```

---

## Task 9: The training driver

**Files:**
- Create: `src/tracking/train/mvq/run.py`, `src/tracking/train/mvq/__main__.py`
- Modify: `docs/training.md`
- Test: `tests/train/mvq/test_run_smoke.py`

**Interfaces:**
- Consumes: every module from Tasks 1-8.
- Produces: `run_training(cfg) -> dict` (the written `mvq_run.json`); `python -m tracking.train.mvq train=mvq_v2 paths=... run.name=...`.

Port `$SRC/train/train_mvq.py:847-1160` (`run_training`). Deviations:

- One root: build ONE `WindowDataset` per window length from `cfg.paths.train_data_root`, not a `ConcatWindowDataset` over four roots. Validation is the val split of that same root.
- Maskless: no prompt annealing, no `prompt_p` schedule, no `prompt_mask`.
- Augmentation is passed to `WindowDataset` (Task 4), not to `make_train_step`.
- `pseudo_weight` check: compare `cfg.train.pseudo_weight` against each pseudo source's `manifest["sources"][sid]["weight"]` and **warn on disagreement**, training at the manifest's value (spec §7.1).
- Write `mvq_run.json` with the resolved config, the per-source mass table from `mix_weights`, the realised mix from `MixCounter` over the first 200 batches, and the calibration block.
- `__main__.py` uses `runpy.run_module("tracking.train.mvq.run", run_name="__main__")`, matching `tracking/curate/__main__.py`.

The smoke test runs `cfg.train.smoke = True`: 2 steps, `batch_size=2`, `total_steps=2`, one window length, no eval, into `tmp_path`. It asserts a checkpoint directory appears and `mvq_run.json` is written. It is gated on the real root being present.

- [ ] **Step 1: Write the failing test**

```python
# tests/train/mvq/test_run_smoke.py
import json
import os

import pytest

ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/unified_v2"
pytestmark = pytest.mark.skipif(not os.path.isdir(ROOT), reason="unified root not present")


def test_two_step_smoke_run_writes_a_checkpoint_and_a_manifest(tmp_path):
    from hydra import compose, initialize_config_dir

    from tracking.train.mvq.run import run_training
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    with initialize_config_dir(config_dir=os.path.join(repo, "configs"), version_base=None):
        cfg = compose(config_name="train", overrides=[
            "train=mvq_v2", f"paths.train_data_root={ROOT}",
            f"paths.runs_root={tmp_path}", "run.name=smoke",
            "train.smoke=true", "train.total_steps=2", "train.batch_size=2",
            "train.window_lengths=[1]", "train.num_workers=2",
            "train.loader_workers=threads", "train.eval_every=1000000",
        ])
    meta = run_training(cfg)
    run_dir = tmp_path / "smoke"
    assert (run_dir / "mvq_run.json").is_file()
    assert (run_dir / "ckpt").is_dir()
    assert meta["train"]["total_steps"] == 2
    assert "sources" in meta["train_data"]


def test_the_module_entry_point_exists():
    import importlib
    assert importlib.util.find_spec("tracking.train.mvq.__main__") is not None
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/train/mvq/test_run_smoke.py -q`
Expected: FAIL — `No module named 'tracking.train.mvq.run'`

- [ ] **Step 3: Implement `run.py` and `__main__.py`**

- [ ] **Step 4: Document**

Add to `docs/training.md`: the MVQ training command, the v2 preset table, the checkpoint layout, and the `PYTHONPATH` note.

- [ ] **Step 5: Run the whole suite**

Run: `PYTHONPATH=$PWD/src python -m pytest -q` and `ruff check src tests` → both clean

- [ ] **Step 6: Commit**

```bash
git add src/tracking/train/mvq docs/training.md tests/train/mvq
git commit -m "feat(train): the mvq training driver"
```

---

## Out of scope

CenterDetect training (spec §7.2 — its own plan). SLURM launch scripts. `_photometric`'s remaining 0.142 s/sample cost. Consolidating the three copies of the ImageNet constants — Plan 3 must **import** them, never add a fourth.
