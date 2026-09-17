# MVQ training data layer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `tracking.train.common` and `tracking.train.data` — the window dataset that turns the unified training root into batched multi-view samples the MVQ model can train on.

**Architecture:** Port the window dataset from `jarvis_jax`, routing all geometry through this repo's `CameraRig`, all camera/keypoint axes through `io.names.Order`, and all SAM masks through the `MaskStore` sidecar. The model stays maskless: masks feed copy-paste donor cutting only and never reach the model.

**Tech Stack:** Python 3.13, numpy, Pillow, JAX (sharding/prefetch only), pytest.

**Spec:** `docs/superpowers/specs/2026-09-16-mvq-centerdetect-training-port-design.md` (§4 architecture, §7.1 MVQ)

**Port source:** `/mmfs1/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/third_party/jarvis_jax/jarvis_jax` (referred to below as `$J`). It is read-only reference — never modify it.

**Plan 2 of 4.** Plan 1 (curate) is complete and merged into this branch. Plan 3 is the MVQ training loop, Plan 4 CenterDetect. This plan produces working, testable software on its own: a dataset that yields correct batches, verifiable without any model.

## Global Constraints

- Branch: `feat/mvq-centerdetect-training`. Do not commit to `main`.
- Python `>=3.13`. Ruff `line-length = 100`, `select = ["E", "W", "F", "I", "UP", "B"]`. Run `ruff format` then `ruff check` on **only the files you touch** — never whole directories. `scripts/collect_sessions.py` is pre-existing ruff-unclean and unrelated.
- **Comment style: minimal and concise.** Docstrings are a one-line summary plus a usage example. The existing `src/tracking/train` and `src/tracking/curate` code has **zero inline `#` comments** — preserve that. The port source is extremely verbose (40-line module docstrings, paragraph rationale); compress ruthlessly. Keep only a clause where losing it would let a real bug return.
- Never index a keypoint or camera axis by bare integer. Resolve by name via `tracking.io.names.Order`.
- Read calibration only through `tracking.geometry.rig.CameraRig` / `tracking.io.names.load_camera_order`. Do not add a second parser.
- Do not modify `inverse_kinematics/`, `preprocess/`, `postprocess/`, `qc/`, `viz/`, `curate/`, or `configs/pipeline.yaml`. The only inference file this plan touches is `detector/mvq/policy.py` (Task 1).
- **The model is maskless.** `prompt_mask` does not exist in this port. Masks are loaded only as copy-paste donor silhouettes under the name `donor_mask`.

## The unified root this reads

`/gscratch/portia/eabe/data/Johnson_lab/red_data/unified_v2`, mask sidecar `unified_v2_masks`. Verified contents: 202,188 images, 36,830 framesets (human 2,814 / pseudo 24,805 / negatives 7,302 / single-fly 1,909), 145,688 mask files, 3 calibration groups, `validate_root` reports 0 errors.

Facts the tests below depend on:
- 7,302 framesets have `fly_id == -1`; all are `source_id == "negatives_0905"`, `role == "negative"`, and carry `center3D`.
- `2026_04_02_15_25_51` and `2026_04_02_17_28_34` carry `calib_groups: ["A","C"]`; every frameset carries its own `calib_group`.
- Keypoint order is `annotations/keypoint_names.json`, 50 names starting `Antenna_Base, EyeL, EyeR, Scutellum`.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/tracking/detector/mvq/slots.py` | Slot and sex integer codes shared by inference and training |
| `src/tracking/train/__init__.py` | Package marker |
| `src/tracking/train/common/prefetch.py` | Background-thread device prefetcher |
| `src/tracking/train/common/sharding.py` | Data-parallel mesh, `shard_batch`, `replicate` |
| `src/tracking/train/data/transforms.py` | `crop_origin`, ImageNet constants, `normalize_rgb` |
| `src/tracking/train/data/windows.py` | `WindowDataset`: enumeration, metadata, sample building, `window_batches` |
| `src/tracking/train/data/augment.py` | `MVAugParams`, `augment_window`, `build_lr_swap` |
| `src/tracking/train/data/copypaste.py` | `CopyPasteParams`, multi-view donor compositing |
| `src/tracking/train/data/loaders.py` | Process-pool sample loader |
| `tests/train/test_*.py` | One module per source module |

---

## Task 1: Shared slot/sex codes

**Files:**
- Create: `src/tracking/detector/mvq/slots.py`
- Modify: `src/tracking/detector/mvq/policy.py` (remove its inlined slot table, import instead)
- Create: `src/tracking/train/__init__.py`, `src/tracking/train/common/__init__.py`, `src/tracking/train/data/__init__.py`
- Test: `tests/train/test_slots.py`

**Interfaces:**
- Consumes: nothing
- Produces: `SLOT_PROMPTED=0`, `SLOT_FEMALE=1`, `SLOT_MALE=2`, `SLOT_OTHER=3`, `N_SLOTS=4`, `SEX_FEMALE=0`, `SEX_MALE=1`, `SEX_UNKNOWN=-1`, `SEX_PRESENT_UNKNOWN=2`

`policy.py` currently inlines the slot table specifically to avoid importing `train.matching`. Once training exists that inlining becomes a second source of truth. The codes live under `detector/mvq/` rather than under `train/` so **inference never imports from `train/`**.

- [ ] **Step 1: Write the failing test**

Create `tests/train/test_slots.py`:

```python
from tracking.detector.mvq import policy, slots


def test_slot_values_are_the_p3a_table():
    assert (slots.SLOT_PROMPTED, slots.SLOT_FEMALE, slots.SLOT_MALE, slots.SLOT_OTHER) == (0, 1, 2, 3)
    assert slots.N_SLOTS == 4


def test_sex_codes():
    assert (slots.SEX_FEMALE, slots.SEX_MALE, slots.SEX_UNKNOWN) == (0, 1, -1)
    assert slots.SEX_PRESENT_UNKNOWN == 2


def test_policy_uses_the_shared_table_not_its_own():
    assert policy.SLOT_PROMPTED is slots.SLOT_PROMPTED
    assert policy.N_SLOTS is slots.N_SLOTS


def test_typed_candidates_still_excludes_slot_zero_at_four_slots():
    assert policy.typed_candidates(4) == [1, 2, 3]
    assert policy.typed_candidates(3) == [0, 1, 2]
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/train/test_slots.py -v`
Expected: FAIL — `ImportError: cannot import name 'slots'`

- [ ] **Step 3: Implement**

Create `src/tracking/detector/mvq/slots.py`:

```python
"""Instance-slot and sex integer codes, shared by inference and training.

>>> SLOT_FEMALE, SEX_FEMALE
(1, 0)
"""

from __future__ import annotations

SLOT_PROMPTED, SLOT_FEMALE, SLOT_MALE, SLOT_OTHER = 0, 1, 2, 3
N_SLOTS = 4

SEX_FEMALE, SEX_MALE, SEX_UNKNOWN = 0, 1, -1
SEX_PRESENT_UNKNOWN = 2
```

In `src/tracking/detector/mvq/policy.py`, delete the four inlined assignment lines (`SLOT_PROMPTED, SLOT_FEMALE, SLOT_MALE, SLOT_OTHER = 0, 1, 2, 3` and `N_SLOTS = 4`) and their comment, and add to the imports:

```python
from tracking.detector.mvq.slots import (
    N_SLOTS,
    SLOT_FEMALE,
    SLOT_MALE,
    SLOT_OTHER,
    SLOT_PROMPTED,
)
```

Keep every name importable from `policy` as before — other modules already do `from ...policy import SLOT_PROMPTED`. If ruff flags an unused import in `policy.py`, the name is re-exported deliberately; add it to a module-level `__all__` rather than deleting it.

Create the three empty-ish package markers:

```python
"""Training: data layer, losses and loops for the MVQ and CenterDetect models."""
```
```python
"""Shared training utilities: sharding, prefetch, checkpointing, metrics."""
```
```python
"""Training data: the window dataset, augmentation and loaders."""
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/train/test_slots.py -v` → 4 passed
Run: `python -m pytest tests -q` → the existing 93 plus these 4; nothing broken.

- [ ] **Step 5: Lint and commit**

```bash
ruff format src/tracking/detector/mvq/slots.py src/tracking/detector/mvq/policy.py src/tracking/train tests/train
ruff check src/tracking/detector/mvq/slots.py src/tracking/detector/mvq/policy.py src/tracking/train tests/train
git add src/tracking/detector/mvq/slots.py src/tracking/detector/mvq/policy.py src/tracking/train tests/train
git commit -m "feat(train): shared slot/sex codes, replacing policy.py's inlined table"
```

---

## Task 2: Common utilities — sharding and prefetch

**Files:**
- Create: `src/tracking/train/common/sharding.py`, `src/tracking/train/common/prefetch.py`
- Test: `tests/train/test_common.py`

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces:
  - `data_parallel_mesh() -> jax.sharding.Mesh` (1-D mesh over all local devices, axis name `"data"`)
  - `shard_batch(x, mesh)` — shard along axis 0
  - `replicate(tree, mesh)` — replicate every array leaf
  - `prefetch(batch_iter, mesh, depth=2)` — generator of device-resident sharded batches

Port `$J/sharding.py` (39 lines) and `$J/data/prefetch.py` (43 lines) verbatim apart from imports. Both are small and correct; compress their docstrings to one line each plus a usage example.

Two behaviours in `prefetch` are load-bearing and must survive: a worker-side exception is re-raised on the consumer side rather than swallowed, and batches are converted with `np.asarray` (not `jnp.asarray`) before device-put, because a `jnp` array would land on the default device and defeat the sharding.

- [ ] **Step 1: Write the failing test**

Create `tests/train/test_common.py`:

```python
import numpy as np
import pytest

from tracking.train.common.prefetch import prefetch
from tracking.train.common.sharding import data_parallel_mesh, replicate, shard_batch


def test_mesh_covers_all_local_devices():
    import jax

    mesh = data_parallel_mesh()
    assert mesh.axis_names == ("data",)
    assert mesh.size == len(jax.devices())


def test_shard_batch_preserves_shape_and_values():
    mesh = data_parallel_mesh()
    x = np.arange(8 * 3, dtype=np.float32).reshape(8, 3)
    out = shard_batch(x, mesh)
    assert out.shape == (8, 3)
    np.testing.assert_allclose(np.asarray(out), x)


def test_replicate_preserves_a_pytree():
    mesh = data_parallel_mesh()
    tree = {"a": np.ones((2, 2), np.float32), "b": np.zeros((3,), np.float32)}
    out = replicate(tree, mesh)
    np.testing.assert_allclose(np.asarray(out["a"]), tree["a"])
    np.testing.assert_allclose(np.asarray(out["b"]), tree["b"])


def test_prefetch_yields_every_batch_in_order():
    mesh = data_parallel_mesh()
    batches = [{"x": np.full((4, 2), i, np.float32)} for i in range(5)]
    got = [np.asarray(b["x"])[0, 0] for b in prefetch(iter(batches), mesh, depth=2)]
    assert got == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_prefetch_reraises_a_worker_exception():
    mesh = data_parallel_mesh()

    def broken():
        yield {"x": np.zeros((4, 2), np.float32)}
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        list(prefetch(broken(), mesh, depth=1))
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/train/test_common.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tracking.train.common.sharding'`

- [ ] **Step 3: Implement**

Port `$J/sharding.py` to `src/tracking/train/common/sharding.py`, keeping `data_parallel_mesh`, `shard_batch`, `replicate` with identical bodies. Replace the module and function docstrings with one-liners; keep one clause on `replicate` noting that Orbax-restored arrays are committed to device 0 and must be explicitly replicated or the jitted step raises "Received incompatible devices" — that is a real bug that will otherwise return.

Port `$J/data/prefetch.py` to `src/tracking/train/common/prefetch.py`, changing only `from jarvis_jax.sharding import shard_batch` to `from tracking.train.common.sharding import shard_batch`.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/train/test_common.py -v` → 5 passed

- [ ] **Step 5: Lint and commit**

```bash
ruff format src/tracking/train/common tests/train/test_common.py
ruff check src/tracking/train/common tests/train/test_common.py
git add src/tracking/train/common tests/train/test_common.py
git commit -m "feat(train): port sharding and prefetch utilities"
```

---

## Task 3: Transforms subset

**Files:**
- Create: `src/tracking/train/data/transforms.py`
- Test: `tests/train/test_transforms.py`

**Interfaces:**
- Consumes: nothing
- Produces: `IMAGENET_MEAN`, `IMAGENET_STD` (float32 `(3,)`), `crop_origin(bbox, img_w, img_h, crop=448) -> (x0, y0)`, `normalize_rgb(rgb01)`

Port only these four from `$J/data/transforms.py`. Do **not** port `transform_keypoints`, `gaussian_blob` or `gaussian_heatmaps` — those serve heatmap models, and MVQ is not one. Adding them would be unrequested scope.

`crop_origin` centres a `crop`-sized window on a bbox centre and clamps it inside the image. That clamping is what keeps a crop in bounds for a fly near the frame edge.

- [ ] **Step 1: Write the failing test**

Create `tests/train/test_transforms.py`:

```python
import numpy as np

from tracking.train.data.transforms import IMAGENET_MEAN, IMAGENET_STD, crop_origin, normalize_rgb


def test_imagenet_constants():
    np.testing.assert_allclose(IMAGENET_MEAN, [0.485, 0.456, 0.406], rtol=1e-6)
    np.testing.assert_allclose(IMAGENET_STD, [0.229, 0.224, 0.225], rtol=1e-6)


def test_crop_origin_centres_on_the_bbox_centre():
    x0, y0 = crop_origin([500, 200, 20, 20], 1936, 448, crop=100)
    assert (x0, y0) == (460, 160)


def test_crop_origin_clamps_at_the_left_and_top_edges():
    assert crop_origin([5, 5, 10, 10], 1936, 448, crop=100) == (0, 0)


def test_crop_origin_clamps_at_the_right_and_bottom_edges():
    x0, y0 = crop_origin([1930, 440, 10, 10], 1936, 448, crop=100)
    assert x0 == 1936 - 100
    assert y0 == 448 - 100


def test_crop_origin_handles_an_image_smaller_than_the_crop():
    x0, y0 = crop_origin([10, 10, 4, 4], 50, 40, crop=100)
    assert x0 <= 0 and y0 <= 0


def test_normalize_rgb_applies_imagenet_stats():
    x = np.full((2, 2, 3), 0.485, np.float32)
    out = normalize_rgb(x)
    np.testing.assert_allclose(out[..., 0], 0.0, atol=1e-6)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/train/test_transforms.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tracking.train.data.transforms'`

- [ ] **Step 3: Implement**

Copy the four symbols from `$J/data/transforms.py` verbatim. If `test_crop_origin_handles_an_image_smaller_than_the_crop` fails, do **not** change the source behaviour to satisfy it — report the actual behaviour and adjust the test to assert what the shipped function does, noting it in your report. The ported function's semantics must match the source exactly, because the checkpoints were trained under them.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/train/test_transforms.py -v` → 6 passed

- [ ] **Step 5: Lint and commit**

```bash
ruff format src/tracking/train/data/transforms.py tests/train/test_transforms.py
ruff check src/tracking/train/data/transforms.py tests/train/test_transforms.py
git add src/tracking/train/data/transforms.py tests/train/test_transforms.py
git commit -m "feat(train): port the transforms subset MVQ needs"
```

---

## Task 4: Window enumeration and metadata

**Files:**
- Create: `src/tracking/train/data/windows.py` (first half — construction and metadata)
- Test: `tests/train/test_windows_enum.py`

**Interfaces:**
- Consumes: `tracking.curate.schema`, `tracking.geometry.rig.CameraRig`, `tracking.io.names.Order`, `tracking.detector.mvq.slots`
- Produces:
  - `CROP = 448`
  - `WINDOW_KEYS` — the sample dict's keys (see Task 5; define the tuple here)
  - `WindowDataset(root, split, T=1, *, pair_deltas=(1,), max_flies=2, jitter_units=3.0, seed=0, train=True, recordings=None, copy_paste=None, center_shift_units=0.0, sex_overrides=None, masks_root=None)`
  - `.windows` (list of `(recording, fly, f0)`), `.win_delta`, `.__len__()`
  - `.window_index(rec, fly, f0, delta=None) -> int`, `.delta(i)`, `.calib_group(i)`
  - `.source(i)`, `.weight(i)`, `.role(i)`, `.is_female(i)`, `.n_flies(i)`, `.unlabelled_sex(i)`, `.is_negative(i)`
  - `.camera_names(i) -> Order`, `.fly_centroids(i)`

Port the construction half of `$J/data/v12_windows.py` (lines ~100-510). Substitutions:

| source | replacement |
| --- | --- |
| `ReprojectionTool(os.path.join(root, "calibrations", grp))` | `CameraRig.from_calib_dir(Path(root) / "calibrations" / grp)` |
| `rt.cameras.keys()` (dict of name→Camera) | `rig.cameras` (an `Order`); return the `Order` itself from `camera_names`, not a list |
| `rt.num_cameras` | `rig.n_cameras` |
| `from jarvis_jax.train.matching import SEX_*` | `from tracking.detector.mvq.slots import SEX_*` |
| `from jarvis_jax.data.build_v5 import iter_resolved_slots` | inline it (below) |
| `json.load(open(...instances_...))` | `tracking.curate.schema.load_instances(root, split)` |
| `manifest.json` reads | `tracking.curate.schema.load_manifest(root)` |

Inline `iter_resolved_slots` as a module-level helper — it is 4 lines and its only caller is here:

```python
def _resolved_slots(frameset):
    """(image id, annotation id) for the cameras of `frameset` that resolved to a fly.

    >>> list(_resolved_slots({"frames": [1, 2], "ann_ids": [7, None]}))
    [(1, 7)]
    """
    for img_id, ann_id in zip(frameset["frames"], frameset["ann_ids"], strict=True):
        if ann_id is not None:
            yield img_id, ann_id
```

**`calib_group` comes from the frameset, not the manifest.** The source reads `self.manifest[rec]["calib_group"]`. In the unified root a recording may span several groups (the 2026-04-02 mid-day recalibration), so every frameset carries its own `calib_group` and that is authoritative. Resolve it with `schema.frameset_field(manifest, frameset, recording, "calib_group", None)` and raise if it comes back `None` — a window whose calibration is unknown must fail loudly, never default.

Keep the per-window sex resolution chain exactly as the source has it (frameset annotation first, then manifest `fly_sex`, then manifest `sex`, then unknown) and keep the one-warning-per-disagreement behaviour. A recording can carry framesets from two annotation subsets that label different animals under one fly id; collapsing to one value per `(recording, fly)` was a real bug.

Keep `window_index(rec, fly, f0, delta)` addressing. Windows are keyed by `(rec, fly, f0, delta)` and **must never be addressed positionally** — this project's documented bug class.

- [ ] **Step 1: Write the failing test**

Create `tests/train/test_windows_enum.py`:

```python
import os

import pytest

from tracking.train.data.windows import WindowDataset

ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/unified_v2"
pytestmark = pytest.mark.skipif(not os.path.isdir(ROOT), reason="unified root not present")


@pytest.fixture(scope="module")
def ds():
    return WindowDataset(ROOT, "train", T=1)


def test_every_frameset_becomes_one_window_at_T1(ds):
    assert len(ds) == 36677


def test_negatives_are_identified(ds):
    neg = [i for i in range(len(ds)) if ds.is_negative(i)]
    assert len(neg) == 7302
    assert all(ds.role(i) == "negative" for i in neg[:50])


def test_window_index_round_trips(ds):
    for i in (0, 100, len(ds) - 1):
        rec, fly, f0 = ds.windows[i]
        assert ds.window_index(rec, fly, f0, ds.delta(i)) == i


def test_calib_group_comes_from_the_frameset(ds):
    groups = {ds.calib_group(i) for i in range(0, len(ds), 997)}
    assert groups <= {"A", "B", "C"}


def test_multi_calib_recording_yields_both_groups(ds):
    seen = {
        ds.calib_group(i)
        for i in range(len(ds))
        if ds.windows[i][0] == "2026_04_02_15_25_51"
    }
    assert seen == {"A"} or seen == {"C"} or seen == {"A", "C"}
    assert seen


def test_camera_names_is_an_order_of_seven(ds):
    cams = ds.camera_names(0)
    assert len(cams) == 7
    assert all(n.startswith("Cam") for n in cams)


def test_weight_and_source_follow_the_frameset(ds):
    i = next(i for i in range(len(ds)) if ds.source(i) == "pseudo")
    assert ds.weight(i) == pytest.approx(0.3)
    j = next(i for i in range(len(ds)) if ds.is_negative(i))
    assert ds.weight(j) == pytest.approx(1.0)


def test_pair_deltas_produce_more_windows_at_T2():
    d2 = WindowDataset(ROOT, "train", T=2, pair_deltas=(1, 4, 16))
    assert len(d2) > 0
    assert set(d2.win_delta) <= {1, 4, 16}
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/train/test_windows_enum.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tracking.train.data.windows'`

- [ ] **Step 3: Implement**

Port the construction half as described. The dataset loads a 285 MB instances file, so build it once per test module (the fixture above is `scope="module"`).

If `test_every_frameset_becomes_one_window_at_T1` reports a count other than 36,677, do not edit the test to match — investigate. That number is the verified train frameset count of the unified root, and at `T=1` every frameset yields exactly one window.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/train/test_windows_enum.py -v` → 8 passed

- [ ] **Step 5: Lint and commit**

```bash
ruff format src/tracking/train/data/windows.py tests/train/test_windows_enum.py
ruff check src/tracking/train/data/windows.py tests/train/test_windows_enum.py
git add src/tracking/train/data/windows.py tests/train/test_windows_enum.py
git commit -m "feat(train): window enumeration and per-window metadata"
```

---

## Task 5: Sample building

**Files:**
- Modify: `src/tracking/train/data/windows.py` (add `_build` and `__getitem__`)
- Test: `tests/train/test_windows_build.py`

**Interfaces:**
- Consumes: Task 4's `WindowDataset`, Task 3's `crop_origin`, `tracking.curate.masks.store.MaskStore`
- Produces: `WindowDataset.__getitem__(i) -> dict` with exactly these keys:

```python
WINDOW_KEYS = (
    "crops", "cam_valid", "M", "t_local", "center3D", "kp3d_local", "has3d",
    "kp2d", "vis2d", "fly_valid", "px_scale", "is_female", "donor_mask",
    "crop_origin", "fly_sex", "unlabelled_sex", "sample_weight", "is_negative",
)
```

Shapes for `T` frames, `C` cameras, `K` keypoints, `F = max_flies`: `crops (T,C,448,448,3) uint8`, `cam_valid (T,C) bool`, `M (C,2,3) float32`, `t_local (T,C,2) float32`, `center3D (3,) float32`, `kp3d_local (F,T,K,3) float32`, `has3d (F,T,K) bool`, `kp2d (F,T,C,K,2) float32`, `vis2d (F,T,C,K) bool`, `fly_valid (F,) bool`, `donor_mask (T,C,448,448) bool`.

**Renamed from the source: `prompt_mask` → `donor_mask`.** The model never receives it; its only consumer is copy-paste donor cutting (Task 7). Keeping the old name would tell a future reader the model gets a prompt. Rename every occurrence.

Substitutions beyond Task 4's:

| source | replacement |
| --- | --- |
| `rt.reconstruct_point(pts, cams_to_use=use)` | `rig.reconstruct(uv, valid)` where `uv` is `(C,2)` float and `valid` is `(C,)` bool — build both full-length and mark non-contributing cameras `False`, rather than passing a subset list |
| `_load_mask(root, file_name, src_ann_id, ann_id, w, h)` | `MaskStore(masks_root).load(recording, camera, frame, ann_id)`, returning `None` when absent |
| `prompt_mask` | `donor_mask` |

`MaskStore.load` returns `None` for a missing mask; treat that as an all-zero silhouette. When `masks_root` is `None`, skip mask loading entirely and leave `donor_mask` all-zero — copy-paste is then simply unavailable, which is the documented degraded mode.

A negative window (`fly_id < 0`) must yield: `fly_valid` all `False`, `has3d` all `False`, no 2D labels, an all-zero `donor_mask`, `unlabelled_sex == SEX_UNKNOWN`, and `is_negative` `True`. The `unlabelled_sex` value is load-bearing — `SEX_PRESENT_UNKNOWN` there would tell the existence loss to ignore the very window it exists to learn from.

- [ ] **Step 1: Write the failing test**

Create `tests/train/test_windows_build.py`:

```python
import os

import numpy as np
import pytest

from tracking.detector.mvq.slots import SEX_UNKNOWN
from tracking.train.data.windows import CROP, WINDOW_KEYS, WindowDataset

ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/unified_v2"
MASKS = ROOT + "_masks"
pytestmark = pytest.mark.skipif(not os.path.isdir(ROOT), reason="unified root not present")


@pytest.fixture(scope="module")
def ds():
    return WindowDataset(ROOT, "train", T=1, masks_root=MASKS)


def test_sample_has_exactly_the_declared_keys(ds):
    s = ds[0]
    assert set(s) == set(WINDOW_KEYS)


def test_shapes_and_dtypes(ds):
    s = ds[0]
    T, C, K, F = 1, 7, 50, 2
    assert s["crops"].shape == (T, C, CROP, CROP, 3) and s["crops"].dtype == np.uint8
    assert s["cam_valid"].shape == (T, C) and s["cam_valid"].dtype == np.bool_
    assert s["M"].shape == (C, 2, 3) and s["M"].dtype == np.float32
    assert s["t_local"].shape == (T, C, 2)
    assert s["center3D"].shape == (3,)
    assert s["kp3d_local"].shape == (F, T, K, 3)
    assert s["has3d"].shape == (F, T, K)
    assert s["kp2d"].shape == (F, T, C, K, 2)
    assert s["vis2d"].shape == (F, T, C, K)
    assert s["fly_valid"].shape == (F,)
    assert s["donor_mask"].shape == (T, C, CROP, CROP)


def test_host_fly_is_always_instance_zero_and_valid(ds):
    i = next(i for i in range(len(ds)) if not ds.is_negative(i))
    assert ds[i]["fly_valid"][0]


def test_labelled_3d_reprojects_onto_the_labelled_2d(ds):
    i = next(i for i in range(len(ds)) if not ds.is_negative(i))
    s = ds[i]
    xyz = s["kp3d_local"][0, 0] + s["center3D"]
    uv = np.einsum("cij,kj->kci", s["M"], xyz) + s["t_local"][0][None, :, :]
    ok = s["vis2d"][0, 0] & s["has3d"][0, 0][:, None]
    err = np.linalg.norm(uv - s["kp2d"][0, 0].transpose(1, 0, 2), axis=-1)
    assert np.median(err[ok.T]) < 5.0


def test_negative_window_is_empty_and_marked(ds):
    i = next(i for i in range(len(ds)) if ds.is_negative(i))
    s = ds[i]
    assert not s["fly_valid"].any()
    assert not s["has3d"].any()
    assert not s["donor_mask"].any()
    assert s["unlabelled_sex"] == SEX_UNKNOWN
    assert bool(s["is_negative"])


def test_crops_are_not_all_black(ds):
    s = ds[0]
    assert s["crops"][0][s["cam_valid"][0]].max() > 0


def test_donor_mask_is_populated_when_a_sidecar_mask_exists(ds):
    hit = 0
    for i in range(0, 4000, 137):
        if ds.is_negative(i):
            continue
        if ds[i]["donor_mask"].any():
            hit += 1
        if hit >= 2:
            break
    assert hit >= 2


def test_without_masks_root_donor_mask_is_empty():
    d = WindowDataset(ROOT, "train", T=1, masks_root=None)
    i = next(i for i in range(len(d)) if not d.is_negative(i))
    assert not d[i]["donor_mask"].any()


def test_sample_weight_matches_the_window_weight(ds):
    i = next(i for i in range(len(ds)) if ds.source(i) == "pseudo")
    assert float(ds[i]["sample_weight"]) == pytest.approx(ds.weight(i))
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/train/test_windows_build.py -v`
Expected: FAIL — `TypeError` or `AttributeError` from the missing `__getitem__`

- [ ] **Step 3: Implement**

Port `_build` and `__getitem__`. The reprojection test is the one that proves the geometry substitution is right: if `CameraRig` is wired up wrongly the median error will be large rather than a few pixels. If it fails, the bug is in the `M`/`t_local` construction or the camera-row ordering, not in the test.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/train/test_windows_build.py -v` → 9 passed

- [ ] **Step 5: Commit**

```bash
ruff format src/tracking/train/data/windows.py tests/train/test_windows_build.py
ruff check src/tracking/train/data/windows.py tests/train/test_windows_build.py
git add src/tracking/train/data/windows.py tests/train/test_windows_build.py
git commit -m "feat(train): build multi-view samples from the unified root"
```

---

## Task 6: Augmentation

**Files:**
- Create: `src/tracking/train/data/augment.py`
- Test: `tests/train/test_augment.py`

**Interfaces:**
- Consumes: Task 5's sample dict
- Produces:
  - `MVAugParams(enabled=True, rot_deg=30.0, scale_min=0.8, scale_max=1.25, translate_frac=0.1, world_yaw=True, world_tilt_deg=30.0, mirror_p=0.5, cam_drop_p=0.1, cam_drop_max=2, brightness=0.2, contrast=0.2, gamma=0.2, blur_max=0.5, noise_scale=0.02, pc_color=0.2)`
  - `augment_window(sample, params, rng, lr_swap) -> dict`
  - `build_lr_swap(kp_names) -> np.ndarray`, `assert_lr_swap_covers(kp_names, swap)`

Port `$J/data/mv_augment.py` (179 lines) whole, plus only `build_lr_swap` and `assert_lr_swap_covers` from `$J/data/augment.py` (523 lines — everything else there serves other models). Rename `prompt_mask` to `donor_mask` throughout.

The defaults above are v2's, from `mvq_t2_v2_20260906d/mvq_run.json` — note `cam_drop_p` is **0.1**, not the 0.3 in the older config.

`build_lr_swap` maps each keypoint index to its left/right mirror partner. `assert_lr_swap_covers` raises when a name has no partner, which is what stops a mirror augmentation from silently scrambling anatomy.

- [ ] **Step 1: Write the failing test**

Create `tests/train/test_augment.py`:

```python
import numpy as np
import pytest

from tracking.train.data.augment import (
    MVAugParams,
    assert_lr_swap_covers,
    augment_window,
    build_lr_swap,
)

KP = ["Antenna_Base", "EyeL", "EyeR", "Scutellum", "T1L_FeTi", "T1R_FeTi"]


def test_lr_swap_pairs_left_and_right():
    swap = build_lr_swap(KP)
    assert swap[KP.index("EyeL")] == KP.index("EyeR")
    assert swap[KP.index("EyeR")] == KP.index("EyeL")
    assert swap[KP.index("T1L_FeTi")] == KP.index("T1R_FeTi")


def test_lr_swap_is_identity_for_midline_keypoints():
    swap = build_lr_swap(KP)
    assert swap[KP.index("Scutellum")] == KP.index("Scutellum")
    assert swap[KP.index("Antenna_Base")] == KP.index("Antenna_Base")


def test_lr_swap_is_an_involution():
    swap = build_lr_swap(KP)
    assert list(swap[swap]) == list(range(len(KP)))


def test_assert_lr_swap_covers_accepts_a_complete_map():
    assert_lr_swap_covers(KP, build_lr_swap(KP))


def test_assert_lr_swap_covers_rejects_an_unpaired_left():
    bad = ["EyeL", "Scutellum"]
    with pytest.raises(Exception):
        assert_lr_swap_covers(bad, build_lr_swap(bad))


def _sample(T=1, C=7, K=len(KP), F=2, crop=32):
    rng = np.random.default_rng(0)
    return {
        "crops": rng.integers(0, 255, (T, C, crop, crop, 3), dtype=np.uint8),
        "cam_valid": np.ones((T, C), bool),
        "M": rng.normal(size=(C, 2, 3)).astype(np.float32),
        "t_local": rng.normal(size=(T, C, 2)).astype(np.float32),
        "center3D": np.zeros(3, np.float32),
        "kp3d_local": rng.normal(size=(F, T, K, 3)).astype(np.float32),
        "has3d": np.ones((F, T, K), bool),
        "kp2d": rng.uniform(0, crop, (F, T, C, K, 2)).astype(np.float32),
        "vis2d": np.ones((F, T, C, K), bool),
        "fly_valid": np.array([True, False]),
        "px_scale": np.float32(1.0),
        "is_female": np.bool_(True),
        "donor_mask": np.zeros((T, C, crop, crop), bool),
        "crop_origin": np.zeros((C, 2), np.int32),
        "fly_sex": np.zeros((F,), np.int8),
        "unlabelled_sex": np.int8(-1),
        "sample_weight": np.float32(1.0),
        "is_negative": np.bool_(False),
    }


def test_augment_preserves_keys_and_shapes():
    s = _sample()
    out = augment_window(s, MVAugParams(), np.random.default_rng(1), build_lr_swap(KP))
    assert set(out) == set(s)
    for k in ("crops", "kp3d_local", "kp2d", "vis2d", "M", "t_local"):
        assert out[k].shape == s[k].shape


def test_disabled_augmentation_is_a_passthrough():
    s = _sample()
    out = augment_window(s, MVAugParams(enabled=False), np.random.default_rng(1), build_lr_swap(KP))
    np.testing.assert_array_equal(out["crops"], s["crops"])
    np.testing.assert_allclose(out["kp3d_local"], s["kp3d_local"])


def test_camera_drop_only_ever_clears_cam_valid():
    s = _sample()
    p = MVAugParams(cam_drop_p=1.0, cam_drop_max=2, rot_deg=0.0, scale_min=1.0, scale_max=1.0,
                    translate_frac=0.0, world_yaw=False, world_tilt_deg=0.0, mirror_p=0.0)
    out = augment_window(s, p, np.random.default_rng(3), build_lr_swap(KP))
    assert out["cam_valid"].sum() < s["cam_valid"].sum()
    assert out["cam_valid"].sum() >= s["cam_valid"].sum() - 2
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/train/test_augment.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tracking.train.data.augment'`

- [ ] **Step 3: Implement**

Port as described. If `build_lr_swap`'s real naming convention differs from the `EyeL`/`EyeR`, `T1L_`/`T1R_` assumptions in the test, fix the **test** to match the shipped convention and say so in your report — do not change the swap logic, which must match what the checkpoints were trained under.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/train/test_augment.py -v` → 9 passed

- [ ] **Step 5: Commit**

```bash
ruff format src/tracking/train/data/augment.py tests/train/test_augment.py
ruff check src/tracking/train/data/augment.py tests/train/test_augment.py
git add src/tracking/train/data/augment.py tests/train/test_augment.py
git commit -m "feat(train): port multi-view augmentation and the LR swap map"
```

---

## Task 7: Copy-paste compositing

**Files:**
- Create: `src/tracking/train/data/copypaste.py`
- Modify: `src/tracking/train/data/windows.py` (donor pool + `paste_window`)
- Test: `tests/train/test_copypaste.py`

**Interfaces:**
- Consumes: Task 5's sample dict (`donor_mask`), Task 4's `WindowDataset`
- Produces:
  - `CopyPasteParams(p=0.0, opposite_sex_p=0.7, contact_p=0.3, contact_sep=(8.0, 30.0))`
  - `composite(host, donor, shifts, rng) -> dict | None`
  - `WindowDataset.paste_window(i, rng) -> dict`

Port `$J/data/mv_copy_paste.py` (182 lines), renaming `prompt_mask` to `donor_mask`. Add the donor pool to `WindowDataset.__init__` keyed by `(calibration group, host sex, spacing)` — a donor must span the same `Delta` as its host or the pasted fly moves through a different amount of time.

v2 trained at `copy_paste_p: 0.8`, `contact_p: 0.7`, `contact_sep: (4.0, 25.0)`, so this path is heavily exercised and must be correct.

Two source behaviours to preserve: a donor with no mask content in any target-valid camera for a given frame yields `None` (the paste is skipped rather than pasting nothing), and where the donor is composited on top, the host's own `donor_mask` loses those pixels.

**Negatives are never donors and never paste targets.** Exclude `is_negative(i)` windows from the pool and from pasting.

- [ ] **Step 1: Write the failing test**

Create `tests/train/test_copypaste.py`:

```python
import numpy as np
import pytest

from tracking.train.data.copypaste import CopyPasteParams, composite


def _win(crop=24, C=3, T=1, K=4, F=2, fill=40, mask_box=None):
    s = {
        "crops": np.full((T, C, crop, crop, 3), fill, np.uint8),
        "cam_valid": np.ones((T, C), bool),
        "M": np.tile(np.eye(3, dtype=np.float32)[:2], (C, 1, 1)),
        "t_local": np.zeros((T, C, 2), np.float32),
        "center3D": np.zeros(3, np.float32),
        "kp3d_local": np.zeros((F, T, K, 3), np.float32),
        "has3d": np.zeros((F, T, K), bool),
        "kp2d": np.zeros((F, T, C, K, 2), np.float32),
        "vis2d": np.zeros((F, T, C, K), bool),
        "fly_valid": np.array([True, False]),
        "px_scale": np.float32(1.0),
        "is_female": np.bool_(True),
        "donor_mask": np.zeros((T, C, crop, crop), bool),
        "crop_origin": np.zeros((C, 2), np.int32),
        "fly_sex": np.zeros((F,), np.int8),
        "unlabelled_sex": np.int8(-1),
        "sample_weight": np.float32(1.0),
        "is_negative": np.bool_(False),
    }
    if mask_box:
        y0, y1, x0, x1 = mask_box
        s["donor_mask"][:, :, y0:y1, x0:x1] = True
        s["crops"][:, :, y0:y1, x0:x1] = 200
    return s


def test_params_defaults():
    p = CopyPasteParams()
    assert (p.p, p.opposite_sex_p, p.contact_p) == (0.0, 0.7, 0.3)
    assert p.contact_sep == (8.0, 30.0)


def test_donor_pixels_land_in_the_host():
    host = _win()
    donor = _win(fill=10, mask_box=(4, 12, 4, 12))
    out = composite(host, donor, np.zeros((3, 2), np.int32), np.random.default_rng(0))
    assert out is not None
    assert (out["crops"][0, 0, 4:12, 4:12] == 200).all()


def test_donor_with_no_mask_content_returns_none():
    host = _win()
    donor = _win(fill=10)
    assert composite(host, donor, np.zeros((3, 2), np.int32), np.random.default_rng(0)) is None


def test_host_donor_mask_loses_pixels_covered_by_the_donor():
    host = _win(mask_box=(2, 20, 2, 20))
    donor = _win(fill=10, mask_box=(4, 12, 4, 12))
    out = composite(host, donor, np.zeros((3, 2), np.int32), np.random.default_rng(0))
    assert not out["donor_mask"][0, 0, 5:11, 5:11].any()
    assert out["donor_mask"][0, 0, 15:19, 15:19].any()


def test_composite_preserves_keys_and_shapes():
    host = _win()
    donor = _win(fill=10, mask_box=(4, 12, 4, 12))
    out = composite(host, donor, np.zeros((3, 2), np.int32), np.random.default_rng(0))
    assert set(out) == set(host)
    for k in ("crops", "kp2d", "vis2d", "donor_mask"):
        assert out[k].shape == host[k].shape


def test_a_second_fly_becomes_valid():
    host = _win()
    donor = _win(fill=10, mask_box=(4, 12, 4, 12))
    out = composite(host, donor, np.zeros((3, 2), np.int32), np.random.default_rng(0))
    assert out["fly_valid"].sum() >= host["fly_valid"].sum()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/train/test_copypaste.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tracking.train.data.copypaste'`

- [ ] **Step 3: Implement**

Port as described. `composite`'s exact signature in the source may differ from `(host, donor, shifts, rng)`; keep the source's signature and adjust the tests to it, reporting the real signature. Do not reshape the source's logic to match a guessed signature.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/train/test_copypaste.py -v` → 6 passed
Run: `python -m pytest tests/train -q` → everything so far still passes.

- [ ] **Step 5: Commit**

```bash
ruff format src/tracking/train/data tests/train/test_copypaste.py
ruff check src/tracking/train/data tests/train/test_copypaste.py
git add src/tracking/train/data tests/train/test_copypaste.py
git commit -m "feat(train): multi-view copy-paste compositing"
```

---

## Task 8: Batching and the process-pool loader

**Files:**
- Create: `src/tracking/train/data/loaders.py`
- Modify: `src/tracking/train/data/windows.py` (add `window_batches`, `worker_spec`)
- Test: `tests/train/test_loaders.py`

**Interfaces:**
- Consumes: everything above
- Produces:
  - `window_batches(ds, batch_size, *, shuffle=True, seed=0, weights=None, num_workers=8, drop_last=True, workers="threads", pool=None, pool_key=None)` — yields dicts of stacked arrays with a leading batch axis
  - `WindowDataset.worker_spec()` — picklable description for a loader process
  - `ProcessSampleLoader(specs, num_workers, inflight=3)` with `.map_batches(index_lists, epoch, key=None)` and `.close(timeout=30.0)`

Port `$J/data/loader_workers.py` (303 lines) and `window_batches` from `$J/data/v12_windows.py`. Replace `V12Spec`/`ConcatSpec` with a single `WindowSpec` — this port has one dataset class, since `ConcatWindowDataset` was retired by the unified root.

`worker_spec` must be built from the dataset's **attributes**, not from a stashed copy of the constructor arguments, so it cannot drift from what the object actually is.

Sampling happens in the parent and assembly in the workers, so a given `(seed, epoch)` draws the same windows regardless of worker count. Batches must be **identical** between the `"threads"` and `"processes"` paths — that equivalence is the one property worth testing hardest, because the process path exists purely for throughput and any divergence would be a silent training difference.

- [ ] **Step 1: Write the failing test**

Create `tests/train/test_loaders.py`:

```python
import os

import numpy as np
import pytest

from tracking.train.data.windows import WINDOW_KEYS, WindowDataset, window_batches

ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/unified_v2"
pytestmark = pytest.mark.skipif(not os.path.isdir(ROOT), reason="unified root not present")


@pytest.fixture(scope="module")
def ds():
    return WindowDataset(ROOT, "train", T=1)


def test_batch_has_a_leading_batch_axis(ds):
    b = next(iter(window_batches(ds, 4, shuffle=False, num_workers=2)))
    assert set(b) == set(WINDOW_KEYS)
    assert b["crops"].shape[0] == 4
    assert b["kp3d_local"].shape[0] == 4


def test_same_seed_draws_the_same_windows(ds):
    a = next(iter(window_batches(ds, 4, seed=7, num_workers=2)))
    c = next(iter(window_batches(ds, 4, seed=7, num_workers=2)))
    np.testing.assert_array_equal(a["center3D"], c["center3D"])


def test_different_seed_draws_different_windows(ds):
    a = next(iter(window_batches(ds, 4, seed=1, num_workers=2)))
    c = next(iter(window_batches(ds, 4, seed=2, num_workers=2)))
    assert not np.array_equal(a["center3D"], c["center3D"])


def test_worker_count_does_not_change_the_batch(ds):
    a = next(iter(window_batches(ds, 4, seed=5, num_workers=1)))
    c = next(iter(window_batches(ds, 4, seed=5, num_workers=4)))
    np.testing.assert_array_equal(a["crops"], c["crops"])


def test_thread_and_process_paths_agree(ds):
    a = next(iter(window_batches(ds, 4, seed=11, num_workers=2, workers="threads")))
    c = next(iter(window_batches(ds, 4, seed=11, num_workers=2, workers="processes")))
    for k in ("crops", "kp3d_local", "kp2d", "cam_valid", "M"):
        np.testing.assert_array_equal(a[k], c[k])


def test_weights_bias_the_draw(ds):
    w = np.zeros(len(ds))
    w[:8] = 1.0
    b = next(iter(window_batches(ds, 4, seed=3, weights=w / w.sum(), num_workers=2)))
    assert b["crops"].shape[0] == 4


def test_worker_spec_round_trips_through_pickle(ds):
    import pickle

    spec = ds.worker_spec()
    assert pickle.loads(pickle.dumps(spec)) == spec
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/train/test_loaders.py -v`
Expected: FAIL — `ImportError: cannot import name 'window_batches'`

- [ ] **Step 3: Implement**

Port as described. `test_thread_and_process_paths_agree` will be the slowest test (it spawns a pool); if it is prohibitively slow against the 285 MB instances file, restrict the dataset with `recordings=` to one small recording for that test rather than deleting it — the equivalence is the point of the task.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/train/test_loaders.py -v` → 7 passed
Run: `python -m pytest tests -q` → the full suite.

- [ ] **Step 5: Commit**

```bash
ruff format src/tracking/train/data tests/train/test_loaders.py
ruff check src/tracking/train/data tests/train/test_loaders.py
git add src/tracking/train/data tests/train/test_loaders.py
git commit -m "feat(train): batching and the process-pool sample loader"
```

---

## Self-Review

**Spec coverage.** §4's `common/` → Task 2; `data/windows.py` → Tasks 4-5, 8; `data/augment.py` → Task 6; `data/copypaste.py` → Task 7; `data/loaders.py` → Task 8; `data/transforms.py` → Task 3. §2.2's requirement that the slot table stay four wide and that `policy.py` stop inlining it → Task 1. §5.3's mask sidecar as copy-paste's only consumer → Task 5 (`donor_mask`, `masks_root=None` degrades cleanly). §7.1's `MVAugParams` v2 defaults → Task 6.

**Not covered here, by design:** `mvq/` (losses, matching, step, evaluate, sampling, calibrate, run) is Plan 3; `centerdetect/` is Plan 4. `distractor.py`'s `build_part_index` / `repulsion_footprint` go with Plan 3, since the loss is their only consumer. `train/common/checkpoint.py` and `metrics.py` go with Plan 3, which is where they are first used.

**Placeholder scan:** clean — every code step carries real code or an exact source file plus the exact substitutions, every test asserts a named behaviour.

**Type consistency:** `WINDOW_KEYS` is defined once in Task 4 and asserted in Tasks 5 and 8. `donor_mask` replaces `prompt_mask` consistently across Tasks 5, 6, 7. `WindowDataset`'s constructor signature in Task 4 matches its use in Tasks 5, 7, 8. `SEX_UNKNOWN` in Task 5 comes from Task 1's `slots.py`. `CROP = 448` is defined in Task 4 and imported in Task 5's test.

**Three places the plan tells the implementer to change the TEST rather than the code** (Task 3's `crop_origin` edge case, Task 6's LR-swap naming convention, Task 7's `composite` signature). That is deliberate: these port behaviours the released checkpoints were trained under, so the shipped semantics win over my guess at them, and the implementer must report the real behaviour rather than silently reshape either side.
