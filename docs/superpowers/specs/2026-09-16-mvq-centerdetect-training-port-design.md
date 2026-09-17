# MVQ + CenterDetect training port — design

**Date:** 2026-09-16
**Branch:** `feat/mvq-centerdetect-training`
**Status:** design approved, spec under review

## 1. Goal

Bring MVQ and CenterDetect **training** into this repository, which today is
inference-only, and retire `3d_tracking_dataset` entirely. Two capabilities
must exist when this lands:

1. **Retrain `mvq_v2_maskless` from the data it was trained on.** Given the
   curated training tiers, one command reruns the v2 recipe — same data, same
   config, same seed — and produces a checkpoint that reaches v2's acceptance
   numbers (val MPJPE ≈ 0.0699 mm, cross-fly ≈ 0.020, centre-shift gate PASS).
   Not bit-identical: a different GPU count, JAX version or cuDNN build
   changes reduction order. Equivalence is measured by the acceptance gates,
   not by comparing weights.
2. **Retrain or fine-tune on new data in that same format.** A user with their
   own calibrated multi-camera recordings and keypoint annotations can build a
   training root, validate it, and train or fine-tune both models.

**Regenerating pseudo-labels is explicitly out of scope.** The existing
pseudo-label tiers are treated as given data, the same way the human labels
are. Nothing in this port produces new ones.

Everything else in `3d_tracking_dataset` — ViTPose/EfficientTrack/V2VNet
keypoint training, the densepose/CSE stack, the SAM3 driver, the STAC/IK
lineage this repo already replaced — is **not** migrated. It is frozen where
it stands and retired with the repo.

## 2. Findings that shaped the design

Recorded because several of them contradict the obvious assumption.

### 2.1 The shipped v2 checkpoint is already maskless

`mvq_t2_v2_20260906d/mvq_run.json` has `prompt_p_start: 0.0` and
`prompt_p_end: 0.0`. The prompt path was off for the entire run. `MVQRunner`
therefore runs mask-free at inference (`runner.py:298` — *"No prompt path:
this repo carries no masks, so every window is unprompted"*), and the two are
consistent.

**But v2 still needed SAM3 masks to build its training data**: `copy_paste_p:
0.8` with `copy_paste_contact_p: 0.7`, and copy-paste cuts donors along
`prompt_mask`.

**A mask used to cut a copy-paste donor never enters the model.** "Maskless
model" and "no masks in the data pipeline" are separable concerns. This design
keeps SAM3-derived masks as a separable *sidecar* layer and keeps the model
maskless.

### 2.2 Prompt-off is numerically exact, and the prompt params must survive

`decoder.py:131` computes `add = self.prompt_proj(prompt_tok) * on[:, None]`.
With `prompt_on` all-False the prompt contribution is exactly zero regardless
of the token, so training passing `prompt_mask=None` is numerically identical
to `MVQRunner`'s current all-zeros mask, and skips the `_prompt` computation.

`prompt_proj` / `prompt_ln` are retained as shape-preserved, untrained
parameters. The slot table stays **four** wide (`SLOT_PROMPTED` = 0 never
receives a positive target) because the existence and sex head shapes depend
on `n_instances == 4`, and `policy.typed_candidates` keys its `range(1, n)`
behaviour on `n_instances == N_SLOTS`. Narrowing either would break warm-start
from the released checkpoint and break shipped inference.

### 2.3 What survives on disk

| Asset | Status |
| --- | --- |
| Curated human root `red_data_3d_v12_export0902` + masks (2.2 G) | present |
| Pseudo tiers p3b / negatives / singlefly (40 G) + p3b masks (16 G) | present |
| `mvq_t2_v2_20260906d`, `cd_focal_bg30` | present |
| Raw recordings, human annotation trees | present |
| `mvq_t1_b16_p3b_contact_20260905` — generated the pseudo-labels | gone |
| `processed/courtship/**/pose_mvq_p3b` — the lifts they came from | gone |
| `red_data_3d_v5_valfix` — CenterDetect's training root | gone |

The pseudo-label *artifacts* survive; the *process* that produced them does
not. Since regenerating them is out of scope, this is not a blocker — the
tiers are inputs. It is recorded so nobody later mistakes the design for one
that could remake them.

### 2.4 The human root's images are symlinks

`red_data_3d_v12_export0902/images/**` are symlinks into
`red_data/general_model/...`. The root is **not self-contained** and cannot be
archived as-is. Dereferenced it is 4.1 GB, not the 22 MB `du` reports. The
merge stage dereferences.

### 2.5 The target repo already holds most of the model code

| Already present | Replaces (source lines) |
| --- | --- |
| `detector/mvq/*` | `models/mvq/*` (~900) |
| `detector/backbones/dinov3.py`, `vit.py` | `models/dinov3.py`, `vit.py`, `backbone.py` (419) |
| `detector/centerdetect/model.py` | `efficientnet` + `efficienttrack` + `effnet_b3_std` + `dilation` (1,340) |
| `detector/centerdetect/decode.py` | `eval/centerdetect_decode.py` (189) |
| `geometry/rig.py::CameraRig` | `geometry/reprojection_tool.py` (269) |
| `io/artifacts.py` + `io/names.Order` | ad-hoc keypoint/camera axis handling |
| `detector/mvq/checkpoint.py` | shared half of `train/checkpoint.py` (~110) |

### 2.6 `build_v5.py` is not a general exporter

It discovers recordings from two specific legacy annotation trees
(`general_model/`, `red_data_unified_V3/`) and merges them. It is a one-off
migration tool and is not ported. Only `iter_resolved_slots` (~20 lines)
survives, inlined.

### 2.7 CenterDetect's two-fly bottleneck

The unified root has 308 two-fly images out of 18,319 human-tier images —
**1.7%**, against the 6.7% of the lost v5 root. Oversampling alone is
documented as insufficient for this; copy-paste existed because of it. Since
the mask sidecar is retained, CenterDetect copy-paste is retained.

### 2.8 Post-training temperature calibration is part of the chain

`MVQRunner` reads `meta["calibration"]["exist_temperature"]` and
`["vis_temperature"]` (`runner.py:250-252`, applied at `327-328`). The shipped
v2 `mvq_run.json` has no `calibration` block, so it runs at 1.0 — but a
retrain should be able to produce one.

## 3. Scope decisions

| Question | Decision |
| --- | --- |
| Primary purpose | Retrain v2 from its own data; retrain/fine-tune on new data in that format |
| Models | MVQ **and** CenterDetect |
| Pseudo-label generation | **Not ported.** Existing tiers are input data |
| Training data released | Not by default; format documented; tiered packaging tool provided |
| Recipe fidelity | Keep the recipe intact, cleaned — not simplified |
| Format guidance | Written spec **plus** a validator script |
| Package layout | `src/tracking/train/` and `src/tracking/curate/`; deps in core |
| PyTorch warm start (CD) | **Dropped.** JAX warm start or from scratch |
| Multi-GPU | Full: sharding + process loader + SLURM script |
| Config | Separate Hydra roots; `configs/pipeline.yaml` untouched |
| Geometry | Reuse `CameraRig`; `ReprojectionTool` kept under `tests/` as oracle only |
| Verification | Checkpoint round-trip, geometry parity, loss fixture |
| CD data format | Unified on the v12-shaped root |
| README | Reference presets **and** a fine-tune-from-released-checkpoint quickstart |
| `jarvis_jax` | Fully retired |
| Model input | Maskless |
| Masks | Separable sidecar; format + adapter, **no SAM3 driver ported** |
| Image storage | Full frames, dereferenced |
| Packaging | Tiered: human-only zip and full zip |

## 4. Architecture

```
src/tracking/curate/            # data curation — its own Hydra root and stages
  run.py  stages.py             # python -m tracking.curate, resume-by-artifact
                                 # (not implemented in Plan 1; stages always re-run)
  merge.py                      # tiers -> one unified root (dereferences symlinks)
  validate.py                   # the checker behind scripts/check_training_root.py
  package.py                    # tiered zip + CHECKSUMS + DATASET.md
  masks/
    store.py                    # read/write/validate the sidecar mask layer
    adapter.py                  # import a SAM/SAM2/SAM3 output dir into that layer

src/tracking/train/
  common/       optimizer.py  ema.py  checkpoint.py  sharding.py
                prefetch.py   metrics.py
  data/         windows.py  augment.py  copypaste.py  loaders.py
                transforms.py
  mvq/          config.py  losses.py  matching.py  step.py
                evaluate.py  sampling.py  calibrate.py  run.py
  centerdetect/ config.py  losses.py  dataset.py  copypaste.py
                evaluate.py  run.py
```

### 4.1 Ported substantially intact

`train_mvq.py` (1,179) splits along its seams into
`mvq/{config,step,evaluate,sampling,run}.py`. `losses_mvq.py` (252) and
`matching.py` (54) port verbatim, the latter always called with
`prompt_on=False`. `v12_windows.py` (808) becomes `data/windows.py`, shedding
`_affine_np` and camera-row bookkeeping to `CameraRig` and keypoint-axis
assertions to `Order`. `mv_augment.py`, `mv_copy_paste.py`,
`loader_workers.py`, `prefetch.py`, `sharding.py` port with imports rewritten.
`train_centerdetect.py` (623) becomes `centerdetect/{config,evaluate,run}.py`.

### 4.2 Dropped

Rows marked † were never inside the measured MVQ/CenterDetect import closure;
they are listed so the decision is on the record, and they do not enter the
arithmetic in §4.3.

| Source | Lines | Reason |
| --- | --- | --- |
| `data/pseudo_export.py` † | 428 | Pseudo-label generation out of scope |
| `data/pseudo_gates.py` † | 233 | Pseudo-label generation out of scope |
| `data/v3.py`, `data/v5_2d.py` | 603 | CD unifies on the v12 root |
| `data/build_v5.py` | 542 | One-off migration tool (§2.6) |
| `data/v5_3d.py` | 295 | Only `_load_mask`, `_resolve_sex`, `_frameset_own_sex` survive (~80) |
| `data/augment.py` | 523 | Only `build_lr_swap`, `assert_lr_swap_covers` reachable (~80) |
| `data/copy_paste.py` | 561 | Only `CopyPasteCenterDetectDataset` (~250) |
| `data/distractor.py` | 240 | Only `build_part_index`, `repulsion_footprint` (~90); mask wrapper dropped |
| `data/concat_windows.py` | 347 | One unified root (§5.2) |
| `convert/load_efficienttrack.py` | 106 | Torch warm start dropped |
| `predict/sam3_driver.py` † | 1,637 | Format + adapter instead (§5.3) |
| `geometry/reprojection_tool.py` | 269 | `CameraRig`; retained under `tests/` as oracle |

`detector/mvq/policy.py` currently inlines `SLOT_PROMPTED`/`N_SLOTS` to avoid
importing `train.matching`. Once `train/` exists that inlining becomes a second
source of truth, so `policy.py` imports from `train/mvq/matching.py`. This is
the **only** shipped inference file this work modifies.

### 4.3 Size

The measured import closure of `train_mvq.py` + `scripts/train_mvq.py` is
7,313 lines over 35 files; CenterDetect adds 4,192 lines that the MVQ closure
does not already cover. **In-scope source: 11,505 lines.**

| | lines |
| --- | --- |
| In-scope source | 11,505 |
| Already present in this repo (§2.5) | −3,200 |
| Dead or inapplicable within the closure (§4.2, unmarked rows, net of partial survivors) | −2,700 |
| **Ported** | **≈ 5,600** |
| New (`curate/merge.py` ~350, `validate.py` ~300, `package.py` ~200, `masks/store.py` ~150, `masks/adapter.py` ~120, `mvq/calibrate.py` ~150) | **≈ 1,250** |

Plus ~200 lines of YAML and ~400 of tests. Total new to the repository:
**≈ 7,000 lines.** Large enough that the implementation plan should phase it —
curate, then MVQ, then CenterDetect — with the checkpoint round-trip test
landing in the first training phase.

## 5. The unified dataset

### 5.1 On-disk format

One format, both trainers. No masks in the core root.

```
<root>/
  annotations/
    instances_train.json    keypoint_names, skeleton, categories,
                            images, annotations, framesets
    instances_val.json
    keypoint_names.json     canonical keypoint order — the authority
    split.json               (not implemented in Plan 1; nothing writes, checks, or ships it)
  images/<recording>/<camera>/Frame_<n>.jpg     real files, never symlinks
  calibrations/<group>/Cam*.yaml                DLT projectionMatrix per camera
  manifest.json             version, sources, calib_groups, recordings
```

- `annotations[]`: `id`, `image_id`, `bbox`, `keypoints` (flat `3K`),
  `num_keypoints`, `fly_id`, `sex`, `subset`, optional `src_ann_id`. `bbox`
  trains CenterDetect; `keypoints` is what MVQ triangulates.
- `framesets{}`: keyed `<rec>/Frame_<n>/fly<id>`, holding `recording`,
  `fly_id`, `frames` (image ids, one per camera in calibration-glob order),
  `ann_ids` (parallel; `null` permitted where a camera did not resolve),
  plus provenance (§5.2). MVQ-only; CenterDetect ignores it.
- `center3D` is required **only** on negative framesets (`fly_id: -1`).
  Positives derive it from their 3D labels.
- Calibration projection row 3 must be `[0,0,0,1]` — the MVQ geometry is
  affine/telecentric throughout.
- `manifest.recordings[*].has_masks` is **stale** in the reference export
  (false on 14 of 16 recordings that do have masks). It is an unused field;
  the mask sidecar's own index is authoritative.

### 5.2 Provenance

Root `manifest.json` carries a `sources` table; each frameset carries a
`source_id` into it:

```json
"sources": {
  "human_v12_0902":  {"kind": "human",     "origin": "red_data_3d_v12_export0902",
                      "annotation_subsets": ["..."], "n_framesets": 2814, "weight": 1.0},
  "pseudo_p3b_0905": {"kind": "pseudo",    "checkpoint": "...", "gate_string": "...",
                      "gates": {...}, "review": {...}, "weight": 0.3, "n_framesets": 24805},
  "negatives_0905":  {"kind": "negative",  "checkpoint": "...", "gates": {...}, "role": "negative"},
  "singlefly_0905":  {"kind": "singlefly", "checkpoint": "...", "weight": 0.3}
}
```

Framesets keep their existing `source`, `weight`, `role`, `stratum`, `gates`,
`partners` fields and gain `source_id`. Provenance is carried, never
regenerated: the gate values, generating checkpoint and human review record
that admitted each pseudo-label travel with the frameset into the unified root
and into the shared archive, so a label's origin is always answerable even
though this repo cannot produce a new one.

**No loader change is needed**: `V12WindowDataset._fs_field` already resolves
frameset → per-recording manifest → root default, which is exactly the chain a
`source_id` lookup wants.

`ConcatWindowDataset` is dropped. `_mix_weights` becomes a group-by over
`source_id`: `_balanced_weights` runs per group, negatives take exactly
`negatives_frac` of sampled mass, the remainder splits by frameset count —
the same arithmetic over one dataset. `_max_calib_diff` moves into
`curate/merge.py`, where cross-source calibration agreement is a build-time
property checked once rather than on every training run.

### 5.3 The mask sidecar

```
<root>_masks/
  <recording>/<camera>/Frame_<n>.npz    keyed by annotation id
  index.json                            coverage, source, generating tool
```

Pointed at by a `masks_root` config key and a manifest pointer. It ships or is
omitted without touching the core root's checksums. Its **only** consumer is
copy-paste donor cutting, and it is never a model input. Absent it, copy-paste
is off and everything else trains unchanged.

`curate/masks/adapter.py` converts a SAM/SAM2/SAM3 output directory into this
layer. The SAM3 driver itself is **not** ported; users bring their own
segmentation, and the existing 2.2 GB and 16 GB mask trees are imported.

### 5.4 Sizes and packaging

Dereferenced, maskless:

| tier | framesets | images | size |
| --- | --- | --- | --- |
| human | 2,814 | 19,131 | 4.2 G |
| pseudo p3b | 24,805 | 127,001 | 26 G |
| negatives | 7,302 | 51,114 | 11 G |
| singlefly | 1,909 | 13,363 | 2.7 G |
| **unified** | **36,830** | **210,609** | **~44 G** |

Mask sidecars add ~18 G (2.2 human + 16 p3b).

Images are stored as full frames (1936×448). MVQ reads only a 448 crop, but
CenterDetect localizes in the whole image, so full frames keep one layout for
both trainers and keep the root usable for future re-cropping.

`curate/package.py` validates first and refuses to package an invalid root,
then writes a zip plus `CHECKSUMS` and a `DATASET.md` carrying the `sources`
table, counts, licence and citation. Tiers: `--tier human` (~4.2 G),
`--tier all` (~44 G), `--tier masks` (~18 G).

## 6. Curation stages

`python -m tracking.curate`, resume-by-artifact in the idiom of
`tracking.pipeline.stages` (not implemented in Plan 1; stages always re-run).

| Stage | Input | Output |
| --- | --- | --- |
| `merge` | source tiers | unified root (dereferenced, renumbered, provenance) |
| `import_masks` | SAM output dir | `<root>_masks/` sidecar |
| `validate` | unified root | pass/fail report naming every violation |
| `package` | unified root | tiered zips + `CHECKSUMS` + `DATASET.md` |

The retrain chain is `merge → validate → train`.

### 6.1 Validator checks

`instances_*.keypoint_names == keypoint_names.json`; every `calib_group`
resolves to a directory with one `Cam*.yaml` per camera and camera names
consistent across groups; affine projection row; every `frameset.frames` image
id exists with its file on disk and **is not a symlink**; `ann_ids` parallel to
`frames` with `null` permitted; at least `MIN_CAMS` resolved slots per
frameset; keypoint arrays of length `3K`; negatives carry `center3D`; every
`source_id` resolves in the `sources` table. It **reports rather than fails**
on the zero-annotation image count and the two-fly image fraction — both are
known traps, and a user with 1.7% two-fly should learn that before training.

## 7. Training

### 7.1 MVQ

`config.py` holds `MVQTrainConfig`. Removed: `prompt_p_start`,
`prompt_p_end`, `prompt_anneal_steps` (maskless), and
`pseudo_root`/`singlefly_root`/`negatives_root` (one unified root).
`negatives_frac` keys on `sources[*].kind == "negative"`. `pseudo_weight`
warns when the config disagrees with the manifest's per-source weight rather
than silently training at the tier's own value.

`losses.py` carries `mvq_loss`, `LossWeights` (including the T=2 `persist` /
`persist_margin_units` terms) and `wing_kp_weight`. `step.py` holds the
optimizer's two parameter groups (`backbone_lr_mult`), gradient clipping and
EMA. `evaluate.py` keeps cohort MPJPE over `female`, `two_fly`,
`contact_pair`, `single_fly` and the centre-shift sweep; it shares
`detector/mvq/policy.py` with inference so the figure and the metric cannot
diverge. `sampling.py` holds `_balanced_weights`, the `_mix_weights` group-by
rewrite, and `_MixCounter`'s realised-ratio report over the first 200 batches.
`calibrate.py` fits the post-training existence/visibility temperatures and
writes the `calibration` block `MVQRunner` reads. `run.py` is the driver.

The v2 reference preset, from `mvq_t2_v2_20260906d/mvq_run.json`:
`lr 3e-4`, `warmup 1000`, `total_steps 40000`, `batch_size 32`,
`backbone_lr_mult 0.1`, `ema 0.999`, `window_lengths [1,2]`,
`pair_deltas [1,4,16]`, `jitter_units 10.0`, `female_host_weight 4.27`,
`female_host_target 0.5`, `negatives_frac 0.05`, `copy_paste_p 0.8`,
`copy_paste_contact_p 0.7`, `copy_paste_contact_sep [4.0, 25.0]`,
`wing_kp_mult 2.0`, `loader_workers processes`, `num_workers 24`,
`other_fly_repulsion 20.0`, `persist 0.5`, `persist_margin_units 2.0`,
`cam_drop_p 0.1`.

### 7.2 CenterDetect

Per-image dataset over the unified root's `bbox` annotations with
`balanced_weights` on `num_flies`; `centerdetect_instance_mse` at both output
scales; copy-paste from the mask sidecar; per-epoch evaluation of
`two_peak_rate`, `dists_median_px`, `dists_p90_px` and the single-fly
false-positive rate, all into `metrics.json`. Epoch-based loop with one
checkpoint directory per epoch — what `configs/centerdetect/default.yaml`
documents and what `restore_centerdetect` expects.

### 7.3 Checkpoint layout

MVQ writes `<run_dir>/final/` (model + ema items) and `<run_dir>/ckpt/<step>/`.
CenterDetect writes `<run_dir>/ckpt/epoch_NNN/` plus `metrics.json`. Both are
where the shipped inference side already looks.

## 8. Configuration and CLI

```
configs/train.yaml       defaults: [_self_, paths, model, train, aug]
configs/curate.yaml      defaults: [_self_, paths, curate]
configs/train/{mvq,mvq_v2,centerdetect}.yaml
configs/model/{mvq,centerdetect}.yaml
configs/aug/default.yaml
configs/curate/default.yaml
```

`configs/paths/*.yaml` gains `train_data_root`, `masks_root` and `runs_root`.
`configs/pipeline.yaml` is untouched — its `stages`/`bout_ids`/`run.root` mean
nothing to a trainer.

```bash
python -m tracking.curate stages=[merge,validate,package] paths=mymachine
python scripts/check_training_root.py <root>
python -m tracking.train.mvq          train=mvq_v2 paths=mymachine run.name=...
python -m tracking.train.centerdetect train=centerdetect paths=mymachine run.name=...
scripts/slurm/train_mvq.sh --nodes 1 --gpus 8 ...
```

## 9. Verification

1. **Checkpoint round-trip.** Train two steps on a batch dict built **in
   memory** — no dataset root, so it runs anywhere — save to `final/`, reload
   through the shipped `detector.mvq.checkpoint.load_mvq_model` and
   `detector.centerdetect.model.restore_centerdetect`, and assert parameter
   equality plus a working forward. A trainer whose output the pipeline cannot
   read is worthless; this is the contract that matters.
2. **Geometry parity.** `CameraRig` against `ReprojectionTool` on the real
   calibration directories, with `ReprojectionTool` committed under `tests/`
   purely as the oracle so it never ships in the package. Plus keypoint- and
   camera-order assertions against `keypoint_names.json`.
3. **Loss fixture.** Frozen `mvq_loss` outputs on a small saved input, so a
   refactor that moves the loss fails a number rather than producing a worse
   model discovered a day into a GPU run.

## 10. Documentation

`docs/training.md` covers the on-disk format, the mask sidecar, merge,
validate, package, train, and the fine-tune-from-released-checkpoint
quickstart. The README gains a **Training** section and that quickstart.
`configs/train/mvq_v2.yaml` and `configs/train/centerdetect.yaml` ship as
reference presets.

Stated plainly in both, because each is a real limitation rather than a
caveat:

- **This repository does not generate pseudo-labels.** The pseudo tiers are
  input data with recorded provenance. A user training on their own recordings
  trains on their own annotations plus whatever tiers they supply.
- `cd_focal_bg30`'s own training root no longer exists, so its exact recipe is
  not reproducible at any fidelity. Warm-starting from the released
  CenterDetect checkpoint is the recommended path.
- Without the mask sidecar, copy-paste is off. Since v2 trained at
  `copy_paste_p: 0.8`, a maskless-curated retrain is a materially different
  recipe and should not be described as reproducing v2.
- The prompt pathway is never trained; `prompt_proj`/`prompt_ln` are carried
  as shape-preserved dead weight so warm-start from the released v2 weights
  keeps working.

## 11. Out of scope

- **Generating pseudo-labels**: gating, export, empty-window negatives,
  single-fly passes, review galleries.
- Porting the SAM3 driver.
- Reproducing the P3b bootstrap or any pre-v2 checkpoint.
- ViTPose/EfficientTrack/V2VNet/densepose training.
- Re-cropping the dataset to 448/512 tiles (revisit if archive size becomes
  the binding constraint).
- Background-subtraction silhouettes as a mask substitute (considered;
  dropped in favour of the mask sidecar).
