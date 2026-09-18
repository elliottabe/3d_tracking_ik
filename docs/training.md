# Training data

The on-disk format for MVQ/CenterDetect training data, the optional mask
sidecar, the curation tools that build and check a root, and their exact CLI
invocations. Code: `src/tracking/curate/`. Config: `configs/curate.yaml`,
`configs/curate/default.yaml`.

## On-disk format

One format, both trainers. No masks in the core root.

```
<root>/
  annotations/
    instances_train.json    keypoint_names, skeleton, categories,
                            images, annotations, framesets
    instances_val.json
    keypoint_names.json     canonical keypoint order -- the authority
  images/<recording>/<camera>/Frame_<n>.jpg     real files, never symlinks
  calibrations/<group>/Cam*.yaml                DLT projectionMatrix per camera
  manifest.json             version, sources, calib_groups, recordings
```

`annotations[]`:

| field | meaning |
| --- | --- |
| `id` | annotation id |
| `image_id` | id into `images[]` |
| `bbox` | trains CenterDetect |
| `keypoints` | flat `3K` array; what MVQ triangulates |
| `num_keypoints` | count of labeled keypoints |
| `fly_id` | per-recording fly identity, `-1` for a negative |
| `sex` | fly sex |
| `subset` | annotation-source subset label |
| `src_ann_id` | optional, original annotation id before merge |

`framesets{}`, keyed `<rec>/Frame_<n>/fly<id>`, MVQ-only (CenterDetect ignores
it):

| field | meaning |
| --- | --- |
| `recording` | recording name |
| `fly_id` | fly identity, `-1` for a negative |
| `calib_group` | which `calibrations/<group>` this frameset's `frames`/`kp3d` agree with |
| `frames` | image ids, one per camera in calibration-glob order |
| `ann_ids` | parallel to `frames`; `null` where a camera did not resolve |
| `source_id` | key into `manifest.sources` (see below) |
| `source`, `weight`, `role`, `stratum`, `gates`, `partners` | carried through from the tier that contributed the frameset |
| `center3D` | **required only on negative framesets** (`fly_id < 0`); positives derive it from their 3D labels |

Rules the format enforces:

- Calibration projection **row 2 must be `[0, 0, 0, 1]`** -- the MVQ geometry
  is affine/telecentric throughout.
- `manifest.recordings[*].has_masks` is **stale** in the reference export and
  is not read by anything. The mask sidecar's own `index.json` is the
  authoritative record of what has masks.
- Calibration is a per-frameset property, not a per-recording one. A
  recording legitimately spans more than one `calib_group` when the rig was
  recalibrated mid-session; each frameset still fits its own group exactly.
  `manifest.recordings[rec]` carries `calib_group` when every tier that
  contributed framesets for `rec` agrees, otherwise `calib_groups` (a sorted
  list of the distinct groups) and no `calib_group` key.

## Provenance

`manifest.json` carries a `sources` table; every frameset carries a
`source_id` into it:

```json
"sources": {
  "human_v12_0902":  {"kind": "human", "origin": "...", "n_framesets": 2814, "weight": 1.0},
  "pseudo_p3b_0905": {"kind": "pseudo", "checkpoint": "...", "weight": 0.3, "n_framesets": 24805}
}
```

`kind` is one of `human`, `pseudo`, `negative`, `singlefly`. A source also
carries `origin`, `checkpoint`, `gates`, `review` and `n_framesets` where
applicable. Every `source_id` referenced by a frameset must resolve in this
table -- the validator checks it. Provenance travels with the data: the gate
values, generating checkpoint and review record that admitted a pseudo-label
are never regenerated, only carried forward.

## The mask sidecar

```
<root>_masks/
  <recording>/<camera>/Frame_<n>.npz    keyed by annotation id
  index.json                            n_files, tool, source, recordings
```

A **sibling** directory to `<root>`, not nested inside it -- pointed at by
`paths.masks_root`. Its **only** consumer is copy-paste donor cutting during
CenterDetect/MVQ training; it is **never a model input**. If it is absent,
copy-paste is simply off and everything else trains unchanged. Shipping or
omitting it never touches the core root's checksums.

`src/tracking/curate/masks/adapter.py` imports a SAM/SAM2/SAM3 output
directory into this layer. No SAM driver is ported -- users bring their own
segmentation.

A tier's own `masks/` sidecar is keyed by *that tier's* annotation ids, which
`merge` renumbers into the unified root's id space. `merge_tiers` (not a
verbatim copy) is what produces `<out_root>_masks/`: it remaps every mask's
`ann_ids` through the same per-tier id map used for `annotations[]`, dropping
rows whose id did not survive into the merged root and combining rows from
different tiers that land on the same `(recording, camera, frame)` path.

## This repository does not generate pseudo-labels

The pseudo tiers (`pseudo`, `negative`, `singlefly`) are **input data** with
recorded provenance, not something this code produces. The process that
generated them -- the P3b bootstrap checkpoint and the lift roots it ran
over -- no longer exists on disk and is not reproducible from this
repository. A user training on their own recordings trains on their own
annotations plus whatever tiers they separately supply.

## Tools

Entry point: `python -m tracking.curate` (also `python -m tracking.curate.run`).
Hydra config: `configs/curate.yaml` (`stages`, `dry_run`, `paths`) and
`configs/curate/default.yaml` (`curate.*`: `tiers`, `out_root`, `masks_root`,
`sam_dir`, `mask_tool`, `package_tier`, `package_zip`).

| Stage | Input | Output |
| --- | --- | --- |
| `merge` | `curate.tiers` | unified root at `curate.out_root` (dereferenced, renumbered, provenance) |
| `import_masks` | `curate.sam_dir` | `curate.masks_root` sidecar |
| `validate` | `curate.out_root` | pass/fail report naming every violation |
| `package` | `curate.out_root` | tiered zip + `CHECKSUMS` + `DATASET.md` |

Stages resolve to dependency order regardless of the order given on the
command line. There is no skip-if-exists check: every requested stage always
re-runs in full.

```bash
# see what would run without doing anything
python -m tracking.curate paths=hyak dry_run=true

# merge, validate, package in one call (the curate.yaml default)
python -m tracking.curate paths=hyak stages=[merge,validate,package]

# individual stages
python -m tracking.curate paths=hyak stages=[merge]
python -m tracking.curate paths=hyak stages=[import_masks] curate.sam_dir=/path/to/sam3_out
python -m tracking.curate paths=hyak stages=[validate]
python -m tracking.curate paths=hyak stages=[package] curate.package_tier=human

# validate or check an already-built root directly, no Hydra config needed
python scripts/check_training_root.py <root> --masks <root>_masks
```

### Validator finding codes

`validate_root()` (`src/tracking/curate/validate.py`) returns a flat list of
`Finding(level, code, message)`. Errors block `package`; warnings and info do
not.

| level | code | meaning |
| --- | --- | --- |
| error | `keypoint_names_mismatch` | a split's `keypoint_names` disagrees with `annotations/keypoint_names.json` |
| error | `keypoint_names_unreadable` | keypoint names file missing or unparsable |
| error | `calibration_unreadable` | a calibration group failed to load via `CameraRig` |
| error | `calibration_not_affine` | projection row 2 is not `[0, 0, 0, 1]` |
| error | `camera_names_differ` | calibration groups disagree on camera names |
| error | `split_missing_keys` | a split's instances file lacks `images`, `annotations` or `framesets` (checked by key presence, not emptiness -- an empty-but-present split is legitimate) |
| error | `keypoints_length` | an annotation's `keypoints` array is not `3K` long |
| error | `image_is_symlink` | an image file is a symlink, not a real file |
| error | `image_missing` | an image referenced by a split is not on disk |
| error | `ann_ids_length` | a frameset's `ann_ids` is not parallel in length to `frames` |
| error | `frameset_image_missing` | a frameset references an unknown image id |
| error | `frameset_ann_missing` | a frameset references an unknown annotation id |
| error | `too_few_cameras` | a frameset resolves fewer than `MIN_CAMS` (3) cameras |
| error | `negative_without_center3d` | a negative frameset (`fly_id < 0`) carries no `center3D` |
| error | `unknown_source_id` | a frameset's `source_id` is not in `manifest.sources` |
| error | `frameset_calib_unresolvable` | a frameset has no `calib_group`, and its recording has neither `calib_group` nor `calib_groups` |
| error | `manifest_unreadable` | `manifest.json` missing or unparsable |
| warning | `source_without_checkpoint` | a `pseudo`/`negative`/`singlefly` source records no `checkpoint` |
| warning | `masks_root_missing` | the given `masks_root` does not exist; copy-paste will be off |
| info | `zero_annotation_images` | count of images with no annotation |
| info | `two_fly_fraction` | fraction of images with >= 2 animals -- reported, not failed, so a low fraction (a known bottleneck) is visible before training rather than discovered after |
| info | `recording_multi_calib` | a recording carries more than one `calib_group` across tiers, e.g. a mid-session recalibration |

### Packaging tiers

`package_root(root, out_zip, tier=...)` validates first and **refuses to
package an invalid root** (raises `ValueError` listing every error finding).
`tier="masks"` raises `FileNotFoundError` if the sidecar directory does not
exist, rather than writing an empty zip.

| tier | contents | summary key |
| --- | --- | --- |
| `human` | only sources with `kind == "human"` | `n_framesets`, `n_images`, `n_sources` |
| `all` | every source | `n_framesets`, `n_images`, `n_sources` |
| `masks` | the `<root>_masks/` sidecar | `n_mask_files` |

Every zip carries `CHECKSUMS` (sha256 per member) and `DATASET.md` (the
`sources` table, contents counts, format link).

## Tier sizes

Dereferenced, maskless (spec §5.4):

| tier | framesets | images | size |
| --- | --- | --- | --- |
| human | 2,814 | 19,131 | 4.2 G |
| pseudo p3b | 24,805 | 127,001 | 26 G |
| negatives | 7,302 | 51,114 | 11 G |
| singlefly | 1,909 | 13,363 | 2.7 G |
| **unified** | **36,830** | **210,609** | **~44 G** |

Mask sidecars add ~18 G (2.2 G human + 16 G p3b).

## MVQ training

Entry point: `python -m tracking.train.mvq` (also `python -m tracking.train.mvq.run`).
Hydra config: `configs/train.yaml` (`run.name`, `paths`) plus the `model`,
`train` and `aug` groups. Code: `src/tracking/train/`.

```bash
# the v2 preset on the unified root, run dir <paths.runs_root>/<run.name>
PYTHONPATH=$PWD/src python -m tracking.train.mvq train=mvq_v2 paths=hyak run.name=mvq_t2_v2

# a different root, and a 2-step plumbing check that writes a real run dir
PYTHONPATH=$PWD/src python -m tracking.train.mvq train=mvq_v2 run.name=smoke \
  paths.train_data_root=/path/to/unified_v2 train.smoke=true train.total_steps=2 \
  train.batch_size=2 train.window_lengths=[1] train.loader_workers=threads
```

`PYTHONPATH=$PWD/src` is only needed when the environment's editable install
points at a different checkout; with this checkout installed it can be dropped.

### The v2 preset

`configs/train/mvq_v2.yaml` (on top of `configs/train/mvq.yaml`) is what the
shipped v2 checkpoint trained with:

| key | value | meaning |
| --- | --- | --- |
| `total_steps` / `warmup_steps` | 40000 / 1000 | AdamW, warmup + cosine decay |
| `window_lengths` | `[1, 2]` | one stream per T, alternating by step |
| `pair_deltas` | `[1, 4, 16]` | frame spacings a T=2 window may span |
| `jitter_units` | 10.0 | crop-centre jitter |
| `female_host_target` | 0.5 | solved per source; `female_host_weight` unused while set |
| `copy_paste_p` | 0.8 | second fly pasted in, needs `paths.masks_root` |
| `copy_paste_contact_p` / `_sep` | 0.7 / `[4.0, 25.0]` | how often, and how close |
| `wing_kp_mult` | 2.0 | per-keypoint loss multiplier on wing landmarks |
| `loader_workers` / `num_workers` | `processes` / 24 | one spawn pool shared by both T streams |

Sampling is source-aware: negatives take `negatives_frac` of the mass, the
rest splits by window count, and each source's own behaviour/host-sex balance
is restored inside it. A window's loss weight is its frameset's `weight` from
`manifest.sources`; `train.pseudo_weight` is checked against it and **warns**
on disagreement, training at the manifest's value.

### Run layout

```
<paths.runs_root>/<run.name>/
  mvq_run.json          resolved model/train/loss/aug config, keypoint_names,
                        train_data (per-source mass + realised mix), val, calibration
  ckpt/<step>/          model, opt, ema, ema_meta -- resumable; a rerun of the
                        same run.name picks up from the latest step
  final/                debiased EMA weights + a copy of mvq_run.json
```

`tracking.detector.mvq.checkpoint.load_mvq_model` reads both: the run dir with
`step=<n>`/`"latest"`, or `<run_dir>/final` with `step=None`. Evaluation runs
every `train.eval_every` steps and once at the end; the final pass also fits
the existence/visibility temperatures written to `mvq_run.json["calibration"]`,
which `MVQRunner` divides its logits by (`1.0` is the identity).
