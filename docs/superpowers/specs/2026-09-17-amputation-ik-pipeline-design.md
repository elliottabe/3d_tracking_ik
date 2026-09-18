# Amputation cohort: 3D-CSV ingest through collect — design

**Date:** 2026-09-17
**Branch:** `feat/amputation-ik` (new, off `main`)
**Status:** design approved, spec under review

## 1. Goal

Run the existing IK pipeline end to end over the Johnson lab amputation
cohort at `/gscratch/portia/eabe/data/Johnson_lab/processed/amputation/`:
103 recordings of single, left-front-leg-amputated flies, for which the
3D keypoints already exist as one whole-recording CSV per recording.

One command per recording must carry it from its bout table to
`ik_output_combined_*.h5`, and the whole cohort must submit as a single
SLURM array job.

Three things make this cohort different from every recording the pipeline
has run so far:

1. **The 3D keypoints are given, not inferred.** There is no video and no
   calibration, so `coarse`, `gates` and `fine` cannot run and there is no
   camera rig at all.
2. **The fly is missing a leg.** `data3D.csv` carries 44 keypoints, not the
   anatomy's 50 — the six distal `T1L` markers do not exist.
3. **`data3D.csv` is the entire recording**, not a bout. Bouts are carved
   out of it by a separate summary table.

`kpvideo` and `sidebyside` are **out of scope** — both render against video
this cohort does not have.

## 2. The data, as verified on disk 2026-09-17

Every number below was measured, not assumed.

### 2.1 Layout

103 directories named `2026_MM_DD_HH_MM_SS`, each holding:

| File | What it is |
|---|---|
| `data3D.csv` | whole-recording 3D keypoints, ~742 MB, 66 GB across the cohort |
| `running_bouts_summary.csv` | the bout table |
| `info.yaml` | `number_frames`, `coord_scale`, source provenance |
| `manifest_row.tsv` | cluster-prediction provenance (absent for one recording) |

plus a cohort-level `index.csv` (`flyID, sex, recording, amp`).

### 2.2 `data3D.csv`

Two header rows: keypoint name, then `x,y,z,confidence`. 176 columns = 44
keypoints × 4. Row count matches `info.yaml: number_frames`.

**Units are already the pipeline's world units.** `Antenna_Base→Abd_tip`
measures 27.4 units, i.e. 2.74 mm at `scale.py`'s `WORLD_UNITS_TO_MM = 0.1`,
inside its `[2.0, 3.0]` plausible band. `info.yaml: coord_scale: 10.0` is the
mm→world factor already applied. **No unit conversion is required, and
applying one would be caught only by `assert_plausible_body_scale`.**

Data quality inside bout windows, sampled: **100 % finite xyz, median
confidence 0.91.**

A chunked pass (`pandas.read_csv(..., header=[0,1], chunksize=200_000)`)
over one 742 MB file takes **~10 s**.

### 2.3 The 44 keypoints

The anatomy's 50 `KP_NAMES` minus exactly `T1L_Tro`, `T1L_FeTi`,
`T1L_TiTa`, `T1L_TaT1`, `T1L_TaT3`, `T1L_TaTip`. `T1L_ThxCx` — the coxa
attachment, proximal to the amputation — **is present**. All 103 recordings
carry the identical 44-name set, and `index.csv` reports `amp: T1L` for
every row.

### 2.4 `running_bouts_summary.csv`

Columns `bout,start_frame,end_frame,n_frames,duration_s,...`. Validated
across all 103 recordings:

- **1759 bouts total**, 1–51 per recording, every recording has at least one.
- **All 1759 are `status: accepted`** — no filtering needed.
- **Zero `n_frames` mismatches** against `end_frame - start_frame + 1`, so
  `read_bout_summary`'s consistency check cannot raise.
- **Zero bouts** with `end_frame >= number_frames`.
- **892,657 bout frames total**; median 8,033 per recording, max 33,328.

`io.bouts.read_bout_summary` parses this schema **unmodified**. There is no
`fly_id` column, so its `session_tag` filter is inert.

### 2.5 `index.csv` disagrees with the disk in both directions

It lists **184** recordings; only **93** are on disk. **10** on-disk
recordings — the `2026_07_13` and `2026_07_15` tail — are absent from it.

Consequence: sex is known for 93 of 103 and `unknown` for 10. Not blocking;
`run.py:291` already falls back to `"unknown"` when `sex.json` is missing.

### 2.6 Session tag is derivable

`2026_07_06` → `session0706` matches `manifest_row.tsv` column 2 exactly
across all six sessions. **Nothing needs to read that TSV**, which is
fortunate — `2026_07_09_17_08_07` does not have one.

### 2.7 fps

`187 frames / 0.23375 s = 800.0` exactly, consistent with the rest of the
lab's recordings. Recorded here because `recording.fps` is the sole source
of `source_hz` and scales every `qvel` in the final dataset.

## 3. Findings that shaped the design

Several of these contradict the obvious assumption and are the reason the
design looks the way it does.

### 3.1 `filter_anatomy` already exists, and is data-driven

`anatomy.py:170` takes `tracked_kp_names` and intersects the whole anatomy
config against it — `KP_NAMES`, `KEYPOINT_MODEL_PAIRS`, `body_names`,
`joint_names`, `wing_names`, `end_eff_names`.

**Nothing in this repo calls it with that argument.** `run.py::_load_anatomy`
calls `load_anatomy(detached)` with the defaults (`tracked_kp_names=None`,
`strict=True`). The mechanism is complete; the plumbing to it is missing.

### 3.2 A genuinely 44-long `kp_order` is better than NaN padding

Because `kp_order` is shortened rather than NaN-filled, every downstream
consumer works *by construction* instead of by NaN tolerance:

- `scale.rigid_segment_pairs` guards `if a in names and b in names`, so the
  T1L chain contributes zero pairs and the other five legs still give 16
  (T1R gives 4; T2L/T2R/T3L/T3R give 3 each -- only T1L and T1R have a
  `ThxCx` keypoint, so the other four legs' chains start one joint later).
- `qc.invariants` has no T1L segment to call collapsed.
- `qc.coverage` reports on the 44 keypoints that exist rather than showing
  six permanently-empty rows.

### 3.3 …but `filter_anatomy` would keep the whole T1L leg anyway

`_leg_tracked` keeps a leg when **any** tracked keypoint carries its tag, and
`T1L_ThxCx` is tracked. So `coxa_T1_left…claw_T1_left` stay in `body_names`
and all seven T1L joints stay in `joint_names`.

This is harmless on its own, because:

### 3.4 Those name lists do not gate the solve

`body_names`, `joint_names`, `end_eff_names` and `wing_names` have **zero
consumers outside `anatomy.py`** in `src/`. `filter_anatomy` never prunes the
XML — `build_marker_model` only *adds* marker sites via `MjSpec`. `names_qpos`
comes from `align_joint_dims(mj_model)` on the compiled model, and
`solve_bout` hardcodes `qs_to_opt=np.ones(nq, bool)` at `perframe.py:475`.

**Freezing the unobserved T1L DOFs is therefore a separate, genuinely missing
piece** — not something `filter_anatomy` does today.

### 3.5 The IK root does not depend on the missing leg

`solve_bout`'s solvability set is `ROOT_OPTIMIZATION_KEYPOINT` plus
`JAXLS_ORIENTATION_KEYPOINTS` — `Scutellum`, `WingL_base`, `WingR_base`,
`Antenna_Base`. All four are trunk markers, all four are present. Had the
amputated leg contributed one of them, no frame would have been solvable.

### 3.6 The scorecard already tolerates a missing reprojection

`SessionRow.reproj_ratio` is `float | None`; `_row_from_qc` reads
`qc.get("reproj", {}) or {}`; `write_scorecard` guards
`if row.reproj_ratio is not None`. Omitting the block degrades gracefully
with no change to `qc/session.py`.

`posture_report` likewise survives an all-NaN `residual_px`: `finite_resid`
is empty → `baseline` is NaN → `misfit` is all-False → nothing flagged, while
pitch and height still compute.

### 3.7 `resume == "partial"` raises, and `plan.py` hides it

`resume.stage_is_current` raises `NotImplementedError` for a `"partial"`
stage. `plan.py` only tolerates `coarse`/`fine` because
`_NEVER_SKIP_AT_PLAN_LEVEL` bypasses the check for exactly those two names.
**A new `"partial"` stage must be added to that frozenset or `plan.resolve`
raises at runtime.**

### 3.8 `sex.json` cannot express a lone female

The schema is `{"identity": "sex", "male_fly": <int>}` — "which of the two
courtship flies is male". For a single fly, encoding *female* means writing
`male_fly: 1`, naming a fly that does not exist.

## 4. Design

### 4.1 Config surface

**`configs/recording/amputation.yaml`** — parameterized on `id`, following
`session1.yaml`:

```yaml
id: 2026_07_06_16_55_07
name: "${recording.id}"
assay: amputation
session_dir: "${paths.base_dir}/amputation/${recording.id}"
calib_dir: null
cameras: []
num_animals: 1
fps: 800.0
bouts_csv: "${recording.session_dir}/running_bouts_summary.csv"
kp3d_csv:  "${recording.session_dir}/data3D.csv"
index_csv: "${paths.base_dir}/amputation/index.csv"
```

`run.root` resolves to `processed/amputation/<id>/<run.name>` — a
subdirectory of `session_dir`, so a run's outputs sit beside the
`data3D.csv` they came from without colliding with it.

**Anatomy: no new YAML.** `_load_anatomy` derives `tracked_kp_names` from the
`data3D.csv` header and calls `load_anatomy(..., strict=False)`. A
hand-written `v1_amp_T1L.yaml` was considered and rejected: the tracked set
cannot drift from the data if it *is* the data, and a future cohort
amputated at a different site needs no new file. `anatomy=v1` stays the
single body-model config, so intact and amputee runs share one lineage.

**When `recording.kp3d_csv` is null — every existing courtship and
free-running config — `_load_anatomy` passes `tracked_kp_names=None` and
`strict=True`, exactly as today.** The filtering path is reachable only from
a recording that declares a 3D CSV, so no existing run changes behaviour.

**`configs/ik/amputation.yaml`** — `defaults: [default]` plus
`freeze_dof_patterns: ["*_T1_left"]`, matching all eleven T1L qpos slots
(`coxa_abduct`, `coxa_twist`, `coxa`, `femur_twist`, `femur`, `tibia`,
`tarsus`, `tarsus2`, `tarsus3`, `tarsus4`, `tarsus5`) and no others.
**Eleven, not the seven `configs/anatomy/v1.yaml`'s curated `joint_names`
lists** -- `names_qpos` comes from `align_joint_dims` on the COMPILED model,
which carries four tarsus joints that curated list omits. Measured on the
v1 model 2026-09-17 (`nq = 93`).

### 4.2 The `ingest3d` stage

**`src/tracking/io/kp3d_csv.py`** (new) — the only module that knows this
CSV format:

- `read_kp3d_header(path) -> list[str]` — the 44 names from the two-row
  header. Called by both `_context` and the stage.
- `read_bout_kp3d(path, bouts, kp_order) -> dict[int, (kp3d, conf3d)]` — one
  chunked pass keeping only rows inside the union of the bout ranges (not
  the outer span, which would hold ~450 k rows to use ~5 k). Returns arrays
  already permuted into `kp_order`. Refuses a bout range exceeding the
  file's row count rather than silently truncating.

**Registry entry** in `stages.py`, immediately after `fine`:

```python
Stage("ingest3d", "recording", ("kp3d_csv", "bouts.csv"), ("kp3d.npz", "sex.json"), "partial"),
```

and `"ingest3d"` added to `plan.py`'s `_NEVER_SKIP_AT_PLAN_LEVEL` (§3.7).
The stage owns its own per-bout resume check — skip a bout whose
`kp3d.npz` already exists unless forced — exactly as `fine` does.

**`run.py::_execute`** gains an `ingest3d` branch: read bouts → one CSV pass
→ per bout, `save_npz(bout_dir/"fly0"/"kp3d.npz", arrays={"kp3d","conf3d"},
kp=anatomy.kp_order, cams=None)`, wrapped in `timed(...)`. Values pass
through untouched (§2.2).

`sex.json` is written here from `index.csv` for the 93 recordings that have
an entry, one per **bout** directory (`bouts/bout_XXXXX/sex.json`) — the
location `run.py:291` and `qc/session.py:39` already read, and where `fine`
already writes it. `fly_sex_label` (`qc/posture.py:25`) gains a `sex_by_fly`
form — `{"identity": "sex", "sex_by_fly": {"0": "female"}}` — checked before
the existing `male_fly` fallback, so the file never asserts something false
(§3.8). Four lines; the courtship path is unchanged.

### 4.3 The rigless capability

**`io/recording.py`** — `calib_dir: Path | None`, plus `kp3d_csv` and
`index_csv`, plus:

```python
@property
def has_rig(self) -> bool:
    return self.calib_dir is not None
```

`validate()` skips the calibration and camera-order checks when
`calib_dir is None`, and **refuses a rigless spec that still names
cameras** — a `cameras:` list with no calibration directory to order it
against is exactly the silent-permutation class the existing `require_same`
check exists to prevent. `Order([])` is legal.

**`run.py::_context`** — `ctx["rig"] = _rig(spec) if spec.has_rig else None`.

**`bout_stages.postprocess_bout_fly`** — load `kp2d.npz` only when
`rig is not None`; pass `kp2d=None, conf2d=None, rig=None` onward.

**`qc/bout.py::bout_qc`** — when `rig is None`, **omit** the `reproj` and
`loo` keys and pass `residual_px=np.full(T, np.nan)` to `posture_report`.
Omitting rather than writing `"reproj": null` is deliberate: a null claims
the check ran and found nothing, where absence says it never ran.

**`stages.py`** — `postprocess`'s `reads` tuple is left ALONE.

An earlier draft of this design trimmed `kp2d.npz` from it, on the claim that
`reads` is documentation and never enforced. **That claim was wrong.**
`slurm/graph.py::_depends_on` consumes `Stage.reads` against `Stage.writes`
to build real `--dependency=afterok:` edges, so trimming it deletes the
`postprocess -> fine` edge for every rigged campaign. Transitivity happens to
keep the ordering correct today (`postprocess -> ik -> preprocess -> fine`),
but that is an unstated property of the graph's shape, not a checked one.

`reads` is therefore the stage's MAXIMAL read set. Over-declaring costs one
redundant edge; under-declaring is a race. The `kp2d.npz` read being
conditional at runtime is a property of the code, not of the registry.

### 4.4 Freezing the T1L DOFs

- `configs/ik/default.yaml` gains top-level `freeze_dof_patterns: []`,
  deliberately **outside** the `per_frame:` block, which that file's own
  header documents as the solver-numerics surface `solver_settings_from_cfg`
  reads. A skeletal fact is not a numerics knob.
- `solve_bout` (`perframe.py:475`) and the offsets fit's full-pose pass
  (`offsets_fit.py:313`) take `freeze_dof_patterns=()` and build
  `qs_to_opt = ~dof_mask(anatomy.names_qpos, patterns)`. `dof_mask` already
  exists at `perframe.py:58`. **Both** are required — freezing only the
  solve would fit marker offsets against a floppy T1L.
- `run.py` passes `cfg.ik.freeze_dof_patterns` in the `ik` and
  `fit_fly_constants` branches.

**Two guards.** A pattern matching nothing is a typo that silently freezes
nothing — announce it loudly, as `build_marker_model` already does for
unknown `SITES_TO_FREEZE`. A pattern matching everything leaves no free DOF
and must refuse rather than "solve" a frozen model.

`solver.py:368` computes `full_q = jnp.where(qs_to_opt, q, frozen_q)` with
`frozen_q` the warm start, so T1L sits at the model's rest pose, constant
across every frame — visibly inert rather than plausible-looking noise.
`nq` and `names_qpos` are unchanged, so `outputs.h5` stays schema-compatible
with intact-fly runs and cohorts remain directly comparable.

### 4.5 The SLURM array

**`scripts/slurm/amputation_array.sh`** — one submission, no dependency
graph. Per-recording chains were rejected: 103 recordings × ~6 jobs is heavy
scheduler churn for bouts averaging 500 frames.

**Manifest drift is the one real hazard.** If each task re-globs the data
directory at run time, a directory appearing or being removed mid-campaign
shifts every index after it, and tasks silently process the wrong recording
or double-process one. So the script globs **once at submit time**, keeps
recordings having both `data3D.csv` and a non-empty
`running_bouts_summary.csv`, and writes a numbered manifest to
`slurm_logs/amputation_<timestamp>.manifest`. Each task reads line
`$SLURM_ARRAY_TASK_ID + 1`. The manifest is the campaign's record of what
ran; `--array=0-$((N-1))%C` is sized from its line count.

Job body per task:

```
python -m tracking.run recording=amputation recording.id='<id>' \
    ik=amputation anatomy=v1 run.name=<name> \
    stages=[bouts,ingest3d,preprocess,ik,postprocess,collect]
```

`recording.id` stays single-quoted: `session_pipeline.sh` documents the
measured failure where Hydra reads an all-digits-and-underscores value as an
int literal and drops every underscore (`2026_07_06_16_55_07` →
`20260706165507`), then 404s on a directory that never existed.

`fit_fly_constants` and `fit_run_floor` are **not** listed — `plan.resolve`
inserts both barriers because `ik` and `postprocess` are present. Listing
them would be harmless but would misrepresent who decides the order.

The body reuses `session_pipeline.sh`'s `_GPU_SETUP` verbatim: `module load
cuda`, the env's bundled CUDA wheels on `LD_LIBRARY_PATH`, then a
`jax.default_backend()=='gpu'` assertion exiting non-zero. Not optional —
that script records a real incident where a silent CPU fallback wrote a
complete, plausible-looking set of artifacts, caught only by an unrelated
downstream failure.

Flags: `--run-name` (required, no `debug` default), `--slurm`
(`gpu_l40s` default, `ckpt_all` available), `--concurrency` for `%C`,
`--recordings` to restrict to a subset, `--dry-run` printing the manifest
and `sbatch` argv without submitting. Logs to `slurm_logs/%x-%A_%a.out`.

Each task gets its own `run.root` by construction, so
`session_pipeline.sh`'s `--run-root`-with-many-recordings refusal has no
analogue here.

## 5. Testing

TDD. Small synthetic fixtures except where noted.

- **`io/kp3d_csv.py`** — two-row header parse; chunked slicing returns
  exactly the requested frame ranges, verified against a whole-file read;
  permutation into `kp_order` is by name; a bout range exceeding the row
  count refuses rather than truncating.
- **Anatomy filtering** — `load_anatomy(strict=False)` with the 44 tracked
  names yields a 44-long `kp_order` with no `T1L_Tro…TaTip`, and
  `rigid_segment_pairs` returns 16 pairs from the remaining five legs
  (20 is the INTACT anatomy's count; only T1L/T1R carry `ThxCx`).
- **Rigless spec** — `validate()` passes with `calib_dir=None` and empty
  `cameras`; refuses `calib_dir=None` with a non-empty `cameras`.
- **`bout_qc(rig=None)`** — no `reproj`/`loo` keys; `posture` still
  computed; `collect_rows`/`write_scorecard` produce a scorecard with
  `median_reproj_ratio: null`.
- **DOF freeze** — `["*_T1_left"]` masks exactly 11 slots and leaves the root
  freejoint free; a pattern matching nothing announces; a pattern matching
  everything refuses.
- **`plan.resolve`** with `ingest3d` in `stages` does not raise (§3.7) —
  this would otherwise surface only at runtime.
- **`sex_by_fly`** — read correctly for a lone fly; the `male_fly` courtship
  path unchanged.

## 6. Rollout

Each step gates the next.

1. **One recording** — `2026_07_06_16_55_07` (11 bouts, 6,090 frames),
   locally, end to end. Inspect `session_qc.md`, and confirm in `outputs.h5`
   that the T1L qpos columns are **constant**. That is the direct evidence
   the freeze worked and cannot be inferred from a green exit code.
2. **A 5-recording array**, including `2026_07_09_10_22_32` (51 bouts,
   33,328 frames) to size `--time` honestly.
3. **All 103.**

## 7. Known data gaps

Neither blocks the run; both belong in any scorecard built from this cohort.

- **Sex is `unknown` for 10 recordings** — the `2026_07_13` and
  `2026_07_15` tail, absent from `index.csv` (§2.5).
- **`2026_07_09_17_08_07` has no `manifest_row.tsv`.** Inert, since the
  session tag is derived from the directory name (§2.6).

## 8. Out of scope

- `coarse`, `gates`, `fine` — no video, no calibration.
- `kpvideo`, `sidebyside` — render against video this cohort does not have.
- Cross-recording aggregation. `scripts/collect_sessions.py::find_run_roots`
  globs `{assay}/{session}/*/{run_name}` (four levels) and will not match the
  flat `amputation/<id>/<run_name>` layout. The per-recording `collect` stage
  is unaffected. Teaching `find_run_roots` a flat shape is a deliberate
  follow-up, not part of this work.
