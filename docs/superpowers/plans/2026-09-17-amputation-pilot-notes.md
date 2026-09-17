# Amputation pilot run — findings

**Date:** 2026-09-17
**Recording:** `2026_07_06_16_55_07` (11 bouts, 6,090 bout frames)
**Run root:** `/gscratch/portia/eabe/data/Johnson_lab/processed/amputation/2026_07_06_16_55_07/ik_pilot`
**Node:** g3120 (L40S), JAX backend `gpu`, `cuda:0`
**Command:**

```
python -m tracking.run recording=amputation ik=amputation \
    recording.id="'2026_07_06_16_55_07'" run.name=ik_pilot \
    'stages=[bouts,ingest3d,preprocess,ik,postprocess,collect]'
```

Exit 0. Wall clock **38m09s** (user 55m40s).

## 1. The T1L freeze — the finding this pilot existed to produce

A green exit code cannot distinguish "the freeze reached the solver" from
"the freeze was configured and ignored". Measured directly out of
`outputs.h5`, with the intact right front leg as a control:

| leg | slots | worst peak-to-peak across all 11 bouts |
|---|---|---|
| **T1L** (amputated, frozen) | 11 | **0.000e+00** |
| T1R (intact, control) | 11 | 1.742e+00 |

Every one of the eleven `*_T1_left` qpos columns is *exactly* constant in
every bout, while the contralateral leg moves normally. The freeze works and
it is not over-broad.

The eleven slots are `coxa_abduct`, `coxa_twist`, `coxa`, `femur_twist`,
`femur`, `tibia`, `tarsus`, `tarsus2`, `tarsus3`, `tarsus4`, `tarsus5` — the
four tarsus joints that `configs/anatomy/v1.yaml`'s curated `joint_names`
list omits are present in `names_qpos` and are frozen too, which is the
point of the count correction made before execution began.

## 2. Body scale

`scale.json` reports `scale: 0.0107387`, **`n_pairs_used: 16`**, implied body
length **2.688 mm** — inside `scale.py`'s plausible band `[2.0, 3.0]`.

The 16 independently confirms the pre-execution ruling that the filtered
anatomy yields 16 rigid-segment pairs, not the 20 the plan first claimed:
only `T1L`/`T1R` carry a `ThxCx` keypoint, so `T2*`/`T3*` contribute three
pairs each, and the amputated `T1L` contributes none.

## 3. Rigless QC degraded exactly as designed

`bouts/bout_00001/fly0/qc.json` top-level keys are
`['coverage', 'invariants', 'posture', 'structural_ok']` — **`reproj` and
`loo` are ABSENT, not null.** `session_qc.json` carries
`median_reproj_ratio: null` and scored all 11 bout-flies without complaint.

Coverage names **44 keypoints** with no `T1L_*` distal entries.
`posture.sex` is `female`, read from the cohort `index.csv` via the new
`sex_by_fly` form. `n_collapsed` is 0 in every bout; `n_flagged` is 0.

All 11 bouts status `ok`; `missing_fraction` is 0 for ten of them and
0.0013 for `bout_00011`.

## 4. Artifacts written

`bouts.csv`, `floor.json`, `scale.json`, `offsets_fly0.h5`, `session_qc.json`,
`session_qc.md`, `timing.json`, and
`ik_output_combined_v1_ik_pilot.h5`, plus `bouts/bout_000NN/fly0/` holding
`kp3d.npz`, `kp3d_filt.npz`, `stac_ik.h5`, `outputs.h5`, `fitted.npz`,
`qc.json`, and `bouts/bout_000NN/sex.json`.

## 5. Per-stage timing, and what it means for the array

| stage | seconds | note |
|---|---|---|
| `bouts` | 0.002 | |
| `ingest3d` | 8.2 | one chunked pass over the 742 MB CSV, 11 bouts |
| `preprocess` | 0.13 | per bout-fly |
| **`fit_fly_constants`** | **581.2** | dominant cost; pooled offsets fit, largely independent of bout count |
| `fit_run_floor` | 1.1 | |
| `ik` | 148.8 | last bout-fly's 237 frames |
| `postprocess` | 1.6 | per bout-fly |
| `collect` | 0.69 | |

`fit_fly_constants` is ~10 minutes and does not scale with bout count, so it
is close to a fixed per-recording floor. The rest scales with frames.

This recording has 6,090 bout frames against a cohort median of 8,033 and a
maximum of 33,328 (`2026_07_09_10_22_32`, 51 bouts). Extrapolating the
frame-proportional part, the heaviest recording should land well under two
hours. **`--time=12:00:00` from `configs/slurm/gpu_l40s.yaml` is ample**; no
change needed before the array runs.

## 6. Not yet done

Steps 6's scale-up (5 recordings, then all 103) submits real SLURM jobs
against a shared allocation and is left for an explicit decision rather than
taken automatically.
