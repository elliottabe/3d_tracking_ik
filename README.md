# 3d_tracking_ik

Multi-animal 3D tracking pipeline for [Whole-body 3D kinematics of freely behaving Drosophila](https://www.biorxiv.org/content/10.64898/2026.05.03.722293)

Mask-free multi-camera fly tracking to articulated IK fits. Raw video from a
calibrated multi-camera rig goes in; per-frame joint angles for a MuJoCo
*Drosophila* body model come out.

```
video + calibration
  → coarse   sweep the recording at a wide stride, locate each fly
  → gates    turn coarse tracks into a bout table   (or: bouts, from a supplied CSV)
  → fine     lift every frame of each bout to 50 3D landmarks
  → kpvideo  render the lift over the raw views (visualization)
  → preprocess   gate, interpolate and smooth the landmark trajectories
  → fit_fly_constants / fit_run_floor   per-fly body scale and marker offsets, arena floor
  → ik       per-frame Levenberg-Marquardt fit of the body model
  → postprocess  forward kinematics, derived kinematics, reprojection QC
  → collect  session scorecard and the combined dataset
```

Detection uses two networks: a per-camera center detector (EfficientNet-b1 +
BiFPN) that localizes flies, and MVQ, a multi-view query transformer
(DINOv3 ViT-B/16 backbone) that lifts the seven cropped views of one fly
directly to 3D landmarks without per-camera triangulation. The IK stage fits
a 93-DOF MuJoCo model to those landmarks, one frame at a time.

## Install

Linux with an NVIDIA GPU. Developed and run with NVIDIA L40S cards, driver 580.178.04; any recent
x86-64 Linux with a CUDA 12-capable driver should work. `jax[cuda12]` ships
its own CUDA runtime, so no system CUDA toolkit is needed — only the driver.
macOS and Windows are not supported: the pipeline needs CUDA for the detector
and MJX, and EGL for headless MuJoCo rendering.

```bash
conda env create -f environment.yaml && conda activate 3d_tracking_ik
uv pip install --python "$CONDA_PREFIX/bin/python" -e .
python -c "import tracking; print(tracking.__version__)"
```

The conda environment pins Python 3.13 and brings in ffmpeg and the EGL
libraries, and sets `MUJOCO_GL=egl` so MuJoCo renders headless on a GPU node.
Renders and IK will not run on a login node.

Verify JAX sees the GPU before a long run:

```bash
python -c "import jax; print(jax.devices())"     # expect [CudaDevice(id=0), ...]
```

The stack this was last run against: JAX 0.11.1, Flax 0.12.9, Optax 0.2.8,
Orbax 0.12.4, MuJoCo / MJX 3.13.0, NumPy 2.5.3, Hydra 1.3.6. `jaxls` has no
PyPI wheel and is pinned by commit in `pyproject.toml` — an unrelated package
of the same name exists on PyPI and will not work.

## Model weights

The two detector checkpoints are too large to ship in this repository and are
archived separately:

> **Data availability.** Model checkpoints (MVQ v2, 782 MB; CenterDetect,
> 8.9 MB) are deposited at [here](https://drive.google.com/drive/folders/1flBiyFmJYWPA6EIN4Xoh_2Lc5VT2_AoV?usp=drive_link).

Download and unpack them, then point `paths.ckpt_dir` at the directory that
holds `jax_mvq_runs/` and `jax_centerdetect_runs/`. To verify placement before
starting a run:

```bash
python scripts/fetch_assets.py             # human-readable report
python scripts/fetch_assets.py --json      # machine-readable
```

It resolves the same config a real run does and names any missing asset in a
couple of seconds, rather than failing forty minutes into a GPU job.

## Configure

Configuration is Hydra, rooted at `configs/pipeline.yaml`. Two groups need
editing for a new setup:

**Machine paths.** Copy the template and fill in the three directories:

```bash
cp configs/paths/template.yaml configs/paths/mymachine.yaml
```

`data_dir` (raw recordings), `base_dir` (where run outputs land), and
`ckpt_dir` (the unpacked checkpoints). `configs/paths/hyak.yaml` is a
filled-in example. Select yours with `paths=mymachine` on every command, or
make it the default in `configs/pipeline.yaml`.

**Recordings.** Each dataset gets a `configs/recording/*.yaml` naming its
session directory, its cameras in calibration-glob order, the number of
animals, and `fps` — which is required, since it is the sole source of the
output sample rate. See `configs/recording/session0.yaml` for a worked
example.

Calibration is read from the recording's own `calibration/` directory: one
`Cam*.yaml` per camera, holding a DLT projection matrix.

## Run

The driver is one entry point. Every stage reads the same config, and every
stage's output is its own checkpoint — a stage whose artifact already exists
and whose signature matches is skipped, so an interrupted run resumes by
being re-invoked.

Print the plan without executing anything:

```bash
python -m tracking.run recording=session0 paths=mymachine dry_run=true
```

Run a whole recording:

```bash
python -m tracking.run recording=session0 paths=mymachine \
    mvq=v2 run.name=pose_v2
```

Run part of the pipeline, on one bout:

```bash
python -m tracking.run recording=session0 paths=mymachine \
    stages=[fine,preprocess,ik] bout_ids=[28] run.name=pose_v2
```

Stage names, in dependency order: `coarse`, `gates` (or `bouts`), `fine`,
`kpvideo`, `preprocess`, `fit_fly_constants`, `fit_run_floor`, `ik`,
`postprocess`, `sidebyside`, `collect`. A recording that already has a bout
table supplies it via `recording.bouts_csv` and skips `coarse`/`gates`
entirely.

**On a SLURM cluster**, submit a whole session as one dependency graph —
per-recording chains with array jobs over bouts, and one `collect` gated on
all of them:

```bash
scripts/slurm/session_pipeline.sh --dry-run --recordings session0 --bout-ids 28
scripts/slurm/session_pipeline.sh --recordings 'session1:*' --slurm gpu_l40s
```

`--dry-run` prints every `sbatch` invocation without submitting. Keep
concurrent JAX pipelines at four or fewer per 128 GB node.

**Resampling** the combined dataset to another rate is a separate CLI, not a
pipeline stage:

```bash
python -m tracking.postprocess.resample --in base.h5 --out padded_1000hz.h5 \
    --target-hz 1000
```

## Outputs

A run lands at `<base_dir>/<assay>/<recording>/<run.name>/`:

```
scale.json                  per-fly body scale
offsets_fly<f>.h5           per-fly STAC marker offsets
floor.json                  fitted arena floor plane
bouts/bout_<id>/
    mvq_meta.json, sex.json     per-bout lift metadata
    kpvideo.mp4                 the lift drawn over the raw views
    fly<f>/
        kp2d.npz, kp3d.npz      lifted landmarks (2D px, 3D world units)
        kp3d_filt.npz           filtered and gap-bridged
        stac_ik.h5              fitted qpos, marker offsets, FK sites
        fitted.npz              fitted landmarks, for overlay and QC
        outputs.h5              qpos, root transform, scale
        qc.json                 reprojection, coverage, rigid invariants
session_qc.{json,md}        per-bout scorecard
ik_output_combined_<anatomy>_<name>.h5    the combined dataset
timing.json                 per-stage wall clock and GPU hours
```

## Figures

`Notebook_figures/` holds the paper figures: one notebook each for Figures 1,
2, 3 and 5, and a script for Figure 4 (`Notebook_figures/Figure_4/`, with its
own README). They need a few packages the pipeline itself does not:

```bash
uv pip install --python "$CONDA_PREFIX/bin/python" -e ".[figures]"
```

That brings in scikit-learn, seaborn and scikit-image, plus
`mujoco_visualizer` for Figure 4's rendered panel.

**Optional GPU acceleration.** Figure 2's embedding runs on RAPIDS when it is
available and falls back to scikit-learn when it is not, so the figure is
reproducible either way — RAPIDS only makes it faster. It needs NVIDIA's
package index, which a Python extra cannot carry, so name it explicitly:

```bash
uv pip install --python "$CONDA_PREFIX/bin/python" \
    --extra-index-url=https://pypi.nvidia.com -e ".[rapids]"
```

Installing it into the pipeline environment **downgrades numpy 2.5.3 to 2.4.6
and pandas 3.0.5 to 3.0.3**, because `cudf` pins them — and numpy is the
version this stack's JAX and MuJoCo builds are tested against. Prefer a
separate environment for Figure 2 over perturbing a working pipeline install.

## Citation

If you use this pipeline, please cite the paper:

```bibtex
@article{ispizua_wholebody_2026,
  author  = {Ispizua, J. I. and Abe, E. T. T. and Yan, J. and Othayoth, R. and
             Sawtelle, S. and Atkins, F. and Shiozaki, H. and Meier, N. R. and
             Wong, J. and Tran, T. T. and Mori, C. K. and Voigts, J. and
             Stern, D. L. and Brunton, B. W. and Tuthill, J. C. and
             Johnson, R. E.},
  title   = {Whole-body {3D} kinematics of freely behaving \textit{Drosophila}},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.05.03.722293},
  url     = {https://doi.org/10.64898/2026.05.03.722293},
  note    = {J. I. Ispizua and E. T. T. Abe contributed equally}
}
```
