# Amputation Cohort IK Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the existing IK pipeline end to end over 103 Johnson lab amputation recordings whose 3D keypoints already exist as one whole-recording CSV each, and submit the cohort as a single SLURM array job.

**Architecture:** A new `ingest3d` recording-scope stage slices `data3D.csv` into per-bout `kp3d.npz`, sitting where `fine` sits in the registry. "No rig, no 2D" becomes a declared property of a recording rather than a crash. The anatomy's existing-but-never-called `filter_anatomy(tracked_kp_names=...)` is wired up so the amputated fly gets a genuinely 44-long `kp_order`, and the eleven now-unobserved T1L DOFs are frozen so they cannot masquerade as data.

**Tech Stack:** Python 3.11+, Hydra/OmegaConf, NumPy, pandas, MuJoCo/MJX, JAX, pytest, SLURM.

**Spec:** `docs/superpowers/specs/2026-09-17-amputation-ik-pipeline-design.md`

## Global Constraints

- **Branch:** `feat/amputation-ik`, worktree at `/mmfs1/gscratch/portia/eabe/Research/MyRepos/3d_tracking_ik-amputation`. Base is `main`.
- **`main` has no test harness.** No `tests/`, no `pytest` dev dependency, no `[tool.pytest.ini_options]`. Task 1 creates all three. Do not assume any test infrastructure exists.
- **Units:** `data3D.csv` is already in the pipeline's world units (`scale.WORLD_UNITS_TO_MM = 0.1`). **Never** apply `info.yaml: coord_scale`. A wrong conversion surfaces only as an `assert_plausible_body_scale` failure.
- **Keypoints:** the CSV has 44; the anatomy has 50. The six absent are `T1L_Tro`, `T1L_FeTi`, `T1L_TiTa`, `T1L_TaT1`, `T1L_TaT3`, `T1L_TaTip`. `T1L_ThxCx` **is present**.
- **fps is 800.0** for every recording in this cohort.
- **Existing behaviour must not change.** Every change is gated on a recording declaring `kp3d_csv` / a null `calib_dir`. Courtship and free-running runs must compose and behave exactly as before.
- **Style:** ruff, `line-length = 100`, `select = ["E","W","F","I","UP","B"]`. Comments are minimal and concise — docstrings and short clarifiers, not the paragraph-length rationale blocks found in older modules.
- **Commit messages** end with:
  ```
  Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01Xw9Rimizss33cv14UaVcaN
  ```

---

### Task 1: Test harness and config surface

Bootstraps pytest (absent on `main`) and adds the three config files. Deliverable: `recording=amputation ik=amputation` composes and resolves to the right paths.

**Files:**
- Modify: `pyproject.toml` (add pytest dev dep + `[tool.pytest.ini_options]`)
- Create: `configs/recording/amputation.yaml`
- Create: `configs/ik/amputation.yaml`
- Modify: `configs/ik/default.yaml` (add `freeze_dof_patterns: []`)
- Test: `tests/test_amputation_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: config group entries `recording=amputation` (fields `id`, `name`, `assay`, `session_dir`, `calib_dir: None`, `cameras: []`, `num_animals: 1`, `fps: 800.0`, `bouts_csv`, `kp3d_csv`, `index_csv`) and `ik=amputation` (`freeze_dof_patterns: ["*_T1_left"]`). A `tests/` root with `testpaths = ["tests"]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_amputation_config.py`:

```python
"""The amputation config group composes and resolves to the cohort's real paths."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

import tracking.utils.path_utils  # noqa: F401 -- registers ${repo_root:}

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _compose(*overrides):
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        return compose(config_name="pipeline", overrides=list(overrides))


def test_amputation_recording_resolves_flat_run_root():
    cfg = _compose("recording=amputation", "run.name=ik_v1")
    assert cfg.recording.name == "2026_07_06_16_55_07"
    assert cfg.recording.assay == "amputation"
    assert cfg.run.root.endswith("/processed/amputation/2026_07_06_16_55_07/ik_v1")
    assert cfg.recording.session_dir.endswith("/processed/amputation/2026_07_06_16_55_07")


def test_amputation_recording_is_rigless_and_single_fly():
    cfg = _compose("recording=amputation")
    assert cfg.recording.calib_dir is None
    assert list(cfg.recording.cameras) == []
    assert cfg.recording.num_animals == 1
    assert cfg.recording.fps == 800.0


def test_amputation_recording_id_override_flows_into_every_path():
    # SINGLE-QUOTED, exactly as scripts/slurm/amputation_array.sh emits it.
    cfg = _compose("recording=amputation", "recording.id='2026_07_09_10_22_32'")
    assert cfg.recording.name == "2026_07_09_10_22_32"
    assert cfg.recording.bouts_csv.endswith(
        "/2026_07_09_10_22_32/running_bouts_summary.csv"
    )
    assert cfg.recording.kp3d_csv.endswith("/2026_07_09_10_22_32/data3D.csv")


def test_unquoted_recording_id_is_mangled_into_an_int():
    """The trap the slurm script quotes around, pinned as a regression test.

    Hydra's override grammar reads an all-digits-and-underscores value as a
    Python int literal and drops every underscore, so an unquoted id silently
    becomes a recording that never existed.
    """
    cfg = _compose("recording=amputation", "recording.id=2026_07_09_10_22_32")
    assert cfg.recording.id == 20260709102232
    assert not str(cfg.recording.kp3d_csv).endswith("/2026_07_09_10_22_32/data3D.csv")


def test_ik_amputation_inherits_per_frame_and_adds_freeze():
    cfg = _compose("ik=amputation")
    assert list(cfg.ik.freeze_dof_patterns) == ["*_T1_left"]
    assert cfg.ik.per_frame.n_iter == 500
    assert cfg.ik.per_frame.batch == 512


def test_ik_default_freezes_nothing():
    cfg = _compose()
    assert list(cfg.ik.freeze_dof_patterns) == []


@pytest.mark.parametrize("group", ["session0", "session1", "stairs"])
def test_existing_recordings_still_compose(group):
    cfg = _compose(f"recording={group}")
    assert cfg.recording.calib_dir is not None
    assert len(cfg.recording.cameras) == 7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_amputation_config.py -v`
Expected: collection error or FAIL — `pytest` may not be installed, and `configs/recording/amputation.yaml` does not exist.

- [ ] **Step 3: Add the pytest harness**

In `pyproject.toml`, add `pytest` to the dev extra and append the config section:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

If the dev extra does not exist yet, add `dev = ["ruff", "pytest"]` under `[project.optional-dependencies]`. Install with `pip install -e '.[dev]'` if pytest is missing.

- [ ] **Step 4: Write the config files**

`configs/recording/amputation.yaml`:

```yaml
# The Johnson lab amputation cohort: 103 single-fly recordings whose 3D
# keypoints already exist as one whole-recording data3D.csv each.
#
#   python -m tracking.run recording=amputation ik=amputation \
#       recording.id=2026_07_06_16_55_07 run.name=ik_v1
#
# Parameterized on `id` like session1.yaml -- every other field is identical
# across the cohort (verified on disk 2026-09-17).

id: "2026_07_06_16_55_07"

name: "${recording.id}"
assay: amputation

# The cohort lives under the PROCESSED root, not Video_recordings: these are
# already-lifted 3D predictions, not raw video. run.root is a subdirectory of
# this, so a run's outputs land beside the data3D.csv they came from.
session_dir: "${paths.base_dir}/amputation/${recording.id}"

# No calibration exists for this cohort, and no 2D is ever read. A `cameras`
# list here with no calib_dir to order it against is refused by
# RecordingSpec.validate().
calib_dir: null
cameras: []

num_animals: 1

# Measured off this cohort's own bout table: 187 frames / 0.23375 s = 800.0.
fps: 800.0

bouts_csv: "${recording.session_dir}/running_bouts_summary.csv"

# The whole-recording 3D table `ingest3d` slices into per-bout kp3d.npz.
# Its presence is also what puts `_load_anatomy` on the filtered path.
kp3d_csv: "${recording.session_dir}/data3D.csv"

# Cohort-level fly metadata (flyID, sex, recording, amp). Covers 93 of the
# 103 on-disk recordings; the rest get sex "unknown".
index_csv: "${paths.base_dir}/amputation/index.csv"
```

`configs/ik/amputation.yaml`:

```yaml
# The IK settings for the amputation cohort: default numerics, plus the
# left front leg's eleven DOFs frozen.
#
# Every T1L keypoint distal to the coxa is absent from data3D.csv, so those
# joints are unobserved. Left free they would drift under the temporal prior
# and land in outputs.h5 looking like measurements.
defaults:
  - default

freeze_dof_patterns: ["*_T1_left"]
```

In `configs/ik/default.yaml`, append at top level (NOT inside `per_frame:`, which that file's header documents as the solver-numerics surface `solver_settings_from_cfg` reads):

```yaml
# fnmatch patterns over `anatomy.names_qpos`; matching qpos slots are held at
# their warm-start value instead of being optimised. Empty means solve every
# DOF, which is what every intact-fly recording wants.
freeze_dof_patterns: []
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_amputation_config.py -v`
Expected: 9 passed (5 plain tests + the 3-way parametrized `test_existing_recordings_still_compose` + the new unquoted-id regression test).

- [ ] **Step 6: Lint and commit**

```bash
# Scoped: Notebook_figures/*.ipynb carries 784 pre-existing ruff errors on main.
ruff check src tests scripts configs && ruff format --check src tests
git add pyproject.toml configs/recording/amputation.yaml configs/ik/amputation.yaml configs/ik/default.yaml tests/test_amputation_config.py
git commit -m "$(cat <<'EOF'
feat(config): amputation recording group, frozen-T1L ik group, pytest harness

main carries no test harness at all; this adds pytest, testpaths and the
tests/ root alongside the first configs that need them.

configs/ik/default.yaml gains `freeze_dof_patterns: []` at top level rather
than inside `per_frame:`, which is the surface solver_settings_from_cfg
reads -- a skeletal fact is not a numerics knob.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Xw9Rimizss33cv14UaVcaN
EOF
)"
```

---

### Task 2: Rigless `RecordingSpec`

**Files:**
- Modify: `src/tracking/io/recording.py`
- Test: `tests/io/test_recording_rigless.py`

**Interfaces:**
- Consumes: Task 1's `configs/recording/amputation.yaml`.
- Produces: `RecordingSpec` with `calib_dir: Path | None`, new fields `kp3d_csv: Path | None` and `index_csv: Path | None`, and a `has_rig -> bool` property. `validate()` skips calibration checks when `calib_dir is None` and raises `ValueError` if `cameras` is non-empty in that case.

- [ ] **Step 1: Write the failing test**

Create `tests/io/test_recording_rigless.py`:

```python
"""A recording may declare that it has no camera rig, and must then name no cameras."""

import pytest
from omegaconf import OmegaConf

from tracking.io.recording import RecordingSpec


def _cfg(tmp_path, **over):
    base = {
        "name": "2026_07_06_16_55_07",
        "session_dir": str(tmp_path),
        "calib_dir": None,
        "cameras": [],
        "num_animals": 1,
        "fps": 800.0,
        "bouts_csv": None,
        "kp3d_csv": str(tmp_path / "data3D.csv"),
        "index_csv": None,
    }
    base.update(over)
    return OmegaConf.create(base)


def test_rigless_spec_validates(tmp_path):
    spec = RecordingSpec.from_config(_cfg(tmp_path))
    spec.validate()
    assert spec.has_rig is False
    assert spec.calib_dir is None
    assert spec.kp3d_csv == tmp_path / "data3D.csv"


def test_rigless_spec_refuses_named_cameras(tmp_path):
    spec = RecordingSpec.from_config(_cfg(tmp_path, cameras=["Cam2012630"]))
    with pytest.raises(ValueError, match="names cameras"):
        spec.validate()


def test_missing_kp3d_csv_is_none_not_a_path(tmp_path):
    cfg = _cfg(tmp_path)
    del cfg["kp3d_csv"]
    spec = RecordingSpec.from_config(cfg)
    assert spec.kp3d_csv is None
    assert spec.index_csv is None


def test_rigged_spec_still_requires_its_calib_dir(tmp_path):
    spec = RecordingSpec.from_config(
        _cfg(tmp_path, calib_dir=str(tmp_path / "nope"), cameras=["Cam1"])
    )
    assert spec.has_rig is True
    with pytest.raises(ValueError, match="calib_dir does not exist"):
        spec.validate()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/io/test_recording_rigless.py -v`
Expected: FAIL — `RecordingSpec.from_config` raises on `calib_dir=None` (`Path(None)`), and there is no `has_rig`.

- [ ] **Step 3: Implement**

In `src/tracking/io/recording.py`, change the dataclass fields and `from_config`/`validate`:

```python
@dataclass(frozen=True)
class RecordingSpec:
    """Typed description of one recording session."""

    name: str
    session_dir: Path
    calib_dir: Path | None
    cameras: Order
    num_animals: int
    fps: float
    bouts_csv: Path | None
    kp3d_csv: Path | None = None
    index_csv: Path | None = None

    @classmethod
    def from_config(cls, cfg) -> RecordingSpec:
        def _opt(key):
            value = cfg.get(key, None) if hasattr(cfg, "get") else getattr(cfg, key, None)
            return Path(value) if value is not None else None

        fps = cfg.get("fps", None) if hasattr(cfg, "get") else getattr(cfg, "fps", None)
        if fps is None or not (float(fps) > 0):
            raise ValueError(
                "recording.fps is required and must be > 0; it is the sole source of "
                "source_hz and must not be guessed"
            )

        return cls(
            name=str(cfg.name),
            session_dir=Path(cfg.session_dir),
            calib_dir=_opt("calib_dir"),
            cameras=Order(cfg.cameras),
            num_animals=int(cfg.num_animals),
            fps=float(fps),
            bouts_csv=_opt("bouts_csv"),
            kp3d_csv=_opt("kp3d_csv"),
            index_csv=_opt("index_csv"),
        )

    @property
    def has_rig(self) -> bool:
        """Whether this recording has calibrated cameras to reproject against."""
        return self.calib_dir is not None

    def validate(self) -> None:
        if not self.session_dir.is_dir():
            raise ValueError(f"recording.session_dir does not exist: {self.session_dir}")
        if not self.has_rig:
            if len(self.cameras):
                raise ValueError(
                    f"recording {self.name!r} has no calib_dir but names cameras "
                    f"{list(self.cameras)}; there is nothing to order them against, "
                    f"and a wrong order silently permutes every camera axis"
                )
            return
        if not self.calib_dir.is_dir():
            raise ValueError(f"recording.calib_dir does not exist: {self.calib_dir}")
        require_same(self.cameras, load_camera_order(self.calib_dir), what="camera")
```

Leave `video_path` unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/io/ tests/test_amputation_config.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
ruff check . && git add src/tracking/io/recording.py tests/io/test_recording_rigless.py
git commit -m "$(cat <<'EOF'
feat(io): a recording may declare it has no camera rig

calib_dir becomes optional, and kp3d_csv/index_csv join the spec. A rigless
spec that still names cameras is refused: there is nothing to order them
against, and a wrong camera order silently permutes every camera axis
downstream rather than failing.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Xw9Rimizss33cv14UaVcaN
EOF
)"
```

---

### Task 3: `io/kp3d_csv.py` — reading the cohort's CSV

**Files:**
- Create: `src/tracking/io/kp3d_csv.py`
- Test: `tests/io/test_kp3d_csv.py`

**Interfaces:**
- Consumes: `tracking.io.names.Order`, `as_order`; `tracking.io.bouts.BoutSpec`.
- Produces:
  - `read_kp3d_header(path) -> list[str]` — keypoint names in column order.
  - `read_bout_kp3d(path, bouts, *, kp_order, chunksize=200_000) -> dict[int, tuple[np.ndarray, np.ndarray]]` — `{bout.idx: (kp3d (T,K,3), conf3d (T,K))}`, permuted into `kp_order`.

- [ ] **Step 1: Write the failing test**

Create `tests/io/test_kp3d_csv.py`:

```python
"""Reading a two-row-header whole-recording 3D keypoint CSV."""

import numpy as np
import pytest

from tracking.io.bouts import BoutSpec
from tracking.io.kp3d_csv import read_bout_kp3d, read_kp3d_header
from tracking.io.names import Order, OrderMismatch

KPS = ["Scutellum", "Abd_tip", "T1R_FeTi"]


def _write_csv(path, n_frames, kps=KPS):
    header1 = ",".join(k for k in kps for _ in range(4))
    header2 = ",".join(["x", "y", "z", "confidence"] * len(kps))
    lines = [header1, header2]
    for t in range(n_frames):
        row = []
        for k_i in range(len(kps)):
            base = t * 100 + k_i * 10
            row += [f"{base}.0", f"{base + 1}.0", f"{base + 2}.0", "0.9"]
        lines.append(",".join(row))
    path.write_text("\n".join(lines) + "\n")
    return path


def _bout(idx, start, end):
    return BoutSpec(
        idx=idx, start_frame=start, end_frame=end,
        n_frames=end - start + 1, source="summary",
    )


def test_header_names_are_read_in_column_order(tmp_path):
    csv = _write_csv(tmp_path / "data3D.csv", 5)
    assert read_kp3d_header(csv) == KPS


def test_header_rejects_unexpected_coordinate_row(tmp_path):
    csv = tmp_path / "bad.csv"
    csv.write_text("A,A,A,A\nx,y,z,score\n1,2,3,4\n")
    with pytest.raises(ValueError, match="expected"):
        read_kp3d_header(csv)


def test_header_rejects_mismatched_name_block(tmp_path):
    csv = tmp_path / "bad.csv"
    csv.write_text("A,A,B,A\nx,y,z,confidence\n1,2,3,4\n")
    with pytest.raises(ValueError, match="one keypoint name"):
        read_kp3d_header(csv)


def test_bout_slices_are_exact_and_permuted(tmp_path):
    csv = _write_csv(tmp_path / "data3D.csv", 50)
    order = Order(["Abd_tip", "T1R_FeTi", "Scutellum"])
    out = read_bout_kp3d(csv, [_bout(1, 10, 12), _bout(2, 40, 40)], kp_order=order, chunksize=7)

    kp3d, conf = out[1]
    assert kp3d.shape == (3, 3, 3)
    assert conf.shape == (3, 3)
    # frame 10, "Abd_tip" is kps index 1 -> base = 10*100 + 1*10 = 1010
    np.testing.assert_allclose(kp3d[0, 0], [1010.0, 1011.0, 1012.0])
    # frame 10, "Scutellum" is kps index 0 -> base = 1000
    np.testing.assert_allclose(kp3d[0, 2], [1000.0, 1001.0, 1002.0])
    np.testing.assert_allclose(conf, 0.9)

    assert out[2][0].shape == (1, 3, 3)
    np.testing.assert_allclose(out[2][0][0, 2], [4000.0, 4001.0, 4002.0])


def test_chunk_boundary_does_not_split_a_bout(tmp_path):
    csv = _write_csv(tmp_path / "data3D.csv", 50)
    order = Order(KPS)
    whole = read_bout_kp3d(csv, [_bout(1, 5, 30)], kp_order=order, chunksize=1000)[1][0]
    split = read_bout_kp3d(csv, [_bout(1, 5, 30)], kp_order=order, chunksize=4)[1][0]
    np.testing.assert_array_equal(whole, split)


def test_bout_past_end_of_file_refuses(tmp_path):
    csv = _write_csv(tmp_path / "data3D.csv", 20)
    with pytest.raises(ValueError, match="yielded"):
        read_bout_kp3d(csv, [_bout(1, 15, 25)], kp_order=Order(KPS))


def test_kp_order_naming_an_absent_keypoint_refuses(tmp_path):
    csv = _write_csv(tmp_path / "data3D.csv", 5)
    with pytest.raises(OrderMismatch):
        read_bout_kp3d(csv, [_bout(1, 0, 2)], kp_order=Order(["Scutellum", "T1L_TaTip"]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/io/test_kp3d_csv.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tracking.io.kp3d_csv'`.

- [ ] **Step 3: Implement**

Create `src/tracking/io/kp3d_csv.py`:

```python
"""Reading a whole-recording 3D keypoint CSV into per-bout arrays.

The amputation cohort's `data3D.csv` carries two header rows -- keypoint name
repeated four times, then `x,y,z,confidence` -- and one row per frame of the
entire recording. Values are already in the pipeline's world units
(`preprocess.scale.WORLD_UNITS_TO_MM`); nothing here rescales them.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from tracking.io.names import Order, as_order

__all__ = ["read_kp3d_header", "read_bout_kp3d"]

_COORDS = ("x", "y", "z", "confidence")


def read_kp3d_header(path) -> list[str]:
    """The keypoint names of a two-row-header 3D CSV, in column order."""
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        try:
            names = next(reader)
            coords = next(reader)
        except StopIteration:
            raise ValueError(f"{path}: fewer than two header rows") from None

    if len(names) != len(coords):
        raise ValueError(
            f"{path}: header rows disagree -- {len(names)} names, {len(coords)} coordinates"
        )
    if not names or len(names) % 4:
        raise ValueError(f"{path}: {len(names)} columns is not a multiple of 4 (x,y,z,confidence)")

    out: list[str] = []
    for i in range(0, len(names), 4):
        block = tuple(c.strip() for c in coords[i : i + 4])
        if block != _COORDS:
            raise ValueError(f"{path}: columns {i}..{i + 3} are {block}, expected {_COORDS}")
        group = {n.strip() for n in names[i : i + 4]}
        if len(group) != 1:
            raise ValueError(
                f"{path}: columns {i}..{i + 3} must all carry one keypoint name, got {sorted(group)}"
            )
        out.append(names[i].strip())
    return out


def read_bout_kp3d(
    path,
    bouts: Sequence,
    *,
    kp_order: Order | Sequence[str],
    chunksize: int = 200_000,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """`{bout.idx: (kp3d (T,K,3), conf3d (T,K))}`, one chunked pass over `path`.

    Rows are frame-indexed from 0. Only rows inside a requested bout are kept,
    so a 760k-frame file costs one pass and a few thousand retained rows.
    """
    path = Path(path)
    order = as_order(kp_order)
    file_order = Order(read_kp3d_header(path))
    perm = file_order.permutation_to(order)
    n_file_kp = len(file_order)

    wanted = {int(b.idx): (int(b.start_frame), int(b.end_frame)) for b in bouts}
    if not wanted:
        return {}
    lo = min(s for s, _ in wanted.values())
    hi = max(e for _, e in wanted.values())

    parts: dict[int, list[np.ndarray]] = {idx: [] for idx in wanted}
    n_rows = 0
    for chunk in pd.read_csv(
        path, skiprows=2, header=None, chunksize=chunksize, dtype=np.float64
    ):
        first, last = int(chunk.index[0]), int(chunk.index[-1])
        n_rows += len(chunk)
        if last < lo or first > hi:
            continue
        for idx, (start, end) in wanted.items():
            if end < first or start > last:
                continue
            parts[idx].append(chunk.loc[max(start, first) : min(end, last)].to_numpy())

    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for idx, (start, end) in wanted.items():
        want = end - start + 1
        got = np.concatenate(parts[idx]) if parts[idx] else np.empty((0, 4 * n_file_kp))
        if got.shape[0] != want:
            raise ValueError(
                f"{path}: bout {idx} wants frames {start}..{end} ({want} rows) but the "
                f"file yielded {got.shape[0]}; the file has {n_rows} frames"
            )
        block = got.reshape(want, n_file_kp, 4)
        out[idx] = (block[:, perm, :3].copy(), block[:, perm, 3].copy())
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/io/test_kp3d_csv.py -v`
Expected: 7 passed.

- [ ] **Step 5: Verify against real data**

Run:
```bash
python -c "
from tracking.io.kp3d_csv import read_kp3d_header, read_bout_kp3d
from tracking.io.bouts import read_bout_summary
from tracking.io.names import Order
d='/gscratch/portia/eabe/data/Johnson_lab/processed/amputation/2026_07_06_16_55_07'
names=read_kp3d_header(f'{d}/data3D.csv'); print(len(names),'keypoints')
b=read_bout_summary(f'{d}/running_bouts_summary.csv'); print(len(b),'bouts')
out=read_bout_kp3d(f'{d}/data3D.csv', b, kp_order=Order(names))
import numpy as np
for i in sorted(out)[:3]:
    k,c=out[i]; print(i, k.shape, 'finite %.1f%%'%(100*np.isfinite(k).all(-1).mean()), 'conf %.3f'%np.nanmedian(c))
"
```
Expected: `44 keypoints`, `11 bouts`, and each bout ~100% finite with median confidence ~0.91.

- [ ] **Step 6: Commit**

```bash
ruff check . && git add src/tracking/io/kp3d_csv.py tests/io/test_kp3d_csv.py
git commit -m "$(cat <<'EOF'
feat(io): read a whole-recording 3D keypoint CSV into per-bout arrays

One chunked pass keeps only rows inside a requested bout, so a 760k-frame
742 MB file costs ~10s and a few thousand retained rows rather than loading
the whole table.

A bout range the file cannot satisfy raises rather than silently returning
a short array, and the header parse refuses anything that is not
name-x-y-z-confidence blocks.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Xw9Rimizss33cv14UaVcaN
EOF
)"
```

---

### Task 4: Wire `filter_anatomy` to the recording's tracked keypoints

**Files:**
- Modify: `src/tracking/run.py` (`_load_anatomy`, ~line 51)
- Test: `tests/test_anatomy_tracked_keypoints.py`

**Interfaces:**
- Consumes: `read_kp3d_header` (Task 3), `cfg.recording.kp3d_csv` (Task 1).
- Produces: `_load_anatomy(cfg)` returning a 44-keypoint `Anatomy` when `recording.kp3d_csv` is set, and the unchanged 50-keypoint strict `Anatomy` when it is not.

- [ ] **Step 1: Write the failing test**

Create `tests/test_anatomy_tracked_keypoints.py`:

```python
"""The anatomy is intersected with what a recording actually tracked."""

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import tracking.utils.path_utils  # noqa: F401
from tracking.preprocess.scale import rigid_segment_pairs
from tracking.run import _load_anatomy

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
T1L_DISTAL = ["T1L_Tro", "T1L_FeTi", "T1L_TiTa", "T1L_TaT1", "T1L_TaT3", "T1L_TaTip"]


def _compose(*overrides):
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        return compose(config_name="pipeline", overrides=list(overrides))


def _csv(tmp_path, names):
    path = tmp_path / "data3D.csv"
    path.write_text(
        ",".join(n for n in names for _ in range(4))
        + "\n"
        + ",".join(["x", "y", "z", "confidence"] * len(names))
        + "\n"
    )
    return path


def test_no_kp3d_csv_keeps_the_full_strict_anatomy():
    anatomy = _load_anatomy(_compose("recording=session0"))
    assert len(anatomy.kp_order) == 50
    for name in T1L_DISTAL:
        assert name in anatomy.kp_order


def test_tracked_header_drops_the_amputated_keypoints(tmp_path):
    cfg = _compose("recording=amputation")
    full = OmegaConf.to_container(cfg.anatomy, resolve=False)["model"]["KP_NAMES"]
    tracked = [k for k in full if k not in T1L_DISTAL]
    cfg.recording.kp3d_csv = str(_csv(tmp_path, tracked))

    anatomy = _load_anatomy(cfg)
    assert len(anatomy.kp_order) == 44
    for name in T1L_DISTAL:
        assert name not in anatomy.kp_order
    assert "T1L_ThxCx" in anatomy.kp_order


def test_filtered_anatomy_still_measures_five_legs(tmp_path):
    cfg = _compose("recording=amputation")
    full = OmegaConf.to_container(cfg.anatomy, resolve=False)["model"]["KP_NAMES"]
    tracked = [k for k in full if k not in T1L_DISTAL]
    cfg.recording.kp3d_csv = str(_csv(tmp_path, tracked))

    anatomy = _load_anatomy(cfg)
    # 16, not 20: only T1L and T1R carry a ThxCx keypoint, so T2L/T2R/T3L/T3R
    # contribute 3 pairs each and T1R 4. 20 is the INTACT anatomy's count.
    pairs = rigid_segment_pairs(anatomy.kp_order)
    assert len(pairs) == 16
    assert not any(a.startswith("T1L") or b.startswith("T1L") for a, b in pairs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_anatomy_tracked_keypoints.py -v`
Expected: the two filtered tests FAIL with 50 keypoints, not 44 — `_load_anatomy` ignores `kp3d_csv`.

- [ ] **Step 3: Implement**

In `src/tracking/run.py`, replace `_load_anatomy`:

```python
def _load_anatomy(cfg: DictConfig):
    """`load_anatomy`, intersected with what this recording actually tracked.

    A recording that names a `kp3d_csv` carries its own keypoint set, which may
    be a subset of the anatomy's (an amputated fly has no markers distal to the
    amputation). Deriving the set from the file's own header means it cannot
    drift from the data. Every other recording takes the unchanged strict path.
    """
    from tracking.inverse_kinematics.anatomy import load_anatomy

    detached = OmegaConf.create(OmegaConf.to_container(cfg.anatomy, resolve=False))
    kp3d_csv = cfg.recording.get("kp3d_csv", None)
    if kp3d_csv is None:
        return load_anatomy(detached)

    from tracking.io.kp3d_csv import read_kp3d_header

    return load_anatomy(
        detached, tracked_kp_names=read_kp3d_header(kp3d_csv), strict=False
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_anatomy_tracked_keypoints.py -v`
Expected: 3 passed. Note the `[anatomy] filter_anatomy: dropping keypoint ...` lines on stderr — that is the announce path working.

- [ ] **Step 5: Commit**

```bash
ruff check . && git add src/tracking/run.py tests/test_anatomy_tracked_keypoints.py
git commit -m "$(cat <<'EOF'
feat(run): intersect the anatomy with the keypoints a recording tracked

filter_anatomy has taken `tracked_kp_names` since it was written but nothing
ever passed it. A recording declaring a kp3d_csv now derives the set from
that file's own header, so it cannot drift from the data and a cohort
amputated at a different site needs no new config.

Recordings without a kp3d_csv keep the strict 50-keypoint path unchanged.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Xw9Rimizss33cv14UaVcaN
EOF
)"
```

---

### Task 5: Freeze unobserved DOFs in the solve and the offsets fit

**Files:**
- Modify: `src/tracking/inverse_kinematics/perframe.py` (add `_qs_to_opt`; `solve_bout` ~line 362 signature, ~line 475 `qs_to_opt`)
- Modify: `src/tracking/inverse_kinematics/offsets_fit.py` (`pose_optimization` ~line 313, `fit_offsets_from_flat` ~line 390)
- Modify: `src/tracking/pipeline/bout_stages.py` (`ik_bout_fly` ~line 249, `fit_fly_constants` ~line 113)
- Modify: `src/tracking/run.py` (`ik` and `fit_fly_constants` branches)
- Test: `tests/inverse_kinematics/test_freeze_dofs.py`

**Interfaces:**
- Consumes: `cfg.ik.freeze_dof_patterns` (Task 1), `perframe.dof_mask` (existing, line 58).
- Produces: `perframe._qs_to_opt(anatomy, patterns) -> np.ndarray` `(nq,)` bool. A `freeze_dof_patterns: Sequence[str] = ()` keyword on `solve_bout`, `pose_optimization`, `fit_offsets_from_flat`, `fit_offsets`, `ik_bout_fly` and `fit_fly_constants`.

- [ ] **Step 1: Write the failing test**

Create `tests/inverse_kinematics/test_freeze_dofs.py`:

```python
"""Unobserved DOFs are held at their warm start rather than optimised."""

from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

import tracking.utils.path_utils  # noqa: F401
from tracking.inverse_kinematics.perframe import _qs_to_opt
from tracking.run import _load_anatomy

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


@pytest.fixture(scope="module")
def anatomy():
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        cfg = compose(config_name="pipeline", overrides=["recording=session0"])
    return _load_anatomy(cfg)


def test_no_patterns_optimises_everything(anatomy):
    mask = _qs_to_opt(anatomy, ())
    assert mask.shape == (anatomy.nq,)
    assert mask.all()


def test_t1l_pattern_freezes_exactly_the_left_front_leg(anatomy):
    # ELEVEN, not the seven configs/anatomy/v1.yaml's curated joint_names lists:
    # names_qpos comes from align_joint_dims on the COMPILED model, which carries
    # tarsus2..tarsus5_T1_left as well. Measured on the v1 model 2026-09-17.
    mask = _qs_to_opt(anatomy, ["*_T1_left"])
    frozen = [n for n, free in zip(anatomy.names_qpos, mask) if not free]
    assert len(frozen) == 11
    assert all(n.endswith("_T1_left") for n in frozen)
    # the root freejoint is never frozen by a leg pattern
    assert mask[:7].all()


def test_pattern_matching_nothing_warns_and_freezes_nothing(anatomy, capsys):
    mask = _qs_to_opt(anatomy, ["*_T9_left"])
    assert mask.all()
    assert "matched no qpos slot" in capsys.readouterr().err


def test_pattern_matching_everything_refuses(anatomy):
    with pytest.raises(ValueError, match="nothing left to solve"):
        _qs_to_opt(anatomy, ["*"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/inverse_kinematics/test_freeze_dofs.py -v`
Expected: FAIL with `ImportError: cannot import name '_qs_to_opt'`.

- [ ] **Step 3: Implement `_qs_to_opt` and use it in `solve_bout`**

In `src/tracking/inverse_kinematics/perframe.py`, add after `dof_mask` (line 58):

```python
def _qs_to_opt(anatomy, patterns: Sequence[str] | None) -> np.ndarray:
    """`(nq,)` bool: which qpos slots the solver may move.

    A DOF with no keypoint observing it is not free information -- left
    optimised it drifts under the temporal prior and lands in outputs.h5
    looking like a measurement.
    """
    nq = int(anatomy.nq)
    patterns = tuple(patterns or ())
    if not patterns:
        return np.ones(nq, dtype=bool)

    frozen = dof_mask(anatomy.names_qpos, patterns)
    if frozen.all():
        raise ValueError(
            f"freeze_dof_patterns {list(patterns)} froze all {nq} qpos slots; "
            f"there is nothing left to solve"
        )
    if not frozen.any():
        announce(
            "perframe",
            "_qs_to_opt",
            f"freeze_dof_patterns {list(patterns)} matched no qpos slot, so NOTHING "
            f"is frozen -- check them against anatomy.names_qpos",
        )
    return ~frozen
```

Add `announce` to the module's `tracking.conventions` import if absent, and `Sequence` to the `collections.abc` import.

In `solve_bout`, add the keyword to the signature:

```python
def solve_bout(
    anatomy,
    kp3d_model: np.ndarray,
    *,
    offsets: np.ndarray,
    settings: SolverSettings | None = None,
    per_frame_cfg: Mapping[str, Any] | None = None,
    solve_mask: np.ndarray | None = None,
    freeze_dof_patterns: Sequence[str] = (),
) -> PerFrameResult:
```

and at the `common = dict(...)` block (~line 475) replace `qs_to_opt=np.ones(nq, bool)` with:

```python
        qs_to_opt=_qs_to_opt(anatomy, freeze_dof_patterns),
```

- [ ] **Step 4: Run the unit test to verify it passes**

Run: `python -m pytest tests/inverse_kinematics/test_freeze_dofs.py -v`
Expected: 4 passed.

- [ ] **Step 5: Thread the patterns through the offsets fit**

In `src/tracking/inverse_kinematics/offsets_fit.py`:

- Import `_qs_to_opt` from `tracking.inverse_kinematics.perframe`.
```python
# top of file
from tracking.inverse_kinematics.perframe import _qs_to_opt

def pose_optimization(
    anatomy,
    mjx_model,
    mjx_data,
    kp_data,
    *,
    root_kp_idx,
    orientation_idx=None,
    settings=None,
    solver=None,
    freeze_dof_patterns: Sequence[str] = (),
):
    ...
    # replaces `qs_to_opt = np.ones(int(anatomy.nq), dtype=bool)` (~line 313)
    qs_to_opt = _qs_to_opt(anatomy, freeze_dof_patterns)
```

Then add `freeze_dof_patterns: Sequence[str] = ()` to `fit_offsets_from_flat`'s keyword-only arguments (~line 390) and to `fit_offsets`, and forward it at **every** `pose_optimization(...)` call site in the file:

```bash
grep -n "pose_optimization(" src/tracking/inverse_kinematics/offsets_fit.py
```

Each call gains `freeze_dof_patterns=freeze_dof_patterns`. Leave `root_optimization`'s own `qs_to_opt` (~line 247, the root-only mask) alone — it optimises only the root's first `root_dims` slots, which a leg pattern never touches.

Freezing only the per-frame solve would fit marker offsets against a floppy T1L, so both are required.

- [ ] **Step 6: Thread the patterns from the config**

In `src/tracking/pipeline/bout_stages.py`:
- `ik_bout_fly`: add `freeze_dof_patterns: Sequence[str] = ()` and forward it to `solve_bout`.
- `fit_fly_constants`: add `freeze_dof_patterns: Sequence[str] = ()` and forward it to `fit_offsets`.

In `src/tracking/run.py`, in the `ik` branch add to the `ik_bout_fly(...)` call:

```python
                freeze_dof_patterns=_plain(cfg.ik.get("freeze_dof_patterns", [])),
```

and the identical line in the `fit_fly_constants(...)` call.

- [ ] **Step 7: Run the full suite**

Run: `python -m pytest tests/ -v`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
ruff check . && git add -A src/tracking tests/inverse_kinematics
git commit -m "$(cat <<'EOF'
feat(ik): freeze DOFs no keypoint observes

An amputated leg's joints have no markers constraining them. Left optimised
they drift under the temporal prior and reach outputs.h5 looking like
measurements; frozen they sit at the warm start, visibly inert.

Applied to both the per-frame solve and the offsets fit -- freezing only the
former would fit marker offsets against a floppy leg.

A pattern matching nothing announces rather than silently freezing nothing;
a pattern matching every slot refuses.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Xw9Rimizss33cv14UaVcaN
EOF
)"
```

---

### Task 6: The `ingest3d` stage, and `sex.json` for a lone fly

**Files:**
- Modify: `src/tracking/pipeline/stages.py` (registry)
- Modify: `src/tracking/pipeline/plan.py` (`_NEVER_SKIP_AT_PLAN_LEVEL`, line 24)
- Modify: `src/tracking/qc/posture.py` (`fly_sex_label`, line 25)
- Create: `src/tracking/pipeline/ingest_stages.py`
- Modify: `src/tracking/run.py` (`_execute` branch)
- Test: `tests/pipeline/test_ingest3d.py`

**Interfaces:**
- Consumes: `read_bout_kp3d` (Task 3), `RecordingSpec.kp3d_csv`/`index_csv` (Task 2), `io.artifacts.save_npz`, `io.bouts.read_bouts_csv`.
- Produces:
  - `ingest_stages.ingest3d_recording(run_root, *, spec, kp_order, bouts, force=False) -> dict` — writes `bouts/bout_XXXXX/fly0/kp3d.npz` and `bouts/bout_XXXXX/sex.json`.
  - `ingest_stages.sex_json_for(index_csv, recording) -> dict | None`.
  - `fly_sex_label` understands `{"identity": "sex", "sex_by_fly": {"0": "female"}}`.

- [ ] **Step 1: Write the failing test**

Create `tests/pipeline/test_ingest3d.py`:

```python
"""Slicing a whole-recording 3D CSV into the per-bout artifacts the pipeline reads."""

from pathlib import Path

import numpy as np
import pytest

from tracking.io.bouts import BoutSpec, write_bouts_csv
from tracking.io.names import Order
from tracking.pipeline.ingest_stages import ingest3d_recording, sex_json_for
from tracking.pipeline.plan import resolve
from tracking.pipeline.stages import stage_by_name
from tracking.qc.posture import fly_sex_label

KPS = ["Scutellum", "Abd_tip", "T1R_FeTi"]


def _csv(path, n_frames):
    rows = [
        ",".join(k for k in KPS for _ in range(4)),
        ",".join(["x", "y", "z", "confidence"] * len(KPS)),
    ]
    for t in range(n_frames):
        row = []
        for k_i in range(len(KPS)):
            base = t * 100 + k_i * 10
            row += [f"{base}.0", f"{base + 1}.0", f"{base + 2}.0", "0.9"]
        rows.append(",".join(row))
    path.write_text("\n".join(rows) + "\n")
    return path


class _Spec:
    def __init__(self, kp3d_csv, index_csv=None, name="rec1"):
        self.kp3d_csv = kp3d_csv
        self.index_csv = index_csv
        self.name = name


def test_ingest_writes_one_kp3d_per_bout(tmp_path):
    csv = _csv(tmp_path / "data3D.csv", 50)
    bouts = [BoutSpec(1, 10, 12, 3, "summary"), BoutSpec(2, 40, 41, 2, "summary")]
    run_root = tmp_path / "run"

    ingest3d_recording(run_root, spec=_Spec(csv), kp_order=Order(KPS), bouts=bouts)

    for idx, n in ((1, 3), (2, 2)):
        path = run_root / "bouts" / f"bout_{idx:05d}" / "fly0" / "kp3d.npz"
        assert path.exists()
        with np.load(path, allow_pickle=True) as z:
            assert z["kp3d"].shape == (n, 3, 3)
            assert z["conf3d"].shape == (n, 3)
            assert [str(s) for s in z["kp_names"]] == KPS


def test_ingest_is_resumable_and_skips_written_bouts(tmp_path):
    csv = _csv(tmp_path / "data3D.csv", 50)
    bouts = [BoutSpec(1, 10, 12, 3, "summary")]
    run_root = tmp_path / "run"

    first = ingest3d_recording(run_root, spec=_Spec(csv), kp_order=Order(KPS), bouts=bouts)
    second = ingest3d_recording(run_root, spec=_Spec(csv), kp_order=Order(KPS), bouts=bouts)
    assert first["n_written"] == 1
    assert second["n_written"] == 0
    assert second["n_skipped"] == 1

    forced = ingest3d_recording(
        run_root, spec=_Spec(csv), kp_order=Order(KPS), bouts=bouts, force=True
    )
    assert forced["n_written"] == 1


def test_sex_column_is_found_by_header_not_first_matching_cell(tmp_path):
    """A flyID of "F" must not be mistaken for the sex column."""
    index = tmp_path / "index.csv"
    index.write_text("flyID,sex,,amp\nF,m,recY,T1L\n")
    assert sex_json_for(index, "recY")["sex_by_fly"] == {"0": "male"}


def test_sex_json_read_from_the_cohort_index(tmp_path):
    index = tmp_path / "index.csv"
    index.write_text("flyID,sex,,amp\n1,f,rec1,T1L\n2,m,rec2,T1L\n")
    assert sex_json_for(index, "rec1")["sex_by_fly"] == {"0": "female"}
    assert sex_json_for(index, "rec2")["sex_by_fly"] == {"0": "male"}
    assert sex_json_for(index, "absent") is None


def test_fly_sex_label_understands_sex_by_fly():
    assert fly_sex_label({"identity": "sex", "sex_by_fly": {"0": "female"}}, 0) == "female"
    assert fly_sex_label({"identity": "sex", "sex_by_fly": {"0": "male"}}, 0) == "male"
    # the courtship form is untouched
    assert fly_sex_label({"identity": "sex", "male_fly": 1}, 1) == "male"
    assert fly_sex_label({"identity": "sex", "male_fly": 1}, 0) == "female"
    assert fly_sex_label({"identity": "sex", "sex_by_fly": {}}, 0) == "unknown"


def test_ingest_writes_sex_json_beside_each_bout(tmp_path):
    csv = _csv(tmp_path / "data3D.csv", 50)
    index = tmp_path / "index.csv"
    index.write_text("flyID,sex,,amp\n1,f,rec1,T1L\n")
    run_root = tmp_path / "run"

    ingest3d_recording(
        run_root,
        spec=_Spec(csv, index_csv=index),
        kp_order=Order(KPS),
        bouts=[BoutSpec(1, 10, 12, 3, "summary")],
    )
    import json

    written = json.loads((run_root / "bouts" / "bout_00001" / "sex.json").read_text())
    assert fly_sex_label(written, 0) == "female"


def test_ingest3d_is_registered_and_plannable(tmp_path):
    from omegaconf import OmegaConf

    assert stage_by_name("ingest3d").scope == "recording"
    cfg = OmegaConf.create(
        {"run": {"root": str(tmp_path)}, "recording": {"name": "rec1", "num_animals": 1}}
    )
    works = resolve(cfg, stages=["ingest3d"], bout_ids=[1])
    assert [w.stage for w in works] == ["ingest3d"]
    assert works[0].skip_reason is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/pipeline/test_ingest3d.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tracking.pipeline.ingest_stages'`.

- [ ] **Step 3: Register the stage**

In `src/tracking/pipeline/stages.py`, insert immediately after the `fine` entry:

```python
    # The 3D-given entry: a recording whose keypoints are already lifted and
    # live in one whole-recording CSV. Peer of `fine`, which infers them.
    Stage(
        "ingest3d",
        "recording",
        ("kp3d_csv", "bouts.csv"),
        ("kp3d.npz", "sex.json"),
        "partial",
    ),
```

In `src/tracking/pipeline/plan.py` (line 24):

```python
_NEVER_SKIP_AT_PLAN_LEVEL = frozenset({"coarse", "fine", "ingest3d"})
```

`resume.stage_is_current` raises `NotImplementedError` for a `"partial"` stage; this frozenset is what keeps `plan.resolve` from reaching it. The stage owns its own per-bout resume check, as `fine` does.

- [ ] **Step 4: Teach `fly_sex_label` the single-fly form**

In `src/tracking/qc/posture.py`, replace `fly_sex_label`:

```python
def fly_sex_label(sex_json: dict, fly: int) -> str:
    """`"female"` / `"male"` / `"unknown"`, read from `sex.json`."""
    if str(sex_json.get("identity")) != "sex":
        return "unknown"
    by_fly = sex_json.get("sex_by_fly")
    if isinstance(by_fly, dict):
        label = by_fly.get(str(fly))
        return label if label in ("male", "female") else "unknown"
    male = sex_json.get("male_fly")
    if male is None:
        return "unknown"
    return "male" if int(fly) == int(male) else "female"
```

`sex_by_fly` is checked first because the courtship `male_fly` form cannot express a lone female — it would have to name a fly that does not exist.

- [ ] **Step 5: Implement the stage**

Create `src/tracking/pipeline/ingest_stages.py`:

```python
"""The 3D-given ingest: a whole-recording keypoint CSV into per-bout artifacts."""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from tracking.conventions import announce
from tracking.io.artifacts import save_npz
from tracking.io.kp3d_csv import read_bout_kp3d
from tracking.io.names import Order, as_order

__all__ = ["ingest3d_recording", "sex_json_for"]

_SEX_LABELS = {"f": "female", "m": "male", "female": "female", "male": "male"}


def sex_json_for(index_csv, recording: str) -> dict[str, Any] | None:
    """This recording's `sex.json` payload from the cohort index, or None.

    The recording's own column is unnamed in the index header, so it is found
    by value. The sex column IS named, so it is read by header position -- a
    left-to-right scan for the first sex-looking cell would return a flyID of
    "F" or "M" before ever reaching the real sex column.

    A recording absent from the index yields None, which the caller writes as
    no file at all -- `fly_sex_label` then reports "unknown".
    """
    with open(index_csv, newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return None

    header = [c.strip().lower() for c in rows[0]]
    sex_col = header.index("sex") if "sex" in header else None

    for row in rows[1:]:
        cells = [c.strip() for c in row]
        if recording not in cells:
            continue
        if sex_col is not None and sex_col < len(cells):
            cells = [cells[sex_col]]
        for cell in cells:
            label = _SEX_LABELS.get(cell.lower())
            if label:
                return {
                    "identity": "sex",
                    "sex_by_fly": {"0": label},
                    "method": "cohort_index",
                    "authority": "cohort_index",
                    "source": str(index_csv),
                }
        return None
    return None


def _write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def ingest3d_recording(
    run_root,
    *,
    spec,
    kp_order: Order | Sequence[str],
    bouts: Sequence,
    force: bool = False,
) -> dict[str, Any]:
    """Slice `spec.kp3d_csv` into one `fly0/kp3d.npz` per bout in `bouts`."""
    run_root = Path(run_root)
    order = as_order(kp_order)
    if spec.kp3d_csv is None:
        raise ValueError(
            "ingest3d needs recording.kp3d_csv; this recording declares none, so "
            "there is no 3D table to slice"
        )

    def _bout_dir(idx: int) -> Path:
        return run_root / "bouts" / f"bout_{int(idx):05d}"

    todo = [b for b in bouts if force or not (_bout_dir(b.idx) / "fly0" / "kp3d.npz").exists()]
    n_skipped = len(bouts) - len(todo)
    announce(
        "pipeline.ingest_stages",
        "ingest3d_recording",
        f"{len(todo)} bout(s) to write, {n_skipped} already present at {run_root}",
    )

    sex = sex_json_for(spec.index_csv, spec.name) if spec.index_csv is not None else None

    by_bout = read_bout_kp3d(spec.kp3d_csv, todo, kp_order=order) if todo else {}
    for bout in todo:
        kp3d, conf3d = by_bout[bout.idx]
        bout_dir = _bout_dir(bout.idx)
        save_npz(
            bout_dir / "fly0" / "kp3d.npz",
            arrays={"kp3d": np.asarray(kp3d), "conf3d": np.asarray(conf3d)},
            kp=order,
            cams=None,
        )
        if sex is not None:
            _write_json(bout_dir / "sex.json", sex)

    return {
        "run_root": str(run_root),
        "n_written": len(todo),
        "n_skipped": n_skipped,
        "sex": None if sex is None else sex["sex_by_fly"].get("0"),
    }
```

- [ ] **Step 6: Wire it into the driver**

In `src/tracking/run.py`, add to `_execute` immediately after the `bouts` branch:

```python
    if stage == "ingest3d":
        from tracking.io.bouts import read_bouts_csv, select
        from tracking.pipeline.ingest_stages import ingest3d_recording

        bouts = select(read_bouts_csv(run_root / "bouts.csv"), bout_ids)
        with timed(timing_path, stage, n_items=len(bouts)):
            ingest3d_recording(
                run_root, spec=spec, kp_order=anatomy.kp_order, bouts=bouts
            )
        return
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `python -m pytest tests/pipeline/test_ingest3d.py -v`
Expected: 7 passed.

- [ ] **Step 8: Run the full suite and commit**

```bash
python -m pytest tests/ -q && ruff check .
git add -A src/tracking tests/pipeline
git commit -m "$(cat <<'EOF'
feat(pipeline): ingest3d, the 3D-given entry beside fine

A recording whose keypoints are already lifted carries them as one
whole-recording CSV. ingest3d slices it into the per-bout kp3d.npz every
downstream stage already reads, in a single pass over the file.

Registered "partial" like coarse/fine, so it must also join
_NEVER_SKIP_AT_PLAN_LEVEL -- resume.stage_is_current raises for that resume
mode, and the frozenset is what keeps plan.resolve away from it. The stage
owns its own per-bout skip check instead.

fly_sex_label gains a sex_by_fly form: the courtship male_fly schema cannot
express a lone female without naming a fly that does not exist.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Xw9Rimizss33cv14UaVcaN
EOF
)"
```

---

### Task 7: Postprocess and QC without a camera rig

**Files:**
- Modify: `src/tracking/run.py` (`_context`, ~line 358)
- Modify: `src/tracking/pipeline/bout_stages.py` (`postprocess_bout_fly`, ~line 292)
- Modify: `src/tracking/qc/bout.py` (`bout_qc`, line 22)
- Test: `tests/qc/test_bout_qc_rigless.py`

**Interfaces:**
- Consumes: `RecordingSpec.has_rig` (Task 2).
- Produces: `bout_qc(..., kp2d=None, conf2d=None, rig=None)` returning a report with **no** `reproj` and **no** `loo` keys. `postprocess_bout_fly` accepts `rig=None` and skips loading `kp2d.npz`.

- [ ] **Step 1: Write the failing test**

Create `tests/qc/test_bout_qc_rigless.py`:

```python
"""Scoring a bout that never had 2D to reproject against."""

import numpy as np

from tracking.detector.coarse import FloorPlane
from tracking.io.names import Order
from tracking.qc.bout import bout_qc
from tracking.qc.session import _blank_row, write_scorecard

KPS = ["Scutellum", "WingL_base", "WingR_base", "Antenna_Base", "Abd_tip", "Abd_A4"]


def _pose(n=40):
    rng = np.random.default_rng(0)
    base = np.array(
        [[0, 0, 5.0], [-2, 1, 5], [-2, -1, 5], [4, 0, 5], [-6, 0, 5], [-4, 0, 5]]
    )
    return base[None] + rng.normal(0, 0.01, size=(n, len(KPS), 3))


def _floor():
    return FloorPlane(np.array([0.0, 0.0, 1.0]), 0.0, "level", 0.0)


def test_rigless_qc_omits_reproj_and_loo():
    kp = _pose()
    report = bout_qc(
        kp2d=None,
        conf2d=None,
        kp3d_raw=kp,
        conf3d=np.full(kp.shape[:2], 0.9),
        fitted_world=kp,
        kp_order=Order(KPS),
        rig=None,
        floor=_floor(),
        sex="female",
    )
    assert "reproj" not in report
    assert "loo" not in report
    assert report["structural_ok"] is True
    assert "posture" in report
    assert "coverage" in report
    assert "invariants" in report


def test_rigless_posture_flags_nothing_on_residuals():
    kp = _pose()
    report = bout_qc(
        kp2d=None, conf2d=None, kp3d_raw=kp, conf3d=np.full(kp.shape[:2], 0.9),
        fitted_world=kp, kp_order=Order(KPS), rig=None, floor=_floor(), sex="female",
    )
    assert report["posture"]["n_flagged"] == 0


def test_scorecard_tolerates_a_missing_reproj_ratio(tmp_path):
    # _blank_row exists for exactly this: every metric None, status carried.
    rows = [_blank_row("bout_00001", 0, "female", "ok")]
    assert rows[0].reproj_ratio is None

    report = write_scorecard(
        tmp_path, rows, json_path=tmp_path / "qc.json", md_path=tmp_path / "qc.md"
    )
    assert (tmp_path / "qc.json").exists()
    assert (tmp_path / "qc.md").exists()
    assert report is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/qc/test_bout_qc_rigless.py -v`
Expected: FAIL — `bout_qc` calls `ik_reproj_report(kp2d, ...)` with `None`.

- [ ] **Step 3: Implement**

In `src/tracking/qc/bout.py`, make the 2D block conditional:

```python
def bout_qc(
    *,
    kp2d,
    conf2d,
    kp3d_raw,
    conf3d,
    fitted_world,
    kp_order,
    rig,
    floor,
    sex: str,
    conf_thresh: float = 0.3,
) -> dict:
    """Run every check for one bout-fly and merge the reports.

    `rig=None` marks a recording with no calibrated cameras: the reprojection
    and leave-one-out blocks are OMITTED, not nulled -- a null would claim the
    check ran and found nothing.
    """
    kp3d_raw = np.asarray(kp3d_raw, np.float64)
    fitted_world = np.asarray(fitted_world, np.float64)
    if not np.isfinite(fitted_world).any():
        raise QCStructuralError(
            "every fitted frame is non-finite; this bout-fly has no pose to score"
        )

    inv = invariant_report(kp3d_raw, kp_order)
    if inv["n_collapsed"]:
        collapsed = [k for k, v in inv["invariants"].items() if v["collapsed"]]
        raise QCStructuralError(
            f"collapsed rigid segment(s) {collapsed}: a rigid segment's length "
            f"cannot be near zero, and a collapsed pair is SMOOTHER than a real "
            f"landmark -- every jitter and confidence metric will rate it as good"
        )

    report: dict = {}
    if rig is None:
        resid = np.full(fitted_world.shape[0], np.nan)
    else:
        # Call order matches the pre-refactor version exactly: ik_reproj_report,
        # then per_frame_reproj, then loo_report. A swap changes which exception
        # a malformed rig surfaces first.
        report["reproj"] = ik_reproj_report(
            kp2d, conf2d, fitted_world, kp3d_raw, rig, conf_thresh=conf_thresh
        )
        resid = per_frame_reproj(kp2d, conf2d, fitted_world, rig, conf_thresh=conf_thresh)
        report["loo"] = loo_report(kp2d, conf2d, rig, conf_thresh=conf_thresh)

    report["invariants"] = inv
    report["coverage"] = coverage_report(kp3d_raw, conf3d, kp_order, conf_thresh=conf_thresh)
    report["posture"] = posture_report(
        fitted_world, kp_order, floor, residual_px=resid, sex=sex
    )
    report["structural_ok"] = True
    return report
```

In `src/tracking/pipeline/bout_stages.py`, `postprocess_bout_fly`, replace the unconditional `kp2d` load (~line 319):

```python
    if rig is None:
        kp2d = conf2d = None
    else:
        kp2d, conf2d = load_bout_kp2d(bout_dir / "kp2d.npz", kp_order=kp_order, cameras=cameras)
```

and pass `kp2d=kp2d, conf2d=conf2d` into `bout_qc`.

In `src/tracking/run.py`, `_context`:

```python
        ctx["rig"] = _rig(spec) if spec.has_rig else None
```

**Leave `src/tracking/pipeline/stages.py`'s `postprocess` `reads` tuple ALONE.** It stays `("stac_ik.h5", "kp2d.npz", "floor.json")`.

`reads` is NOT documentation: `slurm/graph.py::_depends_on` consumes it against `Stage.writes` to build real `--dependency=afterok:` edges. Trimming `kp2d.npz` deletes the `postprocess -> fine` edge for every rigged campaign. It is over-declaration — `reads` is the stage's maximal read set, and the conditional load is a runtime property. One redundant edge is harmless; a missing edge is a race.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/qc/ -v`
Expected: all pass.

- [ ] **Step 5: Run the full suite and commit**

```bash
python -m pytest tests/ -q && ruff check .
git add -A src/tracking tests/qc
git commit -m "$(cat <<'EOF'
feat(qc): score a bout that has no 2D to reproject against

A recording with no calibrated cameras gets no reproj and no loo block. They
are OMITTED rather than nulled: a null claims the check ran and found
nothing, where absence says it never ran.

posture_report already survives an all-NaN residual -- baseline is NaN, the
misfit mask is all-False, and pitch/height still compute -- and the
scorecard already treats reproj_ratio as optional.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Xw9Rimizss33cv14UaVcaN
EOF
)"
```

---

### Task 8: The SLURM array script

**Files:**
- Create: `scripts/slurm/amputation_array.sh`
- Test: `tests/slurm/test_amputation_array.py`

**Interfaces:**
- Consumes: the `recording=amputation` / `ik=amputation` config groups (Task 1) and every stage from Tasks 6–7.
- Produces: a submit-time manifest at `slurm_logs/amputation_<timestamp>.manifest`, one recording per line, and one `sbatch --array=0-N%C` invocation.

- [ ] **Step 1: Write the failing test**

Create `tests/slurm/test_amputation_array.py`:

```python
"""The amputation campaign freezes its recording list at submit time."""

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "slurm" / "amputation_array.sh"


def _data_root(tmp_path, recordings, with_bouts=True):
    root = tmp_path / "amputation"
    for rec in recordings:
        d = root / rec
        d.mkdir(parents=True)
        (d / "data3D.csv").write_text("A,A,A,A\nx,y,z,confidence\n")
        if with_bouts:
            (d / "running_bouts_summary.csv").write_text(
                "bout,start_frame,end_frame,n_frames\n1,0,9,10\n"
            )
    return root


def _run(*args, cwd=None):
    return subprocess.run(
        ["bash", str(SCRIPT), *args], capture_output=True, text=True, cwd=cwd or REPO
    )


def test_dry_run_writes_a_manifest_of_every_usable_recording(tmp_path):
    root = _data_root(tmp_path, ["2026_07_06_16_55_07", "2026_07_06_17_11_20"])
    out = _run("--dry-run", "--run-name", "ik_v1", "--data-root", str(root),
               "--manifest-dir", str(tmp_path))
    assert out.returncode == 0, out.stderr
    manifest = next(tmp_path.glob("amputation_*.manifest"))
    assert manifest.read_text().split() == [
        "2026_07_06_16_55_07", "2026_07_06_17_11_20",
    ]
    assert "--array=0-1" in out.stdout


def test_recording_without_a_bout_table_is_skipped_by_name(tmp_path):
    root = _data_root(tmp_path, ["2026_07_06_16_55_07"])
    bare = root / "2026_07_06_17_11_20"
    bare.mkdir()
    (bare / "data3D.csv").write_text("A,A,A,A\nx,y,z,confidence\n")

    out = _run("--dry-run", "--run-name", "ik_v1", "--data-root", str(root),
               "--manifest-dir", str(tmp_path))
    assert out.returncode == 0, out.stderr
    assert "2026_07_06_17_11_20" in out.stderr
    manifest = next(tmp_path.glob("amputation_*.manifest"))
    assert manifest.read_text().split() == ["2026_07_06_16_55_07"]


def test_requeue_flag_comes_from_the_slurm_profile(tmp_path):
    root = _data_root(tmp_path, ["2026_07_06_16_55_07"])
    ckpt = _run("--dry-run", "--run-name", "ik_v1", "--data-root", str(root),
                "--manifest-dir", str(tmp_path), "--slurm", "ckpt_all")
    assert ckpt.returncode == 0, ckpt.stderr
    assert "--requeue" in ckpt.stdout

    l40s = _run("--dry-run", "--run-name", "ik_v1", "--data-root", str(root),
                "--manifest-dir", str(tmp_path), "--slurm", "gpu_l40s")
    assert l40s.returncode == 0, l40s.stderr
    assert "--requeue" not in l40s.stdout


def test_missing_data_root_is_named_distinctly(tmp_path):
    out = _run("--dry-run", "--run-name", "ik_v1",
               "--data-root", str(tmp_path / "nope"), "--manifest-dir", str(tmp_path))
    assert out.returncode != 0
    assert "--data-root does not exist" in out.stderr


def test_run_name_is_required(tmp_path):
    root = _data_root(tmp_path, ["2026_07_06_16_55_07"])
    out = _run("--dry-run", "--data-root", str(root))
    assert out.returncode != 0
    assert "run-name" in out.stderr


def test_no_usable_recording_refuses(tmp_path):
    root = tmp_path / "amputation"
    root.mkdir()
    out = _run("--dry-run", "--run-name", "ik_v1", "--data-root", str(root),
               "--manifest-dir", str(tmp_path))
    assert out.returncode != 0
    assert "no recording" in out.stderr


def test_dry_run_command_quotes_the_recording_id(tmp_path):
    root = _data_root(tmp_path, ["2026_07_06_16_55_07"])
    out = _run("--dry-run", "--run-name", "ik_v1", "--data-root", str(root),
               "--manifest-dir", str(tmp_path))
    assert "recording.id='" in out.stdout
    assert "stages=[bouts,ingest3d,preprocess,ik,postprocess,collect]" in out.stdout
    assert "ik=amputation" in out.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/slurm/test_amputation_array.py -v`
Expected: FAIL — the script does not exist.

- [ ] **Step 3: Implement**

Create `scripts/slurm/amputation_array.sh` (`chmod +x`). Structure:

```bash
#!/bin/bash
# Submit the amputation cohort as ONE SLURM array: one task per recording,
# each running the whole chain end to end.
#
#   scripts/slurm/amputation_array.sh --run-name ik_v1 [--dry-run]
#
# Per-recording dependency chains (scripts/slurm/session_pipeline.sh's shape)
# were rejected for this cohort: bouts average ~500 frames, so 103 chains of
# ~6 jobs each is scheduler churn for work that fits in one task.
#
# THE MANIFEST IS FROZEN AT SUBMIT TIME. If each task re-globbed the data
# root, a directory appearing or being removed mid-campaign would shift every
# index after it and tasks would silently process the wrong recording, or one
# twice. The glob happens once, here; each task reads its own line.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DATA_ROOT="/gscratch/portia/eabe/data/Johnson_lab/processed/amputation"
MANIFEST_DIR="$REPO/slurm_logs"
RUN_NAME=""
SLURM_CFG=gpu_l40s
CONCURRENCY=10
RECORDINGS=""
DRY=0

while [ $# -gt 0 ]; do
    case "$1" in
        --run-name)      RUN_NAME="$2"; shift 2 ;;
        --data-root)     DATA_ROOT="$2"; shift 2 ;;
        --manifest-dir)  MANIFEST_DIR="$2"; shift 2 ;;
        --slurm)         SLURM_CFG="$2"; shift 2 ;;
        --concurrency)   CONCURRENCY="$2"; shift 2 ;;
        --recordings)    RECORDINGS="$2"; shift 2 ;;
        --dry-run)       DRY=1; shift ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

[ -n "$RUN_NAME" ] || {
    echo "usage: $0 --run-name NAME [--data-root DIR] [--manifest-dir DIR]" >&2
    echo "          [--slurm gpu_l40s|ckpt_all] [--concurrency N]" >&2
    echo "          [--recordings id1,id2,...] [--dry-run]" >&2
    echo "  --run-name is REQUIRED: the pipeline's own default is 'debug'," >&2
    echo "  which must never name a real campaign's outputs." >&2
    exit 2
}
```

Continue the script with the discovery, manifest, and submission:

```bash
# A missing data root is a different operator error from an empty one, and the
# generic "no usable recording" refusal below would hide which it was.
[ -d "$DATA_ROOT" ] || {
    echo "refusing: --data-root does not exist: $DATA_ROOT" >&2
    exit 2
}

# -- Discovery. A recording is usable only if it has BOTH the 3D table and a
#    bout table with at least one data row. A skip is always named on stderr:
#    "N tasks submitted" vs "103 directories exist" is exactly the discrepancy
#    an operator needs told.
if [ -n "$RECORDINGS" ]; then
    CANDIDATES="$(tr ',' '\n' <<< "$RECORDINGS" | grep . || true)"
else
    CANDIDATES="$(find "$DATA_ROOT" -mindepth 1 -maxdepth 1 -type d -name '2026_*' \
                  -printf '%f\n' 2>/dev/null | sort || true)"
fi

USABLE=""
while IFS= read -r rec; do
    [ -n "$rec" ] || continue
    d="$DATA_ROOT/$rec"
    if [ ! -f "$d/data3D.csv" ]; then
        echo "skip $rec: no data3D.csv" >&2; continue
    fi
    if [ ! -f "$d/running_bouts_summary.csv" ]; then
        echo "skip $rec: no running_bouts_summary.csv" >&2; continue
    fi
    if [ "$(wc -l < "$d/running_bouts_summary.csv")" -lt 2 ]; then
        echo "skip $rec: running_bouts_summary.csv has no data rows" >&2; continue
    fi
    USABLE="$USABLE$rec"$'\n'
done <<< "$CANDIDATES"

USABLE="$(printf '%s' "$USABLE" | grep . || true)"
N="$(printf '%s\n' "$USABLE" | grep -c . || true)"
if [ "$N" -eq 0 ]; then
    echo "refusing: no recording under $DATA_ROOT has both a data3D.csv and a" >&2
    echo "non-empty running_bouts_summary.csv" >&2
    exit 2
fi

# -- Freeze the list. Tasks index THIS file, never a fresh glob.
mkdir -p "$MANIFEST_DIR"
# ABSOLUTE, always: the job body does `cd "$REPO"` before reading the manifest,
# so a relative --manifest-dir would resolve against $REPO on the compute node
# and silently find nothing -- defeating the one guarantee this script exists
# to provide.
MANIFEST_DIR="$(cd "$MANIFEST_DIR" && pwd)"
MANIFEST="$MANIFEST_DIR/amputation_$(date +%Y%m%d-%H%M%S).manifest"
printf '%s\n' "$USABLE" > "$MANIFEST"

echo "Data root  : $DATA_ROOT"
echo "Manifest   : $MANIFEST ($N recording(s))"
echo "Run name   : $RUN_NAME"
echo "Slurm cfg  : $SLURM_CFG"
echo

SLURM_YAML="$REPO/configs/slurm/${SLURM_CFG}.yaml"
[ -f "$SLURM_YAML" ] || { echo "no such slurm config: $SLURM_YAML" >&2; exit 2; }
_y() { grep -E "^$1:" "$SLURM_YAML" | head -1 | sed -E "s/^$1:[[:space:]]*//; s/^['\"]//; s/['\"]$//"; }

RESOURCE_FLAGS=(
    "--partition=$(_y partition)" "--account=$(_y account)"
    "--time=$(_y time)" "--cpus-per-task=$(_y cpus_per_task)" "--mem=$(_y mem)"
)
[ -n "$(_y gres)" ]       && RESOURCE_FLAGS+=("--gres=$(_y gres)")
[ -n "$(_y constraint)" ] && RESOURCE_FLAGS+=("--constraint=$(_y constraint)")
[ -n "$(_y exclude)" ]    && RESOURCE_FLAGS+=("--exclude=$(_y exclude)")
# ckpt-all is preemptible and its profile sets `requeue: true`; without this a
# preempted array task is simply lost. session_pipeline.sh does the same
# (`if slurm_cfg.get("requeue")`). Compared against "true" rather than tested
# for non-emptiness so that `requeue: false` does NOT add the flag.
[ "$(_y requeue)" = "true" ] && RESOURCE_FLAGS+=("--requeue")

# -- VERIFIED FACT 1, copied from session_pipeline.sh: `module load cuda` alone
#    measurably leaves JAX on cpu on this cluster; the env's own bundled wheels
#    on LD_LIBRARY_PATH are what actually put devices on jax.devices(). The
#    backend assertion is not optional -- a silent CPU fallback does not crash,
#    it writes a complete and wrong set of artifacts.
_GPU_SETUP='module load cuda; export MUJOCO_GL=egl; '
_GPU_SETUP+='SP=$(python -c "import nvidia, pathlib; print(pathlib.Path(nvidia.__file__).parent)"); '
_GPU_SETUP+='export LD_LIBRARY_PATH="$(ls -d "$SP"/*/lib | tr '"'"'\n'"'"' '"'"':'"'"')$LD_LIBRARY_PATH"; '
_GPU_SETUP+='unset JAX_PLATFORMS; '
_GPU_SETUP+='python -c "import jax,sys; sys.exit(0 if jax.default_backend()==\"gpu\" else 1)" '
_GPU_SETUP+='|| { echo "FATAL: JAX backend is not gpu; refusing to burn a GPU allocation on CPU" >&2; exit 1; }; '

# `recording.id` stays SINGLE-QUOTED inside the body: unquoted, Hydra's
# override grammar reads an all-digits-and-underscores value as a Python int
# literal and silently drops every underscore (2026_07_06_16_55_07 ->
# 20260706165507), which then 404s on a directory that never existed.
BODY="$_GPU_SETUP"
BODY+="cd $(printf '%q' "$REPO"); "
BODY+="_REC=\"\$(sed -n \"\$((SLURM_ARRAY_TASK_ID + 1))p\" $(printf '%q' "$MANIFEST"))\"; "
BODY+='[ -n "$_REC" ] || { echo "no manifest line for task $SLURM_ARRAY_TASK_ID" >&2; exit 1; }; '
BODY+="echo \"recording: \$_REC\"; "
BODY+="python -m tracking.run recording=amputation \"recording.id='\$_REC'\" "
BODY+="ik=amputation anatomy=v1 run.name=$(printf '%q' "$RUN_NAME") "
BODY+="'stages=[bouts,ingest3d,preprocess,ik,postprocess,collect]'"

FLAGS=("--job-name=amputation-$RUN_NAME"
       "--array=0-$((N - 1))%$CONCURRENCY"
       "${RESOURCE_FLAGS[@]}"
       "--output=$REPO/slurm_logs/%x-%A_%a.out"
       "--error=$REPO/slurm_logs/%x-%A_%a.out")

# The body is printed VERBATIM, not %q-escaped: an operator reading a dry run
# needs to see the `recording.id='...'` quoting exactly as the job will get it,
# and %q would rewrite those quotes into something that no longer reads as the
# thing being checked.
if [ "$DRY" -eq 1 ]; then
    printf '+ sbatch'
    printf ' %s' "${FLAGS[@]}"
    printf ' --wrap %s\n' "$BODY"
    echo "(dry-run) $N task(s) described above; nothing submitted"
    exit 0
fi

# Only past the dry branch: sbatch creates the log FILE but never its
# directory, and a dry run must leave nothing behind but its manifest.
mkdir -p "$REPO/slurm_logs"
sbatch "${FLAGS[@]}" --wrap "$BODY"
echo "Monitor : squeue -u \$USER"
```

Note `--array=0-$((N-1))%$CONCURRENCY` uses the manifest's own line count, so the array can never index past the frozen list.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/slurm/test_amputation_array.py -v`
Expected: 7 passed.

- [ ] **Step 5: Dry-run against the real cohort**

Run: `scripts/slurm/amputation_array.sh --run-name ik_v1 --dry-run`
Expected: a manifest of **103** recordings, `--array=0-102%10`, and one printed `sbatch` line. Submits nothing.

- [ ] **Step 6: Commit**

```bash
ruff check . && git add scripts/slurm/amputation_array.sh tests/slurm/test_amputation_array.py
git commit -m "$(cat <<'EOF'
feat(slurm): submit the amputation cohort as one array, one task per recording

The recording list is globbed ONCE at submit time and frozen into a
manifest. Re-globbing inside each task would let a directory appearing or
vanishing mid-campaign shift every later index, silently processing the
wrong recording or one twice.

Bouts here average ~500 frames, so a whole recording fits in one task and
the per-bout dependency chains session_pipeline.sh builds would be pure
scheduler churn.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Xw9Rimizss33cv14UaVcaN
EOF
)"
```

---

### Task 9: Pilot one recording end to end

Validation, not new code. The only step that proves the freeze worked.

**Files:**
- Create: `docs/superpowers/plans/2026-09-17-amputation-pilot-notes.md`

- [ ] **Step 1: Dry-run the plan**

Run:
```bash
python -m tracking.run recording=amputation ik=amputation \
    recording.id=2026_07_06_16_55_07 run.name=ik_pilot \
    stages=[bouts,ingest3d,preprocess,ik,postprocess,collect] dry_run=true
```
Expected: **5 work items**, `run.root` ending `/processed/amputation/2026_07_06_16_55_07/ik_pilot`, `fit_fly_constants` and `fit_run_floor` inserted automatically, nothing created on disk.

Only the RECORDING-scoped stages appear. `preprocess`/`ik`/`postprocess` are bout-scoped and expand over `bout_ids`, which `plan.py` resolves at EXECUTION time — a dry run deliberately never reads `bouts.csv`, so it cannot know the bout count yet. Their absence here is correct, not a missing stage.

- [ ] **Step 2: Run it for real**

Run the same command without `dry_run=true`.
Expected: exit 0. Watch for `[anatomy] filter_anatomy: dropping keypoint 'T1L_...'` — six lines, once.

- [ ] **Step 3: Confirm the T1L DOFs are actually frozen**

Run:
```bash
python - <<'PY'
import h5py, numpy as np, glob
p = sorted(glob.glob("/gscratch/portia/eabe/data/Johnson_lab/processed/amputation/"
                     "2026_07_06_16_55_07/ik_pilot/bouts/*/fly0/outputs.h5"))[0]
with h5py.File(p) as f:
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in f["names_qpos"][:]]
    q = f["qpos"][:]
t1l = [i for i, n in enumerate(names) if n.endswith("_T1_left")]
print(p, q.shape, "T1L slots:", len(t1l))
for i in t1l:
    col = q[np.isfinite(q[:, i]), i]
    print(f"  {names[i]:<24} ptp={col.ptp():.3e}")
PY
```
Expected: 11 slots, every `ptp` at or near 0. **A non-zero spread means the freeze did not reach the solver — stop and fix Task 5 before going further.**

- [ ] **Step 4: Read the scorecard**

Run: `cat .../ik_pilot/session_qc.md` and `python -c "import json;d=json.load(open('.../ik_pilot/session_qc.json'));print(json.dumps(d,indent=2)[:2000])"`

Check: 11 bouts scored, `median_reproj_ratio` is `null` (no rig), coverage names 44 keypoints and no T1L distal, sex reads `female` for this recording, and `ik_output_combined_v1_ik_pilot.h5` exists.

- [ ] **Step 5: Record the findings and commit**

Write `docs/superpowers/plans/2026-09-17-amputation-pilot-notes.md` with the measured wall-clock per stage (from `ik_pilot/timing.json`), the T1L `ptp` values, and the scorecard summary. This sizes `--time` for Task 8's array honestly.

```bash
git add docs/superpowers/plans/2026-09-17-amputation-pilot-notes.md
git commit -m "$(cat <<'EOF'
docs: amputation pilot run findings

One recording end to end, with the T1L qpos spread measured directly --
the only evidence that the DOF freeze reached the solver rather than
merely being configured.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Xw9Rimizss33cv14UaVcaN
EOF
)"
```

- [ ] **Step 6: Scale up**

Only after Step 3 passes:
```bash
scripts/slurm/amputation_array.sh --run-name ik_v1 --dry-run \
  --recordings 2026_07_06_16_55_07,2026_07_06_17_11_20,2026_07_06_17_25_43,2026_07_09_10_22_32,2026_07_15_09_58_38
```
then without `--dry-run`. `2026_07_09_10_22_32` is the heaviest recording (51 bouts, 33,328 frames) and `2026_07_15_09_58_38` is one of the ten with no `index.csv` entry, so this batch exercises both extremes. Then the full 103.
