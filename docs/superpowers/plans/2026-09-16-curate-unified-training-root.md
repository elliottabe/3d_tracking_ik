# Curate: the unified training root — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `tracking.curate` — the subsystem that merges the four existing training tiers into one validated, provenance-carrying, shareable root that both trainers read.

**Architecture:** A Hydra-driven stage pipeline (`merge → import_masks → validate → package`) mirroring `tracking.pipeline.stages`. Geometry goes through the existing `CameraRig`; keypoint and camera axes through `io.names.Order`. SAM masks live in a separable `<root>_masks/` sidecar, never in the core root.

**Tech Stack:** Python 3.13, numpy, Hydra/OmegaConf, opencv (calibration YAML), Pillow, pytest.

**Spec:** `docs/superpowers/specs/2026-09-16-mvq-centerdetect-training-port-design.md`

**Plan 1 of 3.** Plan 2 is MVQ training, Plan 3 CenterDetect training. This plan is complete and testable on its own: its deliverable is a validated unified root plus the tools that build, check and package one.

## Global Constraints

- Branch: `feat/mvq-centerdetect-training`. Do not commit to `main`.
- Python `>=3.13`. Ruff `line-length = 100`, `select = ["E", "W", "F", "I", "UP", "B"]`. Run `ruff check src tests` and `ruff format src tests` before every commit.
- **Comment style:** minimal and concise. Docstrings state what/args/returns and carry a usage example. No paragraph rationale, no multi-line explanatory blocks. This is deliberate — the surrounding code and the source being ported are the opposite, and must not be imitated.
- Never index a keypoint or camera axis by bare integer. Resolve by name through `tracking.io.names.Order`.
- Calibration projection matrices are `(3, 4)` float64 with row 2 (0-based) equal to `[0, 0, 0, 1]` — affine/telecentric.
- `configs/pipeline.yaml` is not modified by this plan.
- Do not modify anything under `src/tracking/detector/`, `inverse_kinematics/`, `preprocess/`, `postprocess/`, `qc/`, or `viz/`.
- Read calibration files only through `tracking.geometry.rig.CameraRig` / `tracking.io.names.load_camera_order`. Do not add a second parser.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/tracking/curate/__init__.py` | Package marker, version-free |
| `src/tracking/curate/calib.py` | Calibration fingerprinting and group labelling |
| `src/tracking/curate/schema.py` | Read/write the root's JSON: manifest, sources, instances |
| `src/tracking/curate/validate.py` | Every root check; returns findings, raises nothing |
| `src/tracking/curate/merge.py` | N tiers → one unified root |
| `src/tracking/curate/package.py` | Tiered zip + CHECKSUMS + DATASET.md |
| `src/tracking/curate/masks/store.py` | Read/write/index the mask sidecar |
| `src/tracking/curate/masks/adapter.py` | SAM output dir → sidecar |
| `src/tracking/curate/stages.py` | Stage registry |
| `src/tracking/curate/run.py` | `python -m tracking.curate` Hydra entry |
| `scripts/check_training_root.py` | Thin CLI over `validate` |
| `configs/curate.yaml`, `configs/curate/default.yaml` | Hydra root + group |
| `tests/conftest.py` | Synthetic-root builder fixture |
| `tests/curate/test_*.py` | One test module per source module |

---

## Task 1: Test scaffolding and calibration fingerprints

**Files:**
- Create: `src/tracking/curate/__init__.py`, `src/tracking/curate/calib.py`
- Create: `tests/conftest.py`, `tests/curate/test_calib.py`
- Modify: `pyproject.toml` (add `pytest` to the `dev` extra, add `[tool.pytest.ini_options]`)

**Interfaces:**
- Consumes: `tracking.geometry.rig.CameraRig`, `tracking.io.names.Order`
- Produces:
  - `calib_fingerprint(calib_dir) -> str` (16-hex)
  - `group_calibrations(dirs: dict[str, str | Path]) -> dict[str, str]`
  - pytest fixture `make_calib_dir(tmp_path, name, matrices) -> Path`
  - pytest fixture `synthetic_root(tmp_path) -> Path` (used from Task 3 on)

The source `calib_groups.py` hardcodes `NUM_CAMERAS = 7` and parses the YAML with a regex. Both are replaced: camera count is inferred per directory, and reading goes through `CameraRig` so the repo has one calibration parser.

- [ ] **Step 1: Add pytest config**

In `pyproject.toml`, change the `dev` extra and append a pytest section:

```toml
[project.optional-dependencies]
dev = ["ruff", "pytest"]
```

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

- [ ] **Step 2: Write the conftest fixtures**

Create `tests/conftest.py`:

```python
"""Shared fixtures: synthetic calibration dirs and a minimal training root."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
from PIL import Image

CAMERAS = ("Cam01", "Cam02", "Cam03")
KP_NAMES = ["Antenna_Base", "EyeL", "EyeR", "Scutellum"]
W, H = 64, 48


def affine_matrix(seed: int) -> np.ndarray:
    """(3, 4) affine DLT matrix; row 2 is [0, 0, 0, 1].

    >>> affine_matrix(0)[2].tolist()
    [0.0, 0.0, 0.0, 1.0]
    """
    rng = np.random.default_rng(seed)
    m = np.zeros((3, 4))
    m[:2, :3] = rng.normal(scale=2.0, size=(2, 3))
    m[:2, 3] = rng.uniform(10, 30, size=2)
    m[2, 3] = 1.0
    return m


@pytest.fixture
def make_calib_dir(tmp_path):
    """Write a calibration dir of OpenCV FileStorage YAMLs and return its path."""

    def _make(name: str, seed: int = 0, cameras=CAMERAS):
        d = tmp_path / name
        d.mkdir(parents=True, exist_ok=True)
        for i, cam in enumerate(cameras):
            fs = cv2.FileStorage(str(d / f"{cam}.yaml"), cv2.FILE_STORAGE_WRITE)
            fs.write("projectionMatrix", affine_matrix(seed * 100 + i))
            fs.release()
        return d

    return _make


def _write_image(path, seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (H, W, 3), dtype=np.uint8)).save(path)


@pytest.fixture
def make_tier(tmp_path, make_calib_dir):
    """Build a minimal tier root. Returns its path.

    >>> root = make_tier("human", n_frames=2)          # doctest: +SKIP
    >>> (root / "annotations" / "instances_train.json").exists()   # doctest: +SKIP
    True
    """

    def _make(name, *, n_frames=2, recording="rec1", calib_seed=0, source=None, fly_ids=(0,)):
        root = tmp_path / name
        (root / "annotations").mkdir(parents=True)
        calib = make_calib_dir(f"{name}_calib", seed=calib_seed)
        group_dir = root / "calibrations" / "A"
        group_dir.mkdir(parents=True)
        for f in sorted(calib.glob("Cam*.yaml")):
            (group_dir / f.name).write_bytes(f.read_bytes())

        images, annotations, framesets = [], [], {}
        img_id = ann_id = 1
        for f in range(n_frames):
            frame = 100 + f
            per_fly_ids = {}
            frame_img_ids = []
            for cam in CAMERAS:
                rel = f"{recording}/{cam}/Frame_{frame}.jpg"
                _write_image(root / "images" / rel, img_id)
                images.append(
                    {"id": img_id, "width": W, "height": H,
                     "recording": recording, "file_name": rel}
                )
                frame_img_ids.append(img_id)
                for fly in fly_ids:
                    kp = []
                    for k in range(len(KP_NAMES)):
                        kp += [5.0 + k + fly, 6.0 + k + fly, 2]
                    annotations.append(
                        {"id": ann_id, "image_id": img_id, "bbox": [4.0, 5.0, 8.0, 8.0],
                         "keypoints": kp, "num_keypoints": len(KP_NAMES),
                         "fly_id": fly, "sex": "female", "subset": name}
                    )
                    per_fly_ids.setdefault(fly, []).append(ann_id)
                    ann_id += 1
                img_id += 1
            for fly in fly_ids:
                fs = {"recording": recording, "fly_id": fly,
                      "frames": frame_img_ids, "ann_ids": per_fly_ids[fly]}
                if source:
                    fs.update(source)
                framesets[f"{recording}/Frame_{frame}/fly{fly}"] = fs

        coco = {"keypoint_names": KP_NAMES, "skeleton": [], "categories": [],
                "images": images, "annotations": annotations, "framesets": framesets}
        (root / "annotations" / "instances_train.json").write_text(json.dumps(coco))
        (root / "annotations" / "instances_val.json").write_text(
            json.dumps({**coco, "images": [], "annotations": [], "framesets": {}})
        )
        (root / "annotations" / "keypoint_names.json").write_text(json.dumps(KP_NAMES))
        (root / "manifest.json").write_text(
            json.dumps({"version": name, "calib_groups": ["A"],
                        "recordings": {recording: {"calib_group": "A", "split": "train"}}})
        )
        return root

    return _make
```

- [ ] **Step 3: Write the failing test**

Create `tests/curate/test_calib.py`:

```python
import pytest

from tracking.curate.calib import calib_fingerprint, group_calibrations


def test_fingerprint_is_stable(make_calib_dir):
    a = make_calib_dir("a", seed=1)
    b = make_calib_dir("b", seed=1)
    assert calib_fingerprint(a) == calib_fingerprint(b)


def test_fingerprint_separates_different_calibrations(make_calib_dir):
    assert calib_fingerprint(make_calib_dir("a", seed=1)) != calib_fingerprint(
        make_calib_dir("b", seed=2)
    )


def test_largest_group_is_A(make_calib_dir):
    dirs = {"r1": make_calib_dir("c1", seed=1), "r2": make_calib_dir("c2", seed=1),
            "r3": make_calib_dir("c3", seed=2)}
    groups = group_calibrations(dirs)
    assert groups["r1"] == groups["r2"] == "A"
    assert groups["r3"] == "B"


def test_camera_count_inferred_not_hardcoded(make_calib_dir):
    d = make_calib_dir("two", seed=1, cameras=("Cam01", "Cam02"))
    assert calib_fingerprint(d)


def test_inconsistent_camera_names_raise(make_calib_dir):
    dirs = {"r1": make_calib_dir("c1", seed=1),
            "r2": make_calib_dir("c2", seed=1, cameras=("Cam01", "Cam09", "Cam03"))}
    with pytest.raises(ValueError, match="camera names"):
        group_calibrations(dirs)
```

- [ ] **Step 4: Run to verify it fails**

Run: `python -m pytest tests/curate/test_calib.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tracking.curate'`

- [ ] **Step 5: Implement**

Create `src/tracking/curate/__init__.py`:

```python
"""Curation: build, check and package the unified training root."""
```

Create `src/tracking/curate/calib.py`:

```python
"""Calibration identity: fingerprint a calibration dir, label recordings by group."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import numpy as np

from tracking.geometry.rig import CameraRig

_ROUND = 6


def calib_fingerprint(calib_dir: str | Path) -> str:
    """16-hex digest of a calibration's rounded coefficients.

    Rounded numerics, not file bytes: the same calibration written with fewer
    decimal places must fingerprint equal.

    >>> calib_fingerprint("calibrations/A")            # doctest: +SKIP
    '3f1a9c2b7d4e5061'
    """
    rig = CameraRig.from_calib_dir(calib_dir)
    payload = repr(np.round(rig.matrices_f64, _ROUND).tolist())
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def group_calibrations(dirs: dict[str, str | Path]) -> dict[str, str]:
    """Map recording name -> group label; the largest group is 'A'.

    Raises if the directories disagree on camera names.

    >>> group_calibrations({"r1": "calib/x", "r2": "calib/y"})    # doctest: +SKIP
    {'r1': 'A', 'r2': 'A'}
    """
    cameras = {rec: CameraRig.from_calib_dir(d).cameras.names for rec, d in dirs.items()}
    distinct = set(cameras.values())
    if len(distinct) > 1:
        raise ValueError(f"calibration dirs disagree on camera names: {sorted(distinct)}")
    fp = {rec: calib_fingerprint(d) for rec, d in dirs.items()}
    counts = Counter(fp.values())
    ordered = sorted(counts, key=lambda h: (-counts[h], min(r for r in fp if fp[r] == h)))
    label = {h: chr(ord("A") + i) for i, h in enumerate(ordered)}
    return {rec: label[h] for rec, h in fp.items()}
```

- [ ] **Step 6: Run to verify it passes**

Run: `python -m pytest tests/curate/test_calib.py -v`
Expected: 5 passed

- [ ] **Step 7: Lint and commit**

```bash
ruff format src/tracking/curate tests && ruff check src/tracking/curate tests
git add pyproject.toml src/tracking/curate tests
git commit -m "feat(curate): calibration fingerprinting and group labelling"
```

---

## Task 2: Root schema read/write

**Files:**
- Create: `src/tracking/curate/schema.py`
- Create: `tests/curate/test_schema.py`

**Interfaces:**
- Consumes: `tracking.io.names.Order`
- Produces:
  - `SourceEntry(id, kind, weight, origin, checkpoint, gates, review, n_framesets, extra)`
  - `load_instances(root, split) -> dict`, `save_instances(root, split, coco) -> None`
  - `load_manifest(root) -> dict`, `save_manifest(root, manifest) -> None`
  - `keypoint_order(root) -> Order`
  - `frameset_field(manifest, frameset, key, default)` — the frameset → recording → root default chain
  - `SOURCE_KINDS = ("human", "pseudo", "negative", "singlefly")`

- [ ] **Step 1: Write the failing test**

Create `tests/curate/test_schema.py`:

```python
import json

import pytest

from tracking.curate import schema
from tracking.io.names import Order, OrderMismatch


def test_keypoint_order_matches_canonical(make_tier):
    root = make_tier("t")
    assert schema.keypoint_order(root) == Order(["Antenna_Base", "EyeL", "EyeR", "Scutellum"])


def test_keypoint_order_rejects_disagreement(make_tier):
    root = make_tier("t")
    (root / "annotations" / "keypoint_names.json").write_text(json.dumps(["A", "B"]))
    with pytest.raises(OrderMismatch, match="keypoint_names"):
        schema.keypoint_order(root)


def test_round_trip_instances(make_tier):
    root = make_tier("t")
    coco = schema.load_instances(root, "train")
    schema.save_instances(root, "train", coco)
    assert schema.load_instances(root, "train") == coco


def test_source_entry_rejects_unknown_kind():
    with pytest.raises(ValueError, match="kind"):
        schema.SourceEntry(id="x", kind="bogus", weight=1.0)


def test_source_entry_round_trips_through_dict():
    e = schema.SourceEntry(id="p", kind="pseudo", weight=0.3, checkpoint="/ckpt",
                           gates={"exist_min": 0.8})
    assert schema.SourceEntry.from_dict("p", e.to_dict()) == e


def test_frameset_field_chain():
    manifest = {"weight": 9.0, "recordings": {"r": {"weight": 5.0}}}
    assert schema.frameset_field(manifest, {"weight": 1.0}, "r", "weight", 0.0) == 1.0
    assert schema.frameset_field(manifest, {}, "r", "weight", 0.0) == 5.0
    assert schema.frameset_field(manifest, {}, "other", "weight", 0.0) == 9.0
    assert schema.frameset_field(manifest, {}, "other", "role", "anchor") == "anchor"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/curate/test_schema.py -v`
Expected: FAIL — `ImportError: cannot import name 'schema'`

- [ ] **Step 3: Implement**

Create `src/tracking/curate/schema.py`:

```python
"""The unified training root's on-disk JSON: manifest, sources, instances."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from tracking.io.names import Order, OrderMismatch

SOURCE_KINDS = ("human", "pseudo", "negative", "singlefly")
SPLITS = ("train", "val")


@dataclass(frozen=True)
class SourceEntry:
    """One row of `manifest.sources`: where a set of framesets came from."""

    id: str
    kind: str
    weight: float = 1.0
    origin: str | None = None
    checkpoint: str | None = None
    gates: dict = field(default_factory=dict)
    review: dict = field(default_factory=dict)
    n_framesets: int = 0
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in SOURCE_KINDS:
            raise ValueError(f"source kind {self.kind!r} not in {SOURCE_KINDS}")

    def to_dict(self) -> dict:
        d = {"kind": self.kind, "weight": self.weight, "n_framesets": self.n_framesets}
        for k in ("origin", "checkpoint"):
            if getattr(self, k) is not None:
                d[k] = getattr(self, k)
        for k in ("gates", "review", "extra"):
            if getattr(self, k):
                d[k] = getattr(self, k)
        return d

    @classmethod
    def from_dict(cls, sid: str, d: dict) -> SourceEntry:
        return cls(
            id=sid, kind=d["kind"], weight=float(d.get("weight", 1.0)),
            origin=d.get("origin"), checkpoint=d.get("checkpoint"),
            gates=d.get("gates", {}), review=d.get("review", {}),
            n_framesets=int(d.get("n_framesets", 0)), extra=d.get("extra", {}),
        )


def instances_path(root: str | Path, split: str) -> Path:
    if split not in SPLITS:
        raise ValueError(f"split {split!r} not in {SPLITS}")
    return Path(root) / "annotations" / f"instances_{split}.json"


def load_instances(root: str | Path, split: str) -> dict:
    """Read `annotations/instances_<split>.json`.

    >>> coco = load_instances("dataset", "train")      # doctest: +SKIP
    >>> sorted(coco)[:3]                               # doctest: +SKIP
    ['annotations', 'categories', 'framesets']
    """
    return json.loads(instances_path(root, split).read_text())


def save_instances(root: str | Path, split: str, coco: dict) -> None:
    p = instances_path(root, split)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(coco))


def load_manifest(root: str | Path) -> dict:
    return json.loads((Path(root) / "manifest.json").read_text())


def save_manifest(root: str | Path, manifest: dict) -> None:
    (Path(root) / "manifest.json").write_text(json.dumps(manifest, indent=2))


def keypoint_order(root: str | Path) -> Order:
    """Canonical keypoint order, cross-checked against every split's copy.

    >>> keypoint_order("dataset")                      # doctest: +SKIP
    Order(names=('Antenna_Base', 'EyeL', ...))
    """
    root = Path(root)
    canon = json.loads((root / "annotations" / "keypoint_names.json").read_text())
    for split in SPLITS:
        p = instances_path(root, split)
        if not p.exists():
            continue
        names = json.loads(p.read_text())["keypoint_names"]
        if names != canon:
            raise OrderMismatch(
                f"instances_{split}.json keypoint_names != annotations/keypoint_names.json"
            )
    return Order(canon)


def frameset_field(manifest: dict, frameset: dict, recording: str, key: str, default):
    """Resolve `key`: frameset, then the recording's manifest entry, then root, then default.

    >>> frameset_field({"weight": 0.3}, {}, "r", "weight", 1.0)
    0.3
    """
    v = frameset.get(key)
    if v is not None:
        return v
    rec = manifest.get("recordings", {}).get(recording, {})
    v = rec.get(key)
    if v is not None:
        return v
    v = manifest.get(key)
    return default if v is None else v


def sources(manifest: dict) -> dict[str, SourceEntry]:
    """`manifest.sources` as `SourceEntry` objects, keyed by id."""
    return {k: SourceEntry.from_dict(k, v) for k, v in manifest.get("sources", {}).items()}
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/curate/test_schema.py -v`
Expected: 6 passed

- [ ] **Step 5: Lint and commit**

```bash
ruff format src/tracking/curate tests && ruff check src/tracking/curate tests
git add src/tracking/curate/schema.py tests/curate/test_schema.py
git commit -m "feat(curate): training-root schema read/write"
```

---

## Task 3: Validator and CLI

**Files:**
- Create: `src/tracking/curate/validate.py`, `scripts/check_training_root.py`
- Create: `tests/curate/test_validate.py`

**Interfaces:**
- Consumes: `schema.*`, `calib.calib_fingerprint`, `tracking.geometry.rig.CameraRig`, `tracking.io.names.load_camera_order`
- Produces:
  - `Finding(level, code, message)` with `level in {"error", "warning", "info"}`
  - `validate_root(root, *, masks_root=None) -> list[Finding]`
  - `format_findings(findings) -> str`
  - `MIN_CAMS = 3`

Findings are returned, never raised: the caller decides. `error` fails a run; `warning` and `info` are reported. The zero-annotation-image count and the two-fly image fraction are `info` — the spec requires reporting them, not failing on them.

- [ ] **Step 1: Write the failing test**

Create `tests/curate/test_validate.py`:

```python
import json

import cv2
import numpy as np

from tracking.curate import validate


def codes(findings, level=None):
    return {f.code for f in findings if level is None or f.level == level}


def test_clean_root_has_no_errors(make_tier):
    assert codes(validate.validate_root(make_tier("t")), "error") == set()


def test_reports_two_fly_fraction_as_info(make_tier):
    findings = validate.validate_root(make_tier("t", fly_ids=(0, 1)))
    info = [f for f in findings if f.code == "two_fly_fraction"]
    assert len(info) == 1 and info[0].level == "info"


def test_keypoint_name_mismatch_is_an_error(make_tier):
    root = make_tier("t")
    (root / "annotations" / "keypoint_names.json").write_text(json.dumps(["A", "B"]))
    assert "keypoint_names_mismatch" in codes(validate.validate_root(root), "error")


def test_symlinked_image_is_an_error(make_tier, tmp_path):
    root = make_tier("t")
    coco = json.loads((root / "annotations" / "instances_train.json").read_text())
    rel = coco["images"][0]["file_name"]
    target = tmp_path / "elsewhere.jpg"
    (root / "images" / rel).rename(target)
    (root / "images" / rel).symlink_to(target)
    assert "image_is_symlink" in codes(validate.validate_root(root), "error")


def test_missing_image_is_an_error(make_tier):
    root = make_tier("t")
    coco = json.loads((root / "annotations" / "instances_train.json").read_text())
    (root / "images" / coco["images"][0]["file_name"]).unlink()
    assert "image_missing" in codes(validate.validate_root(root), "error")


def test_non_affine_calibration_is_an_error(make_tier):
    root = make_tier("t")
    p = root / "calibrations" / "A" / "Cam01.yaml"
    m = np.zeros((3, 4))
    m[:2, :3] = 1.0
    m[2] = [0.1, 0.0, 0.0, 1.0]
    fs = cv2.FileStorage(str(p), cv2.FILE_STORAGE_WRITE)
    fs.write("projectionMatrix", m)
    fs.release()
    assert "calibration_not_affine" in codes(validate.validate_root(root), "error")


def test_ann_ids_length_mismatch_is_an_error(make_tier):
    root = make_tier("t")
    coco = json.loads((root / "annotations" / "instances_train.json").read_text())
    key = next(iter(coco["framesets"]))
    coco["framesets"][key]["ann_ids"] = coco["framesets"][key]["ann_ids"][:1]
    (root / "annotations" / "instances_train.json").write_text(json.dumps(coco))
    assert "ann_ids_length" in codes(validate.validate_root(root), "error")


def test_too_few_resolved_slots_is_an_error(make_tier):
    root = make_tier("t")
    coco = json.loads((root / "annotations" / "instances_train.json").read_text())
    key = next(iter(coco["framesets"]))
    coco["framesets"][key]["ann_ids"] = [coco["framesets"][key]["ann_ids"][0], None, None]
    (root / "annotations" / "instances_train.json").write_text(json.dumps(coco))
    assert "too_few_cameras" in codes(validate.validate_root(root), "error")


def test_negative_without_center3d_is_an_error(make_tier):
    root = make_tier("t")
    coco = json.loads((root / "annotations" / "instances_train.json").read_text())
    key = next(iter(coco["framesets"]))
    coco["framesets"][key]["fly_id"] = -1
    (root / "annotations" / "instances_train.json").write_text(json.dumps(coco))
    assert "negative_without_center3d" in codes(validate.validate_root(root), "error")


def test_unresolved_source_id_is_an_error(make_tier):
    root = make_tier("t", source={"source_id": "nope"})
    assert "unknown_source_id" in codes(validate.validate_root(root), "error")


def test_keypoint_array_length_is_checked(make_tier):
    root = make_tier("t")
    coco = json.loads((root / "annotations" / "instances_train.json").read_text())
    coco["annotations"][0]["keypoints"] = [1.0, 2.0]
    (root / "annotations" / "instances_train.json").write_text(json.dumps(coco))
    assert "keypoints_length" in codes(validate.validate_root(root), "error")


def test_format_findings_mentions_every_error(make_tier):
    root = make_tier("t")
    (root / "annotations" / "keypoint_names.json").write_text(json.dumps(["A"]))
    text = validate.format_findings(validate.validate_root(root))
    assert "keypoint_names_mismatch" in text
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/curate/test_validate.py -v`
Expected: FAIL — `ImportError: cannot import name 'validate'`

- [ ] **Step 3: Implement**

Create `src/tracking/curate/validate.py`:

```python
"""Check a unified training root. Returns findings; never raises on bad data."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tracking.curate import schema
from tracking.geometry.rig import CameraRig
from tracking.io.names import OrderMismatch

MIN_CAMS = 3
LEVELS = ("error", "warning", "info")


@dataclass(frozen=True)
class Finding:
    level: str
    code: str
    message: str


def _err(code, msg):
    return Finding("error", code, msg)


def _check_keypoints(root, out):
    try:
        return schema.keypoint_order(root)
    except OrderMismatch as exc:
        out.append(_err("keypoint_names_mismatch", str(exc)))
    except (OSError, json.JSONDecodeError) as exc:
        out.append(_err("keypoint_names_unreadable", str(exc)))
    return None


def _check_calibrations(root, out):
    """Affine row, one Cam*.yaml per camera, camera names equal across groups."""
    names_seen = {}
    for group_dir in sorted((Path(root) / "calibrations").glob("*")):
        if not group_dir.is_dir():
            continue
        try:
            rig = CameraRig.from_calib_dir(group_dir)
        except (OrderMismatch, ValueError, OSError) as exc:
            out.append(_err("calibration_unreadable", f"{group_dir.name}: {exc}"))
            continue
        names_seen[group_dir.name] = rig.cameras.names
        for cam, m in zip(rig.cameras.names, rig.matrices_f64, strict=True):
            if not np.allclose(m[2], [0.0, 0.0, 0.0, 1.0]):
                out.append(_err(
                    "calibration_not_affine",
                    f"{group_dir.name}/{cam}: projection row 2 is {m[2].tolist()}, "
                    f"expected [0, 0, 0, 1]",
                ))
    if len(set(names_seen.values())) > 1:
        out.append(_err("camera_names_differ", f"calibration groups disagree: {names_seen}"))
    return names_seen


def _check_split(root, split, kp_order, sources, out):
    try:
        coco = schema.load_instances(root, split)
    except FileNotFoundError:
        return
    images = {i["id"]: i for i in coco["images"]}
    anns = {a["id"]: a for a in coco["annotations"]}
    n_kp = len(kp_order) if kp_order is not None else None

    for a in coco["annotations"]:
        if n_kp is not None and len(a.get("keypoints", [])) != 3 * n_kp:
            out.append(_err(
                "keypoints_length",
                f"{split} annotation {a['id']}: {len(a.get('keypoints', []))} values, "
                f"expected {3 * n_kp}",
            ))
            break

    for rel, img_id in ((i["file_name"], i["id"]) for i in coco["images"]):
        p = Path(root) / "images" / rel
        if p.is_symlink():
            out.append(_err("image_is_symlink", f"{split} image {img_id}: {rel} is a symlink"))
            break
        if not p.exists():
            out.append(_err("image_missing", f"{split} image {img_id}: {rel} not on disk"))
            break

    for key, fs in coco["framesets"].items():
        frames, ann_ids = fs.get("frames", []), fs.get("ann_ids", [])
        if len(frames) != len(ann_ids):
            out.append(_err("ann_ids_length",
                            f"{key}: {len(frames)} frames vs {len(ann_ids)} ann_ids"))
            continue
        missing = [i for i in frames if i not in images]
        if missing:
            out.append(_err("frameset_image_missing", f"{key}: unknown image ids {missing[:4]}"))
        unknown = [i for i in ann_ids if i is not None and i not in anns]
        if unknown:
            out.append(_err("frameset_ann_missing", f"{key}: unknown ann ids {unknown[:4]}"))
        if sum(i is not None for i in ann_ids) < MIN_CAMS:
            out.append(_err("too_few_cameras",
                            f"{key}: {sum(i is not None for i in ann_ids)} resolved "
                            f"cameras, need >= {MIN_CAMS}"))
        if int(fs.get("fly_id", 0)) < 0 and fs.get("center3D") is None:
            out.append(_err("negative_without_center3d", f"{key}: negative carries no center3D"))
        sid = fs.get("source_id")
        if sid is not None and sid not in sources:
            out.append(_err("unknown_source_id", f"{key}: source_id {sid!r} not in manifest"))

    per_image = Counter(a["image_id"] for a in coco["annotations"])
    n_zero = len(images) - len(per_image)
    n_two = sum(1 for v in per_image.values() if v >= 2)
    if n_zero:
        out.append(Finding("info", "zero_annotation_images",
                           f"{split}: {n_zero} image(s) carry no annotation"))
    if images:
        out.append(Finding("info", "two_fly_fraction",
                           f"{split}: {n_two}/{len(images)} images have >= 2 animals "
                           f"({100 * n_two / len(images):.1f}%)"))


def validate_root(root: str | Path, *, masks_root: str | Path | None = None) -> list[Finding]:
    """Every check from the format spec, as a flat finding list.

    >>> [f.code for f in validate_root("dataset") if f.level == "error"]   # doctest: +SKIP
    []
    """
    out: list[Finding] = []
    kp_order = _check_keypoints(root, out)
    _check_calibrations(root, out)
    try:
        manifest = schema.load_manifest(root)
    except (OSError, json.JSONDecodeError) as exc:
        out.append(_err("manifest_unreadable", str(exc)))
        return out
    srcs = schema.sources(manifest)
    for sid, entry in srcs.items():
        if entry.kind in ("pseudo", "negative", "singlefly") and not entry.checkpoint:
            out.append(Finding("warning", "source_without_checkpoint",
                               f"source {sid!r} ({entry.kind}) records no checkpoint"))
    for split in schema.SPLITS:
        _check_split(root, split, kp_order, srcs, out)
    if masks_root is not None and not Path(masks_root).exists():
        out.append(Finding("warning", "masks_root_missing",
                           f"{masks_root} does not exist; copy-paste will be off"))
    return out


def format_findings(findings: list[Finding]) -> str:
    """Human-readable report, errors first.

    >>> print(format_findings([Finding("error", "x", "bad")]))   # doctest: +SKIP
    ERROR   x: bad
    1 error, 0 warnings, 0 info
    """
    order = {lvl: i for i, lvl in enumerate(LEVELS)}
    lines = [f"{f.level.upper():<7} {f.code}: {f.message}"
             for f in sorted(findings, key=lambda f: order[f.level])]
    counts = Counter(f.level for f in findings)
    lines.append(f"{counts['error']} error, {counts['warning']} warnings, {counts['info']} info")
    return "\n".join(lines)
```

Create `scripts/check_training_root.py`:

```python
#!/usr/bin/env python
"""Validate a training root and name every violation.

    python scripts/check_training_root.py /path/to/root [--masks /path/to/root_masks]

Exit code 0 when there are no errors, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys

from tracking.curate.validate import format_findings, validate_root


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root")
    ap.add_argument("--masks", default=None)
    args = ap.parse_args()
    findings = validate_root(args.root, masks_root=args.masks)
    print(format_findings(findings))
    return 1 if any(f.level == "error" for f in findings) else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/curate/test_validate.py -v`
Expected: 12 passed

- [ ] **Step 5: Check the validator against the real reference root**

Run:
```bash
python scripts/check_training_root.py \
  /gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v12_export0902
```
Expected: `image_is_symlink` errors (the reference root's images are symlinks — spec §2.4) and a `two_fly_fraction` info line near 1.7%. Record the output in the commit message. This is the check that proves the validator sees the real defect the merge stage exists to fix.

- [ ] **Step 6: Lint and commit**

```bash
ruff format src/tracking/curate scripts tests && ruff check src/tracking/curate scripts tests
git add src/tracking/curate/validate.py scripts/check_training_root.py tests/curate/test_validate.py
git commit -m "feat(curate): training-root validator and CLI"
```

---

## Task 4: Mask sidecar store and adapter

**Files:**
- Create: `src/tracking/curate/masks/__init__.py`, `store.py`, `adapter.py`
- Create: `tests/curate/test_masks.py`

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces:
  - `MaskStore(root)` with `.has(recording, camera, frame) -> bool`, `.load(recording, camera, frame, ann_id) -> np.ndarray | None`, `.save(recording, camera, frame, masks: dict[int, np.ndarray]) -> None`, `.write_index() -> dict`, `.coverage() -> float`
  - `sidecar_path(root) -> Path` — `<root>_masks`
  - `import_masks(src_dir, out_root, *, tool: str) -> dict` (summary)

Sidecar layout: `<root>_masks/<recording>/<camera>/Frame_<n>.npz`, each npz holding `ann_ids` (int array) and `masks` (bool `(N, H, W)`), plus `index.json`.

- [ ] **Step 1: Write the failing test**

Create `tests/curate/test_masks.py`:

```python
import json

import numpy as np

from tracking.curate.masks.adapter import import_masks
from tracking.curate.masks.store import MaskStore, sidecar_path


def test_sidecar_path_is_a_sibling(tmp_path):
    assert sidecar_path(tmp_path / "ds") == tmp_path / "ds_masks"


def test_save_and_load_round_trip(tmp_path):
    store = MaskStore(tmp_path / "m")
    m = np.zeros((4, 5), bool)
    m[1, 2] = True
    store.save("rec", "Cam01", 7, {11: m})
    assert np.array_equal(store.load("rec", "Cam01", 7, 11), m)


def test_load_returns_none_for_unknown_ann(tmp_path):
    store = MaskStore(tmp_path / "m")
    store.save("rec", "Cam01", 7, {11: np.zeros((2, 2), bool)})
    assert store.load("rec", "Cam01", 7, 99) is None


def test_has_is_false_when_absent(tmp_path):
    assert not MaskStore(tmp_path / "m").has("rec", "Cam01", 7)


def test_write_index_records_counts(tmp_path):
    store = MaskStore(tmp_path / "m")
    store.save("rec", "Cam01", 7, {11: np.zeros((2, 2), bool)})
    store.save("rec", "Cam02", 7, {12: np.zeros((2, 2), bool)})
    index = store.write_index(tool="sam3")
    assert index["n_files"] == 2 and index["tool"] == "sam3"
    assert json.loads((tmp_path / "m" / "index.json").read_text())["n_files"] == 2


def test_import_masks_copies_npz_tree(tmp_path):
    src = tmp_path / "src" / "rec" / "Cam01"
    src.mkdir(parents=True)
    np.savez_compressed(src / "Frame_7.npz", ann_ids=np.array([11]),
                        masks=np.zeros((1, 2, 2), bool))
    summary = import_masks(tmp_path / "src", tmp_path / "out_masks", tool="sam3")
    assert summary["n_files"] == 1
    assert MaskStore(tmp_path / "out_masks").has("rec", "Cam01", 7)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/curate/test_masks.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tracking.curate.masks'`

- [ ] **Step 3: Implement**

Create `src/tracking/curate/masks/__init__.py`:

```python
"""Optional SAM mask sidecar. Only consumer: copy-paste donor cutting."""
```

Create `src/tracking/curate/masks/store.py`:

```python
"""The `<root>_masks/` sidecar: per-(recording, camera, frame) instance masks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def sidecar_path(root: str | Path) -> Path:
    """The mask sidecar that belongs to `root`.

    >>> sidecar_path("/data/dataset")
    PosixPath('/data/dataset_masks')
    """
    root = Path(root)
    return root.with_name(root.name + "_masks")


class MaskStore:
    """Read/write instance masks keyed by annotation id.

    >>> store = MaskStore("dataset_masks")                       # doctest: +SKIP
    >>> store.save("rec", "Cam01", 7, {11: mask})                # doctest: +SKIP
    >>> store.load("rec", "Cam01", 7, 11).shape                  # doctest: +SKIP
    (448, 1936)
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def path(self, recording: str, camera: str, frame: int) -> Path:
        return self.root / recording / camera / f"Frame_{int(frame)}.npz"

    def has(self, recording: str, camera: str, frame: int) -> bool:
        return self.path(recording, camera, frame).exists()

    def load(self, recording: str, camera: str, frame: int, ann_id: int):
        """Mask for one annotation, or None when absent."""
        p = self.path(recording, camera, frame)
        if not p.exists():
            return None
        with np.load(p) as z:
            ids = z["ann_ids"].tolist()
            if int(ann_id) not in ids:
                return None
            return z["masks"][ids.index(int(ann_id))].astype(bool)

    def save(self, recording: str, camera: str, frame: int, masks: dict[int, np.ndarray]) -> None:
        p = self.path(recording, camera, frame)
        p.parent.mkdir(parents=True, exist_ok=True)
        ids = sorted(masks)
        np.savez_compressed(
            p,
            ann_ids=np.array(ids, np.int64),
            masks=np.stack([np.asarray(masks[i], bool) for i in ids]),
        )

    def files(self):
        return sorted(self.root.rglob("Frame_*.npz"))

    def coverage(self, n_expected: int) -> float:
        """Fraction of `n_expected` images that have a mask file."""
        return len(self.files()) / n_expected if n_expected else 0.0

    def write_index(self, *, tool: str, source: str | None = None) -> dict:
        """Write `index.json` describing the layer, and return it."""
        files = self.files()
        index = {
            "n_files": len(files),
            "tool": tool,
            "source": source,
            "recordings": sorted({p.relative_to(self.root).parts[0] for p in files}),
        }
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "index.json").write_text(json.dumps(index, indent=2))
        return index
```

Create `src/tracking/curate/masks/adapter.py`:

```python
"""Import a SAM/SAM2/SAM3 output tree into the mask sidecar format."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from tracking.curate.masks.store import MaskStore


def import_masks(src_dir: str | Path, out_root: str | Path, *, tool: str = "sam3") -> dict:
    """Copy `<src>/<rec>/<cam>/Frame_<n>.npz` into the sidecar, checking each file.

    Each npz must hold `ann_ids` (N,) and `masks` (N, H, W).

    >>> import_masks("sam3_out", "dataset_masks", tool="sam3")   # doctest: +SKIP
    {'n_files': 19019, 'tool': 'sam3', ...}
    """
    src_dir, store = Path(src_dir), MaskStore(out_root)
    n = 0
    for p in sorted(src_dir.rglob("Frame_*.npz")):
        rel = p.relative_to(src_dir)
        if len(rel.parts) != 3:
            raise ValueError(f"{p}: expected <recording>/<camera>/Frame_<n>.npz")
        with np.load(p) as z:
            if "ann_ids" not in z or "masks" not in z:
                raise ValueError(f"{p}: needs 'ann_ids' and 'masks' arrays")
            if len(z["ann_ids"]) != len(z["masks"]):
                raise ValueError(f"{p}: ann_ids and masks disagree in length")
        dst = store.root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dst)
        n += 1
    return store.write_index(tool=tool, source=str(src_dir)) | {"n_files": n}
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/curate/test_masks.py -v`
Expected: 6 passed

- [ ] **Step 5: Lint and commit**

```bash
ruff format src/tracking/curate tests && ruff check src/tracking/curate tests
git add src/tracking/curate/masks tests/curate/test_masks.py
git commit -m "feat(curate): mask sidecar store and SAM import adapter"
```

---

## Task 5: Merge — id renumbering and image dereferencing

**Files:**
- Create: `src/tracking/curate/merge.py`
- Create: `tests/curate/test_merge.py`

**Interfaces:**
- Consumes: `schema.*`, `calib.calib_fingerprint`
- Produces:
  - `TierSpec(path, source_id, kind, weight, checkpoint, gates, review)`
  - `merge_tiers(tiers: list[TierSpec], out_root, *, splits=("train", "val"), copy_images=True) -> dict`

Ids are renumbered into one space in tier order. Images are **copied, dereferencing symlinks** (spec §2.4). `file_name` is unchanged, so a recording present in two tiers shares one image file; the merge asserts byte-identity when that happens rather than silently keeping one.

- [ ] **Step 1: Write the failing test**

Create `tests/curate/test_merge.py`:

```python
import json

import pytest

from tracking.curate import schema
from tracking.curate.merge import TierSpec, merge_tiers


def spec(path, sid, kind="human", weight=1.0, **kw):
    return TierSpec(path=path, source_id=sid, kind=kind, weight=weight, **kw)


def test_ids_are_renumbered_without_collision(make_tier, tmp_path):
    a = make_tier("a", recording="rec1")
    b = make_tier("b", recording="rec2")
    merge_tiers([spec(a, "a"), spec(b, "b", kind="pseudo", weight=0.3, checkpoint="/c")],
                tmp_path / "out")
    coco = schema.load_instances(tmp_path / "out", "train")
    ids = [i["id"] for i in coco["images"]]
    assert len(ids) == len(set(ids))
    ann_ids = [a_["id"] for a_ in coco["annotations"]]
    assert len(ann_ids) == len(set(ann_ids))


def test_frameset_references_survive_renumbering(make_tier, tmp_path):
    a = make_tier("a", recording="rec1")
    b = make_tier("b", recording="rec2")
    merge_tiers([spec(a, "a"), spec(b, "b")], tmp_path / "out")
    coco = schema.load_instances(tmp_path / "out", "train")
    images = {i["id"] for i in coco["images"]}
    anns = {x["id"] for x in coco["annotations"]}
    for fs in coco["framesets"].values():
        assert set(fs["frames"]) <= images
        assert {i for i in fs["ann_ids"] if i is not None} <= anns


def test_images_are_dereferenced(make_tier, tmp_path):
    a = make_tier("a", recording="rec1")
    coco = json.loads((a / "annotations" / "instances_train.json").read_text())
    rel = coco["images"][0]["file_name"]
    target = tmp_path / "away.jpg"
    (a / "images" / rel).rename(target)
    (a / "images" / rel).symlink_to(target)
    merge_tiers([spec(a, "a")], tmp_path / "out")
    assert not (tmp_path / "out" / "images" / rel).is_symlink()
    assert (tmp_path / "out" / "images" / rel).exists()


def test_every_frameset_gets_its_source_id(make_tier, tmp_path):
    a = make_tier("a", recording="rec1")
    merge_tiers([spec(a, "human_v1")], tmp_path / "out")
    coco = schema.load_instances(tmp_path / "out", "train")
    assert {fs["source_id"] for fs in coco["framesets"].values()} == {"human_v1"}


def test_duplicate_source_id_is_rejected(make_tier, tmp_path):
    a, b = make_tier("a", recording="rec1"), make_tier("b", recording="rec2")
    with pytest.raises(ValueError, match="duplicate source_id"):
        merge_tiers([spec(a, "x"), spec(b, "x")], tmp_path / "out")


def test_conflicting_image_bytes_are_rejected(make_tier, tmp_path):
    a = make_tier("a", recording="rec1")
    b = make_tier("b", recording="rec1", calib_seed=0)
    coco = json.loads((b / "annotations" / "instances_train.json").read_text())
    (b / "images" / coco["images"][0]["file_name"]).write_bytes(b"different")
    with pytest.raises(ValueError, match="differs between tiers"):
        merge_tiers([spec(a, "a"), spec(b, "b")], tmp_path / "out")


def test_keypoint_name_disagreement_is_rejected(make_tier, tmp_path):
    a, b = make_tier("a", recording="rec1"), make_tier("b", recording="rec2")
    (b / "annotations" / "keypoint_names.json").write_text(json.dumps(["X", "Y"]))
    coco = json.loads((b / "annotations" / "instances_train.json").read_text())
    coco["keypoint_names"] = ["X", "Y"]
    (b / "annotations" / "instances_train.json").write_text(json.dumps(coco))
    with pytest.raises(ValueError, match="keypoint_names"):
        merge_tiers([spec(a, "a"), spec(b, "b")], tmp_path / "out")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/curate/test_merge.py -v`
Expected: FAIL — `ImportError: cannot import name 'merge'`

- [ ] **Step 3: Implement**

Create `src/tracking/curate/merge.py`:

```python
"""Merge training tiers into one unified root with per-frameset provenance."""

from __future__ import annotations

import filecmp
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from tracking.curate import schema
from tracking.curate.calib import calib_fingerprint


@dataclass(frozen=True)
class TierSpec:
    """One input tier and the provenance it contributes."""

    path: str | Path
    source_id: str
    kind: str = "human"
    weight: float = 1.0
    checkpoint: str | None = None
    gates: dict = field(default_factory=dict)
    review: dict = field(default_factory=dict)


def _copy_image(src: Path, dst: Path) -> None:
    """Copy following symlinks; refuse a byte-level conflict at the same path."""
    if dst.exists():
        if not filecmp.cmp(src, dst, shallow=False):
            raise ValueError(f"{dst.name} differs between tiers at the same path: {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)  # follows symlinks


def _merge_calibrations(tiers, out_root):
    """Union calibration groups by fingerprint; return {(tier, old_group): new_group}."""
    out_calib = Path(out_root) / "calibrations"
    out_calib.mkdir(parents=True, exist_ok=True)
    by_fp: dict[str, str] = {}
    mapping: dict[tuple[str, str], str] = {}
    for tier in tiers:
        for d in sorted((Path(tier.path) / "calibrations").glob("*")):
            if not d.is_dir():
                continue
            fp = calib_fingerprint(d)
            if fp not in by_fp:
                label = chr(ord("A") + len(by_fp))
                by_fp[fp] = label
                shutil.copytree(d, out_calib / label, dirs_exist_ok=True)
            mapping[(tier.source_id, d.name)] = by_fp[fp]
    return mapping


def merge_tiers(tiers, out_root, *, splits=schema.SPLITS, copy_images=True) -> dict:
    """Merge `tiers` into `out_root`; return the written manifest.

    >>> merge_tiers([TierSpec("human_root", "human_v12", "human")], "unified")  # doctest: +SKIP
    {'version': 'unified', 'sources': {...}, ...}
    """
    tiers = list(tiers)
    ids = [t.source_id for t in tiers]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate source_id among tiers: {ids}")

    canon = None
    for tier in tiers:
        names = schema.keypoint_order(tier.path).names
        if canon is None:
            canon = names
        elif names != canon:
            raise ValueError(
                f"tier {tier.source_id!r} keypoint_names differ from {tiers[0].source_id!r}"
            )

    out_root = Path(out_root)
    (out_root / "annotations").mkdir(parents=True, exist_ok=True)
    calib_map = _merge_calibrations(tiers, out_root)

    sources: dict[str, dict] = {}
    recordings: dict[str, dict] = {}
    counts = dict.fromkeys(ids, 0)

    for split in splits:
        images, annotations, framesets = [], [], {}
        next_img = next_ann = 1
        for tier in tiers:
            try:
                coco = schema.load_instances(tier.path, split)
            except FileNotFoundError:
                continue
            man = schema.load_manifest(tier.path)
            img_map, ann_map = {}, {}
            for im in coco["images"]:
                img_map[im["id"]] = next_img
                images.append({**im, "id": next_img})
                if copy_images:
                    _copy_image(Path(tier.path) / "images" / im["file_name"],
                                out_root / "images" / im["file_name"])
                next_img += 1
            for an in coco["annotations"]:
                ann_map[an["id"]] = next_ann
                annotations.append({**an, "id": next_ann, "image_id": img_map[an["image_id"]]})
                next_ann += 1
            for key, fs in coco["framesets"].items():
                rec = fs["recording"]
                old_group = man.get("recordings", {}).get(rec, {}).get("calib_group", "A")
                recordings.setdefault(rec, {})["calib_group"] = calib_map[
                    (tier.source_id, old_group)
                ]
                framesets[f"{tier.source_id}/{key}"] = {
                    **fs,
                    "frames": [img_map[i] for i in fs["frames"]],
                    "ann_ids": [None if i is None else ann_map[i] for i in fs["ann_ids"]],
                    "source_id": tier.source_id,
                    "source": fs.get("source", tier.kind),
                    "weight": fs.get("weight", tier.weight),
                }
                counts[tier.source_id] += 1
        schema.save_instances(out_root, split, {
            "keypoint_names": list(canon), "skeleton": [], "categories": [],
            "images": images, "annotations": annotations, "framesets": framesets,
        })

    (out_root / "annotations" / "keypoint_names.json").write_text(json.dumps(list(canon)))
    for tier in tiers:
        sources[tier.source_id] = schema.SourceEntry(
            id=tier.source_id, kind=tier.kind, weight=tier.weight,
            origin=str(tier.path), checkpoint=tier.checkpoint,
            gates=tier.gates, review=tier.review, n_framesets=counts[tier.source_id],
        ).to_dict()
    manifest = {
        "version": out_root.name,
        "sources": sources,
        "calib_groups": sorted({v for v in calib_map.values()}),
        "recordings": recordings,
    }
    schema.save_manifest(out_root, manifest)
    return manifest
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/curate/test_merge.py -v`
Expected: 7 passed

- [ ] **Step 5: Verify a merged root validates clean**

Add to `tests/curate/test_merge.py`:

```python
def test_merged_root_validates_clean(make_tier, tmp_path):
    from tracking.curate.validate import validate_root

    a, b = make_tier("a", recording="rec1"), make_tier("b", recording="rec2")
    merge_tiers([spec(a, "a"), spec(b, "b", kind="pseudo", weight=0.3, checkpoint="/c")],
                tmp_path / "out")
    assert [f for f in validate_root(tmp_path / "out") if f.level == "error"] == []
```

Run: `python -m pytest tests/curate/test_merge.py -v`
Expected: 8 passed

- [ ] **Step 6: Lint and commit**

```bash
ruff format src/tracking/curate tests && ruff check src/tracking/curate tests
git add src/tracking/curate/merge.py tests/curate/test_merge.py
git commit -m "feat(curate): merge tiers into a unified root with provenance"
```

---

## Task 6: Package — tiered archives

**Files:**
- Create: `src/tracking/curate/package.py`
- Create: `tests/curate/test_package.py`

**Interfaces:**
- Consumes: `schema.*`, `validate.validate_root`, `masks.store.sidecar_path`
- Produces:
  - `package_root(root, out_zip, *, tier="all", masks_root=None) -> dict`
  - `TIERS = ("human", "all", "masks")`
  - `dataset_markdown(manifest, counts) -> str`

`tier="human"` keeps only framesets whose source kind is `human`, and only the images those framesets reference. Packaging refuses an invalid root.

- [ ] **Step 1: Write the failing test**

Create `tests/curate/test_package.py`:

```python
import json
import zipfile

import pytest

from tracking.curate.merge import TierSpec, merge_tiers
from tracking.curate.package import TIERS, package_root


def build(make_tier, tmp_path):
    a = make_tier("a", recording="rec1")
    b = make_tier("b", recording="rec2")
    out = tmp_path / "unified"
    merge_tiers(
        [TierSpec(a, "human_v1", "human", 1.0),
         TierSpec(b, "pseudo_v1", "pseudo", 0.3, checkpoint="/ckpt")],
        out,
    )
    return out


def test_tiers_are_named(make_tier, tmp_path):
    assert TIERS == ("human", "all", "masks")


def test_all_tier_contains_both_sources(make_tier, tmp_path):
    root = build(make_tier, tmp_path)
    summary = package_root(root, tmp_path / "all.zip", tier="all")
    assert summary["n_framesets"] == 4
    with zipfile.ZipFile(tmp_path / "all.zip") as z:
        assert any(n.endswith("DATASET.md") for n in z.namelist())
        assert any(n.endswith("CHECKSUMS") for n in z.namelist())


def test_human_tier_drops_pseudo_framesets(make_tier, tmp_path):
    root = build(make_tier, tmp_path)
    summary = package_root(root, tmp_path / "human.zip", tier="human")
    assert summary["n_framesets"] == 2
    with zipfile.ZipFile(tmp_path / "human.zip") as z:
        coco = json.loads(z.read("annotations/instances_train.json"))
        assert {fs["source_id"] for fs in coco["framesets"].values()} == {"human_v1"}
        assert all("rec2" not in n for n in z.namelist())


def test_invalid_root_is_refused(make_tier, tmp_path):
    root = build(make_tier, tmp_path)
    (root / "annotations" / "keypoint_names.json").write_text(json.dumps(["Z"]))
    with pytest.raises(ValueError, match="refusing to package"):
        package_root(root, tmp_path / "bad.zip", tier="all")


def test_checksums_cover_every_member(make_tier, tmp_path):
    root = build(make_tier, tmp_path)
    package_root(root, tmp_path / "all.zip", tier="all")
    with zipfile.ZipFile(tmp_path / "all.zip") as z:
        listed = {line.split("  ", 1)[1] for line in z.read("CHECKSUMS").decode().splitlines()}
        members = {n for n in z.namelist() if n not in ("CHECKSUMS", "DATASET.md")}
    assert listed == members
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/curate/test_package.py -v`
Expected: FAIL — `ImportError: cannot import name 'package'`

- [ ] **Step 3: Implement**

Create `src/tracking/curate/package.py`:

```python
"""Package a validated root as a tiered zip with checksums and a data sheet."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from tracking.curate import schema
from tracking.curate.masks.store import sidecar_path
from tracking.curate.validate import format_findings, validate_root

TIERS = ("human", "all", "masks")


def _subset(coco: dict, keep_ids: set[str]) -> dict:
    """Restrict a split to `keep_ids` source ids, dropping now-unreferenced rows."""
    framesets = {k: v for k, v in coco["framesets"].items() if v.get("source_id") in keep_ids}
    img_ids = {i for fs in framesets.values() for i in fs["frames"]}
    ann_ids = {i for fs in framesets.values() for i in fs["ann_ids"] if i is not None}
    return {
        **coco,
        "images": [i for i in coco["images"] if i["id"] in img_ids],
        "annotations": [a for a in coco["annotations"] if a["id"] in ann_ids],
        "framesets": framesets,
    }


def dataset_markdown(manifest: dict, counts: dict) -> str:
    """The DATASET.md that travels inside the archive."""
    lines = [f"# {manifest.get('version', 'training root')}", "", "## Sources", "",
             "| id | kind | weight | framesets | checkpoint |",
             "| --- | --- | --- | --- | --- |"]
    for sid, s in sorted(manifest.get("sources", {}).items()):
        lines.append(f"| {sid} | {s['kind']} | {s.get('weight', 1.0)} | "
                     f"{s.get('n_framesets', 0)} | {s.get('checkpoint') or '-'} |")
    lines += ["", "## Contents", ""]
    lines += [f"- {k}: {v}" for k, v in sorted(counts.items())]
    lines += ["", "Format: `docs/training.md`.", ""]
    return "\n".join(lines)


def package_root(root, out_zip, *, tier="all", masks_root=None) -> dict:
    """Validate, then write a tiered zip beside `CHECKSUMS` and `DATASET.md`.

    >>> package_root("unified", "human.zip", tier="human")       # doctest: +SKIP
    {'tier': 'human', 'n_framesets': 2814, 'n_images': 19131}
    """
    if tier not in TIERS:
        raise ValueError(f"tier {tier!r} not in {TIERS}")
    root, out_zip = Path(root), Path(out_zip)
    findings = validate_root(root, masks_root=masks_root)
    if any(f.level == "error" for f in findings):
        raise ValueError(f"refusing to package an invalid root:\n{format_findings(findings)}")

    manifest = schema.load_manifest(root)
    if tier == "masks":
        src = Path(masks_root) if masks_root else sidecar_path(root)
        members = {str(p.relative_to(src)): p for p in sorted(src.rglob("*")) if p.is_file()}
        return _write_zip(out_zip, members, manifest, {"mask files": len(members)}, tier)

    keep = {sid for sid, s in manifest.get("sources", {}).items()
            if tier == "all" or s["kind"] == "human"}
    members: dict[str, Path] = {}
    staged: dict[str, bytes] = {}
    n_fs = n_img = 0
    for split in schema.SPLITS:
        try:
            coco = schema.load_instances(root, split)
        except FileNotFoundError:
            continue
        sub = _subset(coco, keep)
        staged[f"annotations/instances_{split}.json"] = json.dumps(sub).encode()
        n_fs += len(sub["framesets"])
        n_img += len(sub["images"])
        for im in sub["images"]:
            members[f"images/{im['file_name']}"] = root / "images" / im["file_name"]

    staged["annotations/keypoint_names.json"] = (
        root / "annotations" / "keypoint_names.json"
    ).read_bytes()
    kept_sources = {k: v for k, v in manifest.get("sources", {}).items() if k in keep}
    staged["manifest.json"] = json.dumps({**manifest, "sources": kept_sources}, indent=2).encode()
    for p in sorted((root / "calibrations").rglob("*")):
        if p.is_file():
            members[str(p.relative_to(root))] = p

    counts = {"framesets": n_fs, "images": n_img, "sources": len(kept_sources)}
    return _write_zip(out_zip, members, {**manifest, "sources": kept_sources}, counts, tier,
                      staged=staged)


def _write_zip(out_zip: Path, members: dict[str, Path], manifest, counts, tier,
               *, staged: dict[str, bytes] | None = None) -> dict:
    staged = staged or {}
    digests = []
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in sorted(staged.items()):
            z.writestr(name, data)
            digests.append((hashlib.sha256(data).hexdigest(), name))
        for name, path in sorted(members.items()):
            data = path.read_bytes()
            z.writestr(name, data)
            digests.append((hashlib.sha256(data).hexdigest(), name))
        z.writestr("CHECKSUMS", "".join(f"{h}  {n}\n" for h, n in sorted(digests, key=lambda x: x[1])))
        z.writestr("DATASET.md", dataset_markdown(manifest, counts))
    return {"tier": tier, "zip": str(out_zip), "n_members": len(digests), **{
        f"n_{k}": v for k, v in counts.items()
    }}
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/curate/test_package.py -v`
Expected: 5 passed

Note: `test_all_tier_contains_both_sources` and `test_human_tier_drops_pseudo_framesets` assert `n_framesets` of 4 and 2 — the fixture writes 2 frames × 1 fly per tier, so two tiers give 4 framesets total and the human tier gives 2.

- [ ] **Step 5: Lint and commit**

```bash
ruff format src/tracking/curate tests && ruff check src/tracking/curate tests
git add src/tracking/curate/package.py tests/curate/test_package.py
git commit -m "feat(curate): tiered packaging with checksums and data sheet"
```

---

## Task 7: Stage registry, Hydra entry and configs

**Files:**
- Create: `src/tracking/curate/stages.py`, `src/tracking/curate/run.py`
- Create: `configs/curate.yaml`, `configs/curate/default.yaml`
- Modify: `configs/paths/hyak.yaml`, `configs/paths/template.yaml` (add three keys)
- Create: `tests/curate/test_stages.py`

**Interfaces:**
- Consumes: `merge.merge_tiers`, `validate.validate_root`, `package.package_root`, `masks.adapter.import_masks`
- Produces:
  - `STAGES: tuple[Stage, ...]`, `Stage(name, reads, writes)`, `ordered(names) -> list[str]`, `stage_by_name(name) -> Stage`
  - `python -m tracking.curate stages=[merge,validate,package]`

- [ ] **Step 1: Write the failing test**

Create `tests/curate/test_stages.py`:

```python
import pytest

from tracking.curate.stages import STAGES, ordered, stage_by_name


def test_declaration_order_is_dependency_order():
    assert [s.name for s in STAGES] == ["merge", "import_masks", "validate", "package"]


def test_ordered_sorts_into_declaration_order():
    assert ordered(["package", "merge", "validate"]) == ["merge", "validate", "package"]


def test_ordered_rejects_unknown_stage():
    with pytest.raises(KeyError, match="nope"):
        ordered(["nope"])


def test_stage_by_name_round_trips():
    assert stage_by_name("merge").name == "merge"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/curate/test_stages.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tracking.curate.stages'`

- [ ] **Step 3: Implement the registry**

Create `src/tracking/curate/stages.py`:

```python
"""The curation stage registry. Declaration order is dependency order."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Stage", "STAGES", "stage_by_name", "ordered"]


@dataclass(frozen=True)
class Stage:
    name: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]


STAGES: tuple[Stage, ...] = (
    Stage("merge", ("tiers",), ("annotations/", "images/", "calibrations/", "manifest.json")),
    Stage("import_masks", ("sam_dir",), ("<root>_masks/",)),
    Stage("validate", ("manifest.json",), ("report",)),
    Stage("package", ("manifest.json",), ("zip",)),
)

_BY_NAME = {s.name: s for s in STAGES}


def stage_by_name(name: str) -> Stage:
    """>>> stage_by_name("merge").name
    'merge'
    """
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError(f"{name!r} is not a curate stage; known: {list(_BY_NAME)}") from None


def ordered(names) -> list[str]:
    """Requested stage names, sorted into declaration order.

    >>> ordered(["package", "merge"])
    ['merge', 'package']
    """
    wanted = {stage_by_name(n).name for n in names}
    return [s.name for s in STAGES if s.name in wanted]
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/curate/test_stages.py -v`
Expected: 4 passed

- [ ] **Step 5: Add the path keys**

Append to `configs/paths/hyak.yaml`:

```yaml
# Curated training root + its optional mask sidecar, and where training runs land.
train_data_root: "${paths.data_dir}/red_data/unified_v2"
masks_root: "${paths.train_data_root}_masks"
runs_root: "${paths.data_dir}"
```

Append to `configs/paths/template.yaml`:

```yaml
# Curated training root + its optional mask sidecar, and where training runs land.
train_data_root: "/CHANGEME/path/to/unified_training_root"
masks_root: "${paths.train_data_root}_masks"
runs_root: "/CHANGEME/path/to/runs"
```

- [ ] **Step 6: Write the configs**

Create `configs/curate.yaml`:

```yaml
# Hydra root for the curation driver.
#
#   python -m tracking.curate paths=hyak stages=[merge,validate,package]
#   python -m tracking.curate paths=hyak stages=[validate] dry_run=true
defaults:
  - _self_
  - paths: hyak
  - curate: default

stages: [merge, validate, package]
dry_run: false

hydra:
  job:
    chdir: false
  run:
    dir: ${paths.base_dir}/.hydra-logs/curate/${now:%Y-%m-%d}/${now:%H-%M-%S}
```

Create `configs/curate/default.yaml`:

```yaml
# @package curate
out_root: ${paths.train_data_root}
masks_root: ${paths.masks_root}

# Input tiers, merged in this order. `path` is a root in the documented format.
tiers:
  - path: "${paths.data_dir}/red_data/red_data_3d_v12_export0902"
    source_id: human_v12_0902
    kind: human
    weight: 1.0
  - path: "${paths.data_dir}/red_data_3d_v12_pseudo_p3b_20260905"
    source_id: pseudo_p3b_0905
    kind: pseudo
    weight: 0.3
  - path: "${paths.data_dir}/red_data_3d_v12_pseudo_negatives_20260905"
    source_id: negatives_0905
    kind: negative
    weight: 1.0
  - path: "${paths.data_dir}/red_data_3d_v12_pseudo_singlefly_20260905"
    source_id: singlefly_0905
    kind: singlefly
    weight: 0.3

# import_masks
sam_dir: null
mask_tool: sam3

# package
package_tier: all
package_zip: "${paths.base_dir}/unified_v2_${curate.package_tier}.zip"
```

- [ ] **Step 7: Implement the driver**

Create `src/tracking/curate/run.py`:

```python
"""`python -m tracking.curate` -- the curation Hydra entry point."""

from __future__ import annotations

import sys

import hydra
from omegaconf import DictConfig, OmegaConf

import tracking.utils.path_utils  # noqa: F401  registers ${repo_root:}
from tracking.curate.masks.adapter import import_masks
from tracking.curate.merge import TierSpec, merge_tiers
from tracking.curate.package import package_root
from tracking.curate.stages import ordered
from tracking.curate.validate import format_findings, validate_root

__all__ = ["main"]


def _tiers(cfg) -> list[TierSpec]:
    return [
        TierSpec(
            path=t["path"], source_id=t["source_id"], kind=t.get("kind", "human"),
            weight=float(t.get("weight", 1.0)), checkpoint=t.get("checkpoint"),
            gates=dict(t.get("gates") or {}), review=dict(t.get("review") or {}),
        )
        for t in OmegaConf.to_container(cfg.curate.tiers, resolve=True)
    ]


def _run_stage(name, cfg) -> int:
    c = cfg.curate
    if name == "merge":
        manifest = merge_tiers(_tiers(cfg), c.out_root)
        print(f"[curate] merged {len(manifest['sources'])} source(s) into {c.out_root}")
    elif name == "import_masks":
        if not c.sam_dir:
            print("[curate] import_masks: curate.sam_dir is null, skipping")
            return 0
        print(f"[curate] {import_masks(c.sam_dir, c.masks_root, tool=c.mask_tool)}")
    elif name == "validate":
        findings = validate_root(c.out_root, masks_root=c.masks_root)
        print(format_findings(findings))
        return 1 if any(f.level == "error" for f in findings) else 0
    elif name == "package":
        print(f"[curate] {package_root(c.out_root, c.package_zip, tier=c.package_tier, masks_root=c.masks_root)}")
    return 0


@hydra.main(version_base=None, config_path="../../../configs", config_name="curate")
def main(cfg: DictConfig) -> int:
    names = ordered(cfg.stages)
    if cfg.dry_run:
        print(f"[curate] stages (dependency order): {names}")
        print(f"[curate] out_root = {cfg.curate.out_root}")
        return 0
    for name in names:
        print(f"[curate] === {name} ===")
        rc = _run_stage(name, cfg)
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Create `src/tracking/curate/__main__.py`:

```python
import sys

from tracking.curate.run import main

sys.exit(main())
```

- [ ] **Step 8: Verify the driver resolves**

Run: `python -m tracking.curate paths=hyak dry_run=true`
Expected: prints `stages (dependency order): ['merge', 'validate', 'package']` and the resolved `out_root`, creating no output root.

- [ ] **Step 9: Lint and commit**

```bash
ruff format src/tracking/curate tests && ruff check src/tracking/curate tests
git add src/tracking/curate configs tests/curate/test_stages.py
git commit -m "feat(curate): stage registry, Hydra driver and configs"
```

---

## Task 8: Format documentation

**Files:**
- Create: `docs/training.md`
- Modify: `README.md` (add a Training data section pointing at it)

**Interfaces:**
- Consumes: everything above
- Produces: no code

- [ ] **Step 1: Write `docs/training.md`**

Cover, in this order, with a runnable command for each: the on-disk format (the directory tree from spec §5.1, the `annotations[]` and `framesets{}` field tables, the `center3D`-on-negatives rule, the affine calibration requirement, the note that `manifest.recordings[*].has_masks` is stale and unused); the `sources` table and `source_id` provenance (spec §5.2); the mask sidecar layout, its single consumer, and that absence means copy-paste off (spec §5.3); then `merge`, `import_masks`, `validate`, `package` with the exact CLI invocations from Task 7's configs; then the tier sizes table from spec §5.4.

State plainly, as its own short section: **this repository does not generate pseudo-labels** — the pseudo tiers are input data with recorded provenance, and the process that produced them (the P3b bootstrap checkpoint and its lift roots) no longer exists.

- [ ] **Step 2: Link it from the README**

Insert a section after **Outputs**, before **Figures**:

```markdown
## Training data

Training reads a single curated root that merges human labels with the
pseudo-label tiers, each frameset carrying its own provenance. The format, the
optional SAM mask sidecar, and the merge/validate/package tools are documented
in [`docs/training.md`](docs/training.md).

```bash
python -m tracking.curate paths=mymachine stages=[merge,validate,package]
python scripts/check_training_root.py <root>
```

This repository does not generate pseudo-labels; the tiers are inputs.
```

- [ ] **Step 3: Verify every documented command runs**

Run each CLI in `docs/training.md` with `--help` or `dry_run=true`:
```bash
python -m tracking.curate paths=hyak dry_run=true
python scripts/check_training_root.py --help
```
Expected: both succeed. Fix any command that does not match the implementation.

- [ ] **Step 4: Run the whole suite and lint**

```bash
python -m pytest tests -v
ruff format --check src tests && ruff check src tests
```
Expected: all tests pass, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add docs/training.md README.md
git commit -m "docs: training-root format, curation tools and provenance"
```

---

## Self-Review

**Spec coverage.** §5.1 format → Tasks 2, 3, 8. §5.2 provenance and the `ConcatWindowDataset` retirement rationale → Task 5 (`source_id` on every frameset; the sampler change itself belongs to Plan 2). §5.3 mask sidecar → Task 4. §5.4 sizes and packaging → Task 6. §6 stages → Task 7. §6.1 validator checks → Task 3, one test per check. §2.4 symlinks → Task 3 (detects) and Task 5 (fixes). §2.6 `build_v5` not ported → nothing to do. §8 config and path keys → Task 7.

**Not covered here, by design:** §4.1 training modules, §7 training, §9 the three verification tests, §2.8 calibration. Those are Plans 2 and 3. The geometry-parity test (spec §9.2) belongs with Plan 2's window dataset, which is the first consumer of `CameraRig` in a training path; Task 1 here already routes calibration reading through `CameraRig`, so no second parser exists for it to diverge from.

**Placeholder scan:** clean — every code step carries real code, every test asserts a named behaviour, no "similar to Task N".

**Type consistency:** `TierSpec` fields match between Task 5's definition and Task 7's `_tiers`. `Finding(level, code, message)` matches between Task 3's definition, its tests and Task 7's driver. `SourceEntry.to_dict()` output keys (`kind`, `weight`, `n_framesets`, `checkpoint`) match what Task 6's `dataset_markdown` and `_subset` read. `MaskStore.write_index(tool=...)` matches `import_masks`. `schema.SPLITS` is used consistently in Tasks 2, 3, 5, 6.

**Fixed during review:** `merge.py` originally wrote `keypoint_names.json` via `schema.json.dumps`, relying on `schema` re-exporting `json`. It now imports `json` directly.
