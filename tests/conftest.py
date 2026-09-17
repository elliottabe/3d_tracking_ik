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
    """Build a minimal tier root with both splits populated. Returns its path.

    Ids are unique across the tier's two split files, as a real tier's are:
    `instances_val.json` continues the numbering `instances_train.json` ends on.

    >>> root = make_tier("human", n_frames=2, val_frames=1)        # doctest: +SKIP
    >>> (root / "annotations" / "instances_train.json").exists()   # doctest: +SKIP
    True
    """

    def _make(
        name,
        *,
        n_frames=2,
        recording="rec1",
        calib_seed=0,
        source=None,
        fly_ids=(0,),
        val_frames=1,
        first_frame=100,
        val_first_frame=200,
    ):
        root = tmp_path / name
        (root / "annotations").mkdir(parents=True)
        calib = make_calib_dir(f"{name}_calib", seed=calib_seed)
        group_dir = root / "calibrations" / "A"
        group_dir.mkdir(parents=True)
        for f in sorted(calib.glob("Cam*.yaml")):
            (group_dir / f.name).write_bytes(f.read_bytes())

        def split_coco(frames, img_id, ann_id):
            images, annotations, framesets = [], [], {}
            for frame in frames:
                per_fly_ids = {}
                frame_img_ids = []
                for cam in CAMERAS:
                    rel = f"{recording}/{cam}/Frame_{frame}.jpg"
                    _write_image(root / "images" / rel, img_id)
                    images.append(
                        {
                            "id": img_id,
                            "width": W,
                            "height": H,
                            "recording": recording,
                            "file_name": rel,
                        }
                    )
                    frame_img_ids.append(img_id)
                    for fly in fly_ids:
                        kp = []
                        for k in range(len(KP_NAMES)):
                            kp += [5.0 + k + fly, 6.0 + k + fly, 2]
                        annotations.append(
                            {
                                "id": ann_id,
                                "image_id": img_id,
                                "bbox": [4.0, 5.0, 8.0, 8.0],
                                "keypoints": kp,
                                "num_keypoints": len(KP_NAMES),
                                "fly_id": fly,
                                "sex": "female",
                                "subset": name,
                            }
                        )
                        per_fly_ids.setdefault(fly, []).append(ann_id)
                        ann_id += 1
                    img_id += 1
                for fly in fly_ids:
                    fs = {
                        "recording": recording,
                        "fly_id": fly,
                        "frames": frame_img_ids,
                        "ann_ids": per_fly_ids[fly],
                    }
                    if source:
                        fs.update(source)
                    framesets[f"{recording}/Frame_{frame}/fly{fly}"] = fs
            coco = {
                "keypoint_names": KP_NAMES,
                "skeleton": [],
                "categories": [],
                "images": images,
                "annotations": annotations,
                "framesets": framesets,
            }
            return coco, img_id, ann_id

        train, img_id, ann_id = split_coco(range(first_frame, first_frame + n_frames), 1, 1)
        val, _, _ = split_coco(range(val_first_frame, val_first_frame + val_frames), img_id, ann_id)
        (root / "annotations" / "instances_train.json").write_text(json.dumps(train))
        (root / "annotations" / "instances_val.json").write_text(json.dumps(val))
        (root / "annotations" / "keypoint_names.json").write_text(json.dumps(KP_NAMES))
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "version": name,
                    "calib_groups": ["A"],
                    "recordings": {recording: {"calib_group": "A", "split": "train"}},
                }
            )
        )
        return root

    return _make
