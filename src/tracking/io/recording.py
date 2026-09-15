"""Recording specification: the typed description of one multi-camera session.

A "recording" is one multi-camera video session of flies: where its videos
and calibration live, how many animals are in it, and its frame rate. This
module reads that description out of a config and refuses two ways it could
silently corrupt everything downstream:

- a missing or non-positive `fps`, which is the sole source of `source_hz`
  stamped into the final analysis dataset;
- a `cameras` list that is not in the calibration glob order, which would
  otherwise permute every camera axis downstream without any error — one
  camera's keypoints plotted onto another camera's image, which still looks
  almost plausible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tracking.io.names import Order, load_camera_order, require_same


@dataclass(frozen=True)
class RecordingSpec:
    """Typed description of one multi-camera recording session."""

    name: str
    session_dir: Path
    calib_dir: Path
    cameras: Order
    num_animals: int
    fps: float
    bouts_csv: Path | None

    @classmethod
    def from_config(cls, cfg) -> RecordingSpec:
        fps = cfg.get("fps", None) if hasattr(cfg, "get") else getattr(cfg, "fps", None)
        if fps is None or not (float(fps) > 0):
            raise ValueError(
                "recording.fps is required and must be > 0; it is the sole source of "
                "source_hz and must not be guessed"
            )

        bouts_csv = cfg.get("bouts_csv", None) if hasattr(cfg, "get") else cfg.bouts_csv
        bouts_csv_path = Path(bouts_csv) if bouts_csv is not None else None

        return cls(
            name=str(cfg.name),
            session_dir=Path(cfg.session_dir),
            calib_dir=Path(cfg.calib_dir),
            cameras=Order(cfg.cameras),
            num_animals=int(cfg.num_animals),
            fps=float(fps),
            bouts_csv=bouts_csv_path,
        )

    def validate(self) -> None:
        if not self.session_dir.is_dir():
            raise ValueError(f"recording.session_dir does not exist: {self.session_dir}")
        if not self.calib_dir.is_dir():
            raise ValueError(f"recording.calib_dir does not exist: {self.calib_dir}")
        require_same(self.cameras, load_camera_order(self.calib_dir), what="camera")

    @property
    def has_bout_summary(self) -> bool:
        return self.bouts_csv is not None and self.bouts_csv.exists()

    def video_path(self, camera: str) -> Path:
        self.cameras.index(camera)
        return self.session_dir / f"{camera}.mp4"
