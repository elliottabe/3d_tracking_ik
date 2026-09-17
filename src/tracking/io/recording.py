"""Recording specification: the typed description of one multi-camera session."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tracking.io.names import Order, load_camera_order, require_same


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

    @property
    def has_bout_summary(self) -> bool:
        return self.bouts_csv is not None and self.bouts_csv.exists()

    def video_path(self, camera: str) -> Path:
        self.cameras.index(camera)
        return self.session_dir / f"{camera}.mp4"
