"""Calibrated multi-camera rig: DLT projection and triangulation, by name.

NumPy DLT reprojection and triangulation -- no torch, no jax -- addressed by
camera NAME rather than integer index.

Calibration directory layout:
  <calib_dir>/
    Cam<id>.yaml   (one file per camera, OpenCV FileStorage YAML)
  Each YAML has a 'projectionMatrix' node: a 3x4 matrix (rows=3, cols=4).

The canonical camera order is `tracking.io.names.load_camera_order`'s glob
order (`sorted(Cam*.yaml)`), so it is defined by the calibration directory
and nothing else -- see the keypoint/camera order warning in CLAUDE.md.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from tracking.io.names import Order, load_camera_order


def _read_projection_matrix(path: str) -> np.ndarray:
    """Read the 'projectionMatrix' node from an OpenCV FileStorage YAML file."""
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    mat = fs.getNode("projectionMatrix").mat()
    fs.release()
    if mat is None or mat.size == 0:
        raise ValueError(f"Node 'projectionMatrix' not found or empty in '{path}'")
    return mat  # (3, 4) float64


class CameraRig:
    """Calibrated rig of cameras, addressed by name, for DLT project/reconstruct.

    Parameters
    ----------
    cameras : Order
        Canonical camera name order (the calibration glob order).
    matrices_f64 : (C, 3, 4) float64
        Untransposed projection matrices, in `cameras` order.
    """

    def __init__(self, cameras: Order, matrices_f64: np.ndarray):
        self.cameras = cameras
        self._matrices_f64 = np.asarray(matrices_f64, dtype=np.float64)

        # (C, 4, 3): each 3x4 matrix transposed to (4,3), float32 -- the
        # layout the neural lifter consumes (p_h @ M gives the homogeneous
        # image point).
        self._matrices_f32 = np.ascontiguousarray(
            self._matrices_f64.transpose(0, 2, 1).astype(np.float32)
        )

    @classmethod
    def from_calib_dir(cls, calib_dir: str | Path) -> CameraRig:
        cameras = load_camera_order(calib_dir)
        calib_dir = Path(calib_dir)
        matrices = np.stack(
            [_read_projection_matrix(str(calib_dir / f"{name}.yaml")) for name in cameras]
        ).astype(np.float64)
        return cls(cameras, matrices)

    @property
    def n_cameras(self) -> int:
        return len(self.cameras)

    @property
    def matrices_f32(self) -> np.ndarray:
        """(C, 4, 3) float32 -- each projection matrix transposed. For the lifter."""
        return self._matrices_f32

    @property
    def matrices_f64(self) -> np.ndarray:
        """(C, 3, 4) float64 -- untransposed projection matrices. For QC/DLT."""
        return self._matrices_f64

    def project(self, xyz: np.ndarray) -> np.ndarray:
        """DLT-project 3D point(s) onto every camera.

        Parameters
        ----------
        xyz : (..., 3) array_like
            3D point(s) in world coordinates (any leading shape, including none).

        Returns
        -------
        uv : (..., C, 2) float64
            Pixel coordinates, perspective-divided.

        Uses float64 and `np.einsum`, NOT a BLAS matmul (which reassociates
        and drifts ~1e-13) and NOT `matrices_f32` (which would move every
        reprojection by ~1e-2 px and silently change every downstream QC
        number) -- see CLAUDE.md's keypoint/camera order + numerics warning.
        """
        xyz = np.asarray(xyz, dtype=np.float64)
        ph = np.concatenate([xyz, np.ones((*xyz.shape[:-1], 1))], axis=-1)
        proj = np.einsum("...k,cjk->...cj", ph, self._matrices_f64)
        return proj[..., :2] / proj[..., 2:3]

    def reconstruct(self, uv: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """Triangulate one 3D point via DLT + SVD from the views marked valid.

        Parameters
        ----------
        uv : (C, 2) array_like
            Observed 2D pixel coordinates, one row per camera (this rig's order).
        valid : (C,) bool array_like
            Which camera observations to use.

        Returns
        -------
        xyz : (3,) float64
            Triangulated 3D point, or `[nan, nan, nan]` if fewer than 2 views
            are valid.
        """
        uv = np.asarray(uv, dtype=np.float64)
        valid = np.asarray(valid, dtype=bool)
        idx = np.flatnonzero(valid)
        if idx.size < 2:
            return np.full(3, np.nan)

        n = idx.size
        A = np.zeros((2 * n, 4), dtype=np.float64)
        for i, cam_idx in enumerate(idx):
            P = self._matrices_f64[cam_idx]  # (3, 4)
            u_uv = uv[cam_idx]  # (2,)
            A[2 * i : 2 * i + 2] = u_uv.reshape(2, 1) * P[2].reshape(1, 4) - P[0:2]

        Vh = np.linalg.svd(A)[2]
        X_h = Vh[-1]  # (4,) homogeneous 3D point
        return (X_h / X_h[3])[:3]

    def reconstruct_batch(self, uv: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """Triangulate N points from per-point-varying valid-camera sets.

        Parameters
        ----------
        uv : (N, C, 2) array_like
            Observed 2D pixel coordinates.
        valid : (N, C) bool array_like
            Which camera observations to use, per point.

        Returns
        -------
        xyz : (N, 3) float64
            Triangulated points; NaN rows where fewer than 2 views are valid.
        """
        uv = np.asarray(uv, dtype=np.float64)
        valid = np.asarray(valid, dtype=bool)
        N = uv.shape[0]
        out = np.empty((N, 3), dtype=np.float64)
        for i in range(N):
            out[i] = self.reconstruct(uv[i], valid[i])
        return out

    def subset(self, names: Sequence[str]) -> CameraRig:
        """Return a rig restricted to the named cameras, in the given order."""
        idx = [self.cameras.index(name) for name in names]
        return CameraRig(Order(names), self._matrices_f64[idx])
