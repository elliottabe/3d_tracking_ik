"""The per-frame model -> world similarity transform ("bridge").

The IK solver's marker sites live in the body model's frame; the 3D keypoints
they were fitted to live in world units. This module fits, per frame,

    world_point = s * (R @ model_point) + t

by Umeyama least squares. Everything downstream inherits these units and this
notion of a missing frame, so a wrong transform here is wrong everywhere.

Two hazards:

- `ok=False` must NEVER be handed a substitute identity transform. An identity
  places the model at the world origin at unit scale, which renders as a
  plausible WRONG pose rather than as missing data. Consume `frame(i)` and
  treat `None` as NaN.
- Fit against re-FK'd sites
  (`postprocess.kinematics.fitted_site_xpos_from_qpos`), NOT `stac_ik.h5`'s
  `marker_sites`, which is FK'd from the INITIAL marker model rather than the
  fitted `offsets` stored beside it -- 2.4e-3 model units apart. A per-frame
  similarity fit tilts under systematically displaced source points: using it
  cost 65-77% of a 1.5-2x reprojection regression on bout 28.

The fit is ~0.8% sensitive in scale to which keypoint array it is given, and
nothing downstream reveals which one was used -- hence `source` is a required,
recorded field.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Bridges", "umeyama", "compute_bridges", "model_to_world", "world_to_model"]


@dataclass(frozen=True)
class Bridges:
    """Per-frame model -> world similarity transforms, `T` frames.

    `s`/`R`/`t` are meaningful only where `ok[t]` is True; elsewhere they hold
    placeholders. Use `frame(i)` rather than the raw arrays, so a missing frame
    cannot be mistaken for a fitted one.
    """

    s: np.ndarray  # (T,)      model -> world scale
    R: np.ndarray  # (T, 3, 3)
    t: np.ndarray  # (T, 3)    world units
    ok: np.ndarray  # (T,) bool
    source: str  # which keypoint array was fitted

    def frame(self, i: int) -> tuple[float, np.ndarray, np.ndarray] | None:
        """`(s, R, t)` for frame `i`, or `None` if `ok[i]` is False.

        The only representation downstream code should consume -- it cannot
        drift from `ok`, unlike reading `s[i]`/`R[i]`/`t[i]` directly.
        """
        if not self.ok[i]:
            return None
        return float(self.s[i]), np.array(self.R[i]), np.array(self.t[i])


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """(s, R, t) minimising ||s R src + t - dst||^2.

    `src` and `dst` are NOT interchangeable: swapping them solves the
    inverse-direction transform, a materially different `(s, R, t)` rather
    than its algebraic inverse re-expressed.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    src_c, dst_c = src - mu_s, dst - mu_d
    cov = dst_c.T @ src_c / len(src)
    U, D, Vt = np.linalg.svd(cov)
    sign = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        sign[2, 2] = -1
    R = U @ sign @ Vt
    s = np.trace(np.diag(D) @ sign) / ((src_c**2).sum() / len(src))
    t = mu_d - s * R @ mu_s
    return s, R, t


def compute_bridges(
    marker_sites: np.ndarray,
    kp3d_units: np.ndarray,
    conf3d: np.ndarray,
    *,
    kp_scale: float,
    source: str,
    min_keypoints: int = 3,
) -> Bridges:
    """Fit one Umeyama similarity transform per frame, model sites -> world keypoints.

    Args:
        marker_sites: `(T, K, 3)` FK'd marker sites in MODEL units.
        kp3d_units: `(T, K, 3)` triangulated keypoints in WORLD units, same
            `(t, k)` indexing as `marker_sites`.
        conf3d: `(T, K)` confidence for `kp3d_units`, gated as `> 0` (not
            `isfinite`) -- a zero but finite confidence must not enter the fit.
        kp_scale: seeds the `s = 1/kp_scale` placeholder for `ok=False` frames.
        source: which keypoint array was fitted (e.g. `"kp3d_filt.npz"`),
            recorded on the result; see the module docstring.
        min_keypoints: minimum usable keypoints for a transform to be
            determined at all.

    Returns:
        A `Bridges` with `ok[t]` True where at least `min_keypoints` keypoints
        were usable (finite `kp3d_units`, `conf3d > 0`, finite `marker_sites`),
        the SVD converged, and the fit was finite. Every other frame is
        `ok=False` with placeholder `s`/`R`/`t` that `frame(i)` refuses.
    """
    marker_sites = np.asarray(marker_sites)
    kp3d_units = np.asarray(kp3d_units)
    conf3d = np.asarray(conf3d)
    n_frames = marker_sites.shape[0]

    s_const = 1.0 / float(kp_scale)
    s_out = np.full((n_frames,), s_const, dtype=np.float32)
    R_out = np.broadcast_to(np.eye(3, dtype=np.float32), (n_frames, 3, 3)).copy()
    t_out = np.zeros((n_frames, 3), dtype=np.float32)
    ok = np.zeros((n_frames,), dtype=bool)

    for frame in range(n_frames):
        sites_t = marker_sites[frame]
        usable = (
            np.isfinite(kp3d_units[frame]).all(-1)
            & (conf3d[frame] > 0)
            & np.isfinite(sites_t).all(-1)
        )
        if usable.sum() < min_keypoints:
            continue
        try:
            s, R, t = umeyama(sites_t[usable], kp3d_units[frame][usable])
        except np.linalg.LinAlgError:
            continue
        if not (np.isfinite(s) and np.isfinite(R).all() and np.isfinite(t).all()):
            continue
        s_out[frame] = s
        R_out[frame] = R
        t_out[frame] = t
        ok[frame] = True

    return Bridges(s=s_out, R=R_out, t=t_out, ok=ok, source=source)


def model_to_world(pts_model: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """`(N, 3)` model-frame points -> world units, via `world = s * (R @ model) + t`."""
    pts_model = np.asarray(pts_model)
    return s * (np.asarray(R) @ pts_model.T).T + np.asarray(t)


def world_to_model(pts_world: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """`(N, 3)` world-unit points -> model frame; the inverse of `model_to_world`."""
    pts_world = np.asarray(pts_world)
    R = np.asarray(R)
    return (R.T @ ((pts_world - np.asarray(t)) / s).T).T
