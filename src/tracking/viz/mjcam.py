"""MuJoCo cameras that observe the model from the SAME viewpoint as the rig's
calibrated cameras, so a render and the real video are comparable pixel for
pixel.
"""

from __future__ import annotations

import mujoco
import numpy as np

__all__ = [
    "affine_camera_rows",
    "mat_to_quat",
    "mujoco_camera_from_affine",
    "project_with_mujoco_camera",
    "render_width_for_square_pixels",
    "add_point_geoms",
    "OBSERVED_RGBA",
    "FITTED_RGBA",
]


def affine_camera_rows(cam_mat_4x3):
    """`(m0, o0, m1, o1)` for the `ph @ M` convention: `u = m0.X + o0`."""
    M = np.asarray(cam_mat_4x3, float)
    if M.shape != (4, 3):
        raise ValueError(f"expected a (4, 3) camera matrix, got {M.shape}")
    third = M[:, 2]
    if not (np.allclose(third[:3], 0) and np.isclose(third[3], 1)):
        raise ValueError(
            f"camera is not affine (third column {third} != [0,0,0,1]); this "
            f"builder only handles the rig's affine calibration, and an "
            f"orthographic MuJoCo camera cannot represent a perspective one"
        )
    return M[:3, 0].copy(), float(M[3, 0]), M[:3, 1].copy(), float(M[3, 1])


def mat_to_quat(rot: np.ndarray) -> np.ndarray:
    """`(3, 3)` rotation -> MuJoCo quaternion `(w, x, y, z)`."""
    rot = np.asarray(rot, float)
    q = np.empty(4)
    tr = np.trace(rot)
    if tr > 0:
        k = 0.5 / np.sqrt(1.0 + tr)
        q[:] = (
            0.25 / k,
            (rot[2, 1] - rot[1, 2]) * k,
            (rot[0, 2] - rot[2, 0]) * k,
            (rot[1, 0] - rot[0, 1]) * k,
        )
    else:
        i = int(np.argmax(np.diag(rot)))
        j, k_ = (i + 1) % 3, (i + 2) % 3
        r = np.sqrt(1.0 + rot[i, i] - rot[j, j] - rot[k_, k_])
        v = np.zeros(3)
        v[i] = 0.5 * r
        v[j] = (rot[j, i] + rot[i, j]) / (2 * r)
        v[k_] = (rot[k_, i] + rot[i, k_]) / (2 * r)
        q[0] = (rot[k_, j] - rot[j, k_]) / (2 * r)
        q[1:] = v
    return q / np.linalg.norm(q)


def mujoco_camera_from_affine(cam_mat_4x3, img_wh, s, rot, trans, anchor_model, back_off=None):
    """`(pos, quat, fovy)` for an ORTHOGRAPHIC MuJoCo camera in MODEL space.

    Args:
        cam_mat_4x3: the rig camera in `ph @ M` convention (see
            `affine_camera_rows`).
        img_wh: `(width, height)` of the real image in px.
        s, rot, trans: the model -> world similarity,
            `X_world = s * rot @ X_model + trans` -- i.e. exactly what
            `Bridges.frame(t)` returns.
        anchor_model: a 3-vector in model space to centre the view on.
        back_off: how far to pull the camera back along its own `+z`. An
            orthographic view has no perspective, so this changes ONLY
            clipping; it just has to clear the scene.

    The camera rows are pushed through the similarity into model space, and the
    optical axis is then placed so the anchor lands at the image centre.
    """
    width, height = float(img_wh[0]), float(img_wh[1])
    m0, o0, m1, o1 = affine_camera_rows(cam_mat_4x3)

    # Push the camera rows through the similarity into model space.
    m0m = s * (rot.T @ m0)
    o0m = float(m0 @ trans + o0)
    m1m = s * (rot.T @ m1)
    o1m = float(m1 @ trans + o1)

    right = m0m / np.linalg.norm(m0m)
    up = -m1m / np.linalg.norm(m1m)  # image v grows DOWN, world up is -v
    zc = np.cross(right, up)  # MuJoCo cameras look along -z
    zc /= np.linalg.norm(zc)
    # Re-orthogonalise: the affine rows need not be exactly perpendicular.
    up = np.cross(zc, right)
    up /= np.linalg.norm(up)

    fovy = height / np.linalg.norm(m1m)  # FULL visible height, MODEL units

    # Put the optical axis through the pixel centre: solve u(X) = W/2 and
    # v(X) = H/2 for X = anchor + a*right + b*up.
    a0 = np.array([m0m @ right, m0m @ up])
    a1 = np.array([m1m @ right, m1m @ up])
    rhs = np.array(
        [
            width / 2.0 - (m0m @ anchor_model + o0m),
            height / 2.0 - (m1m @ anchor_model + o1m),
        ]
    )
    ab = np.linalg.solve(np.vstack([a0, a1]), rhs)
    centre = np.asarray(anchor_model, float) + ab[0] * right + ab[1] * up

    if back_off is None:
        back_off = 10.0 * fovy
    pos = centre + back_off * zc
    quat = mat_to_quat(np.column_stack([right, up, zc]))
    return pos, quat, float(fovy)


def render_width_for_square_pixels(cam_mat_4x3, img_wh) -> int:
    """Width to RENDER at, so a resize back to `img_wh[0]` gives the real camera."""
    m0, _o0, m1, _o1 = affine_camera_rows(cam_mat_4x3)
    return int(round(float(img_wh[0]) * np.linalg.norm(m1) / np.linalg.norm(m0)))


def project_with_mujoco_camera(x_model, pos, quat, fovy, img_wh, render_width=None):
    """Where MuJoCo will draw `x_model` `(N, 3)`, in pixels of the FINAL image."""
    width, height = float(img_wh[0]), float(img_wh[1])
    x_gain = 1.0 if render_width is None else width / float(render_width)
    w, x, y, z = np.asarray(quat, float)
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    d = (np.asarray(x_model, float) - np.asarray(pos, float)) @ rot  # into camera axes
    ppu = height / fovy  # px per model unit (square, as MuJoCo renders it)
    return np.column_stack([width / 2.0 + d[:, 0] * ppu * x_gain, height / 2.0 - d[:, 1] * ppu])


# Keypoint colours for geom overlays.
OBSERVED_RGBA = (0.25, 0.80, 0.95, 1.0)  # cyan  = observed / detector
FITTED_RGBA = (0.17, 0.54, 0.24, 1.0)  # green = fit


def add_point_geoms(scene, points, rgba, size):
    """Draw MODEL-space points as spheres in an already-updated `MjvScene`."""
    mat = np.eye(3).flatten()
    for p in np.asarray(points, float):
        if scene.ngeom >= scene.maxgeom or not np.isfinite(p).all():
            continue
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            g,
            int(mujoco.mjtGeom.mjGEOM_SPHERE),
            np.array([size, size, size]),
            p.astype(np.float64),
            mat,
            np.asarray(rgba, np.float32),
        )
        scene.ngeom += 1
