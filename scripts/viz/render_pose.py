"""Render the fitted MuJoCo pose, so the fit can be looked at rather than scored.

Usage:
    python scripts/viz/render_pose.py --fly 1 --frames 120 250 380 \
        --out figures/<date>-pose
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import h5py  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from tracking.inverse_kinematics.anatomy import load_anatomy  # noqa: E402

REF = (
    "/gscratch/portia/eabe/data/Johnson_lab/processed/courtship/Session0/"
    "2025_10_20_13_20_04/pose_maskfree_v2"
)
# (azimuth, elevation) -- three views, because a single one hides a limb
# folded along the line of sight, which is the pose error worth catching.
VIEWS = {"side": (90.0, -10.0), "top": (90.0, -89.0), "front": (0.0, -10.0)}

# tracking.viz.colors' shared language: cyan = observed/detector, green = fit.
OBSERVED_RGBA = (0.25, 0.80, 0.95, 1.0)
FITTED_RGBA = (0.17, 0.54, 0.24, 1.0)


def add_point_geoms(scene, points, rgba, size):
    """Draw world points as spheres in an already-updated MjvScene."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fly", type=int, default=1)
    ap.add_argument("--frames", type=int, nargs="+", default=[60, 180, 300, 420])
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=480)
    ap.add_argument(
        "--title",
        default="before vs after",
        help="what this comparison IS -- a figure captioned as a "
        "different experiment is worse than no figure",
    )
    ap.add_argument(
        "--metric", default="max change", help="name of the per-row quantity stored as `dmetric`"
    )
    ap.add_argument(
        "--overlay",
        action="store_true",
        help="draw fitted marker sites (green) and observed keypoints (cyan)",
    )
    ap.add_argument(
        "--compare", help="npz with `before`/`after` (T, nq) qpos to render side by side, labelled"
    )
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    repo = Path(__file__).resolve().parents[2]
    anatomy = load_anatomy(OmegaConf.load(repo / "configs/anatomy/v1.yaml"))
    model = anatomy.mj_model
    with h5py.File(f"{REF}/offsets_fly{args.fly}.h5", "r") as f:
        qpos = np.asarray(f["qpos"][:], np.float64)
        offsets = np.asarray(f["offsets"][:], np.float64).reshape(-1, 3)
    model.site_pos[anatomy.site_idxs] = offsets

    if args.compare:
        _render_comparison(args, model, anatomy)
        return

    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.size, width=args.size)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    opt.geomgroup[3] = 0  # the marker sites live in group 3; show the BODY

    tiles = []
    for t in args.frames:
        if t >= len(qpos) or not np.isfinite(qpos[t]).all():
            print(f"  frame {t}: skipped (out of range or NaN qpos)")
            continue
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        pts = data.site_xpos[anatomy.site_idxs]
        centre = pts.mean(axis=0)
        extent = float(np.linalg.norm(pts - centre, axis=1).max())
        cam.lookat[:] = centre
        cam.distance = max(extent * 3.5, 1e-3)
        row = []
        for az, el in VIEWS.values():
            cam.azimuth, cam.elevation = az, el
            renderer.update_scene(data, camera=cam, scene_option=opt)
            row.append(renderer.render())
        tiles.append(np.concatenate(row, axis=1))
        print(f"  frame {t}: rendered {len(VIEWS)} views")

    if not tiles:
        raise SystemExit("nothing rendered")
    sheet = np.concatenate(tiles, axis=0)
    sex = "female" if args.fly == 0 else "male"
    path = out / f"pose_fly{args.fly}_{sex}.png"
    imageio.imwrite(path, sheet)
    print(f"wrote {path}  ({len(tiles)} frames x {len(VIEWS)} views: {', '.join(VIEWS)})")


def _render_comparison(args, model, anatomy):
    """Before/after for the same frames, labelled, one row per frame."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    z = np.load(args.compare)
    before, after = z["before"], z["after"]
    observed = None
    if args.overlay:
        src = (
            z["kp_source"].item()
            if "kp_source" in z
            else (f"{REF}/bouts/bout_00028/fly{args.fly}/stac_ik.h5")
        )
        with h5py.File(src, "r") as f:
            observed = np.asarray(f["kp_data"][: len(before)], np.float64).reshape(
                len(before), -1, 3
            )
    if "offsets" in z:
        model.site_pos[anatomy.site_idxs] = z["offsets"].reshape(-1, 3)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.size, width=args.size)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    opt.geomgroup[3] = 0

    views = list(VIEWS.items())
    fig, axes = plt.subplots(
        len(args.frames), 2 * len(views), figsize=(3.2 * 2 * len(views), 3.2 * len(args.frames))
    )
    axes = np.atleast_2d(axes)
    for r, t in enumerate(args.frames):
        for v, (vname, (az, el)) in enumerate(views):
            for c, (label, q) in enumerate((("before", before), ("after", after))):
                data.qpos[:] = q[t]
                mujoco.mj_forward(model, data)
                pts = data.site_xpos[anatomy.site_idxs]
                centre = pts.mean(axis=0)
                cam.lookat[:] = centre
                cam.distance = max(float(np.linalg.norm(pts - centre, axis=1).max()) * 3.0, 1e-3)
                cam.azimuth, cam.elevation = az, el
                renderer.update_scene(data, camera=cam, scene_option=opt)
                if args.overlay:
                    span = float(np.linalg.norm(pts - centre, axis=1).max())
                    add_point_geoms(renderer.scene, pts, FITTED_RGBA, span * 0.035)
                    if observed is not None:
                        add_point_geoms(renderer.scene, observed[t], OBSERVED_RGBA, span * 0.035)
                ax = axes[r, v * 2 + c]
                ax.imshow(renderer.render())
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(f"{vname} -- {label}", fontsize=10)
                if v == 0 and c == 0:
                    key = "dmetric" if "dmetric" in z else ("dwing" if "dwing" in z else None)
                    dv = float(z[key][t]) if key else float("nan")
                    ax.set_ylabel(f"frame {t}\n{args.metric} {dv:.1f} deg", fontsize=9)
        print(f"  frame {t}: rendered")
    title = args.title
    if args.overlay:
        title += "   --   green = fitted marker sites, cyan = observed keypoints"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "wing_refine_before_after.png"
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
