"""Render a fitted MuJoCo pose from any run root, so a fit can be looked at.

Works for every assay without being told which: a run root is self-describing
(`tracking.viz.runroot`), so the fly count and the keypoint order come from its
own artifacts. A courtship run reports two flies and 50 keypoints, a
single-animal run one fly and 50, an amputation run one fly and 44 -- the same
code path renders all three.

Usage:
    # stills: three views x the chosen frames, tiled into one sheet
    python scripts/viz/render_pose.py --run-root <run> --bout 4 --frames 100 400 800

    # slow-motion video of one bout
    python scripts/viz/render_pose.py --run-root <run> --bout 4 --video --out figures/ik

    # no --bout: the offsets-fit sample, as this script originally rendered
    python scripts/viz/render_pose.py --run-root <run> --frames 60 180 300
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

from tracking.viz.mjcam import add_point_geoms  # noqa: E402
from tracking.viz.runroot import (  # noqa: E402
    discover_bouts,
    discover_flies,
    load_bout_observed,
    load_bout_qpos,
    load_run_anatomy,
)
from tracking.viz.video import write_video  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

# (azimuth, elevation) -- three views, because a single one hides a limb
# folded along the line of sight, which is the pose error worth catching.
VIEWS = {"side": (90.0, -10.0), "top": (90.0, -89.0), "front": (0.0, -10.0)}

# tracking.viz.colors' shared language: cyan = observed/detector, green = fit.
OBSERVED_RGBA = (0.25, 0.80, 0.95, 1.0)
FITTED_RGBA = (0.17, 0.54, 0.24, 1.0)


def resolve_fly(run_root, requested: int | None) -> int:
    """`requested`, or the only fly when the run has just one."""
    flies = discover_flies(run_root)
    if requested is not None:
        if requested not in flies:
            raise SystemExit(f"run has flies {flies}, not fly{requested}")
        return requested
    if len(flies) == 1:
        return flies[0]
    raise SystemExit(f"run has flies {flies}; pass --fly to say which")


def pose_and_observed(run_root, *, bout: int | None, fly: int):
    """`(qpos, observed_or_None, label)` for whichever source was asked for."""
    if bout is not None:
        return (
            load_bout_qpos(run_root, bout=bout, fly=fly),
            load_bout_observed(run_root, bout=bout, fly=fly),
            f"bout_{bout:05d}",
        )
    # No bout: the offsets-fit sample, which is what this script used to read.
    with h5py.File(Path(run_root) / f"offsets_fly{fly}.h5", "r") as f:
        return np.asarray(f["qpos"][:], np.float64), None, "offsets_sample"


def _new_scene(anatomy, size: int):
    model = anatomy.mj_model
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=size, width=size)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    opt.geomgroup[3] = 0  # the marker sites live in group 3; show the BODY
    return model, data, renderer, cam, opt


def render_frame(anatomy, scene, q, observed=None, *, overlay=False) -> np.ndarray:
    """One frame: every view in `VIEWS`, tiled horizontally. RGB uint8."""
    model, data, renderer, cam, opt = scene
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    pts = data.site_xpos[anatomy.site_idxs]
    centre = pts.mean(axis=0)
    span = float(np.linalg.norm(pts - centre, axis=1).max())
    cam.lookat[:] = centre
    cam.distance = max(span * 3.5, 1e-3)

    row = []
    for az, el in VIEWS.values():
        cam.azimuth, cam.elevation = az, el
        renderer.update_scene(data, camera=cam, scene_option=opt)
        if overlay:
            add_point_geoms(renderer.scene, pts, FITTED_RGBA, span * 0.035)
            if observed is not None:
                add_point_geoms(renderer.scene, observed, OBSERVED_RGBA, span * 0.035)
        row.append(renderer.render())
    return np.concatenate(row, axis=1)


def _solvable(qpos, t: int) -> bool:
    return 0 <= t < len(qpos) and bool(np.isfinite(qpos[t]).all())


def render_video(anatomy, scene, qpos, observed, args, out_path: Path) -> int:
    """Write every `--stride`-th solvable frame as an mp4. Returns frame count."""
    idx = [t for t in range(0, len(qpos), args.stride) if _solvable(qpos, t)]
    if not idx:
        raise SystemExit("no solvable frame to render")

    def frames():
        for n, t in enumerate(idx):
            obs = observed[t] if (args.overlay and observed is not None) else None
            # write_video takes BGR (it flips to RGB itself); MuJoCo renders RGB.
            yield render_frame(anatomy, scene, qpos[t], obs, overlay=args.overlay)[:, :, ::-1]
            if n % 50 == 0:
                print(f"  frame {n}/{len(idx)} (t={t})")

    write_video(out_path, frames(), fps=args.fps)
    return len(idx)


def render_stills(anatomy, scene, qpos, observed, args, out_path: Path) -> int:
    """Tile the chosen frames into one sheet. Returns the row count."""
    tiles = []
    for t in args.frames:
        if not _solvable(qpos, t):
            print(f"  frame {t}: skipped (out of range or NaN qpos)")
            continue
        obs = observed[t] if (args.overlay and observed is not None) else None
        tiles.append(render_frame(anatomy, scene, qpos[t], obs, overlay=args.overlay))
        print(f"  frame {t}: rendered {len(VIEWS)} views")
    if not tiles:
        raise SystemExit("nothing rendered")
    imageio.imwrite(out_path, np.concatenate(tiles, axis=0))
    return len(tiles)


def render_comparison(anatomy, args, out_dir: Path) -> Path:
    """Before/after qpos for the same frames, labelled, one row per frame."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    z = np.load(args.compare)
    before, after = z["before"], z["after"]
    observed = None
    if args.overlay and "kp_source" in z:
        with h5py.File(z["kp_source"].item(), "r") as f:
            observed = np.asarray(f["kp_data"][: len(before)], np.float64).reshape(
                len(before), -1, 3
            )
    if "offsets" in z:
        anatomy.mj_model.site_pos[anatomy.site_idxs] = z["offsets"].reshape(-1, 3)

    scene = _new_scene(anatomy, args.size)
    views = list(VIEWS.items())
    fig, axes = plt.subplots(
        len(args.frames), 2 * len(views), figsize=(3.2 * 2 * len(views), 3.2 * len(args.frames))
    )
    axes = np.atleast_2d(axes)
    for r, t in enumerate(args.frames):
        for v, (vname, _) in enumerate(views):
            for c, (label, q) in enumerate((("before", before), ("after", after))):
                obs = observed[t] if (args.overlay and observed is not None) else None
                tiled = render_frame(anatomy, scene, q[t], obs, overlay=args.overlay)
                width = tiled.shape[1] // len(views)
                ax = axes[r, v * 2 + c]
                ax.imshow(tiled[:, v * width : (v + 1) * width])
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
    path = out_dir / "compare_before_after.png"
    fig.savefig(path, dpi=130)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-root", required=True, help="a finished run root")
    ap.add_argument("--fly", type=int, default=None, help="default: the only fly, if there is one")
    ap.add_argument("--bout", type=int, default=None, help="default: the offsets-fit sample")
    ap.add_argument("--frames", type=int, nargs="+", default=[60, 180, 300, 420])
    ap.add_argument("--video", action="store_true", help="write an mp4 instead of a stills sheet")
    ap.add_argument("--stride", type=int, default=4, help="video: keep every Nth frame")
    ap.add_argument("--fps", type=int, default=25, help="video: playback rate")
    ap.add_argument("--size", type=int, default=480)
    ap.add_argument("--out", required=True)
    ap.add_argument("--anatomy", default=str(REPO / "configs/anatomy/v1.yaml"))
    ap.add_argument(
        "--overlay",
        action="store_true",
        help="draw fitted marker sites (green) and observed keypoints (cyan)",
    )
    ap.add_argument("--list", action="store_true", help="print what this run holds, render nothing")
    ap.add_argument(
        "--compare", help="npz with `before`/`after` (T, nq) qpos to render side by side, labelled"
    )
    ap.add_argument(
        "--title",
        default="before vs after",
        help="what a --compare figure IS -- a figure captioned as a "
        "different experiment is worse than no figure",
    )
    ap.add_argument(
        "--metric", default="max change", help="name of the per-row quantity stored as `dmetric`"
    )
    args = ap.parse_args()

    run_root = Path(args.run_root)
    if args.list:
        flies = discover_flies(run_root)
        print(f"run root : {run_root}")
        print(f"flies    : {flies}")
        for f in flies:
            print(f"  fly{f} bouts: {discover_bouts(run_root, fly=f)}")
        return

    fly = resolve_fly(run_root, args.fly)
    anatomy = load_run_anatomy(run_root, fly=fly, anatomy_cfg=args.anatomy)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.compare:
        print(f"wrote {render_comparison(anatomy, args, out_dir)}")
        return

    qpos, observed, label = pose_and_observed(run_root, bout=args.bout, fly=fly)
    print(f"fly{fly}  {label}  {qpos.shape[0]} frames  {len(anatomy.kp_order)} keypoints")

    scene = _new_scene(anatomy, args.size)
    # The suffix keeps an overlay render from silently replacing a plain one.
    suffix = "_overlay" if args.overlay else ""
    if args.video:
        path = out_dir / f"pose_fly{fly}_{label}{suffix}.mp4"
        n = render_video(anatomy, scene, qpos, observed, args, path)
        print(f"wrote {path}  ({n} frames @ {args.fps} fps, stride {args.stride})")
    else:
        path = out_dir / f"pose_fly{fly}_{label}{suffix}.png"
        n = render_stills(anatomy, scene, qpos, observed, args, path)
        print(f"wrote {path}  ({n} frames x {len(VIEWS)} views: {', '.join(VIEWS)})")


if __name__ == "__main__":
    main()
