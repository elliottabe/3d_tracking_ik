"""Thin CLI over `tracking.viz.sidebyside.render_sidebyside` (Task 16).

Usage:
    python scripts/viz/sidebyside_rigcam.py --run <run_root> --bout 28 \
        --fly 0 1 --out figures/<date>-sidebyside
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _bout_start_frame(kp2d_dir: Path, given: int | None) -> int:
    """`--bout-start` if given, else read from `kp2d_dir`'s `mvq_meta.json`."""
    if given is not None:
        return int(given)
    meta_path = Path(kp2d_dir) / "mvq_meta.json"
    if not meta_path.exists():
        raise SystemExit(
            f"--bout-start was not given and {meta_path} does not exist; pass "
            f"--bout-start explicitly for a tree with no mvq_meta.json"
        )
    meta = json.loads(meta_path.read_text())
    if "bout_start_frame" not in meta:
        raise SystemExit(f"{meta_path} has no bout_start_frame; pass --bout-start explicitly")
    return int(meta["bout_start_frame"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument(
        "--kp2d-dir",
        type=Path,
        default=None,
        help="where kp2d.npz/kp3d.npz/mvq_meta.json live; defaults to the run's own bout dir",
    )
    ap.add_argument("--session-dir", type=Path, required=True, help="raw video directory")
    ap.add_argument("--bout", type=int, required=True, help="bout index (dir bouts/bout_XXXXX)")
    ap.add_argument(
        "--bout-start", type=int, default=None, help="override; else read from mvq_meta.json"
    )
    ap.add_argument("--fly", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--frames", type=int, nargs="+", default=None)
    ap.add_argument("--n-preview-frames", type=int, default=8)
    ap.add_argument("--cams", type=str, nargs="+", default=None)
    ap.add_argument("--anatomy", type=Path, default=REPO_ROOT / "configs/anatomy/v1.yaml")
    ap.add_argument("--pad", type=int, default=170)
    ap.add_argument(
        "--geom-size",
        type=float,
        default=0.004,
        help="marker sphere radius in MODEL units (the fly spans ~0.25)",
    )
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from omegaconf import OmegaConf

    from tracking.geometry.rig import CameraRig
    from tracking.inverse_kinematics.anatomy import load_anatomy
    from tracking.viz.sidebyside import render_sidebyside

    bout_dir = Path(args.run) / f"bouts/bout_{args.bout:05d}"
    kp2d_dir = args.kp2d_dir if args.kp2d_dir is not None else bout_dir
    bout_start_frame = _bout_start_frame(kp2d_dir, args.bout_start)

    anatomy = load_anatomy(OmegaConf.load(args.anatomy))
    rig = CameraRig.from_calib_dir(args.session_dir / "calibration")

    for fly in args.fly:
        fly_dir = bout_dir / f"fly{fly}"
        fly_out = args.out / f"fly{fly}"
        fly_out.mkdir(parents=True, exist_ok=True)
        result = render_sidebyside(
            fly_dir,
            fly=fly,
            anatomy=anatomy,
            rig=rig,
            video_dir=args.session_dir,
            bout_start_frame=bout_start_frame,
            video_path=fly_out / "sidebyside.mp4",
            still_path=fly_out / "sidebyside_still.png",
            pose_source_path=fly_out / "sidebyside.pose_source.json",
            cameras=args.cams,
            frames=args.frames,
            n_preview_frames=args.n_preview_frames,
            geom_size=args.geom_size,
            pad=args.pad,
            fps=args.fps,
        )
        print(f"fly{fly}: wrote {result['video_path']} and {result['still_path']}")
        if result["camera_check_px"]:
            worst = max(result["camera_check_px"].values())
            print(f"fly{fly}: camera build check, worst median px = {worst:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
