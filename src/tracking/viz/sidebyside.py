"""Side by side: the real camera with keypoints, and the MuJoCo IK render from
the SAME camera with the same keypoints.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2  # noqa: E402
import h5py  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from tracking.inverse_kinematics.bridge import umeyama, world_to_model  # noqa: E402
from tracking.io.names import Order  # noqa: E402
from tracking.postprocess.kinematics import fitted_site_xpos_from_qpos  # noqa: E402
from tracking.preprocess.filter import DEFAULT_FILTER_CFG, filter_kp3d  # noqa: E402
from tracking.viz.colors import PALETTE, leg_chains  # noqa: E402
from tracking.viz.mjcam import (  # noqa: E402
    FITTED_RGBA,
    OBSERVED_RGBA,
    add_point_geoms,
    mujoco_camera_from_affine,
    project_with_mujoco_camera,
    render_width_for_square_pixels,
)
from tracking.viz.video import write_video  # noqa: E402

__all__ = ["render_sidebyside"]


def _build_model(xml, width, height):
    """The anatomy model plus one orthographic camera we can aim per frame."""
    spec = mujoco.MjSpec.from_file(str(xml))
    # room for the square-pixel correction, which renders WIDER than the frame
    spec.visual.global_.offwidth = int(width * 1.1)
    spec.visual.global_.offheight = int(height)
    cam = spec.worldbody.add_camera()
    cam.name = "rigcam"
    cam.proj = mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC
    cam.fovy = 1.0
    cam.pos = [0.0, 0.0, 1.0]
    return spec.compile()


def _draw(img, uv, ok, colour, chains_idx, *, radius=3, thickness=1):
    """Keypoints and their chains, in image coords, skipping missing joints."""
    for chain in chains_idx:
        pts = [(int(round(uv[i, 0])), int(round(uv[i, 1]))) for i in chain if i < len(uv) and ok[i]]
        for a, b in zip(pts[:-1], pts[1:], strict=False):
            cv2.line(img, a, b, colour, thickness, cv2.LINE_AA)
    for i in np.flatnonzero(ok):
        cv2.circle(img, (int(round(uv[i, 0])), int(round(uv[i, 1]))), radius, colour, -1)


def _sample_frames(n_qpos: int, frames, n_preview_frames: int) -> list[int]:
    """Bout-relative frame indices to render -- see this module's docstring."""
    if frames is not None:
        return sorted({int(t) for t in frames if 0 <= int(t) < n_qpos})
    if n_qpos <= 0:
        return []
    n = max(1, min(int(n_preview_frames), n_qpos))
    idx = np.linspace(0, n_qpos - 1, n)
    return sorted({int(round(i)) for i in idx})


def render_sidebyside(
    fly_dir,
    *,
    fly: int,
    anatomy,
    rig,
    video_dir,
    bout_start_frame: int,
    video_path,
    still_path,
    pose_source_path,
    cameras=None,
    frames=None,
    n_preview_frames: int = 8,
    geom_size: float = 0.004,
    pad: int = 170,
    fps: float = 4.0,
) -> dict:
    """Render ONE bout-fly's side-by-side check to `video_path`/`still_path`,
    and stamp `pose_source_path` with `outputs.h5`'s own `pose_source` attr.
    """
    fly_dir = Path(fly_dir)
    video_dir = Path(video_dir)
    kp_order = anatomy.kp_order
    cams = list(cameras) if cameras else list(rig.cameras)
    chains_idx = [[kp_order.index(n) for n in chain] for chain in leg_chains(kp_order)]

    cap = cv2.VideoCapture(str(video_dir / f"{list(rig.cameras)[0]}.mp4"))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    model = _build_model(anatomy.model_xml, width, height)
    data = mujoco.MjData(model)
    if model.nq != anatomy.nq:
        raise ValueError(f"render model nq={model.nq} != anatomy nq={anatomy.nq}")
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "rigcam")
    # One renderer per distinct render width: MuJoCo fixes the viewport at
    # construction, and each camera needs its own square-pixel correction.
    render_w = {
        cn: render_width_for_square_pixels(
            rig.matrices_f64[list(rig.cameras).index(cn)].T, (width, height)
        )
        for cn in cams
    }
    renderers = {w: mujoco.Renderer(model, height=height, width=w) for w in set(render_w.values())}

    with h5py.File(fly_dir / "outputs.h5", "r") as f:
        world = np.asarray(f["kp3d_mm"][()], np.float64)
        stored = [n.decode() if isinstance(n, bytes) else str(n) for n in f["kp_names"][()]]
        qpos = np.asarray(f["qpos"][()], np.float64)
        pose_source = str(f.attrs.get("pose_source", "none"))
    if stored != list(kp_order.names):
        raise ValueError(
            f"{fly_dir / 'outputs.h5'}: kp_names disagree with the anatomy "
            f"keypoint order -- refusing rather than risk pairing the wrong marker"
        )

    k2 = np.load(fly_dir / "kp2d.npz", allow_pickle=True)
    p_cam = Order([str(x) for x in k2["cameras"]]).permutation_to(rig.cameras)
    p_kp = Order([str(x) for x in k2["kp_names"]]).permutation_to(kp_order)
    kp2d = np.asarray(k2["kp2d"], np.float64)[:, p_cam][:, :, p_kp]
    conf2d = np.asarray(k2["conf"], np.float64)[:, p_cam][:, :, p_kp]

    k3 = np.load(fly_dir / "kp3d.npz", allow_pickle=True)
    p3 = Order([str(x) for x in k3["kp_names"]]).permutation_to(kp_order)
    raw3d = np.asarray(k3["kp3d"], np.float64)[:, p3]
    conf3d = np.asarray(k3["conf3d"], np.float64)[:, p3]
    obs_world, _ = filter_kp3d(raw3d, conf3d, kp_order, DEFAULT_FILTER_CFG)

    with h5py.File(fly_dir / "stac_ik.h5", "r") as f:
        offsets = np.asarray(f["offsets"][()], np.float64)
    sites_all = fitted_site_xpos_from_qpos(anatomy, qpos, offsets)

    frame_idxs = _sample_frames(qpos.shape[0], frames, n_preview_frames)
    if not frame_idxs:
        raise ValueError(f"{fly_dir}: qpos has {qpos.shape[0]} frames -- nothing to sample")

    caps = {cn: cv2.VideoCapture(str(video_dir / f"{cn}.mp4")) for cn in cams}
    checks: dict[str, float] = {}
    frame_imgs: list[np.ndarray] = []
    try:
        for t in frame_idxs:
            data.qpos[:] = qpos[t]
            mujoco.mj_forward(model, data)
            sites_model = sites_all[t]

            fitted_world = world[t]
            finite = np.isfinite(fitted_world).all(-1) & np.isfinite(sites_model).all(-1)
            if finite.sum() < 4:
                print(f"fly{fly} frame {t}: only {finite.sum()} fitted sites, skipping")
                continue
            s, rot, trans = umeyama(sites_model[finite], fitted_world[finite])
            anchor = sites_model[finite].mean(0)
            obs_model = world_to_model(obs_world[t], s, rot, trans)

            rows = []
            for cn in cams:
                ci = list(rig.cameras).index(cn)
                pos, quat, fovy = mujoco_camera_from_affine(
                    rig.matrices_f64[ci].T,
                    (width, height),
                    s,
                    rot,
                    trans,
                    anchor,
                    back_off=2.0,
                )
                model.cam_pos[cid] = pos
                model.cam_quat[cid] = quat
                model.cam_fovy[cid] = fovy
                # cam_xpos/cam_xmat are derived in mj_forward: editing the model
                # camera without re-running kinematics renders the PREVIOUS pose.
                mujoco.mj_forward(model, data)
                rw = render_w[cn]
                renderers[rw].update_scene(data, camera="rigcam")
                scene = renderers[rw].scene
                add_point_geoms(scene, sites_model, FITTED_RGBA, geom_size)
                add_point_geoms(scene, obs_model, OBSERVED_RGBA, geom_size)
                rend = cv2.cvtColor(renderers[rw].render(), cv2.COLOR_RGB2BGR)
                if rw != width:
                    rend = cv2.resize(rend, (width, height), interpolation=cv2.INTER_LINEAR)

                mj_uv = project_with_mujoco_camera(
                    sites_model[finite], pos, quat, fovy, (width, height), render_width=rw
                )
                rig_uv = np.asarray(rig.project(fitted_world[finite]))[:, ci, :]
                checks[f"f{t}_{cn}_px"] = float(np.median(np.linalg.norm(mj_uv - rig_uv, axis=-1)))

                caps[cn].set(cv2.CAP_PROP_POS_FRAMES, bout_start_frame + t)
                okf, real = caps[cn].read()
                if not okf:
                    print(f"could not read {cn} frame {bout_start_frame + t}")
                    continue

                vis = conf2d[t, ci] > 0.3
                _draw(real, kp2d[t, ci], vis, PALETTE["white"], chains_idx)

                cx, cy = (int(np.median(rig_uv[:, 0])), int(np.median(rig_uv[:, 1])))
                bw, bh = min(2 * pad, width), min(2 * pad, height)
                x0 = int(np.clip(cx - pad, 0, width - bw))
                y0 = int(np.clip(cy - pad, 0, height - bh))
                x1, y1 = x0 + bw, y0 + bh
                left, right = real[y0:y1, x0:x1].copy(), rend[y0:y1, x0:x1].copy()
                for im, lab in (
                    (left, f"{cn} REAL + observed 2D kp  f{t}"),
                    (right, f"{cn} MuJoCo IK @ rig camera  green=fitted cyan=observed"),
                ):
                    cv2.putText(
                        im,
                        lab,
                        (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.38,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                sep = np.full((left.shape[0], 3, 3), 60, np.uint8)
                rows.append(np.hstack([left, sep, right]))

            if not rows:
                continue
            if len(rows) != len(cams):
                missing = len(cams) - len(rows)
                print(
                    f"sidebyside {fly_dir}: frame {t} rendered {len(rows)}/"
                    f"{len(cams)} cameras ({missing} unreadable) -- skipping "
                    f"it; a short frame cannot share a movie with full ones",
                    file=sys.stderr,
                )
                continue
            h = min(r.shape[0] for r in rows)
            w = min(r.shape[1] for r in rows)
            frame_imgs.append(np.vstack([r[:h, :w] for r in rows]))
    finally:
        for c in caps.values():
            c.release()

    if not frame_imgs:
        raise ValueError(
            f"{fly_dir}: every sampled frame ({frame_idxs}) had too few finite "
            f"fitted sites or an unreadable video frame -- nothing to render"
        )

    shapes = {im.shape for im in frame_imgs}
    if len(shapes) > 1:
        raise ValueError(
            f"{fly_dir}: sampled frames have {len(shapes)} different sizes "
            f"{sorted(shapes)} -- every frame of one movie must match. The crop "
            f"window is fixed-size and short frames are dropped, so this is a "
            f"cause neither guard covers"
        )

    write_video(video_path, frame_imgs, fps=fps, macro_block_size=2)
    cv2.imwrite(str(still_path), frame_imgs[len(frame_imgs) // 2])
    Path(pose_source_path).write_text(json.dumps({"pose_source": pose_source}, indent=2))

    if checks:
        worst = max(checks.values())
        print(f"sidebyside fly{fly} {fly_dir}: camera-build check, worst median px = {worst:.3f}")

    return {
        "video_path": str(video_path),
        "still_path": str(still_path),
        "pose_source_path": str(pose_source_path),
        "n_frames_rendered": len(frame_imgs),
        "camera_check_px": checks,
    }
