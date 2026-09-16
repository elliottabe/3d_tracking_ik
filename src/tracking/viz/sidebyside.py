"""Side by side: the real camera with keypoints, and the MuJoCo IK render from
the SAME camera with the same keypoints.

Every other check in this repo reduces the fit to a number. This one puts the
posed body model beside the animal it was fitted to, from the animal's own
camera, which is the only view in which "does the fit match the fly" is a
question you can answer by looking (CLAUDE.md: a figure is how you notice a
number matches for the wrong reason).

`render_sidebyside` is the ONE implementation: both the pipeline stage
(`pipeline.recording_stages.sidebyside_bout_fly`) and the thin CLI that
replaced the script call it, so there are not two renders to keep in sync.

**What this should show if the fit and the camera build are both right.** The
rendered fly (right) has the SAME pose, orientation and apparent size as the
real fly (left): same body axis, same wing positions, legs on the same side.
The green markers sit ON the rendered body, and the white observed keypoints
sit on the real fly. Left and right should look like the same animal
photographed twice.

**What each kind of failure looks like, so a wrong render is not read as a
right one:**

- A fly that is MIRRORED, rotated 90/180 degrees, or the wrong size = the
  CAMERA build is wrong, not the fit. `camera_check_px` (returned, and
  printed) measures the camera numerically rather than trusting the picture.
- Apparent size drifting between cameras at different depths = the camera was
  built as perspective rather than orthographic. The rig's calibration is
  AFFINE; see `tracking.viz.mjcam`.
- Green markers ON the rendered body but the body in a different POSE from the
  real fly = the camera is right and the IK is wrong. This is the only failure
  this figure is meant to find.
- Green markers OFF the rendered body = the model->world bridge disagrees with
  the render, i.e. the two panels are not in the same space.

**Which frames get rendered.** `frames=None` (the pipeline default) samples
`n_preview_frames` bout-relative indices evenly across the WHOLE bout, so a
40-frame bout and a 4000-frame bout both get a representative preview from
the SAME knob -- nobody has to know or type a per-bout frame count. An
explicit `frames` list (the original script's `--frames`) is used exactly,
clipped to the bout's own length. `sidebyside.mp4` stacks the sampled frames
in order; `sidebyside_still.png` is the middle one of those actually
rendered (some sampled frames are skipped -- see below -- so "the middle
requested index" and "the middle rendered frame" can differ).

A sampled frame is silently skipped, not fatal, when fewer than 4 of its
fitted sites are finite: `umeyama` needs at least that many points to fit a
similarity transform, and a bout can have frames where the IK's own root is
NaN (the same "0 finite root keypoints" case `sidebyside_bout_fly` guards
against for a whole bout-fly). One unusable frame among several sampled ones
should not fail the whole render.

Needs a GPU node and `MUJOCO_GL=egl` (set here, as the script did). Never run
on a login node.
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
    """Bout-relative frame indices to render -- see this module's docstring.

    `frames`, when given, is used exactly (deduplicated, clipped to
    `[0, n_qpos)`): this is the `--frames` flag the original standalone
    script took, now a config knob (`configs/viz/default.yaml`). `None`
    samples `n_preview_frames` indices evenly across `[0, n_qpos)` with
    `np.linspace`, so the same knob gives a representative preview
    regardless of the bout's own length.
    """
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

    Reads, all from `fly_dir` and resolved by NAME (never a bare integer,
    CLAUDE.md): `outputs.h5` (fitted `qpos`/`kp3d_mm`), `kp2d.npz`/`kp3d.npz`
    (the observed 2D/3D this bout-fly was fitted to) and `stac_ik.h5`
    (fitted marker `offsets`). `cameras=None` renders every camera in `rig`;
    an explicit list renders exactly those, in that order.

    The model -> world similarity is refit per FRAME via `umeyama`
    (`kp3d_mm` IS the bridge applied to the fitted sites, so refitting over
    the pair recovers the run's own bridge exactly rather than inventing a
    second map) -- never reused across frames, because the bridge is itself
    per-frame in `outputs.h5`.
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

    # Two models, deliberately. `anatomy.mj_model` carries the added marker
    # sites and is what the marker positions must come from; the RENDER model
    # is a plain compile of the same MJCF plus our camera. Sites add no DOFs,
    # so `qpos` is interchangeable between them.
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

    # The OBSERVED 3D keypoints the solve was fitted to, in WORLD units. From
    # the stamped `kp3d.npz` plus this repo's own filter, never from
    # `kp3d_filt.npz`: that file carries no `kp_names`, so its axis cannot be
    # resolved by name (CLAUDE.md's keypoint-order trap).
    k3 = np.load(fly_dir / "kp3d.npz", allow_pickle=True)
    p3 = Order([str(x) for x in k3["kp_names"]]).permutation_to(kp_order)
    raw3d = np.asarray(k3["kp3d"], np.float64)[:, p3]
    conf3d = np.asarray(k3["conf3d"], np.float64)[:, p3]
    obs_world, _ = filter_kp3d(raw3d, conf3d, kp_order, DEFAULT_FILTER_CFG)

    # The marker positions the bridge was fitted from: FK with the FITTED
    # offsets applied. NOT `stac_ik.h5`'s `marker_sites` -- that is FK'd from
    # the INITIAL marker model and sits ~2.4e-3 model units from the fitted
    # one (`fitted_site_xpos_from_qpos`'s own docstring: a 1.5-2x reprojection
    # regression once traced to exactly this file).
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
                # A FIXED-SIZE window, shifted to stay inside the frame -- never
                # clamped to it. Clamping shrank the crop whenever the fly came
                # near an arena edge, so the size depended on where the fly was:
                # sampled frames came out different sizes and the writer refused
                # the movie with "All images in a movie should have same size".
                # Measured 2026-09-15: 84 of Session1's sidebyside tasks, across
                # 8 of 12 recordings. Session0's four hand-run bouts never hit an
                # edge, which is why this shipped looking fine.
                # The female works the walls, so she hits this far more than the
                # male -- the hard fly is the one whose renders were lost.
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
            # A camera whose read() failed was `continue`d above, so this frame
            # has fewer rows than its neighbours and would vstack SHORTER --
            # the second, independent way to reach "All images in a movie
            # should have same size", and one the fixed-size crop above cannot
            # prevent. Skip such a frame, by name, rather than let a partial
            # one set a size nothing else matches.
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

    # Both size hazards are handled above, so a mismatch here means a THIRD
    # cause nobody has seen yet. Name it: imageio's own "All images in a movie
    # should have same size" says nothing about which frames, which bout-fly,
    # or what the sizes were, and that cost a full campaign pass to diagnose.
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
