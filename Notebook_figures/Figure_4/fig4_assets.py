"""Panel A/H/I/J inputs: video crops, MuJoCo pair renders, SAM3 resolution."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

from fig4_analysis import (  # noqa: F401  -- re-exported for callers
    LocomotionConfig,
    SongAnalysisConfig,
    _load_kp3d,
    _resolve_male_female_slots,
    analyze_fly_song,
    body_pitch_deg_from_quat,
    classify_walking_state,
    compute_centroid_velocity,
    compute_com_height,
    get_fields,
    sam3_camera_index,
    sorted_bout_keys,
    summarize_by_song,
    triangulate_sam3_female_com,
    unpack_sam3_masks_for_frames,
)
from fig4_analysis import load as h5_load  # noqa: F401



# ======================================================================
# from utils/courtship_figure_panels.py
# ======================================================================


def _dlt_load(csv_path: str | Path) -> np.ndarray:
    """Load 11 DLT coefficients (one per line) from a ``*_dlt.csv`` file."""
    coeffs = np.loadtxt(str(csv_path)).astype(float).reshape(-1)
    if coeffs.size != 11:
        raise ValueError(
            f'expected 11 DLT coefficients in {csv_path}, got {coeffs.size}'
        )
    return coeffs

def _dlt_project(coeffs: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Project 3D world points to 2D pixel coords with the standard 11-param DLT."""
    L = np.asarray(coeffs, dtype=float).reshape(11)
    pts = np.asarray(xyz, dtype=float)
    if pts.shape[-1] != 3:
        raise ValueError(f'xyz last dim must be 3; got {pts.shape}')
    X = pts[..., 0]
    Y = pts[..., 1]
    Z = pts[..., 2]
    denom = L[8] * X + L[9] * Y + L[10] * Z + 1.0
    u = (L[0] * X + L[1] * Y + L[2] * Z + L[3]) / denom
    v = (L[4] * X + L[5] * Y + L[6] * Z + L[7]) / denom
    return np.stack([u, v], axis=-1)

def _open_video(mp4_path: str | Path):
    import cv2
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise FileNotFoundError(f'cannot open video: {mp4_path}')
    return cap

def _read_frame(cap, fidx: int, roi: Optional[Tuple[int, int, int, int]] = None):
    """Seek and read frame ``fidx``; return cropped RGB ndarray or ``None``."""
    import cv2
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
    ok, frame = cap.read()
    if not ok:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if roi is not None:
        x, y, w, h = roi
        frame = frame[y:y + h, x:x + w]
    return frame

def floor_align_qpos_pair(
    model,
    qpos_pair: np.ndarray,
    floor_z: Optional[float] = None,
    fly_nq: Optional[int] = None,
    fly_suffixes: Sequence[str] = ('_fly0', '_fly1'),
    floor_geom_name: str = 'floor',
    floor_z_offset: float = 0.0,
) -> np.ndarray:
    """Return a copy of ``qpos_pair`` with each fly's free-joint root-z shifted
    so its lowest-standing geom surface touches ``floor_z`` in every frame.
    """
    import mujoco

    qpos_pair = np.asarray(qpos_pair, dtype=float).copy()
    if qpos_pair.ndim != 2:
        raise ValueError(f'qpos_pair must be (T, nq); got {qpos_pair.shape}')
    T, nq = qpos_pair.shape
    if nq != model.nq:
        raise ValueError(f'qpos_pair nq={nq} != model.nq={model.nq}')
    fly_nq = int(fly_nq or (model.nq // len(fly_suffixes)))
    if len(fly_suffixes) * fly_nq != model.nq:
        raise ValueError(f'{len(fly_suffixes)}*{fly_nq} != model.nq={model.nq}')

    if floor_z is None:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, floor_geom_name)
        floor_z = float(model.geom_pos[gid, 2]) if gid >= 0 else 0.0

    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ''
                  for b in range(model.nbody)]
    geom_fly = np.full(model.ngeom, -1, dtype=int)
    for g in range(model.ngeom):
        bname = body_names[model.geom_bodyid[g]].lower()
        for fi, suf in enumerate(fly_suffixes):
            if suf.lower() in bname:
                geom_fly[g] = fi
                break

    corner_signs = np.array([
        [+1, +1, +1], [+1, +1, -1], [+1, -1, +1], [+1, -1, -1],
        [-1, +1, +1], [-1, +1, -1], [-1, -1, +1], [-1, -1, -1],
    ], dtype=float)
    local_aabb = np.asarray(model.geom_aabb, dtype=float)  # (ngeom, 6)
    rbound = np.asarray(model.geom_rbound, dtype=float)
    data = mujoco.MjData(model)
    for t in range(T):
        data.qpos[:] = qpos_pair[t]
        mujoco.mj_forward(model, data)
        for fi in range(len(fly_suffixes)):
            mask = np.nonzero(geom_fly == fi)[0]
            if mask.size == 0:
                continue
            bottoms = np.full(mask.size, np.inf, dtype=float)
            for k, g in enumerate(mask):
                c = local_aabb[g, :3]
                s = local_aabb[g, 3:]
                if not (np.all(np.isfinite(c)) and np.all(np.isfinite(s))):
                    # Fall back to bounding sphere for geoms with no AABB.
                    bottoms[k] = data.geom_xpos[g, 2] - rbound[g]
                    continue
                corners_local = c + corner_signs * s            # (8, 3)
                R = data.geom_xmat[g].reshape(3, 3)
                corners_world_z = corners_local @ R.T[:, 2] + data.geom_xpos[g, 2]
                bottoms[k] = float(corners_world_z.min())
            bottoms = bottoms[np.isfinite(bottoms)]
            if bottoms.size == 0:
                continue
            shift = float(floor_z - bottoms.min()) - float(floor_z_offset)
            qpos_pair[t, fi * fly_nq + 2] += shift
    return qpos_pair

def estimate_rig_pose_from_qpos(
    data,
    bout_keys: Optional[Iterable[str]] = None,
    fly_nq: int = 93,
    two_flies_per_bout: bool = True,
    exclude_keys: Sequence[str] = ('info',),
) -> Dict[str, object]:
    """Estimate the rig's xy center and yaw from all fly root-xy samples."""
    keys = (list(bout_keys) if bout_keys is not None
            else [k for k in data.keys() if k not in exclude_keys])
    xy_chunks: List[np.ndarray] = []
    for k in keys:
        entry = data.get(k)
        if entry is None or 'qpos' not in entry:
            continue
        qpos = np.asarray(entry['qpos'], dtype=float)
        if qpos.ndim != 2 or qpos.shape[1] < 2:
            continue
        xy_chunks.append(qpos[:, 0:2])
        if two_flies_per_bout and qpos.shape[1] >= 2 * fly_nq:
            xy_chunks.append(qpos[:, fly_nq:fly_nq + 2])
    if not xy_chunks:
        raise ValueError('no xy samples collected from data')
    xy = np.concatenate(xy_chunks, axis=0)
    xy = xy[np.all(np.isfinite(xy), axis=1)]
    if xy.shape[0] < 2:
        raise ValueError(f'not enough finite xy samples: {xy.shape[0]}')
    # PCA on mean-centered data → principal axis direction.
    mean_xy = xy.mean(axis=0)
    cov = np.cov(xy - mean_xy, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order]
    v = evecs[:, 0]
    if v[0] < 0:
        v = -v
    yaw = float(np.arctan2(v[1], v[0]))
    c, s = np.cos(-yaw), np.sin(-yaw)
    R_inv = np.array([[c, -s], [s, c]])
    xy_local = (xy - mean_xy) @ R_inv.T
    lo = xy_local.min(axis=0)
    hi = xy_local.max(axis=0)
    mid_local = 0.5 * (lo + hi)
    # Map midpoint back into the calibration frame.
    c2, s2 = np.cos(yaw), np.sin(yaw)
    R = np.array([[c2, -s2], [s2, c2]])
    center = mean_xy + R @ mid_local
    return {
        'center_xy': (float(center[0]), float(center[1])),
        'yaw_rad': yaw,
        'major_len': float(hi[0] - lo[0]),
        'minor_len': float(hi[1] - lo[1]),
        'n_points': int(xy.shape[0]),
    }

def _courtship_pair_anatomy(cameras: Optional[Sequence[str]] = None):
    """AnatomyConfig with every fly category duplicated per ``_fly0``/``_fly1``."""
    from mujoco_visualizer.config import AnatomyConfig, CategoryRule

    pairs = [
        # (category_name_root, body_substrings_before_suffix, all)
        ('thorax',        ['thorax'],        False),
        ('head',          ['head'],          False),
        ('abdomen',       ['abdomen'],       False),
        ('antenna_left',  ['antenna', 'left'],  True),
        ('antenna_right', ['antenna', 'right'], True),
        ('haltere_left',  ['haltere', 'left'],  True),
        ('haltere_right', ['haltere', 'right'], True),
        ('proboscis',     ['proboscis', 'rostrum', 'labrum', 'labell',
                           'haustellum'], False),
        ('wing_left',     ['wing_left'],  False),
        ('wing_right',    ['wing_right'], False),
        ('T1_left',       ['t1', 'left'],  True),
        ('T1_right',      ['t1', 'right'], True),
        ('T2_left',       ['t2', 'left'],  True),
        ('T2_right',      ['t2', 'right'], True),
        ('T3_left',       ['t3', 'left'],  True),
        ('T3_right',      ['t3', 'right'], True),
    ]

    rules: List[CategoryRule] = []
    for tag in ('fly0', 'fly1'):
        rules.append(CategoryRule(
            name=f'eye_red_{tag}',
            geom_substring=['red', tag], all=True,
        ))
        for cat, subs, require_all in pairs:
            rules.append(CategoryRule(
                name=f'{cat}_{tag}',
                body_substring=list(subs),
                geom_substring=[tag],
                all=require_all,
            ))

    return AnatomyConfig(
        cameras=list(cameras or []),
        categories=rules,
    )

def _remap_settings_keys(settings: dict, suffix: str) -> dict:
    """Return a copy of ``settings`` with color keys suffixed (``_fly0`` etc)."""
    import copy as _copy
    out = _copy.deepcopy(settings)
    if isinstance(out.get('colors'), dict):
        out['colors'] = {f'{k}{suffix}': v for k, v in out['colors'].items()}
    return out

def _find_worldbody_geom(floor_spec, rig_geom_name: str):
    for g in floor_spec.worldbody.geoms:
        if g.name == rig_geom_name:
            return g
    names = [g.name for g in floor_spec.worldbody.geoms]
    raise ValueError(
        f'rig geom {rig_geom_name!r} not found in floor spec worldbody; '
        f'available geoms: {names}'
    )

def _apply_rig_pose_override(
    floor_spec,
    floor_xml: str,
    rig_geom_name: str,
    rig_pos: Optional[Sequence[float]],
    rig_quat: Optional[Sequence[float]],
) -> None:
    """Mutate the spec so the rig mesh RENDERS at the requested world pose."""
    import mujoco

    target = _find_worldbody_geom(floor_spec, rig_geom_name)

    probe_spec = mujoco.MjSpec.from_file(floor_xml)
    probe_model = probe_spec.compile()
    pgid = mujoco.mj_name2id(
        probe_model, mujoco.mjtObj.mjOBJ_GEOM, rig_geom_name,
    )
    probe_target = _find_worldbody_geom(probe_spec, rig_geom_name)
    spec_pos0 = np.asarray(probe_target.pos, dtype=float)
    spec_quat0 = np.asarray(probe_target.quat, dtype=float)
    comp_pos0 = np.asarray(probe_model.geom_pos[pgid], dtype=float)
    comp_quat0 = np.asarray(probe_model.geom_quat[pgid], dtype=float)
    if not np.allclose(spec_quat0, [1.0, 0.0, 0.0, 0.0], atol=1e-8):
        raise NotImplementedError(
            f'rig geom {rig_geom_name!r} has non-identity xml quat '
            f'{spec_quat0.tolist()}; mesh auto-center compensation not '
            f'implemented for that case'
        )
    mesh_auto_pos = comp_pos0 - spec_pos0      # in body frame (= world here)
    mesh_auto_quat = comp_quat0                 # identity ⊗ auto_quat

    desired_compiled_pos = comp_pos0.copy()
    desired_compiled_quat = comp_quat0.copy()
    if rig_pos is not None:
        p = [float(x) for x in rig_pos]
        if len(p) == 2:
            desired_compiled_pos[0] = p[0]
            desired_compiled_pos[1] = p[1]
        elif len(p) == 3:
            desired_compiled_pos = np.asarray(p, dtype=float)
        else:
            raise ValueError(f'rig_pos must have length 2 or 3; got {len(p)}')
    if rig_quat is not None:
        q = [float(x) for x in rig_quat]
        if len(q) != 4:
            raise ValueError(
                f'rig_quat must have length 4 (w, x, y, z); got {len(q)}'
            )
        desired_compiled_quat = np.asarray(q, dtype=float)
        desired_compiled_quat /= np.linalg.norm(desired_compiled_quat)

    spec_quat = _quat_mul(desired_compiled_quat, _quat_conj(mesh_auto_quat))
    spec_pos = desired_compiled_pos - _quat_rotate(spec_quat, mesh_auto_pos)

    target.pos = spec_pos.tolist()
    target.quat = spec_quat.tolist()

def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two scalar-first quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=float)

def _quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)

def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Apply scalar-first quaternion ``q`` to 3-vector ``v``."""
    qv = np.array([0.0, v[0], v[1], v[2]], dtype=float)
    return _quat_mul(_quat_mul(q, qv), _quat_conj(q))[1:]

def build_courtship_pair_visualizer(
    flybody_xml: str,
    floor_xml: str,
    settings_fly0: Optional[str] = None,
    settings_fly1: Optional[str] = None,
    root_body: str = 'thorax',
    spawn_pos=(0.0, 0.0, -0.005),
    rig_geom_name: str = 'Happy_house',
    rig_pos: Optional[Sequence[float]] = None,
    rig_quat: Optional[Sequence[float]] = None,
):
    """Compose a floor + two fly bodies (``_fly0`` male, ``_fly1`` female).

    Parameters
    ----------
    flybody_xml : str
        Path to the single-fly MuJoCo XML (e.g. ``fruitfly_v1_free.xml``).
    floor_xml : str
        Path to a worldbody XML containing a ``geom name="floor"`` plane.
    settings_fly0, settings_fly1 : str or None
        Settings file names (preset names or absolute paths) applied to each
        fly's geoms. Uses the mujoco_visualizer settings resolver, so bare
        names like ``'Earthy_V1_courtship_fly0'`` work. ``None`` skips.
    root_body : str
        Body to attach from the fly spec (default ``'thorax'``).
    spawn_pos : (x, y, z)
        Spawn offset for both flies (applied via the floor frame).
    rig_geom_name : str
        Name of the rig/arena geom in ``floor_xml`` whose pose may be
        overridden (e.g. ``'Happy_house'``). Only consulted when
        ``rig_pos`` or ``rig_quat`` is given.
    rig_pos : sequence of float or None
        Optional override for the rig geom's world xyz. Length-2 keeps the
        XML's original z; length-3 sets xyz fully.
    rig_quat : sequence of float or None
        Optional override for the rig geom's quaternion in MuJoCo's
        scalar-first ``(w, x, y, z)`` order.
    """
    import json
    import mujoco
    from mujoco_visualizer import Visualizer
    from mujoco_visualizer.render_settings import _resolve_settings_path

    fly_spec_0 = mujoco.MjSpec.from_file(flybody_xml)
    fly_spec_1 = mujoco.MjSpec.from_file(flybody_xml)
    floor_spec = mujoco.MjSpec.from_file(floor_xml)
    if rig_pos is not None or rig_quat is not None:
        _apply_rig_pose_override(
            floor_spec, floor_xml, rig_geom_name, rig_pos, rig_quat,
        )
    spawn_frame = floor_spec.worldbody.add_frame(
        pos=list(spawn_pos), quat=[1, 0, 0, 0],
    )
    spawn_frame.attach_body(fly_spec_0.body(root_body), '', '_fly0')
    spawn_frame.attach_body(fly_spec_1.body(root_body), '', '_fly1')

    anatomy = _courtship_pair_anatomy()
    viz = Visualizer(spec=floor_spec, anatomy=anatomy)

    def _load(name: str, suffix: str) -> None:
        try:
            path = _resolve_settings_path(name)
        except FileNotFoundError:
            path = Path(name)
        with open(path) as f:
            raw = json.load(f)
        viz.load_settings(_remap_settings_keys(raw, suffix))

    if settings_fly0 is not None:
        _load(settings_fly0, '_fly0')
    if settings_fly1 is not None:
        _load(settings_fly1, '_fly1')
    return viz

# ======================================================================
# from scripts/figures/export_fig4_bundle.py
# ======================================================================


_ENVELOPE_CACHE: Dict[str, tuple] = {}

def _arena_envelope(recording_dir, pose_dir: str = "pose_v2_20260914") -> tuple:
    """(y_lo, y_hi, z_hi) Scutellum bounds pooled over a recording's bouts."""
    key = (str(recording_dir), str(pose_dir))
    if key in _ENVELOPE_CACHE:
        return _ENVELOPE_CACHE[key]
    import glob as _glob
    arrs = []
    for f in sorted(_glob.glob(str(Path(recording_dir) / pose_dir / "bouts"
                                   / "bout_*" / "fly*" / "kp3d.npz"))):
        try:
            arrs.append(np.load(f)["kp3d"][:, 0, :])
        except Exception:                        # noqa: BLE001
            continue
    if not arrs:
        env = (None, None, None)
    else:
        S = np.concatenate(arrs)
        ylo, yhi = np.nanpercentile(S[:, 1], [1, 99])
        zhi = float(np.nanpercentile(S[:, 2], 99))
        env = (float(ylo), float(yhi), zhi)
    _ENVELOPE_CACHE[key] = env
    return env

def _outside_arena(recording_dir, bout_name: str, fly_dir: str, n: int,
                   pose_dir: str = "pose_v2_20260914"):
    """Boolean mask (len n) of frames whose Scutellum leaves the arena envelope."""
    ylo, yhi, zhi = _arena_envelope(recording_dir, pose_dir)
    if ylo is None:
        return np.zeros(n, bool)
    kp_path = (Path(recording_dir) / pose_dir / "bouts" / bout_name / fly_dir
               / "kp3d.npz")
    if not kp_path.exists():
        return np.zeros(n, bool)
    try:
        scut = np.load(kp_path)["kp3d"][:, 0, :]
    except Exception:                            # noqa: BLE001
        return np.zeros(n, bool)
    m = ((scut[:, 1] > yhi) | (scut[:, 1] < ylo) | (scut[:, 2] > zhi))
    m = np.asarray(m, bool)
    out = np.zeros(n, bool)
    k = min(n, m.size)
    out[:k] = m[:k]
    return out

def _mean_z_by_label(results: List[dict], label: str,
                     bad_masks: Optional[Dict[str, np.ndarray]] = None
                     ) -> np.ndarray:
    out: List[float] = []
    for r in results:
        v = np.asarray(r["male_valid"], dtype=bool)
        lab = np.asarray(r["male_labels"])
        z = np.asarray(r["com_z"], dtype=float)
        m = v & np.isfinite(z) & (lab == label)
        if bad_masks is not None:
            bad = bad_masks.get(r.get("key0"))
            if bad is not None:
                k = min(m.size, bad.size)
                m[:k] &= ~bad[:k]
        if m.any():
            out.append(float(np.mean(z[m])))
    return np.asarray(out, dtype=float)

def analyze_unpaired_males(data, bout_keys, info, pairs, kp_names, *,
                           song_cfg=None, loc_cfg=None, despike=True):
    """Single-fly analysis for males whose partner has no reconstruction."""
    # (import removed by the port: provided at module level)
    song_cfg = song_cfg or SongAnalysisConfig()
    loc_cfg = loc_cfg or LocomotionConfig()

    paired = {k for pr in pairs for k in pr}
    src = list(info.get("source_flies", []))
    male_fly = list(info.get("male_fly", []))
    verified = list(info.get("sex_verified", []))

    out = []
    for i, key in enumerate(bout_keys):
        if key in paired or key not in data:
            continue
        if i >= len(src) or i >= len(male_fly):
            continue
        slot = int(str(src[i]).replace("fly", "") or -1)
        if int(male_fly[i]) != slot:            # this lone fly is the female
            continue
        if verified and i < len(verified) and not bool(verified[i]):
            continue
        try:
            kp, xp, q = get_fields(data[key], despike=despike)
            kp = np.asarray(kp, float)
            valid = np.isfinite(kp).all(axis=tuple(range(1, kp.ndim)))
            song = analyze_fly_song(kp, xp, q, kp_names, cfg=song_cfg,
                                    valid_mask=valid)
            kin = compute_centroid_velocity(kp, kp_names, loc_cfg,
                                            body_length=None)
            com_z, floor_z = compute_com_height(kp, kp_names, loc_cfg)
            speed_bl = kin.get("speed_bl", kin["speed"])
            dw = str(song.get("dominant_wing", "L")).upper()
            side = "L" if dw.startswith("L") else "R"
            labels = np.asarray(song["sides"][side]["frame_labels"])
            metrics = {
                "forward_speed_bl": np.asarray(
                    kin.get("forward_speed_bl", kin["forward_speed"])),
                "speed_bl": np.asarray(speed_bl),
                "turn_rate": np.asarray(kin["turn_rate"]),
                "com_z": np.asarray(com_z),
            }
            out.append({
                "key0": key, "key1": None, "T": int(len(kp)),
                "single_fly": True,
                "song0": song, "song1": None, "sex": None,
                "valid_fly0": valid, "valid_fly1": None, "colocated": None,
                "male_labels": labels, "male_valid": valid,
                "kin": kin, "com_z": com_z, "floor_z": floor_z,
                "walking_state": classify_walking_state(np.asarray(speed_bl), loc_cfg),
                "by_song": summarize_by_song(labels, metrics, valid_mask=valid),
            })
        except Exception as e:                   # noqa: BLE001 - report, don't die
            print(f"[single-fly] {key}: skipped ({type(e).__name__}: {e})")
    return out

VIZ_SETTINGS = ("Earthy_V1_courtship_fly0", "Earthy_V1_courtship_fly1")

VIZ_CAMERA = "track1_fly0"

RENDER_CAM = {"distance": 0.75, "azimuth": 85.0, "elevation": -20.0}

def _pair_qpos(q0: np.ndarray, q1: np.ndarray, n: int) -> np.ndarray:
    """Concatenate two single-fly qpos arrays into the pair layout
    ``[fly0_qpos | fly1_qpos]`` that `build_courtship_pair_visualizer`'s model
    expects (nq = 2 * fly_nq). Pure; split out so the concatenation logic is
    unit-testable without MuJoCo."""
    q0 = np.asarray(q0, dtype=float)[:n]
    q1 = np.asarray(q1, dtype=float)[:n]
    return np.concatenate([q0, q1], axis=-1)


def _free_running_com_z(h5_path) -> np.ndarray:
    """Per-bout mean scutellum height ABOVE THE FLOOR for a free-running h5."""
    # (import removed by the port: provided at module level)

    data = h5_load(str(h5_path))
    info = data.get("info", {}) or {}
    raw = info.get("kp_names", info.get("site_names_egocentric", []))
    if isinstance(raw, dict):
        kp_names = [raw[k] for k in sorted(raw.keys(), key=lambda x: int(x))]
    else:
        kp_names = list(raw)

    cfg = LocomotionConfig()
    keys = sorted_bout_keys(k for k in data.keys() if k != "info")
    means: List[float] = []
    for k in keys:
        kp = np.asarray(data[k]["kp_data"])
        if kp.ndim == 2:
            kp = kp.reshape(kp.shape[0], -1, 3)
        com_z, _floor_z = compute_com_height(kp, kp_names, cfg)
        com_z = com_z[np.isfinite(com_z)]
        if com_z.size == 0:
            continue
        means.append(float(np.nanmean(com_z)))
    return np.asarray(means, dtype=float)

def _pair_center_xyz(kp0: np.ndarray, kp1: np.ndarray, scut_idx: int, T: int) -> np.ndarray:
    """Per-frame midpoint of the two flies' Scutellum, given two already-
    loaded ``(T, n_kp, 3)`` kp3d arrays in the TRUE DLT world frame — feeds
    `panel_video_strip_with_kp`'s `center_xyz`/`crop_wh` auto-centred crop
    instead of a fixed `roi` copied from a different recording (Finding 2:
    """
    kp0 = np.asarray(kp0, dtype=float)
    kp1 = np.asarray(kp1, dtype=float)
    n = min(len(kp0), len(kp1), T)
    return 0.5 * (kp0[:n, scut_idx, :] + kp1[:n, scut_idx, :])

def _video_frame_indices(T: int, n_video: int, fs: float,
                         span_ms: Optional[float]) -> np.ndarray:
    """Frame indices for the panel-A video strip."""
    last = T - 1
    if span_ms is not None:
        last = min(last, int(round(float(span_ms) / 1000.0 * fs)))
    return np.linspace(0, last, n_video, dtype=int)

def _blend_masks(frame, masks, colors, alpha: float) -> np.ndarray:
    """Blend translucent per-fly mask tints into an RGB crop."""
    from matplotlib.colors import to_rgb

    out = np.asarray(frame, dtype=float).copy()
    H, W = out.shape[:2]
    for m, c in zip(masks, colors):
        m = np.asarray(m, dtype=bool)
        if m.shape != (H, W):
            continue
        rgb = np.array(to_rgb(c), dtype=float) * 255.0
        out[m] = (1.0 - alpha) * out[m] + alpha * rgb
    return np.clip(out, 0, 255).astype(np.uint8)

def _project_kp_to_crop(kp_xyz, dlt, kp_scale: float, roi, shape,
                        project=None) -> np.ndarray:
    """Project one frame's keypoints into CROP-LOCAL pixel coordinates."""
    if project is None:
        project = _dlt_project          # defined above in this module

    pts = np.asarray(kp_xyz, dtype=float) * float(kp_scale)
    uv = np.asarray(project(dlt, pts), dtype=float).reshape(-1, 2).copy()
    if roi is not None:
        uv -= np.array([roi[0], roi[1]], dtype=float)
    H, W = int(shape[0]), int(shape[1])
    bad = ~(np.isfinite(uv).all(axis=1)
            & (uv[:, 0] >= 0) & (uv[:, 0] < W)
            & (uv[:, 1] >= 0) & (uv[:, 1] < H))
    uv[bad] = np.nan
    return uv

def _order_pair_male_first(key0_dir: str, key1_dir: str,
                           male_slot: Optional[int]) -> tuple:
    """Indices that put the MALE first in the render pair, as ``(i, j)``."""
    if male_slot is None:
        return (0, 1)
    male_dir = f"fly{int(male_slot)}"
    if key0_dir == male_dir:
        return (0, 1)
    if key1_dir == male_dir:
        return (1, 0)
    return (0, 1)

def _rig_pos_for_session(data, bout_keys, fly_ids, recording: str,
                         offset=(0.0, 0.0)):
    """`rig_pos` placing the Happy_house arena on the flies, or None."""
    # (import removed by the port: provided at module level)

    base = _recording_base(recording)
    keys = [k for k, fid in zip(bout_keys, fly_ids)
            if base in _recording_base(fid)]
    if not keys:
        return None
    rig = estimate_rig_pose_from_qpos(data, bout_keys=keys)
    cx = float(rig["center_xy"][0]) + float(offset[0])
    cy = float(rig["center_xy"][1]) + float(offset[1])
    print(f"rig pose: {len(keys)}/{len(bout_keys)} bouts from {base} -> "
          f"centre ({cx:+.3f}, {cy:+.3f}) cm (offset {tuple(offset)}), "
          f"yaw {np.degrees(rig['yaw_rad']):+.1f} deg, "
          f"extent {rig['major_len']:.2f} x {rig['minor_len']:.2f} cm")
    return (cx, cy)

def _snap_to_renderable(want: np.ndarray, ok_idx: np.ndarray) -> np.ndarray:
    """Snap each desired frame to the nearest one that can actually be drawn."""
    ok_idx = np.asarray(ok_idx)
    if ok_idx.size == 0:
        raise ValueError("no renderable (finite-qpos) frames")
    want = np.asarray(want)
    pos = np.searchsorted(ok_idx, want).clip(0, ok_idx.size - 1)
    prev = (pos - 1).clip(0, ok_idx.size - 1)
    take_prev = np.abs(ok_idx[prev] - want) < np.abs(ok_idx[pos] - want)
    return ok_idx[np.where(take_prev, prev, pos)]

def _recording_base(fly_id: str) -> str:
    """Strip a fly_id down to the RECORDING id the processed tree uses."""
    parts = str(fly_id).rstrip("/").split("/")
    while len(parts) > 1 and (parts[-1].startswith("fly")
                              or parts[-1].startswith("bout")):
        parts.pop()
    return "/".join(parts).rsplit("_fly", 1)[0]

def _processed_fly_dir(fly_id: str) -> str:
    """``'Session1/2026_04_02_16_21_32_fly1'`` -> ``'fly1'``."""
    tail = fly_id.rstrip("/").rsplit("/", 1)[-1]
    if not tail.startswith("fly"):
        tail = tail.rsplit("_", 1)[-1]
    if not tail.startswith("fly"):
        raise ValueError(f"cannot derive fly dir from fly_id {fly_id!r}")
    return tail

def _resolve_sam3_camera_index(cameras, calib_dir, cam: str) -> tuple:
    """Return ``(cam_idx, source)`` for `cam` (e.g. ``"Cam2012630"``) against
    a SAM3 mask npz's own camera axis.
    """
    if cameras is not None:
        names = [str(c) for c in cameras]
        if cam not in names:
            raise ValueError(f"camera {cam!r} not in npz cameras: {names}")
        return names.index(cam), "npz cameras"
    # (import removed by the port: provided at module level)
    return sam3_camera_index(calib_dir, f"{cam}_dlt.csv"), "glob fallback"

def _slot_to_raw_frame(sync_plan, camera: str, slot: int) -> tuple:
    """Convert a CANONICAL slot number to a raw mp4 frame position for
    `camera`. Returns ``(raw_frame, corrected)``; `corrected` is False (and
    `raw_frame == slot`, the pre-fix positional behaviour) when `sync_plan`
    is falsy/`None` or its `status` is not `"reindex"`.
    """
    if not sync_plan or getattr(sync_plan, "status", None) != "reindex":
        return int(slot), False
    # The source repo read this from its vendored jarvis_jax; the target repo
    # owns the same sync-plan slot mapping, so use its implementation.
    from tracking.io.video import slot_positions
    positions, present = slot_positions(sync_plan, camera, int(slot), 1)
    if not present[0] or positions[0] is None:
        raise ValueError(
            f"camera {camera!r} dropped canonical slot {slot} entirely "
            f"(no raw mp4 frame delivers it)")
    return int(positions[0]), True

_DEFAULT_MALE_SLOT = 1

_DEFAULT_FEMALE_SLOT = 0

def _resolve_session_bout(session_dir, sam3_root, recording: str, clip_len: int,
                          tol: int = 1, pose_dir: str = "pose_v2_20260914") -> tuple:
    """Map a combined-h5 exemplar onto its session bout: `start_frame` from
    the recording's bout table (``<pose_dir>/bouts_unified_summary.csv`` if
    present, else the legacy ``courtship_bout_summary.csv``), but the SAM3
    directory name by SCANNING ``sam3_root`` for the one whose own mask
    frame count matches — never by deriving it from the CSV's ``bout_idx``.
    """
    import pandas as pd

    candidates = [Path(session_dir) / pose_dir / "bouts_unified_summary.csv",
                  Path(session_dir) / "courtship_bout_summary.csv"]
    csv_path = next((p for p in candidates if p.exists()), None)
    if csv_path is None:
        raise FileNotFoundError(
            f"no bout table under {session_dir}: looked for "
            + ", ".join(str(p) for p in candidates))
    df = pd.read_csv(csv_path)
    if "fly_id" in df.columns:
        rows = df[df["fly_id"].astype(str) == _recording_base(recording)]
    else:
        rows = df                      # already scoped to this recording
    n = rows["end_frame"] - rows["start_frame"] + 1
    matches = rows[(n - int(clip_len)).abs() <= tol]
    if len(matches) == 0:
        raise ValueError(
            f"no bout in {csv_path} for recording {recording!r} with "
            f"clip_len={clip_len} (tol={tol})")
    if len(matches) > 1:
        idxs = sorted(int(v) for v in matches["bout_idx"])
        raise ValueError(
            f"ambiguous bout match in {csv_path} for recording {recording!r} "
            f"with clip_len={clip_len} (tol={tol}): candidate bout_idx "
            f"{idxs} — refusing to guess")
    start_frame = int(matches.iloc[0]["start_frame"])

    sam3_root = Path(sam3_root)
    counts = []
    for npz_path in sorted(sam3_root.glob("bout_*/sam3_masks.npz")):
        with np.load(npz_path) as z:
            counts.append((npz_path.parent.name, int(z["valid"].shape[-1])))
    dir_matches = [(d, c) for d, c in counts if abs(c - int(clip_len)) <= tol]
    if len(dir_matches) == 0:
        raise ValueError(
            f"no sam3 bout dir under {sam3_root} with mask frame count "
            f"matching clip_len={clip_len} (tol={tol}); available: {counts}")
    if len(dir_matches) > 1:
        dirs = sorted(d for d, _ in dir_matches)
        raise ValueError(
            f"ambiguous sam3 bout dir under {sam3_root} for clip_len="
            f"{clip_len} (tol={tol}): candidate dirs {dirs} — refusing to guess")
    bout_dir = dir_matches[0][0]
    return (bout_dir, start_frame)

def _render_frames(flybody_xml, floor_xml, qpos_pair, frame_idx,
                   camera=VIZ_CAMERA, track_midpoint=True, size=256,
                   rig_pos=None):
    """Bake two-fly courtship-pair MuJoCo frames to uint8 RGB via the styled
    visualizer (`Earthy_V1_courtship_fly0`/`fly1` presets: red fly0, teal
    fly1), floor-aligned.
    """
    import mujoco

    # (import removed by the port: provided at module level)

    viz = build_courtship_pair_visualizer(
        flybody_xml=str(flybody_xml), floor_xml=str(floor_xml),
        settings_fly0=VIZ_SETTINGS[0], settings_fly1=VIZ_SETTINGS[1],
        rig_pos=rig_pos)
    qpos_pair = np.asarray(qpos_pair, dtype=float)
    if qpos_pair.shape[1] != viz.model.nq:
        raise ValueError(f"qpos_pair has {qpos_pair.shape[1]} dof but "
                         f"pair model nq={viz.model.nq}")
    qpos_pair = floor_align_qpos_pair(viz.model, qpos_pair)
    fly_nq = qpos_pair.shape[1] // 2

    out = []
    for fi in frame_idx:
        q_row = qpos_pair[int(fi)]
        if track_midpoint:
            mid = 0.5 * (q_row[0:3] + q_row[fly_nq:fly_nq + 3])
            cam_arg = mujoco.MjvCamera()
            cam_arg.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam_arg.lookat[:] = mid
            cam_arg.distance = RENDER_CAM["distance"]
            cam_arg.azimuth = RENDER_CAM["azimuth"]
            cam_arg.elevation = RENDER_CAM["elevation"]
        else:
            cam_arg = camera
        pixels = viz.render_frame(q_row, camera=cam_arg, height=size, width=size)
        out.append(np.asarray(pixels, dtype=np.uint8))
    return out
