"""Panel A/H/I/J inputs: video crops, MuJoCo pair renders, SAM3 resolution.

Everything that reads something other than the combined h5. Extracted by AST
trace from `3d_tracking_dataset/utils/courtship_figure_panels.py` and
`scripts/figures/export_fig4_bundle.py`, bodies unchanged.

Most of this exists to answer four questions that look trivial and are not,
each of which produces an entirely plausible figure of the wrong animal when
answered wrong:

* Which SAM3 bout is this bout? Ordinal position in the combined h5 is a
  permutation of the tree's numbering, so `_resolve_session_bout` resolves it
  from the recording's own bout summary and refuses an ambiguous match.
* Which mask slot is the male? From a reviewed `sex.json`, never slot order.
* Which camera axis is which camera? `_resolve_sam3_camera_index` maps by NAME:
  the mask npz's stored `cameras` array is a different order from the
  calibration glob order.
* Which video frame is bout frame 0? `_slot_to_raw_frame` goes through the
  recording's `sync_plan.json`, since dropped frames mean slot != raw index.

Needs a GPU node and MUJOCO_GL=egl for the panel-H renders.
"""
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
    """Project 3D world points to 2D pixel coords with the standard 11-param DLT.

    ``xyz`` is broadcast over leading axes; trailing dim must be 3. Returns an
    array with the same leading shape and trailing dim 2 (u, v in pixels).
    """
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

    ``floor_z_offset`` (default 0) is added to the per-frame shift, so a
    positive value pushes the fly further down into the floor (useful when
    the claw-tip touches visually read as floating at grazing camera angles).

    ``qpos_pair`` is laid out as ``[fly0_qpos | fly1_qpos]`` with each fly
    starting in a 7-dof free joint (xyz + quat), so root-z lives at index
    ``fly_nq * k + 2`` for fly ``k``. ``fly_nq`` defaults to ``model.nq // 2``.

    Geoms are assigned to a fly by a case-insensitive substring match on their
    parent body name against ``fly_suffixes`` (e.g. ``'_fly0'``). The lowest
    surface per fly is ``min(geom_xpos[:, 2] - geom_rbound[:])`` — i.e. the
    bottom of each geom's bounding sphere — which works for mesh feet.

    When ``floor_z`` is ``None`` it is auto-detected from the z of the geom
    named ``floor_geom_name`` (plane geoms store z in ``geom_pos``).
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

    # Local AABB corner offsets (8 corners of a unit box in ±1 space). The
    # true world-frame bottom of each geom comes from transforming these by
    # its rotation. Tighter than geom_rbound, so feet actually touch the floor.
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
    """Estimate the rig's xy center and yaw from all fly root-xy samples.

    Fly root xy (qpos indices 0,1 per fly) lives in the camera-calibration
    world frame, but the rig mesh is anchored at the MuJoCo world origin.
    Aggregating every fly's xy across all bouts in ``data`` gives both the
    rig's xy center (sample centroid) and its long-axis orientation
    (principal axis of the centered covariance).

    ``data`` maps bout_key → dict with a ``'qpos'`` array of shape
    ``(T, nq)``. When ``two_flies_per_bout`` is True and ``nq >= 2*fly_nq``,
    fly1's xy is taken from columns ``fly_nq:fly_nq+2`` as well.

    Returns a dict with:
      ``center_xy`` — (cx, cy) of the rig center in the calibration frame
      ``yaw_rad``   — dominant-eigenvector angle, sign-fixed so ``v_x >= 0``
      ``major_len`` — full extent (max-min) along the major axis
      ``minor_len`` — full extent along the minor axis
      ``n_points`` — total number of finite xy samples used

    The rig center is the midpoint of the PCA-aligned bounding box of the
    fly xy samples, not the raw centroid: flies typically spend unequal time
    on different sides of a narrow chamber, so the mean drifts off-center.
    The midpoint is unbiased as long as their range is symmetric about the
    true rig center — a much weaker assumption.
    """
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
    # Rotate samples into the PCA-aligned frame and take the midpoint of the
    # axis-aligned bounding box. This recovers the rig center robustly even
    # when flies spend unequal time on each side of the chamber.
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
    """AnatomyConfig with every fly category duplicated per ``_fly0``/``_fly1``.

    Categories use ``body_substring=[name, flyN], all=true`` so each rule
    matches only the geoms on its intended fly even though both flies share
    the same unsuffixed body names before MjSpec ``attach_body`` renames
    them with the per-fly suffix.
    """
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

    # eye_red must precede `head` so the `head_red` eye geom is bucketed there.
    # Use geom_substring for the fly-tag constraint (geom names carry the
    # ``_fly0`` / ``_fly1`` suffix too), so each rule ANDs the body match
    # with the geom-name tag match. This avoids the edge case where a rule
    # with ``all=False`` over multiple body substrings would match on the
    # tag alone and sweep in unrelated bodies (e.g. proboscis rule grabbing
    # wing geoms because both contain "fly0").
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
    """Return a copy of ``settings`` with color keys suffixed (``_fly0`` etc).

    Only the ``colors`` block is remapped; flags / camera / lighting / floor /
    skybox apply globally and are left untouched.
    """
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
    """Mutate the spec so the rig mesh RENDERS at the requested world pose.

    MuJoCo's mesh compiler re-centers mesh vertices to the mesh COM and aligns
    principal inertial axes to the body frame, so the compiled ``geom_pos`` /
    ``geom_quat`` differ from the spec values. This helper probes that
    mesh-induced shift by compiling a fresh copy of ``floor_xml``, then solves
    for the spec-level ``pos``/``quat`` that produce the desired compiled pose.

    ``rig_pos`` may be length-2 (preserving the XML default's rendered z) or
    length-3. ``rig_quat`` is MuJoCo's scalar-first ``(w, x, y, z)``.
    """
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
        # Math below assumes the xml-declared geom quat is identity; this is
        # the case for floor.xml. Generalizing would require unrotating the
        # auto-center offset by spec_quat0 first.
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

    Returns a :class:`mujoco_visualizer.Visualizer` whose ``self.model.nq``
    equals ``fly_nq * 2`` and whose category buckets are per-fly, so the two
    fly-specific settings JSONs color each fly independently.

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
    """(y_lo, y_hi, z_hi) Scutellum bounds pooled over a recording's bouts.

    The chamber is long and narrow, so a reconstruction that fails near a wall
    leaves the arena in y and/or z while staying plausible in x. Pooling every
    bout-fly in the SAME recording gives a per-recording envelope without
    assuming absolute coordinates (they differ between recordings).

    `pose_dir` names the pose tree under the recording. It is part of the
    cache key because one recording directory can hold several pose trees
    whose envelopes differ.
    """
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
    """Boolean mask (len n) of frames whose Scutellum leaves the arena envelope.

    Diagnosed on Session1/2026_04_02_16_56_37 bout_00003/fly0, panel G's 0.345
    outlier: 16% of frames sit beyond the chamber in y AND above the
    recording's z p99, with conf3d dropping 0.731 -> 0.639 there -- a
    triangulation failure near a wall, not a real climb. The next-highest
    bout (0.219) is 0% outside at conf 0.971, so this gate keeps it: it
    removes the artifact WITHOUT trimming the tail generally.

    `pose_dir` is forwarded to `_arena_envelope` and used for the per-bout
    kp3d lookup below, so both read the SAME pose tree (see that function's
    docstring for why the tree can vary per recording).
    """
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
    """Single-fly analysis for males whose partner has no reconstruction.

    `pair_bouts` only pairs an adjacent (fly0, fly1), so a bout where one fly
    could not be solved contributes NOTHING -- and every Figure 4 panel derives
    from `analyze_all_pairs` results, including the ones that need a single fly
    (wing-angle density, pulse classification, wing phase, z-height). On the
    2026-08-28 re-run three Session0 bouts came back male-only, because the
    mask-agreement gate found the female's keypoints unusable for essentially
    the whole bout (see run_bout.view_mask_agreement / unsolvable.json).
    Dropping a perfectly good male fit for that reason is a waste.

    This reproduces the SINGLE-FLY half of `utils.courtship_loader.analyze_pair`
    by calling the same `utils` functions -- utils is consumed unmodified -- and
    returns dicts with the keys those pooled panels read: `song0`,
    `male_labels`, `male_valid`, `com_z`, `by_song`, `kin`.

    Only a slot the h5 marks as the MALE (`info['male_fly']`, and only where
    `sex_verified`) is analysed: the panels are about male song, and guessing
    the sex of a lone fly is exactly the mistake the wing-song CV metric made.

    Pair-only fields (`song1`, `sex`, `colocated`, `valid_fly1`) are set to
    None, and `single_fly=True` is set, so a pair-only consumer that is handed
    one of these breaks loudly instead of silently reading a male as a pair.
    """
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
            # No partner, so no pair-validity mask: a frame counts when this
            # fly's own keypoints are finite. compute_pair_validity's other
            # gates (colocation, occlusion by the other fly) are meaningless
            # for a lone animal and must not be faked.
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
    """Per-bout mean scutellum height ABOVE THE FLOOR for a free-running h5.

    ``utils.free_walking_loader.load__scutellum_z`` returns RAW z with no
    floor subtraction, while the courtship arms (``pulse_z``/``sine_z``,
    from ``r["com_z"]``) use ``utils.locomotion.compute_com_height``
    (scutellum z minus the 5th-percentile ground-keypoint z for that bout).
    Plotting the two together compared different quantities and overstated
    the courtship-vs-walking height difference roughly 2x (measured: raw
    free-running mean 0.250 vs floor-corrected 0.147, against courtship's
    0.127). Use the same estimator for both sides — this is a figbuilder-
    layer fix (not `utils/`, which is consumed unmodified): a local
    per-bout floor correction that mirrors `compute_com_height` exactly,
    using its own default `LocomotionConfig` so the two sides genuinely
    agree, rather than a single global floor across all bouts (which would
    reintroduce the same class of error `compute_com_height` exists to
    avoid — the floor can differ bout to bout).

    Returns the per-bout array, matching the shape
    ``load__scutellum_z(..., per_bout=True)`` returned so nothing
    downstream changes. Bouts with no finite `com_z` samples are skipped.
    """
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
    the notebook's SESSION0 crop is the wrong window for a SESSION1
    recording; a static crop is wrong for every new session).

    Round 6: this used to reach into the combined h5's `kp_data`, which is
    body-model-RESCALED and RE-CENTERED and is NOT in the DLT world frame
    (measured: its Scutellum midpoint projects to uv (13.6, 426.6), the
    frame's bottom-left corner) — feeding it here centred every crop on
    empty chamber. Callers must now load `kp3d.npz` from the processed pose
    tree (`pose/bouts/<bout>/fly{0,1}/kp3d.npz`, key `"kp3d"`) and pass
    those arrays in; this function stays pure and only does the midpoint
    arithmetic, unit-testable without video/DLT. Clamps to the shorter of
    the two arrays (and `T`), same as `_pair_qpos`.
    """
    kp0 = np.asarray(kp0, dtype=float)
    kp1 = np.asarray(kp1, dtype=float)
    n = min(len(kp0), len(kp1), T)
    return 0.5 * (kp0[:n, scut_idx, :] + kp1[:n, scut_idx, :])

def _video_frame_indices(T: int, n_video: int, fs: float,
                         span_ms: Optional[float]) -> np.ndarray:
    """Frame indices for the panel-A video strip.

    The published Figure 4 samples the FIRST 1500 ms (0 / 501 / 1002 /
    1504 ms), not the whole bout: the courtship beat it illustrates happens
    early, and spreading four frames over all 2508 ms of bout 28 lands the
    last one at 2506 ms on a different part of the interaction.

    `span_ms` is clamped to the bout, so a bout shorter than the span still
    ends on its own last frame instead of indexing past it. `span_ms=None`
    restores the historical whole-bout sampling.
    """
    last = T - 1
    if span_ms is not None:
        last = min(last, int(round(float(span_ms) / 1000.0 * fs)))
    return np.linspace(0, last, n_video, dtype=int)

def _blend_masks(frame, masks, colors, alpha: float) -> np.ndarray:
    """Blend translucent per-fly mask tints into an RGB crop.

    The masks stay RASTER on purpose: they are soft, low-frequency fills, so
    vectorising them would add contour machinery for no visible gain. Only the
    KEYPOINTS become vector (see `figbuilder.panels.courtship.VideoKpPanel`),
    because those are the marks that looked like blobs once baked.

    A mask whose shape does not match the crop is SKIPPED rather than
    broadcast — a silent shape mismatch would tint the wrong pixels.
    """
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
    """Project one frame's keypoints into CROP-LOCAL pixel coordinates.

    Mirrors the projection `panel_video_strip_with_kp` does internally, but
    returns the coordinates instead of drawing them, so the bundle can carry
    them and the panel can scatter them as vector marks.

    Keypoints that do not project, or land outside the crop, come back as NaN
    — the panel drops those rather than drawing a mark at the frame corner.
    """
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
    """Indices that put the MALE first in the render pair, as ``(i, j)``.

    `build_courtship_pair_visualizer` colours by PAIR POSITION —
    `VIZ_SETTINGS[0]` (red) is applied to whichever fly is passed first. The
    z-height panel of this same figure labels Male red and Female blue, so if
    the pair is passed in its raw key0/key1 order the two panels disagree
    about which animal is which.

    key0/key1 is the PAIR ordering and tracks neither the fly0/fly1 directory
    index nor sex — on the published exemplar key0 is the FEMALE, so the raw
    order painted her red and the male teal, the reverse of the figure's own
    legend. Resolve by NAME against the dataset's `male_fly`, never by
    position; this is the same class of trap as the keypoint/camera order
    bugs in CLAUDE.md.

    With no sexing available, the pair's own order is kept rather than
    guessed at.
    """
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
    """`rig_pos` placing the Happy_house arena on the flies, or None.

    The rig mesh is anchored at the MuJoCo world origin while the flies live
    in the DLT calibration frame, so without this the arena sits metres away
    and slabs across the camera instead of reading as the chamber wall the
    female climbs. `estimate_rig_pose_from_qpos` recovers the chamber's xy
    centre from the pooled fly root-xy of a session's bouts.

    Returned length-2 ON PURPOSE: that keeps the XML's own z for the rig, as
    `Courtship_Song_Figures2.ipynb` (cell 8/13) does. Passing a z here instead
    lifts or sinks the arena relative to the floor plane.

    `offset` is the notebook's RIG_CENTER_OFFSET (cm, world xy): the PCA
    centre puts the chamber symmetrically around the flies, which leaves a
    wall between the camera and a female climbing it. Nudging the rig moves
    that wall behind her so she reads as ON it. Measured on the exemplar's
    1500 ms frame by sweeping (+-0.2, +-0.4) in x and y and LOOKING: +0.2 in
    x is the only direction that reveals the reared female against the wall;
    -0.2 x and both y directions hide her further.

    Restricted to bouts from THIS recording: `data` may merge several h5s,
    each with its own DLT calibration, and pooling xy across them averages
    disjoint world frames into a centre that belongs to neither.
    """
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
    """Snap each desired frame to the nearest one that can actually be drawn.

    The render strip illustrates the SAME moments as the panel-A video strip:
    a reader compares the two rows frame-for-frame, so independently sampled
    timepoints invite a comparison that is not being made. But frames whose
    qpos is NaN cannot be posed, so a requested timepoint that lands on a hole
    snaps to the nearest renderable neighbour instead of being dropped —
    keeping the two strips the same length and as close in time as the data
    allows.
    """
    ok_idx = np.asarray(ok_idx)
    if ok_idx.size == 0:
        raise ValueError("no renderable (finite-qpos) frames")
    want = np.asarray(want)
    pos = np.searchsorted(ok_idx, want).clip(0, ok_idx.size - 1)
    prev = (pos - 1).clip(0, ok_idx.size - 1)
    take_prev = np.abs(ok_idx[prev] - want) < np.abs(ok_idx[pos] - want)
    return ok_idx[np.where(take_prev, prev, pos)]

def _recording_base(fly_id: str) -> str:
    """Strip a fly_id down to the RECORDING id the processed tree uses.

        'Session1/2026_04_02_16_21_32_fly1'            -> 'Session1/2026_04_02_16_21_32'
        'Session0/2025_10_20_13_20_04/bout00028/fly0'  -> 'Session0/2025_10_20_13_20_04'

    The second (2026-08) form also names the bout, so trailing `flyN` and
    `boutNNNNN` path segments are dropped before the older `_flyN` suffix
    strip. `courtship_bout_summary.csv`'s `fly_id` column carries neither.
    """
    parts = str(fly_id).rstrip("/").split("/")
    while len(parts) > 1 and (parts[-1].startswith("fly")
                              or parts[-1].startswith("bout")):
        parts.pop()
    return "/".join(parts).rsplit("_fly", 1)[0]

def _processed_fly_dir(fly_id: str) -> str:
    """``'Session1/2026_04_02_16_21_32_fly1'`` -> ``'fly1'``.

    Round 7: the combined h5's key0/key1 ordering is the PAIR ordering and
    does NOT correspond to the processed tree's fly0/fly1 directory names —
    verified: for the exemplar, key0=bout_183 has fly_id
    ``..._fly1`` and key1=bout_182 has fly_id ``..._fly0`` (INVERTED).
    `analyze_pair`'s `male_id='fly0'` refers to the pair's first element
    (key0), not to this fly_id suffix. Always derive the directory from the
    fly_id suffix; never assume key0 -> `fly0` / key1 -> `fly1`.
    """
    # Two fly_id shapes are in the wild:
    #   Old_preds / analysis-v1 : 'Session1/2026_04_02_16_21_32_fly1'  (_flyN suffix)
    #   analysis v1_2026-08     : 'Session0/2025_10_20_13_20_04/bout00028/fly0'
    #                             (path-style, and it names the bout too)
    # Take the last path segment first, then fall back to the '_' split.
    tail = fly_id.rstrip("/").rsplit("/", 1)[-1]
    if not tail.startswith("fly"):
        tail = tail.rsplit("_", 1)[-1]
    if not tail.startswith("fly"):
        raise ValueError(f"cannot derive fly dir from fly_id {fly_id!r}")
    return tail

def _resolve_sam3_camera_index(cameras, calib_dir, cam: str) -> tuple:
    """Return ``(cam_idx, source)`` for `cam` (e.g. ``"Cam2012630"``) against
    a SAM3 mask npz's own camera axis.

    Round 10 finding: SAM3 mask npzs in the PROCESSED tree self-label their
    camera axis via a ``cameras`` array, in an order that is NOT guaranteed
    to match ``sorted(glob('Cam*_dlt.csv'))`` — verified on the exemplar's
    own npz: ``cameras`` puts `Cam2012630` at index 5, while glob-sorted
    calibration order puts it at index 0. The OLD `Video_recordings` tree's
    npzs (``['packed','valid','centroids','shape']``, no ``cameras`` array)
    are the convention `utils/sam3_female_com.py`'s `sam3_camera_index` was
    written against; the round-6 repoint to the processed tree carried that
    glob-order assumption forward, silently permuting the camera axis.

    ``cameras`` is the npz's own array (or `None` when absent — the legacy
    convention, still supported as a fallback). When present, `source` is
    ``"npz cameras"`` and a `cam` absent from it raises `ValueError` naming
    what the npz DOES contain (never silently falls back to index 0). When
    `cameras` is `None`, falls back to `sam3_camera_index` against
    ``calib_dir``'s glob-sorted order, `source` is ``"glob fallback"`` — the
    caller should print which path was taken.
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

    Round 11 finding A: this recording's own `sync_plan.json` has
    `status="reindex"` with one interior drop
    (`Cam2012630: gaps=[{"slot": 21, "lost": 26}]`), long before this
    exemplar's bout (canonical slot ~380781). Reading raw mp4 position ==
    canonical slot directly (the pre-fix behaviour, `utils/
    courtship_figure_panels.py`'s `_read_frame(cap, fidx + video_frame_
    offset, ...)` does exactly this) silently reads a DIFFERENT real
    instant than the one the masks/kp3d describe.

    Reuses the repo's OWN canonical slot<->position mapping —
    `jarvis_jax.predict.frame_sync.SyncCam.pos()`/`.has()` (vendored from
    JohnsonLabJanelia/cluster_pose's `check_sync.py`; also the engine
    behind `viz/core/io.py`'s `sync_positions`/`read_frames_synced`) —
    rather than re-deriving the gap-accumulation arithmetic here, per
    `SyncCam`'s own docstring: "pos(t) = mp4 frame position that delivers
    slot t (= count of present frames before t)" — i.e. `pos(t) = t -
    (frames lost before t)`, confirmed against the pre-existing repo test
    `third_party/jarvis_jax/tests/test_synced_reader.py::
    test_slot_positions_maps_around_gap` (gap at slot 41 losing 3: slot 44
    -> pos 41, `# pos = slot-3`). `SyncCam.has(t)` reports `False` — the
    camera dropped that exact slot entirely — for a slot inside a gap; this
    raises `ValueError` rather than silently reading an adjacent frame.
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

    Returns ``(bout_dir_name, start_frame)``.

    SAM3 bout directory numbering is NOT guaranteed to match the CSV's
    `bout_idx` — verified on the ``Video_recordings`` SAM3 tree (round 5):
    CSV bout_idx 5 (n=778, the exemplar) lived in `bout_00004`, while
    `bout_00005` held CSV bout_idx 6 (n=569); almost certainly parallel
    SAM3 shards writing their outputs in completion order. The processed
    tree this function is now pointed at (round 6:
    `<processed_root>/<recording>/{courtship_bout_summary.csv,sam3_masks}`)
    verifiably does NOT have this permutation (dir `bout_0000N` <-> CSV
    `bout_idx N` exactly, for every row) — but matching by mask frame count
    is kept regardless, since it is correct whether or not the numbering
    happens to line up, and the first three rows of the round-5 permutation
    also lined up before row 4 broke it. Do NOT "simplify" this back to a
    `bout_idx`-derived directory name.

    ``recording`` (e.g. from ``recording_of(ex)``, which reads
    ``info/fly_ids``) carries a trailing ``_flyN`` suffix that the CSV's
    ``fly_id`` column never has, so the suffix is stripped before matching
    and compared by EQUALITY, not substring containment (round-4 fix: the
    prior ``fly_id`` substring-of-``recording`` check ran backwards — the
    CSV id is a substring of the suffixed recording id, never the reverse,
    so it matched zero rows on every real run). A recording id with no
    ``_fly`` suffix is left unchanged by the strip, so the same code path
    handles both forms without a conditional.

    Reads ONLY each npz's tiny ``valid`` array (shape ``(2, 7, N)``) to get
    its frame count ``N`` — never ``packed`` (``(2, 7, N, 448, 242)``);
    decompressing all of them would be very slow.
    """
    import pandas as pd

    # The MVQ tree has no `courtship_bout_summary.csv`; it carries
    # `<pose_dir>/bouts_unified_summary.csv` instead, whose rows agree with
    # the legacy file (bout_idx 1 -> 14045/14557 on Session0) but which has no
    # `fly_id` column, being already scoped to one recording. The backup tree
    # still has the legacy file, so both are supported and the error names
    # both when neither is found.
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

    `rig_pos` places the Happy_house arena on the flies. It was previously
    pinned to None because the old `floor_xml` had no such geom; the
    repo-local `configs/render/floor_happy_house.xml` enables it, and without
    a rig_pos the mesh stays at the MuJoCo world origin while the flies live
    in the DLT calibration frame -- the arena then slabs across the camera
    instead of reading as the chamber wall. Pass the length-2 (x, y) that
    `_rig_pos_for_session` returns, which keeps the XML's own z. None still
    works and still renders a bare floor.

    Mirrors panel_render_strip's own `track_midpoint` + `viz.render_frame`
    path (verified: red fly0, teal fly1, extended wing visible), inlined here
    to return raw arrays without a matplotlib axes round-trip. When
    `track_midpoint` (default) a free camera tracks the fly0/fly1 midpoint
    every frame using `RENDER_CAM`; set it False to use the named `camera`
    (e.g. a model-defined `track1_fly0`) unmodified instead.
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
