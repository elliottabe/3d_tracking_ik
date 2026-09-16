"""Anatomy config <-> MuJoCo model consistency: report and filter absent names."""

from __future__ import annotations

import copy
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from omegaconf import OmegaConf

import tracking.utils.path_utils  # noqa: E402,F401
from tracking.conventions import announce, nearest_key
from tracking.io.names import Order, load_keypoint_order

__all__ = [
    "AnatomyMismatch",
    "validate_anatomy",
    "filter_anatomy",
    "KNOWN_MODEL_KEYS",
    "validate_model_keys",
    "align_joint_dims",
    "build_marker_model",
    "Anatomy",
    "load_anatomy",
]


class AnatomyMismatch(Exception):
    """A `strict=True` `filter_anatomy` call found a configured name the
    model does not have."""


_LEG_TAG_RE = re.compile(r"T([123])_(left|right)")


def _leg_tag(name: str) -> str | None:
    """`"claw_T1_left"` -> `"T1L"`; `None` if `name` carries no leg tag."""
    m = _LEG_TAG_RE.search(name)
    if not m:
        return None
    return f"T{m.group(1)}{'L' if m.group(2) == 'left' else 'R'}"


def _model_names(mj_model, objtype) -> set[str]:
    n_by_type = {
        mujoco.mjtObj.mjOBJ_BODY: mj_model.nbody,
        mujoco.mjtObj.mjOBJ_JOINT: mj_model.njnt,
        mujoco.mjtObj.mjOBJ_SITE: mj_model.nsite,
    }
    n = n_by_type[objtype]
    names = set()
    for i in range(n):
        name = mujoco.mj_id2name(mj_model, objtype, i)
        if name:
            names.add(name)
    return names


def _to_plain(cfg) -> dict[str, Any]:
    """`cfg` (an OmegaConf `DictConfig` or a plain mapping) -> a fresh plain
    `dict`, WITHOUT resolving interpolations (`resolve=False`): a bare
    `model.MJCF_PATH: ${anatomy.mjcf_path}`-style self-reference is only
    resolvable once Hydra has placed this file under a real `anatomy`
    namespace, and `validate_anatomy`/`filter_anatomy` never touch that
    field, so forcing its resolution here would fail for no reason.
    """
    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=False)
    return copy.deepcopy(dict(cfg))


def _body_candidates(cfg, model_cfg) -> set[str]:
    names = set(cfg.get("body_names", []) or [])
    names |= set(cfg.get("end_eff_names", []) or [])
    names |= set((model_cfg.get("KEYPOINT_MODEL_PAIRS", {}) or {}).values())
    return names


def _joint_candidates(cfg) -> set[str]:
    return set(cfg.get("joint_names", []) or []) | set(cfg.get("wing_names", []) or [])


KNOWN_MODEL_KEYS: frozenset[str] = frozenset(
    {
        "MJCF_PATH",
        "name",
        "segment_calibration",
        "N_FRAMES_PER_CLIP",
        "FTOL",
        "ROOT_FTOL",
        "LIMB_FTOL",
        "N_ITERS",
        "N_ITER_Q",
        "N_ITER_M",
        "KP_NAMES",
        "KEYPOINT_MODEL_PAIRS",
        "KEYPOINT_INITIAL_OFFSETS",
        "ROOT_OPTIMIZATION_KEYPOINT",
        "TRUNK_OPTIMIZATION_KEYPOINTS",
        "INDIVIDUAL_PART_OPTIMIZATION",
        "KEYPOINT_COLOR_PAIRS",
        "SCALE_FACTOR",
        "MOCAP_SCALE_FACTOR",
        "SITES_TO_REGULARIZE",
        "SITES_TO_FREEZE",
        "RENDER_FPS",
        "N_SAMPLE_FRAMES",
        "M_REG_COEF",
        "M_OPTIMIZER",
        "M_LEARNING_RATE",
        "JOINT_REG_WEIGHTS",
        "STEPSIZE_Q",
        "USE_JAXLS",
        "JAXLS_LAMBDA_INITIAL",
        "JAXLS_GRADIENT_TOLERANCE",
        "JAXLS_PARAMETER_TOLERANCE",
        "JAXLS_COST_TOLERANCE",
        "JAXLS_SMOOTH_WEIGHT",
        "JAXLS_LINEAR_SOLVER",
        "JAXLS_USE_SE3_ROOT",
        "JAXLS_CHUNK_SIZE",
        "JAXLS_ORIENTATION_KEYPOINTS",
        "KEYPOINT_WEIGHTS",
        "SEGMENT_SCALES",
        "JAXLS_SMOOTH_Q_MULT",
    }
)


def validate_model_keys(cfg) -> list[str]:
    """Sorted keys of `cfg['model']` not in `KNOWN_MODEL_KEYS`; announces each."""
    model_cfg = cfg["model"] if "model" in cfg else {}
    unknown = sorted(k for k in model_cfg if k not in KNOWN_MODEL_KEYS)
    for key in unknown:
        near = nearest_key(key, KNOWN_MODEL_KEYS)
        hint = f" Nearest valid spelling: {near!r}." if near else ""
        announce("anatomy", "validate_model_keys", f"unknown model config key {key!r}.{hint}")
    return unknown


def validate_anatomy(cfg, mj_model) -> dict[str, list[str]]:
    """Report every name in `cfg` absent from `mj_model`, grouped by category."""
    validate_model_keys(cfg)

    model_cfg = cfg.get("model", {}) or {}
    bodies = _model_names(mj_model, mujoco.mjtObj.mjOBJ_BODY)
    joints = _model_names(mj_model, mujoco.mjtObj.mjOBJ_JOINT)
    sites = _model_names(mj_model, mujoco.mjtObj.mjOBJ_SITE)

    kp_names = list(model_cfg.get("KP_NAMES", []) or [])

    return {
        "body": sorted(n for n in _body_candidates(cfg, model_cfg) if n not in bodies),
        "joint": sorted(n for n in _joint_candidates(cfg) if n not in joints),
        "site": sorted(n for n in (cfg.get("site_names", []) or []) if n not in sites),
        "keypoint": sorted(k for k in kp_names if f"aligned[{k}]" not in sites),
    }


def _announce(category: str, name: str, reason: str) -> None:
    print(f"[anatomy] filter_anatomy: dropping {category} {name!r} ({reason})", file=sys.stderr)


def filter_anatomy(cfg, mj_model, *, tracked_kp_names=None, strict: bool = True) -> dict[str, Any]:
    """Return a copy of `cfg` intersected with what `mj_model` has, and (when
    `tracked_kp_names` is given) with what this recording tracked.
    """
    report = validate_anatomy(cfg, mj_model)
    absent = {category: names for category, names in report.items() if names}
    if strict and absent:
        raise AnatomyMismatch(
            f"anatomy config names not present in the model: {absent}; call "
            f"filter_anatomy(..., strict=False) to drop them instead"
        )

    plain = _to_plain(cfg)
    model_cfg = plain.get("model", {}) or {}

    bodies = _model_names(mj_model, mujoco.mjtObj.mjOBJ_BODY)
    joints = _model_names(mj_model, mujoco.mjtObj.mjOBJ_JOINT)
    sites = _model_names(mj_model, mujoco.mjtObj.mjOBJ_SITE)
    tracked = None if tracked_kp_names is None else set(tracked_kp_names)

    def _leg_tracked(name: str) -> bool:
        if tracked is None:
            return True
        tag = _leg_tag(name)
        if tag is None:
            return True
        return any(k.startswith(f"{tag}_") for k in tracked)

    def _filter_list(names, universe: set[str], *, category: str) -> list[str]:
        valid = []
        for n in names:
            if n in universe:
                valid.append(n)
            elif not strict:
                _announce(category, n, "not present in the model")
        if tracked is None:
            return valid
        leg_kept = [n for n in valid if _leg_tracked(n)]
        if not leg_kept and valid:
            return valid  # never propose an anatomy with nothing left (see docstring)
        if not strict:
            for n in valid:
                if n not in leg_kept:
                    _announce(category, n, "no tracked keypoint for its leg this recording")
        return leg_kept

    out = dict(plain)
    out["body_names"] = _filter_list(plain.get("body_names", []) or [], bodies, category="body")
    out["end_eff_names"] = _filter_list(
        plain.get("end_eff_names", []) or [], bodies, category="body"
    )
    out["joint_names"] = _filter_list(plain.get("joint_names", []) or [], joints, category="joint")
    out["wing_names"] = _filter_list(plain.get("wing_names", []) or [], joints, category="joint")
    if "site_names" in plain:
        out["site_names"] = _filter_list(plain.get("site_names", []) or [], sites, category="site")

    kp_names = list(model_cfg.get("KP_NAMES", []) or [])
    kept_kp = []
    for k in kp_names:
        if f"aligned[{k}]" not in sites:
            if not strict:
                _announce("keypoint", k, "no aligned[] site in the model")
            continue
        if tracked is not None and k not in tracked:
            if not strict:
                _announce("keypoint", k, "not tracked this recording")
            continue
        kept_kp.append(k)

    new_model = dict(model_cfg)
    new_model["KP_NAMES"] = kept_kp
    pairs = dict(model_cfg.get("KEYPOINT_MODEL_PAIRS", {}) or {})
    new_model["KEYPOINT_MODEL_PAIRS"] = {k: v for k, v in pairs.items() if k in kept_kp}
    out["model"] = new_model
    return out


_JOINT_TYPE_DIMS: dict[int, int] = {
    int(mujoco.mjtJoint.mjJNT_FREE): 7,
    int(mujoco.mjtJoint.mjJNT_BALL): 4,
    int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
    int(mujoco.mjtJoint.mjJNT_HINGE): 1,
}

_FREE_LB = np.concatenate([-np.inf * np.ones(3), -1.0 * np.ones(4)])
_FREE_UB = np.concatenate([np.inf * np.ones(3), 1.0 * np.ones(4)])

# MuJoCo's own "unlimited" sentinel is `jnt_range == (0, 0)`; substitute a
# generous unconstrained range for each joint type.
_UNCONSTRAINED: dict[int, tuple[np.ndarray, np.ndarray]] = {
    int(mujoco.mjtJoint.mjJNT_FREE): (_FREE_LB, _FREE_UB),
    int(mujoco.mjtJoint.mjJNT_BALL): (-np.ones(4), np.ones(4)),
    int(mujoco.mjtJoint.mjJNT_SLIDE): (np.array([-np.inf]), np.array([np.inf])),
    int(mujoco.mjtJoint.mjJNT_HINGE): (np.array([-2 * np.pi]), np.array([2 * np.pi])),
}

_FREE = int(mujoco.mjtJoint.mjJNT_FREE)


def align_joint_dims(mj_model) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Per-qpos lower/upper bounds and joint name, one entry per qpos SLOT."""
    lb_parts: list[np.ndarray] = []
    ub_parts: list[np.ndarray] = []
    names: list[str] = []
    for j in range(mj_model.njnt):
        jtype = int(mj_model.jnt_type[j])  # coerce: see the mujoco 3.13 note above
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j)
        dims = _JOINT_TYPE_DIMS[jtype]
        if jtype == _FREE:
            lo, hi = _UNCONSTRAINED[jtype]
        else:
            lo_val, hi_val = mj_model.jnt_range[j]
            if lo_val == 0.0 and hi_val == 0.0:
                lo, hi = _UNCONSTRAINED[jtype]
            else:
                lo, hi = np.full(dims, lo_val), np.full(dims, hi_val)
        lb_parts.append(lo)
        ub_parts.append(hi)
        names += [name] * dims

    lb = np.minimum(np.concatenate(lb_parts), 0.0)
    ub = np.concatenate(ub_parts)
    return lb, ub, names


def build_marker_model(cfg, kp_order: Order) -> tuple[mujoco.MjModel, np.ndarray, np.ndarray]:
    """Compile `cfg`'s MJCF with one marker site per keypoint added."""
    model_cfg = cfg["model"]
    if model_cfg.get("segment_calibration"):
        raise NotImplementedError(
            "model.segment_calibration=true was dropped deliberately in this port "
            "(see the comment beside it in configs/anatomy/v1.yaml) and is not "
            "implemented: stac_mjx.rescale.rescale_per_segment is not ported, and "
            "src/ may not import stac_mjx"
        )
    if model_cfg.get("SEGMENT_SCALES"):
        raise NotImplementedError(
            "model.SEGMENT_SCALES is set but per-segment morphing was dropped "
            "deliberately in this port (see configs/anatomy/v1.yaml); "
            "stac_mjx.rescale.rescale_per_segment is not ported"
        )
    scale_factor = model_cfg.get("SCALE_FACTOR", 1)
    if float(scale_factor) != 1.0:
        raise NotImplementedError(
            f"model.SCALE_FACTOR={scale_factor!r} != 1 needs stac_mjx.rescale."
            f"dm_scale_spec, which is not ported (src/ may not import stac_mjx); "
            f"v1's own SCALE_FACTOR is 1 (inert) -- this port only carries the "
            f"identity case"
        )

    spec = mujoco.MjSpec.from_file(str(cfg["mjcf_path"]))
    pairs = model_cfg["KEYPOINT_MODEL_PAIRS"]
    offsets = model_cfg["KEYPOINT_INITIAL_OFFSETS"]
    for kp_name in kp_order.names:
        pos = offsets[kp_name]
        if isinstance(pos, str):
            pos = [float(p) for p in pos.split(" ")]
        spec.body(pairs[kp_name]).add_site(
            name=kp_name,
            size=[0.005, 0.005, 0.005],
            rgba=(0, 0, 0, 0.8),
            pos=pos,
            group=3,
        )

    mj_model = spec.compile()

    site_idxs = np.array(
        [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, name) for name in kp_order.names],
        dtype=np.int32,
    )
    if np.any(site_idxs < 0):
        missing = [n for n, i in zip(kp_order.names, site_idxs, strict=True) if i < 0]
        raise ValueError(f"build_marker_model: site(s) not found after compile: {missing}")

    regularize = set(model_cfg.get("SITES_TO_REGULARIZE", []) or [])
    is_regularized = np.zeros(len(kp_order) * 3, dtype=np.float64)
    for i, name in enumerate(kp_order.names):
        if name in regularize:
            is_regularized[3 * i : 3 * i + 3] = 1.0

    freeze = set(model_cfg.get("SITES_TO_FREEZE", []) or [])
    unknown = sorted(freeze - set(kp_order.names))
    if unknown:
        print(
            f"anatomy {cfg.get('name', '<unnamed>')!r}: SITES_TO_FREEZE names "
            f"{unknown}, which are not keypoints of this anatomy -- ignored. "
            f"Expected after filter_anatomy dropped them; otherwise a typo, and "
            f"those sites are NOT frozen.",
            file=sys.stderr,
        )
    is_frozen = np.zeros(len(kp_order) * 3, dtype=bool)
    for i, name in enumerate(kp_order.names):
        if name in freeze:
            is_frozen[3 * i : 3 * i + 3] = True

    both = sorted(freeze & regularize)
    if both:
        print(
            f"anatomy {cfg.get('name', '<unnamed>')!r}: site(s) {both} are in BOTH "
            f"SITES_TO_FREEZE and SITES_TO_REGULARIZE. Freezing wins; their "
            f"M_REG_COEF term is inert. Remove them from one list to say which "
            f"you meant.",
            file=sys.stderr,
        )

    return mj_model, site_idxs, is_regularized, is_frozen


@dataclass(frozen=True)
class Anatomy:
    """An anatomy config, resolved against its compiled MuJoCo model."""

    name: str
    model_xml: Path
    cfg: Mapping[str, Any]
    mj_model: mujoco.MjModel
    kp_order: Order
    site_idxs: np.ndarray
    is_regularized: np.ndarray
    is_frozen: np.ndarray
    kp_weights: np.ndarray
    names_qpos: tuple[str, ...]
    names_xpos: tuple[str, ...]
    lb: np.ndarray
    ub: np.ndarray
    mocap_scale_factor: float
    has_freejoint: bool

    @property
    def nq(self) -> int:
        return int(self.mj_model.nq)

    @property
    def nv(self) -> int:
        return int(self.mj_model.nv)

    @property
    def nbody(self) -> int:
        return int(self.mj_model.nbody)

    @property
    def nsite(self) -> int:
        return int(self.mj_model.nsite)

    @property
    def n_keypoints(self) -> int:
        return len(self.kp_order)

    def site_idx(self, name: str) -> int:
        """MuJoCo site id for `name` (any site, not just a keypoint marker)."""
        idx = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, name)
        if idx < 0:
            raise ValueError(f"no site named {name!r} in this anatomy's model")
        return idx

    def body_idx(self, name: str) -> int:
        """MuJoCo body id for `name`."""
        idx = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        if idx < 0:
            raise ValueError(f"no body named {name!r} in this anatomy's model")
        return idx


def load_anatomy(cfg, *, tracked_kp_names=None, strict: bool = True) -> Anatomy:
    """Load an anatomy config into an `Anatomy`: the config boundary (R12)."""
    cfg_for_path = cfg if OmegaConf.is_config(cfg) else OmegaConf.create(cfg)
    mjcf_path = Path(str(cfg_for_path.mjcf_path))

    base_model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    filtered = dict(
        filter_anatomy(cfg, base_model, tracked_kp_names=tracked_kp_names, strict=strict)
    )
    filtered["mjcf_path"] = str(mjcf_path)

    kp_order = load_keypoint_order(filtered)
    mj_model, site_idxs, is_regularized, is_frozen = build_marker_model(filtered, kp_order)
    lb, ub, names_qpos = align_joint_dims(mj_model)
    names_xpos = tuple(
        mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
        for i in range(mj_model.nbody)
    )

    model_cfg = filtered["model"]
    weight_cfg = model_cfg.get("KEYPOINT_WEIGHTS", {}) or {}
    kp_weights = np.repeat(
        np.array([float(weight_cfg.get(n, 1.0)) for n in kp_order.names], dtype=np.float64), 3
    )
    mocap_scale_factor = float(model_cfg["MOCAP_SCALE_FACTOR"])
    has_freejoint = bool(mj_model.njnt > 0 and int(mj_model.jnt_type[0]) == _FREE)

    return Anatomy(
        name=str(filtered.get("name", "")),
        model_xml=mjcf_path,
        cfg=filtered,
        mj_model=mj_model,
        kp_order=kp_order,
        site_idxs=site_idxs,
        is_regularized=is_regularized,
        is_frozen=is_frozen,
        kp_weights=kp_weights,
        names_qpos=tuple(names_qpos),
        names_xpos=names_xpos,
        lb=lb,
        ub=ub,
        mocap_scale_factor=mocap_scale_factor,
        has_freejoint=has_freejoint,
    )
