"""Anatomy config <-> MuJoCo model consistency: report and filter absent names.

An anatomy config (`configs/anatomy/v1.yaml`, say) names bodies, joints and
keypoints it expects the MuJoCo model to have. Two things can make that
false: the config drifts from the XML (a name renamed or removed), or the
RECORDING itself tracks less anatomy than the model defines -- an amputation
cohort's fly is missing a leg, so it has no distal keypoint for it.

`validate_anatomy` reports every configured name absent from the model,
pure reporting, grouped by category (body / joint / site / keypoint) --
never raises. `filter_anatomy` acts on that report: it returns a config
whose name lists are intersected with what the model actually has, and (via
`tracked_kp_names`) with what THIS recording actually tracked. `strict=True`
(the default) RAISES on any name absent from the model; `strict=False`
filters and announces every drop to stderr instead.

`v1`'s own config must pass `filter_anatomy(cfg, mj_model, strict=True)`
UNCHANGED -- the executable form of "the v1 base anatomy is complete". It is
also what makes an incomplete anatomy config safe to use at all: a mismatched
or amputee anatomy must filter explicitly and loudly (`strict=False`, one
stderr line per drop) rather than have the solver silently fit a leg the fly
does not have.

Drops announce to stderr, never `warnings.warn`, which dedupes by (message,
category, module, lineno): a batch driver calling `filter_anatomy` once per
bout against the same config would warn on the first bout and go silent on
every one after it in the same process.

`validate_model_keys`/`KNOWN_MODEL_KEYS` are a second, independent check.
`validate_anatomy`'s four categories catch a NAME the model doesn't have;
this one catches a `model:` config KEY the schema doesn't know -- a typo
(`ROOT_OPTIMIZATION_KEYPOINTT`) or a solver feature this repo does not have
(`JAXLS_REG_GATE`). `validate_anatomy` calls it, so `filter_anatomy` and
`load_anatomy` get both checks.

`build_marker_model` iterates `kp_order`, not the config's pair dict, so the
returned `site_idxs` is always in canonical keypoint order regardless of how
the config happens to list its pairs.

`load_anatomy` is the config boundary: the one surface here that ACCEPTS an
OmegaConf `DictConfig`, because it is where Hydra hands the pipeline its
config. `Anatomy` holds plain NumPy and a compiled `mujoco.MjModel` only --
no JAX, no MJX; callers build the MJX model from `Anatomy.mj_model`.
"""

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

# Registers the `${repo_root:}` OmegaConf resolver as an import side effect.
# `configs/anatomy/v1.yaml`'s `root: ${repo_root:}/model` needs it resolved
# before `load_anatomy` can find the MJCF file; nothing else in this module
# imports `tracking.utils`, so without this the resolver may never be
# registered and `cfg.mjcf_path` comes back with the literal `${...}` text.
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


# A leg-tagged name carries `T{1,2,3}_{left,right}` somewhere in it (e.g.
# `claw_T1_left`, `tarsus2_T3_right`, `coxa_abduct_T2_left`). Names with no
# leg tag (thorax, head, wing, abdomen) are never leg-filtered by
# `tracked_kp_names` -- amputation removes a LEG, not the trunk.
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


# The complete `model:` key schema: `configs/anatomy/v1.yaml`'s 36 keys plus
# `SEGMENT_SCALES` (read by `build_marker_model` under `segment_calibration`)
# and `JAXLS_SMOOTH_Q_MULT` (written into the offsets fit's config), neither of
# which v1 carries. Six of the 36 are read by nothing here but stay KNOWN
# anyway, because validating v1's own shipped config must announce nothing.
#
# Deliberately NOT known, so a config that sets one announces rather than being
# silently obeyed by nobody: `JAXLS_ROBUST_DELTA`, `JAXLS_Q_REG_TO_REST`,
# `JAXLS_REG_GATE`, `JAXLS_REG_GATE_SIGMA_DEG` -- the solver has no such
# branches.
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
    """Sorted keys of `cfg['model']` not in `KNOWN_MODEL_KEYS`; announces each.

    A second, independent check from `validate_anatomy`'s four categories:
    those catch a configured NAME (body/joint/site/keypoint) absent from the
    model; this one catches a `model:` config KEY the schema doesn't
    recognise at all -- a misspelling (`ROOT_OPTIMIZATION_KEYPOINTT`) is
    indistinguishable from a legitimately pruned anatomy without a schema,
    and a key naming a dropped solver feature (`JAXLS_REG_GATE`) would
    otherwise be silently obeyed by nobody.

    stderr, never `warnings.warn` -- same reasoning as `filter_anatomy`'s own
    drop announcements (module docstring): a batch driver resolving the same
    anatomy config once per bout must hear about a bad key on every bout, not
    just the first.
    """
    model_cfg = cfg["model"] if "model" in cfg else {}
    unknown = sorted(k for k in model_cfg if k not in KNOWN_MODEL_KEYS)
    for key in unknown:
        near = nearest_key(key, KNOWN_MODEL_KEYS)
        hint = f" Nearest valid spelling: {near!r}." if near else ""
        announce("anatomy", "validate_model_keys", f"unknown model config key {key!r}.{hint}")
    return unknown


def validate_anatomy(cfg, mj_model) -> dict[str, list[str]]:
    """Report every name in `cfg` absent from `mj_model`, grouped by category.

    Pure reporting: never raises, never mutates `cfg`. Returns
    `{"body": [...], "joint": [...], "site": [...], "keypoint": [...]}`,
    each a sorted list of offending names (empty when nothing is missing):

      - "body": `body_names` + `end_eff_names` + `KEYPOINT_MODEL_PAIRS`
        values, checked against the model's bodies.
      - "joint": `joint_names` + `wing_names`, checked against the model's
        joints.
      - "site": an optional `site_names` list (most anatomy configs, `v1`
        included, don't carry one -- always `[]` for them), checked against
        the model's sites.
      - "keypoint": `model.KP_NAMES`, checked via the `aligned[<name>]`
        site-naming convention (CLAUDE.md) -- a keypoint is present iff the
        model has that site, not iff its own name matches anything.

    Also runs `validate_model_keys(cfg)` for its announcement side effect
    (unknown `model:` keys), so every caller of `filter_anatomy`/
    `load_anatomy` gets that check for free without asking for it by name.
    """
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

    `strict=True` (the default) RAISES `AnatomyMismatch` on any configured
    name absent from the MODEL (`validate_anatomy`'s report) -- a real
    config/XML mismatch, not something a caller should ever want silently
    dropped. `v1` must pass this call unchanged; that is this module's own
    correctness test. `tracked_kp_names` narrowing is NOT a mismatch (a
    recording legitimately tracking fewer keypoints than the model defines
    is the expected amputation case) and is applied under `strict=True` too
    -- only model-absence raises.

    `strict=False` additionally filters model-absent names and announces
    every drop to stderr (see the module docstring for why stderr, not
    `warnings.warn`).

    A leg-tagged body/joint (`claw_T1_left`, `femur_twist_T2_right`, ...) is
    dropped under `tracked_kp_names` iff NO tracked keypoint carries that
    leg's tag (`f"{tag}_"` prefix) -- i.e. the whole leg is untracked, the
    amputation case. One guard: if every leg-tagged entry in a list would be
    dropped this way, the
    unfiltered (model-valid) list is kept instead -- `tracked_kp_names`
    losing every leg at once is almost always a caller mistake (empty or
    wrong keypoint list), not a fly with no legs left, so this never
    proposes an anatomy with nothing left in it.

    `KEYPOINT_INITIAL_OFFSETS` / `KEYPOINT_COLOR_PAIRS` / `KEYPOINT_WEIGHTS`
    / `SITES_TO_REGULARIZE` / `JAXLS_ORIENTATION_KEYPOINTS` /
    `INDIVIDUAL_PART_OPTIMIZATION` are left exactly as configured: they are
    keyed LOOKUPS (or, for the last, a leg-tag-keyed list of joint names),
    not lists a caller enumerates to discover which keypoints/legs exist, so
    a stale entry naming a keypoint or leg this call dropped is inert -- a
    caller must still walk from the filtered `KP_NAMES`/`body_names`/
    `joint_names`, never from these.
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


# Keys are plain `int`, not `mujoco.mjtJoint` members: mujoco 3.13 raises
# `KeyError: np.int32(0)` looking a numpy scalar up in an enum-keyed dict
# (pybind11's `__eq__` against a numpy integer changed after 3.11). Hence every
# `int(...)` around an `mjt*` member or a `jnt_type[...]` read below.
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
    """Per-qpos lower/upper bounds and joint name, one entry per qpos SLOT.

    A free joint occupies 7 qpos slots under ONE name, a ball 4, hinge/slide
    1 each, so
    `len(names)` is `mj_model.nq`, not `mj_model.njnt` -- a per-JOINT list
    would silently misalign every later per-qpos lookup. `jnt_range ==
    (0, 0)` is MuJoCo's own "unconstrained" sentinel and is substituted with
    a generous range per joint type; free joints always take that branch
    (their own `jnt_range` is meaningless). The final `lb = min(lb, 0)` clamp
    keeps `qpos = 0` always feasible (matters for the free joint's identity
    quaternion) -- verified inert on v1 (0 of 87 joints have a positive lower
    bound) but load-bearing on `two_hinge.xml`'s "clamped" joint, the only
    fixture that can see it at all.
    """
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
    """Compile `cfg`'s MJCF with one marker site per keypoint added.

    Iterates `kp_order.names` for both the site ADDITIONS and the returned
    `site_idxs`, so the result is always in canonical keypoint order no matter
    how `KEYPOINT_MODEL_PAIRS` happens to list its pairs. (v1's own config
    cannot demonstrate the difference, because its `KEYPOINT_MODEL_PAIRS`
    already happens to agree with `KP_NAMES` order.)

    `cfg["mjcf_path"]` must already be a concrete, resolved path (that is
    `load_anatomy`'s job -- this function never touches OmegaConf). Returns
    `(mj_model, site_idxs, is_regularized)`: `site_idxs` is `(K,)` int32 in
    `kp_order` order; `is_regularized` is `(K*3,)` float64, 1.0 where the
    keypoint is in `SITES_TO_REGULARIZE` (`M_REG_COEF` uses this to pin a
    site's fitted offset toward its initial guess).

    `segment_calibration` / `SEGMENT_SCALES` (subject-specific per-segment
    morphing) and a `SCALE_FACTOR` other than the identity are unimplemented
    features with a config key each -- asking for one RAISES rather than being
    silently ignored.
    """
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

    # `SITES_TO_FREEZE` is the LIMIT of regularization: the offset is held at
    # its MJCF value and never optimised. `M_REG_COEF` is a compromise on both
    # sides -- too small and a marker drifts off the body part it names, too
    # large and the offsets solve diverges (100 and 1000 return NaN) -- while
    # freezing has no coefficient and cannot diverge.
    #
    # Use it where the MJCF's site position is the anatomy you trust, e.g. the
    # wing veins: their model sites sit 0.11x and 0.49x of the wing's
    # half-thickness from the mesh, where an unregularized fit moved them 3.2x
    # OFF it. A marker that leaves the thin wing surface forces the solve to
    # rotate the wing to reach it -- a twisted wing no keypoint residual sees.
    freeze = set(model_cfg.get("SITES_TO_FREEZE", []) or [])
    # A name that is not a keypoint here is INERT, not fatal: an amputation
    # anatomy keeps entries for keypoints it just dropped. Announced anyway,
    # because the other way a name lands here unmatched is a typo, and a
    # misspelt entry freezes nothing while looking like it worked.
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

    # Freezing dominates regularizing, so a site in both lists gets a
    # regularization term that can never do anything. Announce rather than
    # silently prefer one -- a config that says two things about one site is a
    # question, not a default (stderr, not `warnings`, which dedupes per process
    # and would show this once across a whole session's bouts).
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
    """An anatomy config, resolved against its compiled MuJoCo model.

    Every model-derived quantity here -- `nq`/`nv`/`nbody`/`nsite`, the
    marker-site indices, the per-qpos joint names, the joint bounds, and
    `mocap_scale_factor` -- comes from the loaded model or the config, never
    from a literal: v1 is 93/92/68/115(+50); v2_3 (Plan C) is a different
    101/100/74 and maps its wing DOFs onto legs. `mj_model` already carries
    one marker site per keypoint (`build_marker_model`); `site_idxs` locates
    them BY NAME, in `kp_order` order, because on v1 they land at ids
    55..164 with gaps -- no arithmetic on `nsite` reproduces them.

    Pure MuJoCo and NumPy: no JAX, no MJX. `mj_model` binds no device, so this
    is free to construct on CPU in a test with no GPU; the solver builds an MJX
    model from `mj_model` when it needs one.
    """

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
    """Load an anatomy config into an `Anatomy`: the config boundary (R12).

    Unlike every `tracking.preprocess` surface, this ACCEPTS an OmegaConf
    `DictConfig` and converts it -- deliberately: this is the one place in
    the phase where Hydra hands the pipeline its config, so a refusal here
    would just move the `OmegaConf.to_container` call into every caller.

    `cfg["mjcf_path"]` (`configs/anatomy/v1.yaml`'s `root: ${repo_root:}/
    model` / `mjcf_path: "${root}/..."`) is resolved by wrapping `cfg` in an
    `OmegaConf` node and reading that ONE field with normal (lazy) attribute
    access -- never `OmegaConf.to_container(cfg, resolve=True)` on the whole
    tree, which raises: `model.MJCF_PATH: ${anatomy.mjcf_path}` names a
    top-level `anatomy` key that exists only once Hydra has composed this
    file under a real `anatomy:` group, so a blanket resolve fails on a
    config loaded standalone (as every test here loads it) even though
    `mjcf_path` alone resolves fine (`root` and `${repo_root:}` don't depend
    on that missing key). `tracking.utils.path_utils` must have been
    imported for `${repo_root:}` to be registered -- this module imports it
    itself, so importing `tracking.inverse_kinematics.anatomy` is enough.

    Flow: resolve `mjcf_path` -> compile the BASE model (no marker sites) so
    `filter_anatomy` has something to check names against -> `filter_anatomy
    (cfg, base_model, tracked_kp_names=tracked_kp_names, strict=strict)`,
    which reaches `validate_anatomy`, which runs the `validate_model_keys`
    schema check as a side effect -> `build_marker_model` on the FILTERED
    config -> every `Anatomy` field derived from the resulting model.
    """
    cfg_for_path = cfg if OmegaConf.is_config(cfg) else OmegaConf.create(cfg)
    mjcf_path = Path(str(cfg_for_path.mjcf_path))

    base_model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    filtered = dict(
        filter_anatomy(cfg, base_model, tracked_kp_names=tracked_kp_names, strict=strict)
    )
    # `_to_plain` (inside `filter_anatomy`) keeps `mjcf_path` UNRESOLVED
    # (resolve=False, for the reason in this module's docstring); substitute
    # the concrete path computed above so `build_marker_model` -- which takes
    # only plain values, never OmegaConf -- gets something it can compile.
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
