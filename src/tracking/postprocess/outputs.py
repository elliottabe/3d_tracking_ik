"""One bout-fly's `outputs.h5` and `fitted.npz`.

`outputs.h5` is the per-bout-fly analysis artifact: the pose, the model-frame
arrays derived from it, and the per-frame model -> world bridge. Everything in
it is in MODEL units. `fitted.npz` is the one world-unit product, written
because `viz/kpvideo.py`'s `fitted` source reads
`fly<f>/fitted.npz` through `load_npz(...)['kp3d']` and projects it with
`CameraRig.project`, which takes WORLD points.

NO MESH. The visualisation path renders the posed body through MuJoCo's own
renderer, which reads the model's mesh geoms and the committed .obj assets
directly. A stored mesh array would be (T, ~10^4, 3) floats per bout-fly --
the largest thing in the artifact, for no reader.

A frame the bridge could not fit is NaN in `root_se3`, `scale` and
`fitted_world`, and False in `bridge_ok`. It is never given a substitute
identity transform: an identity places the model at the world origin at model
scale, which is off by roughly the arena and reads as a teleport rather than as
a gap.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np

from tracking.inverse_kinematics.bridge import compute_bridges, model_to_world
from tracking.io.artifacts import save_npz
from tracking.io.names import Order, order_sha
from tracking.postprocess.kinematics import (
    THORAX_BODY,
    body_frames_from_qpos,
    egocentric_sites,
    fitted_site_xpos_from_qpos,
    site_xpos_from_qpos,
)

__all__ = [
    "OUTPUTS_DATASETS",
    "FlyOutputs",
    "read_stac_h5",
    "build_fly_outputs",
    "write_outputs_h5",
    "write_fitted_npz",
]

OUTPUTS_DATASETS = (
    "qpos",
    "root_se3",
    "scale",
    "kp3d_mm",
    "site_xpos",
    "xpos_egocentric",
    "bridge_ok",
    "kp_names",
    "names_qpos",
    "names_xpos",
)

_NAME_KEYS = ("kp_names", "names_qpos", "names_xpos")


@dataclass(frozen=True)
class FlyOutputs:
    """One bout-fly's derived arrays. MODEL units except `kp3d_mm`.

    `root_se3` is `qpos[:, :7]` -- the free joint's own pose in the MODEL
    frame, `[x, y, z, qw, qx, qy, qz]`. It is NOT the model->world bridge;
    the bridge contributes only `scale` (its `s`) and `kp3d_mm` (its result).

    `kp3d_mm` carries the FK'd marker sites in WORLD units (0.1 mm). The name
    is the reference artifact's and is a known misnomer -- Global Constraints
    keep it on disk for compatibility. Measured on bout 28 fly0: -0.23..240.53.

    `site_xpos` and `xpos_egocentric` are `(T, n_keypoints, 3)` -- the K
    MARKER sites only, in `kp_order`, NOT all `nsite` model sites (v1:
    nsite=165, n_keypoints=50). The full site axis has no name: `kp_names`/
    `order_sha` stamp a K-long axis, and writing all 165 would put an
    unnameable axis in a file whose whole contract is name-addressability.
    """

    qpos: np.ndarray
    root_se3: np.ndarray
    scale: np.ndarray
    kp3d_mm: np.ndarray
    site_xpos: np.ndarray
    xpos_egocentric: np.ndarray
    bridge_ok: np.ndarray
    summary: dict


def read_stac_h5(path) -> dict[str, Any]:
    """Every dataset of a `stac_ik.h5`, with `|S` name arrays decoded to `str`.

    Attributes come back under `"attrs"`; the solver writes `ik_solver`,
    `ik_candidates` and `ik_summary` there.
    """
    out: dict[str, Any] = {}
    with h5py.File(str(path), "r") as f:
        for key in f.keys():
            value = f[key][()]
            if key in _NAME_KEYS:
                out[key] = [v.decode() if isinstance(v, bytes) else str(v) for v in value]
            elif isinstance(value, bytes):
                out[key] = value.decode()
            else:
                out[key] = np.asarray(value)
        out["attrs"] = {k: f.attrs[k] for k in f.attrs}
    return out


def build_fly_outputs(
    anatomy,
    stac: dict,
    *,
    kp3d_world: np.ndarray,
    conf3d: np.ndarray,
    kp_scale: float,
    source: str = "kp3d_filt",
) -> FlyOutputs:
    """Derive every `outputs.h5` array from one `stac_ik.h5` plus its keypoints.

    `kp3d_world` and `conf3d` are the WORLD-unit keypoints the solve was fitted
    to -- `BoutArtifact.kp3d`/`conf3d` values, or the filtered array. They are
    what the bridge maps the model onto; `kp_scale` only seeds the bridge's
    constant fallback and is the fly's `scale.json` `scale` value.
    """
    qpos = np.asarray(stac["qpos"], np.float64)
    n = len(qpos)

    # The bridge's model points are RE-FK'd here with the fitted offsets, and
    # `stac["marker_sites"]` is deliberately not read: it is FK'd from the
    # INITIAL marker model, 2.4e-3 model units from the fitted one. Fitting a
    # per-frame similarity to systematically displaced points tilts every
    # frame's transform -- on bout 28 that alone cost 3.249 px against 2.450
    # (fly0) and 2.334 against 1.343 (fly1). Re-FK'ing reproduces both.
    marker_sites = fitted_site_xpos_from_qpos(
        anatomy, qpos, np.asarray(stac["offsets"], np.float64)
    )

    # Subset to the K marker/keypoint sites, in kp_order, NOT all nsite=165
    # model sites: `kp_names`/`order_sha` name a K-long axis, and an nsite-long
    # axis would be unnamed by that stamp -- the one thing this file's order
    # contract exists to prevent (v1: nsite=165, n_keypoints=50).
    site_xpos = site_xpos_from_qpos(anatomy, qpos)[:, anatomy.site_idxs, :]
    xpos, xquat = body_frames_from_qpos(anatomy, qpos)
    ego = egocentric_sites(site_xpos, xpos, xquat, anatomy=anatomy, body=THORAX_BODY)

    bridges = compute_bridges(
        marker_sites,
        np.asarray(kp3d_world, np.float64),
        np.asarray(conf3d, np.float32),
        kp_scale=float(kp_scale),
        source=source,
    )

    # The free joint's own pose, model frame. A slice of qpos, not the bridge.
    # Guarded rather than assumed: an anatomy with no free joint has no root
    # SE3 to slice, and taking qpos[:, :7] anyway would silently return seven
    # unrelated joint angles formatted as a pose.
    if not anatomy.has_freejoint:
        raise ValueError(
            f"anatomy {anatomy.name!r} has no free joint; qpos[:, :7] is not a "
            f"root SE3 here and must not be written as one"
        )
    root_se3 = qpos[:, :7].copy()

    scale = np.full(n, np.nan, np.float64)
    kp3d_mm = np.full(marker_sites.shape, np.nan, np.float64)
    for i in range(n):
        fit = bridges.frame(i)
        if fit is None:
            continue
        s, rot, trans = fit
        scale[i] = s
        kp3d_mm[i] = model_to_world(marker_sites[i], s, rot, trans)

    return FlyOutputs(
        qpos=qpos,
        root_se3=root_se3,
        scale=scale,
        kp3d_mm=kp3d_mm,
        site_xpos=site_xpos,
        xpos_egocentric=ego,
        bridge_ok=np.asarray(bridges.ok, bool),
        summary={
            "n_frames": int(n),
            "n_bridge_ok": int(np.count_nonzero(bridges.ok)),
            "bridge_source": bridges.source,
            "egocentric_body": THORAX_BODY,
            "pose_source": str((stac.get("attrs", {}) or {}).get("ik_solver", "per_frame")),
        },
    )


def write_outputs_h5(path, out: FlyOutputs, *, anatomy, kp_order: Order) -> None:
    """Write `outputs.h5` atomically (`.tmp` then `os.replace`).

    `order_sha` is an ATTR and `kp_names` a dataset, mirroring `stac_ik.h5`;
    a reader that finds neither must refuse the file rather than assume an
    order.
    """
    path = str(path)
    tmp = f"{path}.tmp"
    with h5py.File(tmp, "w") as f:
        for key, value in (
            ("qpos", out.qpos),
            ("root_se3", out.root_se3),
            ("scale", out.scale),
            ("kp3d_mm", out.kp3d_mm),
            ("site_xpos", out.site_xpos),
            ("xpos_egocentric", out.xpos_egocentric),
        ):
            f.create_dataset(key, data=np.asarray(value, np.float32))
        f.create_dataset("bridge_ok", data=np.asarray(out.bridge_ok, bool))
        for key, names in (
            ("kp_names", kp_order.names),
            ("names_qpos", anatomy.names_qpos),
            ("names_xpos", anatomy.names_xpos),
        ):
            f.create_dataset(key, data=np.array([str(v) for v in names], dtype="S"))
        f.attrs["order_sha"] = order_sha(kp_order)
        f.attrs["anatomy"] = str(anatomy.name)
        # Per dataset, because this file mixes two unit spaces on purpose.
        f.attrs["units"] = json.dumps(
            {
                "qpos": "model",
                "root_se3": "model",
                "scale": "dimensionless (model->world)",
                "kp3d_mm": "world (0.1 mm) -- NOT mm, name inherited",
                "site_xpos": "model",
                "xpos_egocentric": "model",
            }
        )
        f.attrs["pose_source"] = str(out.summary.get("pose_source", "per_frame"))
        f.attrs["summary"] = json.dumps(dict(out.summary), default=str)
    os.replace(tmp, path)


def write_fitted_npz(path, out: FlyOutputs, *, kp_order: Order) -> None:
    """Write `fitted.npz`: the same array `outputs.h5` stores as `kp3d_mm`.

    Two carriers, one array, because their consumers differ: `viz/kpvideo.py`
    reads `fly<f>/fitted.npz` through `load_npz`, which requires the
    `kp_names`/`order_sha` stamp that an h5 dataset has no mechanism for. The
    array is named `kp3d` inside the npz because that is the key kpvideo reads.
    Do not recompute it -- pass the same `FlyOutputs`.
    """
    save_npz(
        path,
        arrays={"kp3d": np.asarray(out.kp3d_mm, np.float64)},
        kp=kp_order,
        extra={"units": "world", "bridge_source": str(out.summary["bridge_source"])},
    )
