"""One bout-fly's `outputs.h5` and `fitted.npz`."""

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
    """One bout-fly's derived arrays. MODEL units except `kp3d_mm`."""

    qpos: np.ndarray
    root_se3: np.ndarray
    scale: np.ndarray
    kp3d_mm: np.ndarray
    site_xpos: np.ndarray
    xpos_egocentric: np.ndarray
    bridge_ok: np.ndarray
    summary: dict


def read_stac_h5(path) -> dict[str, Any]:
    """Every dataset of a `stac_ik.h5`, with `|S` name arrays decoded to `str`."""
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
    """Derive every `outputs.h5` array from one `stac_ik.h5` plus its keypoints."""
    qpos = np.asarray(stac["qpos"], np.float64)
    n = len(qpos)

    marker_sites = fitted_site_xpos_from_qpos(
        anatomy, qpos, np.asarray(stac["offsets"], np.float64)
    )

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
    """Write `outputs.h5` atomically (`.tmp` then `os.replace`)."""
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
    """Write `fitted.npz`: the same array `outputs.h5` stores as `kp3d_mm`."""
    save_npz(
        path,
        arrays={"kp3d": np.asarray(out.kp3d_mm, np.float64)},
        kp=kp_order,
        extra={"units": "world", "bridge_source": str(out.summary["bridge_source"])},
    )
