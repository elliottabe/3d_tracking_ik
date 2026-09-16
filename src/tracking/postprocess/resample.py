"""Resample a combined dataset to a target rate. A STANDALONE CLI, not a stage."""

from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np

from tracking.postprocess.combine import CONTRACT_DATASETS, EXTRA_DATASETS

__all__ = ["resample_clip", "resample_file", "main"]

_QUAT_DATASETS = ("xquat",)
_TIME_DATASETS = (
    tuple(name for name in CONTRACT_DATASETS if name not in ("clip_lengths", "qpos_names"))
    + EXTRA_DATASETS
)


def _qpos_names_list(qpos_names_group: h5py.Group) -> list[str]:
    """`qpos_names` group (keys are `str(index)`) -> an ordered plain-str list."""
    items = sorted(qpos_names_group.items(), key=lambda kv: int(kv[0]))
    names: list[str] = []
    for _, node in items:
        val = node[()]
        names.append(val.decode() if isinstance(val, bytes) else str(val))
    return names


def _free_joint_quat_slice(qpos_names: list[str]) -> slice | None:
    """`slice(3, 7)` when qpos starts with a 7-DOF free joint, else `None`."""
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in qpos_names]
    if len(names) >= 7 and all(n == "free" for n in names[:7]):
        return slice(3, 7)
    return None


def resample_clip(values: np.ndarray, n_src: int, n_dst: int, *, kind: str) -> np.ndarray:
    """`(T, ...)` -> `(n_dst, ...)`, reading only the first `n_src` rows."""
    src = np.asarray(values[:n_src], np.float64)
    if n_dst == n_src:
        return src.astype(np.float32)
    t_src = np.linspace(0.0, 1.0, n_src)
    t_dst = np.linspace(0.0, 1.0, n_dst)
    flat = src.reshape(n_src, -1)
    if kind == "quat":
        q = src.reshape(n_src, -1, 4).copy()
        for t in range(1, n_src):
            dot = np.sum(q[t] * q[t - 1], axis=-1)
            q[t] = np.where(dot[:, None] < 0.0, -q[t], q[t])
        flat = q.reshape(n_src, -1)
    out = np.stack([np.interp(t_dst, t_src, flat[:, c]) for c in range(flat.shape[1])], axis=1)
    out = out.reshape(n_dst, *src.shape[1:])
    if kind == "quat":
        norm = np.linalg.norm(out, axis=-1, keepdims=True)
        out = out / np.where(norm == 0.0, 1.0, norm)
    return out.astype(np.float32)


def resample_file(src_path, dst_path, *, target_hz: float) -> dict:
    """Write `dst_path` as `src_path` resampled to `target_hz`."""
    tmp = f"{dst_path}.tmp"
    with h5py.File(str(src_path), "r") as src:
        if "source_hz" not in src.attrs:
            raise KeyError(
                f"{src_path} has no `source_hz` attr; the rate is required and is "
                f"never inferred from a frame count"
            )
        source_hz = float(src.attrs["source_hz"])
        lengths = np.asarray(src["clip_lengths"][()], np.int64)
        new_lengths = np.maximum(
            1, np.rint(lengths * (float(target_hz) / source_hz)).astype(np.int64)
        )
        tmax = int(new_lengths.max())
        quat_sl: slice | None = None
        qpos_names_group = src.get("qpos_names")
        if isinstance(qpos_names_group, h5py.Group):
            quat_sl = _free_joint_quat_slice(_qpos_names_list(qpos_names_group))
        with h5py.File(tmp, "w") as dst:
            dst.create_dataset("clip_lengths", data=new_lengths.astype(np.int32))
            for name in src.keys():
                if name in ("clip_lengths",):
                    continue
                node = src[name]
                if isinstance(node, h5py.Group):
                    src.copy(name, dst, name=name)
                    continue
                if name not in _TIME_DATASETS:
                    dst.create_dataset(name, data=node[()])
                    continue
                data = node[()]
                if data.ndim < 2 or data.shape[0] != lengths.shape[0]:
                    raise ValueError(
                        f"'{name}' is a declared time-axis dataset but has shape "
                        f"{data.shape}; expected >= 2 dims with leading dim "
                        f"{lengths.shape[0]} (clip count)"
                    )
                kind = "quat" if name in _QUAT_DATASETS else "linear"
                out = np.zeros((data.shape[0], tmax, *data.shape[2:]), np.float32)
                for i in range(data.shape[0]):
                    n_dst = int(new_lengths[i])
                    out[i, :n_dst] = resample_clip(data[i], int(lengths[i]), n_dst, kind=kind)
                    if name == "qpos" and quat_sl is not None:
                        block = data[i][:, quat_sl][:, None, :]
                        out[i, :n_dst, quat_sl] = resample_clip(
                            block, int(lengths[i]), n_dst, kind="quat"
                        )[:, 0, :]
                dst.create_dataset(name, data=out)
            for key in src.attrs:
                dst.attrs[key] = src.attrs[key]
            dst.attrs["source_hz"] = source_hz
            dst.attrs["target_hz"] = float(target_hz)
    os.replace(tmp, str(dst_path))
    return {"source_hz": source_hz, "target_hz": float(target_hz), "tmax": tmax}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tracking.postprocess.resample")
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--target-hz", dest="target_hz", type=float, required=True)
    args = ap.parse_args(argv)
    report = resample_file(args.src, args.dst, target_hz=args.target_hz)
    print(
        f"{args.src} -> {args.dst}: {report['source_hz']:.1f} Hz -> "
        f"{report['target_hz']:.1f} Hz, Tmax {report['tmax']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
