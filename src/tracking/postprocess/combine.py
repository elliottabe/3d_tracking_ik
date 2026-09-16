"""One session's bout-flies gathered into the downstream analysis h5."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

__all__ = [
    "CONTRACT_DATASETS",
    "EXTRA_DATASETS",
    "EXCLUDE_REASONS",
    "BoutEntry",
    "exclusion_reason",
    "collect_bout_entries",
    "combine_session",
    "combine_many",
]

CONTRACT_DATASETS = (
    "clip_lengths",
    "kp_data",
    "qpos",
    "qpos_names",
    "qvel",
    "xpos",
    "xquat",
)
EXTRA_DATASETS = ("site_xpos", "xpos_egocentric")
EXCLUDE_REASONS = ("no_outputs_h5", "no_solved_frames", "no_bridge", "nan_cost")


@dataclass(frozen=True)
class BoutEntry:
    key: str
    fly: int
    outputs: Path
    stac: Path
    start_frame: int
    end_frame: int
    bucket: str
    fly_id: str
    sex: str
    status: str
    missing_fraction: float | None
    fitted_px: float | None


def exclusion_reason(outputs_path, stac_path) -> str | None:
    """Why this bout-fly must not enter the dataset, or `None` if it may."""
    with h5py.File(str(outputs_path), "r") as f:
        qpos = f["qpos"][()]
        bridge_ok = f["bridge_ok"][()]
    if not np.isfinite(qpos).any():
        return "no_solved_frames"
    if not np.any(bridge_ok):
        return "no_bridge"
    if Path(stac_path).exists():
        with h5py.File(str(stac_path), "r") as f:
            raw = f.attrs.get("ik_summary")
        if raw:
            summary = json.loads(raw)
            costs = [v for k, v in summary.items() if "cost" in k.lower()]
            if costs and not all(np.isfinite(np.asarray(costs, np.float64))):
                return "nan_cost"
    return None


def _bucket_of(run_root: Path) -> str:
    """The assay directory name (`courtship`, `free_running`) above the session."""
    parts = Path(run_root).resolve().parts
    for known in ("courtship", "free_running"):
        if known in parts:
            return known
    return "unknown"


def _fly_id_of(run_root: Path) -> str:
    """`<Session>/<recording>` — the session-tagged form the bout tables use."""
    parts = Path(run_root).resolve().parts
    if len(parts) < 3:
        raise ValueError(
            f"run_root {run_root!r} is too shallow to name a recording; expected "
            f"<assay>/<Session>/<recording>/<pose dir>"
        )
    session, recording = parts[-3], parts[-2]
    return f"{session}/{recording}"


def _bout_frame_range(bout_dir: Path, n_frames: int) -> tuple[int, int]:
    """The bout's ABSOLUTE frame range in the recording, from `mvq_meta.json`."""
    meta = bout_dir / "mvq_meta.json"
    if not meta.exists():
        return 0, int(n_frames)
    try:
        d = json.loads(meta.read_text())
        start = int(d["bout_start_frame"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return 0, int(n_frames)
    return start, start + int(n_frames)


def collect_bout_entries(
    run_root, *, fly_id=None, bucket=None
) -> tuple[list[BoutEntry], list[dict]]:
    """Every `bouts/bout_*/fly*` DIRECTORY under `run_root`, in bout order."""
    from tracking.qc.session import collect_rows

    run_root = Path(run_root)
    bucket = bucket if bucket is not None else _bucket_of(run_root)
    fly_id = fly_id if fly_id is not None else _fly_id_of(run_root)
    qc_by_key = {(row.bout, row.fly): row for row in collect_rows(run_root)}

    usable: list[BoutEntry] = []
    excluded: list[dict] = []
    for fly_dir in sorted(run_root.glob("bouts/bout_*/fly*")):
        if not fly_dir.is_dir():
            continue
        bout_dir = fly_dir.parent
        key = bout_dir.name
        fly = int(fly_dir.name.removeprefix("fly"))
        row = qc_by_key.get((key, fly))
        sex = row.sex if row is not None else "unknown"
        out_path = fly_dir / "outputs.h5"
        if not out_path.exists():
            excluded.append(
                {"key": key, "fly": fly, "fly_id": fly_id, "sex": sex, "reason": "no_outputs_h5"}
            )
            continue
        stac = fly_dir / "stac_ik.h5"
        reason = exclusion_reason(out_path, stac)
        if reason is not None:
            excluded.append(
                {"key": key, "fly": fly, "fly_id": fly_id, "sex": sex, "reason": reason}
            )
            continue
        with h5py.File(str(out_path), "r") as f:
            n = int(f["qpos"].shape[0])
        start_frame, end_frame = _bout_frame_range(bout_dir, n)
        usable.append(
            BoutEntry(
                key=key,
                fly=fly,
                outputs=out_path,
                stac=stac,
                start_frame=start_frame,
                end_frame=end_frame,
                bucket=bucket,
                fly_id=fly_id,
                sex=sex,
                status=row.status if row is not None else "unknown",
                missing_fraction=row.missing_fraction if row is not None else None,
                fitted_px=row.fitted_px if row is not None else None,
            )
        )
    return usable, excluded


def _pad_stack(arrays: list[np.ndarray], tmax: int) -> np.ndarray:
    """`(N, Tmax, ...)` zero-padded; the true lengths live in `clip_lengths`."""
    shape = (len(arrays), tmax, *arrays[0].shape[1:])
    out = np.zeros(shape, np.float32)
    for i, a in enumerate(arrays):
        out[i, : len(a)] = a
    return out


def _fly_uid(fly_id: str, fly: int) -> str:
    """`"<fly_id>/fly<N>"` -- the grouping key a train/val split must key on."""
    return f"{fly_id}/fly{fly}"


def _write_info_group(info, entries: list[BoutEntry], excluded: list[dict]) -> None:
    """Everything under `info/` that is not one of the seven contract
    datasets: identity, quality and exclusion bookkeeping, built from the
    SAME `entries`/`excluded` rows the arrays above came from, plus a
    compound `info/table` built in the same pass so the two representations
    of one row cannot drift apart.
    """
    str_dtype = h5py.string_dtype(encoding="utf-8")

    def s(seq):
        return np.array([str(v) for v in seq], dtype="S")

    info.create_dataset("bout_keys", data=s(e.key for e in entries))
    info.create_dataset("fly_ids", data=s(e.fly_id for e in entries))
    info.create_dataset("buckets", data=s(e.bucket for e in entries))
    info.create_dataset("source_flies", data=np.asarray([e.fly for e in entries], np.int32))
    info.create_dataset("start_frames", data=np.asarray([e.start_frame for e in entries], np.int32))
    info.create_dataset("end_frames", data=np.asarray([e.end_frame for e in entries], np.int32))
    info.create_dataset("sex", data=s(e.sex for e in entries))
    info.create_dataset("fly_uid", data=s(_fly_uid(e.fly_id, e.fly) for e in entries))
    info.create_dataset("status", data=s(e.status for e in entries))
    info.create_dataset(
        "missing_fraction",
        data=np.asarray(
            [np.nan if e.missing_fraction is None else e.missing_fraction for e in entries],
            np.float32,
        ),
    )
    info.create_dataset(
        "fitted_px",
        data=np.asarray(
            [np.nan if e.fitted_px is None else e.fitted_px for e in entries], np.float32
        ),
    )

    ex = info.create_group("excluded")
    ex.create_dataset("bout_keys", data=s(r["key"] for r in excluded))
    ex.create_dataset("flies", data=np.asarray([r["fly"] for r in excluded], np.int32))
    ex.create_dataset("fly_ids", data=s(r["fly_id"] for r in excluded))
    ex.create_dataset("sex", data=s(r.get("sex", "unknown") for r in excluded))
    ex.create_dataset("reasons", data=s(r["reason"] for r in excluded))

    table_dtype = np.dtype(
        [
            ("bout_key", str_dtype),
            ("fly", np.int32),
            ("fly_id", str_dtype),
            ("bucket", str_dtype),
            ("sex", str_dtype),
            ("fly_uid", str_dtype),
            ("status", str_dtype),
            ("missing_fraction", np.float32),
            ("fitted_px", np.float32),
            ("start_frame", np.int32),
            ("end_frame", np.int32),
        ]
    )
    table = np.zeros(len(entries), dtype=table_dtype)
    for i, e in enumerate(entries):
        table[i] = (
            e.key,
            e.fly,
            e.fly_id,
            e.bucket,
            e.sex,
            _fly_uid(e.fly_id, e.fly),
            e.status,
            np.nan if e.missing_fraction is None else e.missing_fraction,
            np.nan if e.fitted_px is None else e.fitted_px,
            e.start_frame,
            e.end_frame,
        )
    info.create_dataset("table", data=table)

    n_male = sum(1 for e in entries if e.sex == "male")
    n_female = sum(1 for e in entries if e.sex == "female")
    info.attrs["n_clips"] = len(entries)
    info.attrs["n_male"] = n_male
    info.attrs["n_female"] = n_female
    info.attrs["n_unknown"] = len(entries) - n_male - n_female
    info.attrs["n_excluded"] = len(excluded)


def _combine(
    entries, *, anatomy, source_hz, git_sha, out_path, source_file, excluded, padded=False
):
    if not entries:
        raise ValueError(
            f"no usable bout-fly outputs to combine (excluded {len(excluded)}); "
            f"a combined file with no clips would assert an empty session as a result"
        )
    FROM_OUTPUTS = ("qpos", "site_xpos", "xpos_egocentric")
    FROM_STAC = ("qvel", "xpos", "xquat")
    per: dict[str, list[np.ndarray]] = {k: [] for k in (*FROM_OUTPUTS, *FROM_STAC)}
    kp_flat: list[np.ndarray] = []
    lengths: list[int] = []
    for e in entries:
        with h5py.File(str(e.outputs), "r") as f:
            for k in FROM_OUTPUTS:
                per[k].append(np.asarray(f[k][()], np.float32))
        with h5py.File(str(e.stac), "r") as f:
            for k in FROM_STAC:
                per[k].append(np.asarray(f[k][()], np.float32))
            # MODEL units -- the same frame and scale as qpos, matching the
            # downstream contract. NOT outputs.h5's world-unit `kp3d_mm`.
            kp = np.asarray(f["kp_data"][()], np.float32)
        kp_flat.append(kp.reshape(len(kp), -1))
        lengths.append(len(kp))
        if len(kp) != len(per["qpos"][-1]):
            raise ValueError(
                f"{e.key} fly{e.fly}: stac_ik.h5 has {len(kp)} frames but "
                f"outputs.h5 has {len(per['qpos'][-1])}; they describe the same "
                f"solve and a mismatch means one is stale"
            )
    tmax = int(max(lengths))

    tmp = f"{out_path}.tmp"
    with h5py.File(tmp, "w") as f:
        f.create_dataset("clip_lengths", data=np.asarray(lengths, np.int32))
        if padded:
            f.create_dataset("kp_data", data=_pad_stack(kp_flat, tmax))
            for k, v in per.items():
                f.create_dataset(k, data=_pad_stack(v, tmax))
        else:
            for i, (kp, key) in enumerate(zip(kp_flat, entries, strict=True)):
                g = f.create_group(f"bout_{i:03d}")
                g.create_dataset("kp_data", data=kp)
                for k, v in per.items():
                    g.create_dataset(k, data=v[i])
                g.attrs["bout_key"] = str(key.key)
                g.attrs["fly"] = int(key.fly)
                g.attrs["n_frames"] = int(len(kp))
        grp = f.create_group("qpos_names")
        for i, name in enumerate(anatomy.names_qpos):
            grp.create_dataset(str(i), data=np.bytes_(str(name)))
        _write_info_group(f.create_group("info"), entries, excluded)
        f.attrs["anatomy"] = str(anatomy.name)
        f.attrs["clip_lengths_are_true"] = True
        f.attrs["layout"] = "padded" if padded else "per_bout_groups"
        f.attrs["git_sha"] = str(git_sha)
        f.attrs["source_file"] = str(source_file)
        f.attrs["source_hz"] = float(source_hz)
        f.attrs["target_hz"] = float(source_hz)
    os.replace(tmp, str(out_path))
    return {
        "n_clips": len(entries),
        "tmax": tmax if padded else None,
        "layout": "padded" if padded else "per_bout_groups",
        "excluded": excluded,
        "out_path": str(out_path),
    }


def combine_session(
    run_root, *, anatomy, source_hz, git_sha, out_path, fly_id=None, bucket=None, padded=False
) -> dict[str, Any]:
    """One session -> one combined h5. The default unit."""
    entries, excluded = collect_bout_entries(run_root, fly_id=fly_id, bucket=bucket)
    return _combine(
        entries,
        anatomy=anatomy,
        source_hz=source_hz,
        git_sha=git_sha,
        out_path=out_path,
        source_file=str(run_root),
        excluded=excluded,
        padded=padded,
    )


def combine_many(
    run_roots, *, anatomy, source_hz, git_sha, out_path, padded=False
) -> dict[str, Any]:
    """Several sessions pooled into one file. EXPLICIT, never the default."""
    entries: list[BoutEntry] = []
    excluded: list[dict] = []
    for root in run_roots:
        e, x = collect_bout_entries(root)
        entries.extend(e)
        excluded.extend(x)
    return _combine(
        entries,
        anatomy=anatomy,
        source_hz=source_hz,
        git_sha=git_sha,
        out_path=out_path,
        source_file=json.dumps([str(r) for r in run_roots]),
        excluded=excluded,
        padded=padded,
    )
