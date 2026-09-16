"""The FINAL collect: name sessions, get one h5 with everything in it."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("JAX_PLATFORMS", "cpu")

DEFAULT_BASE = Path("/gscratch/portia/eabe/data/Johnson_lab/processed")


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def find_run_roots(base: Path, assay: str, sessions: list[str], run_name: str) -> list[Path]:
    """Every `<base>/<assay>/<session>/<recording>/<run_name>` holding bouts."""
    roots: list[Path] = []
    empty: list[str] = []
    for session in sessions:
        found = sorted(
            r for r in base.glob(f"{assay}/{session}/*/{run_name}") if (r / "bouts").is_dir()
        )
        if not found:
            empty.append(session)
        roots.extend(found)
    if empty:
        raise SystemExit(
            f"no run roots for session(s) {empty} under "
            f"{base}/{assay}/<session>/*/{run_name} -- check the session name, the "
            f"assay, or --run-name. Pooling the sessions that DID match would "
            f"quietly give you less than you asked for."
        )
    return roots


def list_sessions(base: Path, assay: str, run_name: str) -> list[str]:
    """Session directories under `assay` that hold at least one such run root."""
    return sorted(
        {
            r.parents[1].name
            for r in base.glob(f"{assay}/*/*/{run_name}")
            if (r / "bouts").is_dir()
        }
    )


def read_headers(run_roots: list[Path]) -> list[dict]:
    """Each root's own per-recording combined h5 header."""
    import h5py

    out = []
    for root in run_roots:
        matches = sorted(root.glob("ik_output_combined_*.h5"))
        if not matches:
            out.append({"root": root, "missing": True})
            continue
        with h5py.File(matches[0]) as f:
            out.append(
                {
                    "root": root,
                    "missing": False,
                    "source_hz": float(f.attrs["source_hz"]),
                    "anatomy": str(f.attrs["anatomy"]),
                    "n": int(len(f["info/sex"])),
                }
            )
    return out


def check_poolable(headers: list[dict], fly_ids: list[str]) -> float:
    """Refuse a pool whose parts disagree. Returns the agreed `source_hz`."""
    missing = [h["root"] for h in headers if h["missing"]]
    if missing:
        raise SystemExit(
            "refusing to pool: these run roots have no combined h5, so `collect` "
            "never ran for them and their bouts would join the pool without ever "
            "having been scored:\n  " + "\n  ".join(str(m) for m in missing)
        )

    rates = {h["source_hz"] for h in headers}
    if len(rates) > 1:
        detail = "\n  ".join(f"{h['source_hz']:g} Hz  {h['root']}" for h in headers)
        raise SystemExit(
            f"refusing to pool: run roots disagree about sample rate {sorted(rates)}.\n"
            f"  {detail}\n"
            f"One `source_hz` would label every clip from the other rate with a rate "
            f"that was never run, and qvel is in units of it."
        )

    models = {h["anatomy"] for h in headers}
    if len(models) > 1:
        detail = "\n  ".join(f"{h['anatomy']:10s} {h['root']}" for h in headers)
        raise SystemExit(
            f"refusing to pool: run roots disagree about anatomy {sorted(models)}.\n"
            f"  {detail}\n"
            f"`qpos_names` is written once for the pool, so one model's names would "
            f"index another model's qpos."
        )

    dupes = sorted({i for i in fly_ids if fly_ids.count(i) > 1})
    if dupes:
        raise SystemExit(
            f"refusing to pool: these fly_ids appear more than once: {dupes}. "
            f"`info/fly_ids` is what splits the pool back apart by recording; "
            f"duplicates merge two recordings under one label."
        )
    return rates.pop()


def write_provenance(out_path: Path, *, headers, fly_ids, sessions, assay, run_name) -> None:
    """Add `info/session` and the `provenance/` recording table."""
    import h5py
    import numpy as np

    present = [(h, fid) for h, fid in zip(headers, fly_ids, strict=True) if not h["missing"]]
    with h5py.File(out_path, "a") as f:
        n_clips = len(f["info/fly_ids"])
        stops = np.cumsum([h["n"] for h, _ in present])
        if stops[-1] != n_clips:
            raise ValueError(
                f"{out_path}: the per-recording clip counts sum to {stops[-1]} but the "
                f"pooled file holds {n_clips}. The index ranges in provenance/ would "
                f"point at the wrong clips, so nothing is written"
            )
        starts = np.concatenate([[0], stops[:-1]])

        stored = [s.decode() if isinstance(s, bytes) else str(s) for s in f["info/fly_ids"][:]]
        per_clip_session = [s.split("/")[0] for s in stored]
        if "session" in f["info"]:
            del f["info/session"]
        f["info"].create_dataset("session", data=np.array(per_clip_session, dtype="S64"))

        if "provenance" in f:
            del f["provenance"]
        g = f.create_group("provenance")
        g.attrs["assay"] = assay
        g.attrs["run_name"] = run_name
        g.attrs["sessions_requested"] = json.dumps(sessions)
        g.attrs["created_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
        g.attrs["script"] = "scripts/collect_sessions.py"
        g.attrs["n_recordings"] = len(present)
        g.create_dataset("fly_id", data=np.array([fid for _, fid in present], dtype="S128"))
        g.create_dataset(
            "session", data=np.array([fid.split("/")[0] for _, fid in present], dtype="S64")
        )
        g.create_dataset(
            "run_root", data=np.array([str(h["root"]) for h, _ in present], dtype="S512")
        )
        g.create_dataset("n_clips", data=np.array([h["n"] for h, _ in present], np.int32))
        g.create_dataset("clip_start", data=starts.astype(np.int32))
        g.create_dataset("clip_stop", data=stops.astype(np.int32))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Pool named sessions' per-recording collects into one h5."
    )
    ap.add_argument("--session", action="append", default=[], help="e.g. Session0; repeatable")
    ap.add_argument("--all-sessions", action="store_true", help="every session under --assay")
    ap.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--assay", default="courtship", help="courtship | free_running")
    ap.add_argument("--run-name", default="pose_v2_20260914")
    ap.add_argument("--out", type=Path, default=None, help="default: <base>/<assay>/<name>.h5")
    ap.add_argument(
        "--rl",
        action="store_true",
        help="RL dataset: the (N, Tmax, ...) zero-padded stack instead of per-bout "
        "groups. Pair with --target-hz for the interpolated half.",
    )
    ap.add_argument(
        "--target-hz",
        type=float,
        default=None,
        help="with --rl, resample the padded file to this rate "
        "(tracking.postprocess.resample). The RL contract file is 1000 Hz.",
    )
    ap.add_argument("--dry-run", action="store_true", help="check and report, write nothing")
    args = ap.parse_args(argv)

    if args.all_sessions and args.session:
        raise SystemExit("--all-sessions and --session are mutually exclusive")
    if args.target_hz and not args.rl:
        raise SystemExit(
            "--target-hz requires --rl: resampling operates on the padded time "
            "axis, so the interpolated dataset is the padded one. Use --rl "
            "--target-hz 1000 for the RL contract file."
        )
    sessions = args.session
    if args.all_sessions:
        sessions = list_sessions(args.base_dir, args.assay, args.run_name)
        if not sessions:
            raise SystemExit(
                f"no sessions under {args.base_dir}/{args.assay}/*/*/{args.run_name}"
            )
    if not sessions:
        available = list_sessions(args.base_dir, args.assay, args.run_name)
        raise SystemExit(
            f"name at least one --session (or --all-sessions). "
            f"Available for {args.assay}/{args.run_name}: {available or '(none)'}"
        )

    roots = find_run_roots(args.base_dir, args.assay, sessions, args.run_name)

    from tracking.postprocess.combine import _fly_id_of

    fly_ids = [_fly_id_of(r) for r in roots]
    headers = read_headers(roots)

    print(f"assay {args.assay!r}, run {args.run_name!r}, sessions {sessions}")
    for h, fid in zip(headers, fly_ids, strict=True):
        print(f"  {fid:44s} {'MISSING collect' if h['missing'] else f'{h['n']:4d} clips'}")

    source_hz = check_poolable(headers, fly_ids)
    total = sum(h["n"] for h in headers)
    anatomy_name = headers[0]["anatomy"]
    print(
        f"agreed: {source_hz:g} Hz, anatomy {anatomy_name}, "
        f"{len(roots)} recordings, {total} clips"
    )

    suffix = "_rl_padded" if args.rl else ""
    if args.rl and args.target_hz:
        suffix = f"_rl_{args.target_hz:g}hz_interp_padded"
    out_path = args.out or (
        args.base_dir
        / args.assay
        / f"ik_output_combined_{anatomy_name}_{args.run_name}_{'+'.join(sessions)}{suffix}.h5"
    )
    print(f"layout: {'padded (RL)' if args.rl else 'per-bout groups'}")
    print(f"output: {out_path}")

    if args.dry_run:
        print("(dry-run) nothing written")
        return 0

    import hydra
    from hydra import compose, initialize_config_dir

    import tracking.utils.path_utils  # noqa: F401 -- registers ${repo_root:} before compose
    from tracking.postprocess.combine import combine_many
    from tracking.run import _load_anatomy

    if hydra.core.global_hydra.GlobalHydra().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        cfg = compose(config_name="pipeline", overrides=[f"run.name={args.run_name}"])
    anatomy = _load_anatomy(cfg)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combine_target = out_path.with_name(out_path.stem + ".base.h5") if args.target_hz else out_path
    report = combine_many(
        roots,
        anatomy=anatomy,
        source_hz=source_hz,
        git_sha=_git_sha(),
        out_path=combine_target,
        padded=args.rl,
    )
    if args.target_hz:
        from tracking.postprocess.resample import resample_file

        print(f"resampling {source_hz:g} Hz -> {args.target_hz:g} Hz ...")
        report["resample"] = resample_file(
            combine_target, out_path, target_hz=args.target_hz
        )
        Path(combine_target).unlink()
    write_provenance(
        out_path,
        headers=headers,
        fly_ids=fly_ids,
        sessions=sessions,
        assay=args.assay,
        run_name=args.run_name,
    )

    print(json.dumps({k: v for k, v in report.items() if k != "entries"}, indent=2, default=str))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
