"""`timing.json`: how long each stage took, and over how many items."""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from pathlib import Path

__all__ = ["record", "load", "shard_dir", "timed"]

SHARD_DIRNAME = "timing.d"


def shard_dir(path) -> Path:
    """Where `path`'s per-stage shards live, beside it."""
    return Path(path).parent / SHARD_DIRNAME


def _read_json(p: Path) -> dict:
    """One JSON object, or `{}` if it is absent or unreadable."""
    if not p.exists():
        return {}
    try:
        got = json.loads(p.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(f"timing: {p} is unreadable ({exc}); starting a fresh ledger", file=sys.stderr)
        return {}
    return got if isinstance(got, dict) else {}


def _write_json(p: Path, obj: dict) -> None:
    """Atomic write through a tmp name UNIQUE TO THIS PROCESS."""
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, p)


def load(path) -> dict:
    """The merged ledger: `timing.json` overlaid with every shard."""
    ledger = _read_json(Path(path))
    for shard in sorted(shard_dir(path).glob("*.json")):
        ledger.update(_read_json(shard))
    return ledger


def record(path, stage: str, seconds: float, *, ok: bool = True, **meta) -> None:
    """Append one stage's timing, replacing only that stage's entry."""
    entry: dict = {"seconds": float(seconds), "ok": bool(ok), **meta}
    n = meta.get("n_items")
    if n:
        entry["seconds_per_item"] = float(seconds) / float(n)

    _write_json(shard_dir(path) / f"{stage}.json", {stage: entry})
    _write_json(Path(path), load(path))


@contextlib.contextmanager
def timed(path, stage: str, **meta):
    """Time a stage, recording on the way out EVEN IF the body raises."""
    t0 = time.time()
    try:
        yield
    except BaseException:
        record(path, stage, time.time() - t0, ok=False, **meta)
        raise
    record(path, stage, time.time() - t0, ok=True, **meta)
