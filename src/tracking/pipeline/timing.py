"""`timing.json`: how long each stage took, and over how many items.

Spec section 9 puts timings on the same footing as numbers -- "a stage that gets
materially slower is a regression to be explained" -- so the parity harness
reads this file, which means it has to survive the runs it measures.

`n_items` is not optional metadata. A bare duration is not comparable across
bouts: the same per-frame solver measured 158 s per 500 frames on a 2007-frame
bout and 329 s per 500 on a 550-frame one, because a one-off JIT compile
amortises over fewer frames. Without the count, "slower" and "shorter" look
identical.

**One ledger, many writers.** A SLURM array puts 30 processes on ONE run root
at once, and a single read-modify-write of one JSON file cannot survive that.
Measured 2026-09-15 on Session0's `postprocess[0-29]`: every task built the
same `timing.json.tmp`, one task's `os.replace` consumed it, and the next
task's raised `FileNotFoundError` -- which failed the array task, which left
`collect` on a dependency that could never be satisfied. The silent half was
worse: a task that read the ledger before another task's write put it back
without that stage, so entries vanished with no error at all.

So `record` never rewrites a shared dict. It writes ONE stage's own file under
`timing.d/`, and `timing.json` is a derived flattening of that directory.
Concurrent flattenings still race, but they race over a file whose source of
truth is the shards, so a stale one self-heals on the next `record` and no
entry can be lost. Per-stage (not per-process) shard names keep a re-run
rewriting the same paths rather than accumulating one file per pid.
"""

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
    """One JSON object, or `{}` if it is absent or unreadable.

    An unreadable ledger is ANNOUNCED to stderr and then replaced, rather than
    raising: timings are diagnostics, and losing a 40-minute stage's result
    because its ledger was truncated would be the tail wagging the dog. stderr
    rather than `warnings` because `warnings` dedupes per process and a session
    run would show this once across hundreds of bouts.
    """
    if not p.exists():
        return {}
    try:
        got = json.loads(p.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(f"timing: {p} is unreadable ({exc}); starting a fresh ledger", file=sys.stderr)
        return {}
    return got if isinstance(got, dict) else {}


def _write_json(p: Path, obj: dict) -> None:
    """Atomic write through a tmp name UNIQUE TO THIS PROCESS.

    The pid is what makes this safe under an array: a shared `.tmp` is not an
    atomic write, it is a rendezvous. See the module docstring.
    """
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, p)


def load(path) -> dict:
    """The merged ledger: `timing.json` overlaid with every shard.

    Reading the flattened file first keeps ledgers written before `timing.d/`
    existed readable; the shards win, because they are the source of truth.
    """
    ledger = _read_json(Path(path))
    for shard in sorted(shard_dir(path).glob("*.json")):
        ledger.update(_read_json(shard))
    return ledger


def record(path, stage: str, seconds: float, *, ok: bool = True, **meta) -> None:
    """Append one stage's timing, replacing only that stage's entry.

    Writes this stage's shard, then re-flattens `timing.json` from the whole
    shard directory. Another process racing the flattening can only produce a
    stale `timing.json`, never a lost entry -- the shards keep them all.
    """
    entry: dict = {"seconds": float(seconds), "ok": bool(ok), **meta}
    n = meta.get("n_items")
    if n:
        entry["seconds_per_item"] = float(seconds) / float(n)

    _write_json(shard_dir(path) / f"{stage}.json", {stage: entry})
    _write_json(Path(path), load(path))


@contextlib.contextmanager
def timed(path, stage: str, **meta):
    """Time a stage, recording on the way out EVEN IF the body raises.

    A stage that crashed at forty minutes is the most interesting timing in the
    file; recording only on success loses exactly the runs worth investigating.
    The entry is stamped `ok: False` so a failed run is never mistaken for a
    fast one.
    """
    t0 = time.time()
    try:
        yield
    except BaseException:
        record(path, stage, time.time() - t0, ok=False, **meta)
        raise
    record(path, stage, time.time() - t0, ok=True, **meta)
