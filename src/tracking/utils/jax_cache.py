"""JAX's persistent compilation cache, on by default for this pipeline.

The IK compiles a large jaxls program: measured on this repo, one solve of an
8-frame problem costs **72.5 s** in a cold process and **0.13 s** once
compiled -- a ratio of about 560. Without a persistent cache every process
pays that again, so a test run, a bout, and a re-run of the same bout each
re-compile from scratch.

With the cache warm, the same first solve in a NEW process costs **22.9 s**
instead of 72.5 -- 50 seconds back per process, on a cache shared across
runs, sessions and machines that see the same shapes.

This is not the whole answer on its own: the offsets fit changes marker site
positions every outer iteration, and had they stayed jaxpr constants each
iteration would be a different program with a different cache key. They are
traced variables instead (`solver.py`'s `OffsetVar`/`FrozenVar`), so one
compiled program serves the whole fit.

The directory defaults to `$TMPDIR/jax_cache` (falling back to
`/tmp/<user>/jax_cache`) and is overridden by `TRACKING_JAX_CACHE_DIR`. Set
it to a shared filesystem path to share compilations across nodes; set it
empty to disable.
"""

from __future__ import annotations

import getpass
import os
from pathlib import Path

__all__ = ["cache_dir", "enable"]

_MIN_COMPILE_SECS = 1.0


def cache_dir() -> Path | None:
    """Where compiled executables are cached, or None when disabled."""
    override = os.environ.get("TRACKING_JAX_CACHE_DIR")
    if override is not None:
        return Path(override) if override.strip() else None
    base = os.environ.get("TMPDIR") or f"/tmp/{getpass.getuser()}"
    return Path(base) / "jax_cache"


def enable() -> Path | None:
    """Turn on JAX's persistent compilation cache. Idempotent; returns the dir.

    Safe to call before or after `jax` is imported, and a no-op when the cache
    is disabled or the directory cannot be created -- a cache that cannot be
    written is a performance problem, never a correctness one, so it must not
    take a pipeline down.
    """
    path = cache_dir()
    if path is None:
        return None
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None

    import jax

    jax.config.update("jax_compilation_cache_dir", str(path))
    # Only cache what is worth caching: below this, the disk round-trip costs
    # more than the compile it saves.
    jax.config.update("jax_persistent_cache_min_compile_time_secs", _MIN_COMPILE_SECS)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    return path
