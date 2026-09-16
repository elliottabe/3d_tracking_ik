"""JAX's persistent compilation cache, on by default for this pipeline."""

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
    """Turn on JAX's persistent compilation cache. Idempotent; returns the dir."""
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
