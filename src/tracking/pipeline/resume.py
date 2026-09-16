"""Is this stage's work already done?"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from tracking.pipeline.stages import Stage

__all__ = ["stage_is_current", "write_signature", "read_signature"]


def _sig_path(out_dir: Path, stage: str) -> Path:
    return Path(out_dir) / f".{stage}.signature"


def read_signature(out_dir, stage: str) -> str | None:
    """The signature recorded for `stage`, or None if there is not one."""
    p = _sig_path(Path(out_dir), stage)
    return p.read_text().strip() if p.exists() else None


def write_signature(out_dir, stage: str, signature: str) -> None:
    """Record `signature` for `stage`, atomically."""
    p = _sig_path(Path(out_dir), stage)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(str(signature))
    os.replace(tmp, p)


def stage_is_current(
    stage: Stage,
    *,
    out_dir,
    artifacts: Sequence[str],
    signature: str | None = None,
) -> bool:
    """True only if re-running `stage` would change nothing.

    Args:
        stage: the registry row (`stages.stage_by_name`).
        out_dir: where this stage's artifacts live.
        artifacts: the artifact FILENAMES to check, resolved by the caller --
            the registry's `writes` carries placeholders like
            `offsets_fly.h5` that only the caller can fill in per fly.
        signature: required when `stage.resume == "signature"`; ignored
            otherwise.

    ALL artifacts must be present, not just one: a stage that crashed between
    its first and third write leaves a directory that passes any weaker check.
    """
    out_dir = Path(out_dir)

    if stage.resume == "always":
        return False

    if stage.resume == "partial":
        raise NotImplementedError(
            f"stage {stage.name!r} resumes on a CHUNKED PARTIAL artifact, whose "
            f"presence proves nothing about completeness (spec section 5). That "
            f"stage owns its own chunk-level resume check; this predicate refuses "
            f"rather than reporting a truncated run as done"
        )

    if not all((out_dir / name).exists() for name in artifacts):
        return False

    if stage.resume == "signature":
        if signature is None:
            raise ValueError(
                f"stage {stage.name!r} resumes on a signature, but none was given. "
                f"Refusing to fall back to an existence check: a different "
                f"checkpoint or threshold writes these same filenames, and "
                f"reusing them would mix two computations in one run silently"
            )
        return read_signature(out_dir, stage.name) == str(signature)

    return True
