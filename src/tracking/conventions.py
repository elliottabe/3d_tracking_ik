"""The two rules every module in this package answers the same way."""

from __future__ import annotations

import difflib
import sys
from collections.abc import Iterable, Mapping
from types import MappingProxyType

__all__ = [
    "NotFit",
    "announce",
    "nearest_key",
    "plain_mapping",
    "refuse_config_object",
    "unknown_key_error",
]


class NotFit(ValueError):
    """This bout-fly cannot be fitted from the data it has. Not a defect."""


def announce(module: str, func: str, message: str) -> None:
    """One stderr line, tagged `[module] func: message`."""
    print(f"[{module}] {func}: {message}", file=sys.stderr)


# `dict` and the shipped read-only views of one. Anything else -- a `DictConfig`
# above all -- is refused at the boundary rather than converted here.
_PLAIN = (dict, MappingProxyType)


def refuse_config_object(value, *, what: str, api: str, allow: tuple[type, ...] = _PLAIN) -> None:
    """Raise `TypeError` unless `value` is a plain mapping. Decision D6."""
    if isinstance(value, allow):
        return
    kind = f"{type(value).__module__}.{type(value).__qualname__}"
    fix = (
        "OmegaConf.to_container(cfg, resolve=True)"
        if type(value).__module__.split(".")[0] == "omegaconf"
        else "dict(...)"
    )
    raise TypeError(
        f"{api} takes a plain dict; {what} is {kind}. Convert at the caller's "
        f"boundary -- {fix} -- rather than here: a silent conversion would hide "
        f"which surfaces are still Hydra-coupled."
    )


def nearest_key(key: str, valid: Iterable[str]) -> str | None:
    """The closest spelling of `key` among `valid`, or None if nothing is close."""
    matches = difflib.get_close_matches(str(key), [str(v) for v in valid], n=1, cutoff=0.6)
    return matches[0] if matches else None


def unknown_key_error(key: str, valid: Iterable[str], *, where: str, api: str) -> ValueError:
    """The error for a key that is not in a COMPLETE schema -- i.e. a typo."""
    valid = sorted(str(v) for v in valid)
    near = nearest_key(key, valid)
    hint = f" Did you mean {near!r}?" if near else ""
    return ValueError(
        f"{api}: unknown config key {key!r} in {where}.{hint} Valid keys there are "
        f"{valid}. An unknown key is refused rather than ignored: the schema is "
        f"complete, so a key that is not in it never takes effect, and a stage "
        f"that silently did not run reports zeros that read as 'nothing needed "
        f"doing' -- measured on the reference bout, misspelling 'bone_length' "
        f"turns that gate off, gives n_bone_flagged=0 instead of 1802, and moves "
        f"128458 of 300873 entries off the reference with no error and no line."
    )


def plain_mapping(value, *, what: str, api: str) -> Mapping:
    """A mapping view of a config node, tolerating None and refusing a Hydra one."""
    if value is None:
        return {}
    refuse_config_object(value, what=what, api=api)
    return value
