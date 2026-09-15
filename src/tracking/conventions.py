"""The two rules every module in this package answers the same way.

Phase 4 was built as four separate tasks and each one drew these two lines in a
different place: a `DictConfig` made `offsets` raise, `filter` accept silently
and `gaps` convert silently, and announcements came out in two formats, so no
single grep found this phase's stderr lines in a batch log. Both are the same
class of defect the modules exist to prevent -- a difference that produces no
symptom -- so they are settled here, once, and imported rather than restated.

**Raise where a complete schema exists. Announce where absence is legitimate.**

`filter.DEFAULT_FILTER_CFG` enumerates every valid filter key, so an unknown key
there is unambiguously a typo and raises (`unknown_key_error`). An anatomy
config's absent key is a legitimately pruned anatomy and stays silent; a key
that is present but whose value will not resolve is a typo or a pruning, and
announces (`announce`). A guard a caller can switch off by omitting an argument
announces too -- never silently, because this repo's most expensive bug class is
a check that was not running and said nothing about it.

This module is repo-wide: `tracking.preprocess`'s four modules were its first
consumer, and `tracking.inverse_kinematics.anatomy` is the second.
"""

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
    """This bout-fly cannot be fitted from the data it has. Not a defect.

    The female is the hard fly (walls, occlusion, OOD poses), and on a real
    session some of her bouts have no usable 3D at all: measured on Session0,
    bouts 8 and 19 are 0.000 finite across every keypoint and every frame,
    while her other 28 bouts range 0.063-1.000 and the male is 1.000
    throughout. The IK correctly refuses those two -- there is no frame to
    warm-start from.

    That refusal is an OUTCOME, not a crash, and the scorecard already has a
    word for it (`not_fit` in `tracking.qc.session`). Raising a bare
    `ValueError` conflated it with a real failure, and the conflation was not
    academic: `run.py` exits non-zero on any bout-fly failure, SLURM marks the
    array task FAILED, and `afterok` then refuses to start `postprocess` --
    so the 58 bout-flies that solved cleanly were never collected because two
    could not be. Subclasses `ValueError` so existing callers that catch
    broadly are unaffected.
    """


def announce(module: str, func: str, message: str) -> None:
    """One stderr line, tagged `[module] func: message`.

    The format matches `tracking.inverse_kinematics.anatomy`, so one grep
    (`^\\[`) finds every announcement this pipeline makes in a batch log.

    **stderr, never `warnings.warn`.** `warnings` dedupes by (message, module,
    lineno), so a batch driver that resolves once per bout would say this on the
    first bout of a recording and stay silent for every bout after it -- which
    is precisely the shape of failure these lines exist to make visible.
    """
    print(f"[{module}] {func}: {message}", file=sys.stderr)


# `dict` and the shipped read-only views of one. Anything else -- a `DictConfig`
# above all -- is refused at the boundary rather than converted here.
_PLAIN = (dict, MappingProxyType)


def refuse_config_object(value, *, what: str, api: str, allow: tuple[type, ...] = _PLAIN) -> None:
    """Raise `TypeError` unless `value` is a plain mapping. Decision D6.

    Ported surfaces take plain dicts and Hydra converts at the caller's
    boundary. A `DictConfig` is a `Mapping`, so without this it walks in and
    fails deep inside -- a struct one with `ConfigKeyError: Key ... is not in
    struct` from an unhelpful place, a non-struct one by "succeeding" and
    handing back a half-converted config, and a struct one missing a stage key
    by defaulting that stage silently OFF. It is refused rather than converted,
    because a quiet conversion hides which surfaces are still Hydra-coupled.

    `allow` narrows the rule for a surface that must mutate its copy:
    `offsets_fit_cfg` deep-copies, and a `MappingProxyType` cannot be
    deep-copied, so it takes `(dict,)`.

    The detection reads the type's module and name, so `src/` never imports
    omegaconf -- the check has to work in a process that has never seen Hydra.
    """
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
    """The error for a key that is not in a COMPLETE schema -- i.e. a typo.

    Returned, not raised, so the caller keeps the traceback shallow. `where`
    names the config path (`cfg` or `cfg['savgol']`) and the message carries the
    nearest valid spelling, because the whole failure mode is that the operator
    believes the key took effect.
    """
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
    """A mapping view of a config node, tolerating None and refusing a Hydra one.

    `None` is how Hydra spells a disabled group, so it reads as absent. Anything
    that is not a plain mapping goes through `refuse_config_object`.
    """
    if value is None:
        return {}
    refuse_config_object(value, what=what, api=api)
    return value
