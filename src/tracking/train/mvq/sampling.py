"""Source-aware sampling for the unified training root.

Ported from `jarvis_jax.train.train_mvq` (`_balanced_weights`, `_mix_weights`,
`_MixCounter`). That trainer concatenated four roots, one dataset object per
root; here every window carries its own `source_id` over one dataset, so
`mix_weights` groups by `ds.source_id(i)` instead of by sub-dataset.
"""

from __future__ import annotations

import collections
import threading

import numpy as np


def balanced_weights(ds, alpha, female_weight, female_host_weight=1.0,
                      female_host_target=None, label=None):
    """Per-window sampling weights, normalised to sum 1.

    `female_weight` (P2) multiplies female-host windows INSIDE the
    behaviour-category balance, so how much of the sampled mass it actually buys
    depends on how the categories fall out. `female_host_weight` (P3b) multiplies
    them again on the FINAL normalised weights, so the mass ratio it produces is
    exactly `female_host_weight x (mass_F / mass_M)`.

    `female_host_target` (v2) SOLVES that multiplier instead of taking it on
    faith: given this dataset's own post-balance female mass `f`, the multiplier
    that lands the female-host mass exactly on `target` is
    `target*(1-f) / ((1-target)*f)`. A hand-tuned `female_host_weight` is only
    correct for the ONE census it was solved on, and per-source balancing
    applies it PER SOURCE, where each source has its own census, so a single
    hand number stacks into whatever the sources happen to average out to.
    When `female_host_target` is set it wins and `female_host_weight` is unused.
    A source with no female host (or no male/other host) cannot reach any
    interior target at all: the multiplier stays 1.0 and a note is printed
    naming `label`.
    """
    is_f = np.array([ds.is_female(i) for i in range(len(ds))], bool)
    cats = [f"{ds.behavior(i)}_{'female' if is_f[i] else 'other'}" for i in range(len(ds))]
    counts = {c: cats.count(c) for c in set(cats)}
    w = np.array([(1.0 / counts[c]) ** alpha for c in cats])
    w *= np.where(is_f, female_weight, 1.0)
    w = w / w.sum()
    mult = float(female_host_weight)
    if female_host_target is not None:
        t = float(female_host_target)
        if not 0.0 < t < 1.0:
            raise ValueError(f"female_host_target must be strictly between 0 and 1, got {t}")
        f = float(w[is_f].sum())
        who = f"[mvq] {label or getattr(ds, 'root', 'train set')}"
        # One-sided by COUNT first (exact), then by mass with a tolerance: an
        # all-female source's `f` comes back as 0.9999999999999998, not 1.0, and
        # a bare `f >= 1.0` would sail past it and "solve" a multiplier of 4e-16
        # -- which reads as a legitimate number in the log and would zero out the
        # female mass of any source that is merely NEARLY one-sided.
        n_f = int(is_f.sum())
        one_sided = n_f == 0 or n_f == len(is_f) or f <= 1e-9 or f >= 1.0 - 1e-9
        if one_sided:
            mult = 1.0
            side = "no female-host window" if n_f == 0 or f <= 1e-9 else "no male/other-host window"
            print(f"{who}: female_host_target={t:.3f} UNATTAINABLE -- this source has {side} "
                  f"(post-balance female mass {f:.4f}); multiplier left at 1.0 and the source's "
                  f"sampling weights are unchanged", flush=True)
        else:
            mult = t * (1.0 - f) / ((1.0 - t) * f)
            print(f"{who}: female_host_target={t:.3f}, post-balance female mass {f:.4f} "
                  f"-> solved female-host multiplier {mult:.4f} (train.female_host_weight="
                  f"{female_host_weight} is unused while a target is set)", flush=True)
    w = w * np.where(is_f, mult, 1.0)
    return w / w.sum()


class _SourceGroup:
    """Windows of one `source_id`, addressed by LOCAL index 0..len-1."""

    def __init__(self, ds, indices):
        self._ds, self._idx = ds, indices

    def __len__(self):
        return len(self._idx)

    def is_female(self, i):
        return self._ds.is_female(self._idx[i])

    def behavior(self, i):
        return self._ds.behavior(self._idx[i])


def mix_weights(ds, tcfg, manifest):
    """(weights (N,), mass dict[str, float]) for the TRAIN split.

    TRAIN-ONLY: the val split legitimately carries fewer sources than the
    manifest declares (no weighted sampling there), and the "0 windows"
    raise below would misfire on it.

    Windows group by `ds.source_id(i)`; `balanced_weights` runs SEPARATELY on
    each group, restoring that source's own behaviour balance and host-sex
    ratio INSIDE it -- applying the pseudo export's own `female_host_weight`
    a second time across sources would over-sample female hosts by ~2x.

    Negative sources (`manifest["sources"][sid]["kind"] == "negative"`) take
    exactly `negatives_frac` of the mass, split evenly among them; the
    remaining mass splits across the other sources by window count.
    """
    frac = float(tcfg.negatives_frac)
    if not 0.0 <= frac < 1.0:
        raise ValueError(f"negatives_frac must be in [0, 1), got {frac} -- at 1.0 the sampler "
                         f"would draw nothing but empty windows")
    by_source = collections.defaultdict(list)
    for i in range(len(ds)):
        by_source[ds.source_id(i)].append(i)
    for sid in manifest.get("sources", {}):
        if sid not in by_source:
            raise ValueError(
                f"train source {sid!r} contributes 0 windows at T={ds.T} -- a source with no "
                f"window at this length would silently drop out of the mix (check pair_deltas: "
                f"a delta no pair of labelled frames spans yields nothing)"
            )

    local_w = {
        sid: balanced_weights(_SourceGroup(ds, idxs), tcfg.balance_alpha, tcfg.female_weight,
                              tcfg.female_host_weight, female_host_target=tcfg.female_host_target,
                              label=f"T={ds.T} source {sid!r}")
        for sid, idxs in by_source.items()
    }
    neg_sids = [sid for sid in by_source
                if manifest.get("sources", {}).get(sid, {}).get("kind") == "negative"]
    pos_sids = [sid for sid in by_source if sid not in neg_sids]
    mass = {}
    if neg_sids:
        for sid in neg_sids:
            mass[sid] = frac / len(neg_sids)
        pos_n = np.array([float(len(by_source[sid])) for sid in pos_sids])
        pos_mass = pos_n / max(pos_n.sum(), 1e-12) * (1.0 - frac)
        mass.update(zip(pos_sids, (float(m) for m in pos_mass), strict=True))
    else:
        n_win = np.array([float(len(by_source[sid])) for sid in pos_sids])
        share = n_win / max(n_win.sum(), 1e-12)
        mass.update(zip(pos_sids, (float(m) for m in share), strict=True))

    w = np.zeros(len(ds), np.float64)
    for sid, idxs in by_source.items():
        w[idxs] = local_w[sid] * mass[sid]
    return w / w.sum(), mass


class MixCounter:
    """Counts which source each drawn window came from.

    The realised real/pseudo/negative mix has to be counted, not inferred: two
    sources can carry the same `sample_weight` (the batch's only per-sample
    provenance signal), so provenance cannot be read back off the batch.
    `window_batches` calls `note_drawn` on every drawn index, on both the
    thread and process paths, so the tally agrees either way.
    """

    def __init__(self, ds):
        self._ds = ds
        self._lock = threading.Lock()
        self.counts = collections.Counter()

    def note_drawn(self, i):
        sid = self._ds.source_id(i)
        with self._lock:
            self.counts[sid] += 1

    def realised(self):
        with self._lock:
            c = dict(self.counts)
        n = max(sum(c.values()), 1)
        return c, n
