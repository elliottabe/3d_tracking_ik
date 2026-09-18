import numpy as np
import pytest

from tracking.train.mvq.sampling import MixCounter, balanced_weights, mix_weights


class FakeDS:
    """Minimal stand-in exposing only what the samplers read."""

    def __init__(self, sources, female, n_flies=None, behavior=None):
        self.sources, self._f = list(sources), list(female)
        self._n = list(n_flies or [1] * len(sources))
        self._b = list(behavior or ["walk"] * len(sources))
        self.T = 1

    def __len__(self):
        return len(self.sources)

    def source_id(self, i):
        return self.sources[i]

    def is_female(self, i):
        return self._f[i]

    def n_flies(self, i):
        return self._n[i]

    def behavior(self, i):
        return self._b[i]


class Cfg:
    balance_alpha = 0.5
    female_weight = 1.0
    female_host_weight = 1.0
    female_host_target = None
    negatives_frac = 0.05


MANIFEST = {"sources": {"human": {"kind": "human"}, "pseudo": {"kind": "pseudo"},
                        "neg": {"kind": "negative"}}}


def test_balanced_weights_sum_to_one():
    ds = FakeDS(["human"] * 6, [True, True, False, False, False, False])
    w = balanced_weights(ds, 0.5, 1.0)
    assert np.isclose(w.sum(), 1.0)


def test_female_host_target_solves_the_multiplier():
    ds = FakeDS(["human"] * 4, [True, False, False, False])
    w = balanced_weights(ds, 0.5, 1.0, female_host_target=0.5)
    is_f = np.array([True, False, False, False])
    assert np.isclose(w[is_f].sum(), 0.5, atol=1e-6)


def test_one_sided_root_leaves_the_multiplier_at_one(capsys):
    ds = FakeDS(["human"] * 3, [True, True, True])
    w = balanced_weights(ds, 0.5, 1.0, female_host_target=0.5, label="all-female")
    assert np.isclose(w.sum(), 1.0)
    assert "UNATTAINABLE" in capsys.readouterr().out


def test_negatives_take_exactly_their_fraction_by_kind_not_name():
    ds = FakeDS(["human"] * 4 + ["pseudo"] * 4 + ["neg"] * 2, [False] * 10)
    w, mass = mix_weights(ds, Cfg(), MANIFEST)
    assert np.isclose(w.sum(), 1.0)
    assert np.isclose(mass["neg"], 0.05)
    assert np.isclose(w[8:].sum(), 0.05, atol=1e-9)


def test_non_negative_mass_splits_by_window_count():
    ds = FakeDS(["human"] * 2 + ["pseudo"] * 6 + ["neg"] * 2, [False] * 10)
    _, mass = mix_weights(ds, Cfg(), MANIFEST)
    assert np.isclose(mass["human"] / mass["pseudo"], 2 / 6, atol=1e-9)


def test_each_source_is_balanced_separately():
    """Each source's own female mass hits the target; global balancing pools them and cannot."""
    female = [True, False, False, False, True, True, True, False]
    ds = FakeDS(["human"] * 4 + ["pseudo"] * 4, female)
    cfg = Cfg()
    cfg.female_host_target = 0.5
    manifest = {"sources": {"human": {"kind": "human"}, "pseudo": {"kind": "pseudo"}}}
    w, mass = mix_weights(ds, cfg, manifest)
    is_f = np.array(female)
    a, b = slice(0, 4), slice(4, 8)
    assert np.isclose(w[a][is_f[a]].sum() / mass["human"], 0.5, atol=1e-6)
    assert np.isclose(w[b][is_f[b]].sum() / mass["pseudo"], 0.5, atol=1e-6)


def test_negatives_frac_of_one_raises():
    ds = FakeDS(["human"] * 4, [False] * 4)
    cfg = Cfg()
    cfg.negatives_frac = 1.0
    with pytest.raises(ValueError, match="negatives_frac"):
        mix_weights(ds, cfg, {"sources": {"human": {"kind": "human"}}})


def test_a_source_the_manifest_declares_but_no_window_carries_raises():
    ds = FakeDS(["human"] * 4, [False] * 4)
    manifest = {"sources": {"human": {"kind": "human"}, "pseudo": {"kind": "pseudo"}}}
    with pytest.raises(ValueError, match="pseudo"):
        mix_weights(ds, Cfg(), manifest)


def test_mix_counter_counts_what_was_drawn():
    ds = FakeDS(["human", "human", "pseudo"], [False] * 3)
    c = MixCounter(ds)
    for i in (0, 0, 2):
        c.note_drawn(i)
    counts, n = c.realised()
    assert counts == {"human": 2, "pseudo": 1} and n == 3
