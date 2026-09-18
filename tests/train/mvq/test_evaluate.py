import os

import numpy as np
import pytest

from tracking.train.mvq.evaluate import cohorts

ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/unified_v2"


def test_evaluate_shares_the_inference_policy():
    import inspect

    import tracking.train.mvq.evaluate as ev
    src = inspect.getsource(ev)
    assert "from tracking.detector.mvq.policy import" in src, "must import the shared policy"
    assert "def policy_instance" not in src, "the policy must be shared, not reimplemented"


class FakeDS:
    def __init__(self, n_flies, female, groups, cents):
        self._n, self._f, self._g, self._c = n_flies, female, groups, cents

    def __len__(self):
        return len(self._n)

    def n_flies(self, i):
        return self._n[i]

    def is_female(self, i):
        return self._f[i]

    def calib_group(self, i):
        return self._g[i]

    def fly_centroids(self, i):
        return self._c[i]


def test_cohorts_cover_the_five_named_sets_and_one_per_group():
    far = np.array([[0.0, 0, 0], [500.0, 0, 0]])
    near = np.array([[0.0, 0, 0], [1.0, 0, 0]])
    ds = FakeDS([1, 2, 2], [True, False, False], ["A", "A", "B"],
                [np.full((2, 3), np.nan), far, near])
    c = cohorts(ds)
    for k in ("female", "two_fly", "single_fly", "contact_pair", "group_A", "group_B"):
        assert k in c, k
    assert list(c["single_fly"]) == [True, False, False]
    assert list(c["two_fly"]) == [False, True, True]
    assert list(c["female"]) == [True, False, False]
    assert list(c["contact_pair"]) == [False, False, True]
    assert list(c["group_A"]) == [True, True, False]


def test_a_single_fly_window_is_never_a_contact_pair():
    ds = FakeDS([1], [False], ["A"], [np.zeros((2, 3))])
    assert not cohorts(ds)["contact_pair"][0]


@pytest.mark.skipif(not os.path.isdir(ROOT), reason="unified root not present")
def test_cohorts_on_the_real_root_are_non_empty():
    from tracking.train.data.windows import WindowDataset
    c = cohorts(WindowDataset(ROOT, split="val", T=1, train=False))
    assert c["female"].any() and c["single_fly"].any()
