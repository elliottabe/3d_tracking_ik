import os

import numpy as np
import pytest

from tracking.train.mvq.evaluate import cohorts, evaluate

ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/unified_v2"
# Single small recording, all single-fly/female -- shared with
# test_loaders.py/test_paste_window.py so its parse cost is a known quantity.
SMALL_REC = "2026_01_29_14_09_33"


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


#: keys `evaluate()` legitimately returns as NaN on the `SMALL_REC` fixture,
#: with the one-line reason each is excused from the finiteness check below.
_EXPECTED_NAN = {
    "exist_rec_slot0": ("SLOT_PROMPTED can only receive a positive existence target "
                        "when prompt_on is True, and it is hardcoded False, so recall "
                        "is undefined for every call to evaluate() in this codebase, "
                        "not just this fixture."),
    "exist_rec_slot2": "male never occurs in this all-female fixture recording.",
    "exist_rec_slot3": "'other' never occurs in this all-female fixture recording.",
}
_ZERO_TWO_FLY_WINDOWS = "this recording has zero two-fly windows."
_EXPECTED_NAN.update({k: _ZERO_TWO_FLY_WINDOWS for k in (
    "cross_fly_frac", "cross_fly_frac_slot0", "cross_fly_frac_slot1",
    "cross_fly_frac_slot2", "cross_fly_frac_slot3", "cross_fly_frac_female",
    "cross_fly_frac_single_fly", "cross_fly_frac_two_fly", "cross_fly_frac_contact_pair",
    "cross_fly_frac_group_A", "cohort_two_fly", "cohort_contact_pair",
    "policy_miss_frac_two_fly", "policy_miss_frac_contact_pair",
    "sex_acc_two_fly", "sex_acc_contact_pair",
)})


@pytest.mark.skipif(not os.path.isdir(ROOT), reason="unified root not present")
def test_evaluate_end_to_end_with_a_ground_truth_stub():
    """Runs the real `evaluate()` against a real single-recording `WindowDataset`.

    The stub feeds the host fly's own GT as every instance's `xyz`, so a
    correctly-wired MPJPE must collapse to ~0. The mesh is CPU-only,
    deliberately, so this can't contend with the other session's GPU. An
    empty cohort (`two_fly`/`contact_pair`, naturally empty on this
    single-fly/all-female fixture) surfaces as NaN under its ordinary key,
    not an absent key or a crash.
    """
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from jax.sharding import Mesh

    from tracking.detector.mvq.slots import N_SLOTS, SLOT_FEMALE
    from tracking.train.common.sharding import replicate
    from tracking.train.data.windows import WindowDataset, window_batches
    from tracking.train.mvq.losses import LossWeights

    ds = WindowDataset(ROOT, "train", T=1, recordings=[SMALL_REC])
    n = len(ds)
    mesh = Mesh(np.array(jax.devices("cpu")), axis_names=("data",))

    # The exact batch `evaluate()` decodes internally (same ds, batch_size=n,
    # shuffle=False, drop_last=False -> dataset-index order, one batch, no
    # padding) -- used only to read off the host fly's own GT for the stub.
    ref = next(window_batches(ds, n, shuffle=False, drop_last=False, num_workers=4))
    gt_host = replicate(jnp.asarray(ref["kp3d_local"][:, 0]), mesh)  # (n,T,K,3)

    class _GTStub(nnx.Module):
        def __init__(self, gt):
            self.gt = gt

        def __call__(self, crops, cam_valid, M, t_local, *, prompt_on=None):
            B, T, C = crops.shape[:3]
            K = self.gt.shape[-2]
            xyz = jnp.broadcast_to(self.gt[:, None], (B, N_SLOTS, T, K, 3))
            exist = jnp.where(jnp.arange(N_SLOTS) == SLOT_FEMALE, 5.0, -5.0)
            sex = jnp.where(jnp.arange(N_SLOTS) == SLOT_FEMALE, 5.0, 0.0)
            return {
                "xyz": xyz,
                "uv": jnp.zeros((B, N_SLOTS, T, C, K, 2)),
                "conf_logit": jnp.zeros((B, N_SLOTS, T, K)),
                "vis_logit": jnp.zeros((B, N_SLOTS, T, C, K)),
                "exist_logit": jnp.broadcast_to(exist, (B, N_SLOTS)),
                "sex_logit": jnp.broadcast_to(sex, (B, N_SLOTS)),
            }

    cohort_masks = cohorts(ds)
    assert not cohort_masks["two_fly"].any()
    assert not cohort_masks["contact_pair"].any()

    res = evaluate(_GTStub(gt_host), ds, n, cohort_masks=cohort_masks,
                    part_of_k=np.arange(len(ds.kp_order)), weights=LossWeights(),
                    mesh=mesh, num_workers=4)

    # point 4 (the one that matters most): exact GT in -> ~0 MPJPE out.
    assert res["mpjpe3d_units"] < 1e-4
    assert res["mpjpe3d_mm"] < 1e-4
    assert res["mpjpe3d_policy_units"] < 1e-4
    assert res["policy_miss_frac"] == 0.0

    # point 2: expected keys present, including per-cohort MPJPE entries.
    for k in ("cohort_female", "cohort_single_fly", "cohort_group_A"):
        assert k in res, k

    # point 1 & 3, inverted: every key is checked, not just a curated subset
    # -- a key not in `_EXPECTED_NAN` must be finite, so a newly added metric
    # is covered by default and has to be DELIBERATELY excused, rather than
    # silently unchecked. A denied key must actually BE NaN (not merely
    # "not asserted"), so the deny-list can't mask a real bug either.
    for k, v in res.items():
        if k in _EXPECTED_NAN:
            assert np.isnan(v), f"{k}: expected NaN ({_EXPECTED_NAN[k]}), got {v}"
        else:
            assert np.isfinite(v), (k, v)
