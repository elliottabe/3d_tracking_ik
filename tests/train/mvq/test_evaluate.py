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


@pytest.mark.skipif(not os.path.isdir(ROOT), reason="unified root not present")
def test_evaluate_end_to_end_with_a_ground_truth_stub():
    """Runs the real `evaluate()` (forward/prefetch/sharding/policy/aggregation
    path) against a real, single-recording `WindowDataset`, with a stub model
    (not `MVQModel`) so the test is fast and immune to GPU memory pressure --
    everything, including the mesh, runs on the CPU device explicitly.

    `SMALL_REC` is single-fly and all-female, so `two_fly`/`contact_pair` are
    this fixture's naturally zero-member cohorts (no need to fabricate an
    extra all-False mask). This implementation's sentinel for a zero-member
    cohort is NaN, present under its normal key (`cohort_two_fly`, etc.), not
    an absent key or a crash -- asserted explicitly below.

    The stub ignores crops/cam_valid/M/t_local and always reports the host
    fly's OWN ground truth on every instance slot: that drives the oracle AND
    policy MPJPE to exactly 0 (mvq_loss keys `mpjpe3d_units`, `mpjpe3d_mm`,
    `mpjpe3d_policy_units`), which is the assertion that proves the metric is
    wired to the labels rather than merely returning a plausible number.
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
    # point 1: finite for every key this fixture has real data for -- slots
    # 0/2/3 are deliberately excluded (slot 0 = SLOT_PROMPTED never receives a
    # positive existence target in the maskless world; slots 2/3 = male/other
    # never occur in this single-fly/all-female recording), same underlying
    # reason `two_fly`/`contact_pair` are the zero-member cohorts below --
    # neither is "an empty-cohort division" bug, both are this fixture having
    # no data for that slice.
    finite_keys = [
        "mpjpe3d_units", "mpjpe3d_mm", "reproj_px", "uv2d_px", "head_vs_reproj_px",
        "mpjpe3d_policy_units", "mpjpe3d_policy_mm", "policy_miss_frac",
        "exist_prec", "exist_rec", "exist_prec_slot1", "exist_rec_slot1", "sex_acc",
        "cohort_female", "cohort_single_fly", "policy_miss_frac_female",
        "policy_miss_frac_single_fly", "sex_acc_female", "sex_acc_single_fly",
    ] + [k for k in res if k.startswith("cohort_group_")]
    for k in finite_keys:
        assert k in res, k
        assert np.isfinite(res[k]), (k, res[k])

    # point 3: a zero-member cohort must not crash and must not produce a
    # value that looks like real data -- this implementation's sentinel is
    # NaN, under the ordinary key (not an absent key).
    for name in ("two_fly", "contact_pair"):
        assert np.isnan(res[f"cohort_{name}"])
        assert np.isnan(res[f"policy_miss_frac_{name}"])
        assert np.isnan(res[f"sex_acc_{name}"])
