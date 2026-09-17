"""Batching and the process-pool sample loader.

`ds` restricts the unified root to `SMALL_REC` (a single small recording,
also used by `test_paste_window.py`): the root's instances file is ~390 MB,
so a module-scoped fixture that builds the full split would pay that parse
cost again in every spawned worker process on `test_thread_and_process_paths_agree`.
"""

import os

import numpy as np
import pytest

from tracking.train.data.windows import WINDOW_KEYS, WindowDataset, window_batches

ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/unified_v2"
pytestmark = pytest.mark.skipif(not os.path.isdir(ROOT), reason="unified root not present")

SMALL_REC = "2026_01_29_14_09_33"


@pytest.fixture(scope="module")
def ds():
    d = WindowDataset(ROOT, "train", T=1, recordings=[SMALL_REC])
    yield d
    pool = getattr(d, "_loader_pool", None)
    if pool is not None:
        pool.close()


def test_batch_has_a_leading_batch_axis(ds):
    b = next(iter(window_batches(ds, 4, shuffle=False, num_workers=2)))
    assert set(b) == set(WINDOW_KEYS)
    assert b["crops"].shape[0] == 4
    assert b["kp3d_local"].shape[0] == 4


def test_same_seed_draws_the_same_windows(ds):
    a = next(iter(window_batches(ds, 4, seed=7, num_workers=2)))
    c = next(iter(window_batches(ds, 4, seed=7, num_workers=2)))
    np.testing.assert_array_equal(a["center3D"], c["center3D"])


def test_different_seed_draws_different_windows(ds):
    a = next(iter(window_batches(ds, 4, seed=1, num_workers=2)))
    c = next(iter(window_batches(ds, 4, seed=2, num_workers=2)))
    assert not np.array_equal(a["center3D"], c["center3D"])


def test_worker_count_does_not_change_the_batch(ds):
    a = next(iter(window_batches(ds, 4, seed=5, num_workers=1)))
    c = next(iter(window_batches(ds, 4, seed=5, num_workers=4)))
    np.testing.assert_array_equal(a["crops"], c["crops"])


def test_thread_and_process_paths_agree(ds):
    a = next(iter(window_batches(ds, 4, seed=11, num_workers=2, workers="threads")))
    c = next(iter(window_batches(ds, 4, seed=11, num_workers=2, workers="processes")))
    for k in ("crops", "kp3d_local", "kp2d", "cam_valid", "M"):
        np.testing.assert_array_equal(a[k], c[k])


def test_weights_bias_the_draw(ds):
    w = np.zeros(len(ds))
    w[:8] = 1.0
    b = next(iter(window_batches(ds, 4, seed=3, weights=w / w.sum(), num_workers=2)))
    assert b["crops"].shape[0] == 4


def test_worker_spec_round_trips_through_pickle(ds):
    import pickle

    spec = ds.worker_spec()
    assert pickle.loads(pickle.dumps(spec)) == spec


def test_worker_spec_refuses_a_subclass(ds):
    class Subclass(WindowDataset):
        pass

    sub = Subclass.__new__(Subclass)
    sub.__dict__.update(ds.__dict__)
    with pytest.raises(TypeError, match="build_dataset"):
        sub.worker_spec()
