"""cv2 thread pinning in the loader worker paths.

Both `loaders._init_worker` (process path) and `windows.window_batches`'s
"threads" branch pin cv2 to one thread, since the worker pool already
provides the parallelism and cv2's own thread pool would otherwise
oversubscribe on top of it.
"""

import multiprocessing
import os

import cv2
import numpy as np
import pytest

from tracking.train.data.loaders import _cv2_num_threads, _init_worker
from tracking.train.data.windows import WindowDataset, window_batches

ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/unified_v2"
SMALL_REC = "2026_01_29_14_09_33"


def test_spawned_worker_observes_cv2_pinned_to_one():
    """The exact initializer `ProcessSampleLoader` uses, in a real spawn pool."""
    ctx = multiprocessing.get_context("spawn")
    pool = ctx.Pool(1, initializer=_init_worker, initargs=({},))
    try:
        n = pool.apply(_cv2_num_threads)
    finally:
        pool.close()
        pool.join()
    assert n == 1


@pytest.mark.skipif(not os.path.isdir(ROOT), reason="unified root not present")
def test_thread_path_pins_cv2_in_process():
    ds = WindowDataset(ROOT, "train", T=1, recordings=[SMALL_REC])
    cv2.setNumThreads(32)
    try:
        next(iter(window_batches(ds, 4, seed=13, num_workers=2, workers="threads")))
        assert cv2.getNumThreads() == 1
    finally:
        cv2.setNumThreads(32)


@pytest.mark.skipif(not os.path.isdir(ROOT), reason="unified root not present")
def test_pinning_does_not_change_batch_output():
    """Thread count must not alter augmentation output: same seed, byte-identical."""
    ds = WindowDataset(ROOT, "train", T=1, recordings=[SMALL_REC])
    cv2.setNumThreads(32)
    a = next(iter(window_batches(ds, 4, seed=17, num_workers=2, workers="threads")))
    cv2.setNumThreads(1)
    b = next(iter(window_batches(ds, 4, seed=17, num_workers=2, workers="threads")))
    for k in a:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)
