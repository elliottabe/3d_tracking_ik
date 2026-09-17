"""Process-based sample workers for `windows.window_batches`.

WHY PROCESSES. `WindowDataset.__getitem__` is mostly pure-Python/numpy work
-- window assembly, centre jitter, copy-paste compositing -- and only the
JPEG decode releases the GIL, so a `ThreadPoolExecutor` starves an 8-GPU
training run (jarvis_jax's `mvq_t2_v2_20260906`: ~11 samples/s against
~15/s eight L40S eat). Processes give each sample its own interpreter, so
the Python half scales with cores instead of with one GIL.

SPAWN, NEVER FORK. By the time the loader starts, the parent may have JAX
and CUDA contexts initialised; `fork` would copy that driver state into a
child that cannot legally use it. The pool uses
`multiprocessing.get_context("spawn")`, so a child inherits nothing -- it
rebuilds the dataset from a picklable `WindowSpec` in the pool initializer,
once per worker, then serves `__getitem__` for the index lists the parent
sends.

SAMPLING STAYS IN THE PARENT. The epoch permutation and `ds.epoch = seed`
are parent-side; only `__getitem__` moves. The epoch travels with every
task, and every per-sample RNG in `WindowDataset` is seeded as
`SeedSequence([self.seed, i, self.epoch, ...])` -- a pure function of
(dataset seed, index, epoch) -- so a worker reproduces the same jitter and
copy-paste donor the thread path would have drawn. Batches are consequently
byte-identical between the two paths (`tests/train/test_loaders.py`).

This module must stay JAX-free: it is imported by the spawned children, and
a worker that imports JAX would try to claim a GPU the parent already owns.
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import multiprocessing
import os
import threading


@dataclasses.dataclass(frozen=True)
class WindowSpec:
    """Everything `WindowDataset.__init__` needs, as picklable values."""

    root: str
    split: str
    T: int
    pair_deltas: tuple
    max_flies: int
    jitter_units: float
    seed: int
    train: bool
    recordings: tuple | None
    copy_paste: object
    center_shift_units: float
    sex_overrides: dict
    masks_root: str | None = None


def build_dataset(spec):
    """Rebuild a `WindowDataset` from its spec.

    The import is local so that importing this module (which the parent does
    before spawning) does not drag in `windows.py`'s own dependencies until a
    worker actually needs them.
    """
    if not isinstance(spec, WindowSpec):
        raise TypeError(f"not a window-dataset spec: {type(spec).__name__}")
    from tracking.train.data.windows import WindowDataset

    return WindowDataset(
        spec.root,
        spec.split,
        T=spec.T,
        pair_deltas=tuple(spec.pair_deltas),
        max_flies=spec.max_flies,
        jitter_units=spec.jitter_units,
        seed=spec.seed,
        train=spec.train,
        recordings=(set(spec.recordings) if spec.recordings is not None else None),
        copy_paste=spec.copy_paste,
        center_shift_units=spec.center_shift_units,
        sex_overrides=dict(spec.sex_overrides or {}),
        masks_root=spec.masks_root,
    )


def dataset_spec(ds):
    """The spec for `ds`, asking the object itself (`ds.worker_spec()`).

    A dataset that cannot describe itself raises rather than silently
    loading something else in the worker.
    """
    fn = getattr(ds, "worker_spec", None)
    if fn is None:
        raise TypeError(
            f"{type(ds).__name__} has no worker_spec(): it cannot be rebuilt in a "
            f"loader worker process (use workers='threads')"
        )
    return fn()


_WORKER_DS = {}


_NO_GPU_ENV = {"JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""}


def _init_worker(specs):
    for k, v in _NO_GPU_ENV.items():
        os.environ.setdefault(k, v)
    global _WORKER_DS
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        _WORKER_DS = {k: build_dataset(s) for k, s in specs.items()}


def _get_item(task):
    key, epoch, i = task
    ds = _WORKER_DS[key]
    ds.epoch = int(epoch)
    return ds[int(i)]


@contextlib.contextmanager
def _child_env():
    """Pin a spawn child to the CPU before it re-imports the parent's
    `__main__` (which can happen before `_init_worker` ever runs)."""
    old = {k: os.environ.get(k) for k in _NO_GPU_ENV}
    os.environ.update(_NO_GPU_ENV)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class ProcessSampleLoader:
    """A spawn pool of workers, each holding its own copy of the dataset(s).

    `specs` is `{key: spec}`; the key selects which dataset a task addresses,
    so several streams sharing one pool address one slice of the node's
    cores instead of running `len(specs) * num_workers` processes.

    `map_batches` keeps `inflight` batches submitted at all times so the
    workers stay busy while the parent collates; results come back in
    submission order, so the batch a consumer sees is the batch the sampler
    drew.
    """

    def __init__(self, specs, num_workers, inflight=3):
        if not isinstance(specs, dict):
            specs = {None: specs}
        self.keys = tuple(specs)
        self.num_workers = max(1, int(num_workers))
        self.inflight = max(1, int(inflight))
        ctx = multiprocessing.get_context("spawn")
        with _child_env():
            self._pool = ctx.Pool(self.num_workers, initializer=_init_worker, initargs=(specs,))
        self._closed = False

    def map_batches(self, index_lists, epoch, key=None):
        pending = collections.deque()
        it = iter(index_lists)

        def submit():
            try:
                nxt = [int(i) for i in next(it)]
            except StopIteration:
                return False
            pending.append(
                (
                    nxt,
                    self._pool.map_async(
                        _get_item, [(key, int(epoch), i) for i in nxt], chunksize=1
                    ),
                )
            )
            return True

        try:
            for _ in range(self.inflight):
                if not submit():
                    break
            while pending:
                bidx, res = pending.popleft()
                samples = res.get()
                submit()
                yield bidx, samples
        finally:
            pending.clear()

    def close(self, timeout=30.0):
        """Shut the pool down, and never block the caller indefinitely.

        Ask politely, then force, then give up. Pool workers are daemonic,
        so any survivor dies with the parent process. Returns True if the
        workers were reaped, False if they were abandoned.
        """
        if self._closed:
            return True
        self._closed = True
        for stage, stop in (("close", self._pool.close), ("terminate", self._pool.terminate)):
            try:
                stop()
            except Exception:
                pass
            joiner = threading.Thread(target=self._pool.join, daemon=True)
            joiner.start()
            joiner.join(timeout)
            if not joiner.is_alive():
                return True
            print(
                f"[loader] pool {stage}() did not finish in {timeout:.0f}s; "
                f"{'forcing' if stage == 'close' else 'abandoning'} "
                f"{self.num_workers} worker processes (they are daemonic and exit "
                f"with this process)",
                flush=True,
            )
        return False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
