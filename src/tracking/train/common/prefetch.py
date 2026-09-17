"""Background-thread device prefetcher overlapping CPU batch IO with GPU compute."""

import queue
import threading

import jax
import numpy as np

from tracking.train.common.sharding import shard_batch


def prefetch(batch_iter, mesh, depth=2):
    """Yield device-resident, sharded batches from a numpy batch iterator,
    buffering up to `depth` ahead on a background thread.

    A worker-side exception is re-raised on the consumer side (not silently
    swallowed). Example: ``for batch in prefetch(iter(batches), mesh):``.
    """
    q = queue.Queue(maxsize=depth)

    def worker():
        try:
            for batch in batch_iter:
                # np.asarray, NOT jnp.asarray: a jnp array lands on the default
                # device first and device_put then shards it DEVICE-TO-DEVICE,
                # racing the running step's NCCL collectives -- 7 GPUs spin at
                # 100 % waiting for the 8th. From host memory every shard is a
                # plain host-to-device copy, which cannot deadlock with a collective.
                dev = jax.tree.map(lambda a: shard_batch(np.asarray(a), mesh), batch)
                q.put(("ok", dev))
        except Exception as e:
            q.put(("err", e))
            return
        q.put(("end", None))

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    while True:
        tag, payload = q.get()
        if tag == "ok":
            yield payload
        elif tag == "end":
            return
        else:
            raise payload
