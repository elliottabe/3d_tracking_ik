"""Background-thread device prefetcher overlapping CPU batch IO with GPU compute."""

import queue
import threading

import jax
import numpy as np

from tracking.train.common.sharding import shard_batch


def prefetch(batch_iter, mesh, depth=2):
    """Yield device-resident, sharded batches buffered ahead on a background thread.

    Example: ``for batch in prefetch(iter(batches), mesh):``.

    Converts batches with np.asarray (not jnp.asarray, to avoid device-to-device
    sharding during NCCL collectives) and re-raises worker exceptions on consumer.
    """
    q = queue.Queue(maxsize=depth)

    def worker():
        try:
            for batch in batch_iter:
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
