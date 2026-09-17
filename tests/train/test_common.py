import numpy as np
import pytest

from tracking.train.common.prefetch import prefetch
from tracking.train.common.sharding import data_parallel_mesh, replicate, shard_batch


def test_mesh_covers_all_local_devices():
    import jax

    mesh = data_parallel_mesh()
    assert mesh.axis_names == ("data",)
    assert mesh.size == len(jax.devices())


def test_shard_batch_preserves_shape_and_values():
    mesh = data_parallel_mesh()
    x = np.arange(8 * 3, dtype=np.float32).reshape(8, 3)
    out = shard_batch(x, mesh)
    assert out.shape == (8, 3)
    np.testing.assert_allclose(np.asarray(out), x)


def test_replicate_preserves_a_pytree():
    mesh = data_parallel_mesh()
    tree = {"a": np.ones((2, 2), np.float32), "b": np.zeros((3,), np.float32)}
    out = replicate(tree, mesh)
    np.testing.assert_allclose(np.asarray(out["a"]), tree["a"])
    np.testing.assert_allclose(np.asarray(out["b"]), tree["b"])


def test_prefetch_yields_every_batch_in_order():
    mesh = data_parallel_mesh()
    batches = [{"x": np.full((8, 2), i, np.float32)} for i in range(5)]
    got = [np.asarray(b["x"])[0, 0] for b in prefetch(iter(batches), mesh, depth=2)]
    assert got == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_prefetch_reraises_a_worker_exception():
    mesh = data_parallel_mesh()

    def broken():
        yield {"x": np.zeros((8, 2), np.float32)}
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        list(prefetch(broken(), mesh, depth=1))
