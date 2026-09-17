"""Data-parallel sharding utilities for multi-GPU inference."""

import jax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


def data_parallel_mesh() -> Mesh:
    """Return a 1-D device mesh over all local JAX devices, axis name 'data'."""
    return Mesh(jax.devices(), axis_names=("data",))


def shard_batch(x, mesh: Mesh):
    """Shard array x along axis 0 across the 'data' axis of mesh.

    Example: ``sharded = shard_batch(batch, mesh)`` for a batch on axis 0.
    """
    return jax.device_put(x, NamedSharding(mesh, P("data")))


def replicate(tree, mesh: Mesh):
    """Replicate every array leaf of tree across all devices in mesh.

    Orbax-restored arrays are committed to device 0, so they must be
    explicitly replicated or the jitted step raises "Received incompatible
    devices". Freshly built params are uncommitted (jit replicates them
    automatically).

    Example: ``replicated_params = replicate(params, mesh)`` before the step.
    """
    return jax.device_put(tree, NamedSharding(mesh, P()))
