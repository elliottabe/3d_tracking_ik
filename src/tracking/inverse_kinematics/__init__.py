"""Articulated STAC/IK: anatomy, the jaxls solver, offsets, and the bridge.

Enables JAX's persistent compilation cache on import. Measured on the
reference fit: 208 s of its 440 s -- **47%** -- is the two initial XLA
compiles (the T=1 root solve and the first T=n_frames pose solve), and a
warm cache turns a 72.5 s compile into 22.9 s. Everything in this package
pays that cost, and it is a cache: failing to write it is a performance
problem, never a correctness one, so `enable()` swallows its own errors.

Set `TRACKING_JAX_CACHE_DIR=""` to disable, or to a path to relocate it.
"""

from tracking.utils.jax_cache import enable as _enable_jax_cache

_enable_jax_cache()

"""Anatomy configs and the STAC/IK fit built on top of them."""
