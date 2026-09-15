"""cuDNN flash-attention wrapper for the backbone and decoder.

`jax.nn.dot_product_attention(..., implementation="cudnn")` needs bf16/fp16
inputs, head dim a multiple of 8 (<=128 on compute capability 8.9), and an
EVEN token count under training. Empirically this even-length requirement holds
REGARDLESS of whether a mask/bias is supplied: T=789 (odd), `mask=None`, no
padding info at all still raises `NotImplementedError: Unsupported sequence
length` under `jax.grad` -- so the token axis must always be padded to even.

What differs by mask presence is HOW the pad is excluded from the softmax:

- When `key_valid` is given (the decoder's real per-camera validity, an
  ARBITRARY pattern across the sequence, not a prefix/suffix), we build an
  explicit boolean mask. cuDNN converts this into an additive bf16 bias
  tensor internally (`has_bias=True`) -- no attention logits/softmax are
  materialised, but this bias tensor (shape (B,1,Tq_pad,Tk_pad), broadcast
  over heads) and its `dbias` gradient are real, sizeable tensors: at the
  shipped decoder shape (2100 queries x 10976 keys) that's ~46 MB/sample
  each way. Measured ~18ms/iter at the (2,789,12,64) backbone shape
  (fix-round-1 job, see notes) -- correct, but not free.
- When `key_valid` is None (the backbone: no camera has ever invalidated a
  patch token, so the ONLY "invalid" position is our own even-length pad),
  we exclude it EXACTLY via `key_value_seq_lengths` (cuDNN's native
  `MaskType.PADDING`, no bias/mask tensor at all -- `has_bias` stays False).
  This is both exact (no pad-key contamination of the softmax, unlike an
  earlier version of this helper that left the pad unmasked) and faster
  than the boolean-mask path: measured ~7.9ms/iter vs ~18ms/iter at the same
  shape (fix-round-1 job). The seq-length mechanism only supports a single
  prefix-valid/suffix-invalid split per batch row, which is exactly what our
  own even-length padding is (the real tokens first, one zero pad row after)
  -- it CANNOT express the decoder's arbitrary `key_valid` pattern, which is
  why the decoder keeps the boolean-mask path.

Layout is (B, N, heads, hd) ("BTNH") -- the shape `jax.nn.dot_product_attention`
expects -- NOT the (B, heads, N, hd) layout `dinov3.py`'s explicit path uses
internally, nor the flat (B, N, D) layout `fusion.py::masked_attention` takes.
Callers transpose/reshape into BTNH before calling and back out after.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _pad_even_tokens(x):
    """Pad axis 1 (tokens) with one zero row if its length is odd.

    Returns (padded_x, original_length).
    """
    n = x.shape[1]
    if n % 2 == 0:
        return x, n
    pad_width = [(0, 0)] * x.ndim
    pad_width[1] = (0, 1)
    return jnp.pad(x, pad_width), n


def flash_attention(q, k, v, key_valid=None, *, _impl: str = "cudnn"):
    """q (B,Tq,heads,hd), k/v (B,Tk,heads,hd), key_valid (B,Tk) bool or None
    -> (B,Tq,heads,hd) fp32.

    Pads the token axis of q and of k/v to an even length (always required
    under training, regardless of masking -- see module docstring). When
    `key_valid` is given, builds a (B,1,Tq_pad,Tk_pad) boolean mask from it
    (padded key columns False, broadcast identically over every query row
    including the padded one(s) -- since every row shares the same >=1 valid
    key, no row ever goes all-False, so no row can softmax to NaN and poison
    the gradient of k/v shared with real rows). When `key_valid` is None and
    padding actually happened, excludes the pad key(s) exactly via
    `key_value_seq_lengths` instead (no mask/bias tensor at all -- see
    module docstring for why this can't replace the decoder's boolean
    mask). Casts to bf16 only for `_impl == "cudnn"` (the kernel's required
    dtype), calls `jax.nn.dot_product_attention`, slices the padding back
    off, returns fp32.

    `_impl` is a private escape hatch (default "cudnn") so CPU tests can pass
    `_impl="xla"` to exercise the padding/mask construction above without a
    GPU. The bf16 cast is skipped for `_impl="xla"`: that path exists to
    isolate the padding/mask logic to fp32 precision (1e-5 parity against the
    explicit path), not to re-test bf16 rounding, which the GPU parity test
    covers directly on the real cudnn kernel. On a host where the cudnn
    kernel is unavailable, the default call raises `NotImplementedError` from
    JAX; this wraps it in a `RuntimeError` (naming `attn_impl`, and carrying
    the original exception text) so the failure is legible from model
    config, not a bare JAX internals trace.
    """
    q_pad, Tq = _pad_even_tokens(q)
    k_pad, Tk = _pad_even_tokens(k)
    v_pad, _ = _pad_even_tokens(v)
    Tk_p = k_pad.shape[1]

    mask = None
    key_value_seq_lengths = None
    if key_valid is not None:
        Tq_p = q_pad.shape[1]
        kv = jnp.pad(key_valid, ((0, 0), (0, Tk_p - Tk)), constant_values=False)
        mask = jnp.broadcast_to(kv[:, None, None, :], (kv.shape[0], 1, Tq_p, Tk_p))
    elif Tk_p != Tk:
        # No explicit invalid keys -- the only "padding" is our own
        # even-length pad -- so exclude it exactly via cuDNN's native
        # MaskType.PADDING (key_value_seq_lengths), never materialising a
        # mask/bias tensor. query_seq_lengths is left unset: the padded
        # query row(s) are discarded by the final slice regardless of
        # whether cuDNN treats them as "valid" queries, and letting it
        # default to Tq_pad (all valid) is correct and simpler.
        key_value_seq_lengths = jnp.full((k.shape[0],), Tk, dtype=jnp.int32)

    if _impl == "cudnn":
        bf = jnp.bfloat16
        q_pad, k_pad, v_pad = q_pad.astype(bf), k_pad.astype(bf), v_pad.astype(bf)
    try:
        out = jax.nn.dot_product_attention(
            q_pad,
            k_pad,
            v_pad,
            mask=mask,
            key_value_seq_lengths=key_value_seq_lengths,
            implementation=_impl,
        )
    except NotImplementedError as e:
        raise RuntimeError(
            "flash_attention: cuDNN attention is unavailable on this host "
            "(no compatible GPU, or an unsupported shape/dtype). Set the "
            "model config's attn_impl to 'xla' to use the explicit path "
            f"instead. Original error: {e}"
        ) from e
    return out[:, :Tq].astype(jnp.float32)
