"""cuDNN flash-attention wrapper for the backbone and decoder."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _pad_even_tokens(x):
    """Pad axis 1 (tokens) with one zero row if its length is odd."""
    n = x.shape[1]
    if n % 2 == 0:
        return x, n
    pad_width = [(0, 0)] * x.ndim
    pad_width[1] = (0, 1)
    return jnp.pad(x, pad_width), n


def flash_attention(q, k, v, key_valid=None, *, _impl: str = "cudnn"):
    """q (B,Tq,heads,hd), k/v (B,Tk,heads,hd), key_valid (B,Tk) bool or None
    -> (B,Tq,heads,hd) fp32.
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
