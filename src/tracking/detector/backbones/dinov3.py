"""DINOv3 ViT (patch 16, RoPE, registers, LayerScale) in Flax NNX.

Adapted from jax-ml/bonsai `bonsai/models/dinov3/{modeling,params}.py`
(Copyright 2026 The JAX Authors, Apache-2.0), with three changes for this
repo: NHWC input, patch-tokens-only default output, and a generic
safetensors loader that mirrors the HuggingFace `DINOv3ViTModel` key layout
(module attribute names below are chosen to MATCH those keys, so the loader
needs no regex table). RoPE is applied in bf16 exactly as the reference does,
which is where the ~2e-3 parity tolerance comes from.
"""

from __future__ import annotations

import dataclasses
import glob
import os

import jax
import jax.numpy as jnp
from flax import nnx

HF_REPOS = {
    "dinov3_b16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov3_l16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}


@dataclasses.dataclass(frozen=True)
class DINOv3Config:
    patch: int = 16
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    num_registers: int = 4
    rope_base: float = 100.0
    layer_norm_eps: float = 1e-5
    in_ch: int = 3
    rope_dtype: str = "bfloat16"
    # "xla" (default; the explicit fp32-softmax path below) or "cudnn" (the
    # flash-attention path in mvq/attention.py -- no `key_valid` mask, since
    # no camera ever invalidates a backbone patch token; the only "invalid"
    # position is attention.py's own even-length pad, which it excludes
    # exactly via `key_value_seq_lengths`, not by leaving it unmasked).
    attn_impl: str = "xla"

    def __post_init__(self):
        if self.attn_impl not in ("xla", "cudnn"):
            raise ValueError(f"attn_impl must be 'xla' or 'cudnn', got {self.attn_impl!r}")

    @classmethod
    def vitb16(cls):
        return cls(embed_dim=768, depth=12, num_heads=12)

    @classmethod
    def vitl16(cls):
        return cls(embed_dim=1024, depth=24, num_heads=16)

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads


def rope_cos_sin(h: int, w: int, head_dim: int, base: float):
    """Axial 2D RoPE tables for an h x w patch grid: (h*w, head_dim) each.
    Coordinates are pixel-centre aligned and normalised per axis to [-1, 1]
    ("separate" normalisation, the DINOv3 default)."""
    ch = (jnp.arange(0.5, h, dtype=jnp.float32) / h) * 2.0 - 1.0
    cw = (jnp.arange(0.5, w, dtype=jnp.float32) / w) * 2.0 - 1.0
    coords = jnp.stack(jnp.meshgrid(ch, cw, indexing="ij"), axis=-1).reshape(-1, 2)  # (HW,2)
    inv_freq = 1.0 / base ** jnp.arange(0.0, 1.0, 4.0 / head_dim, dtype=jnp.float32)  # (D/4,)
    ang = 2.0 * jnp.pi * coords[:, :, None] * inv_freq[None, None, :]  # (HW,2,D/4)
    ang = jnp.tile(ang.reshape(coords.shape[0], -1), (1, 2))  # (HW,D)
    return jnp.cos(ang), jnp.sin(ang)


def _rotate_half(x):
    d = x.shape[-1] // 2
    return jnp.concatenate([-x[..., d:], x[..., :d]], axis=-1)


def _apply_rope(q, k, cos, sin, n_prefix, dtype):
    """Rotate the PATCH positions of q/k (B,heads,N,hd); prefix tokens untouched."""
    q = q.astype(dtype)
    k = k.astype(dtype)
    c = cos.astype(dtype)[None, None]
    s = sin.astype(dtype)[None, None]
    qp, qk = q[:, :, :n_prefix], q[:, :, n_prefix:]
    kp, kk = k[:, :, :n_prefix], k[:, :, n_prefix:]
    qk = qk * c + _rotate_half(qk) * s
    kk = kk * c + _rotate_half(kk) * s
    return (
        jnp.concatenate([qp, qk], 2).astype(jnp.float32),
        jnp.concatenate([kp, kk], 2).astype(jnp.float32),
    )


class _Embeddings(nnx.Module):
    def __init__(self, cfg: DINOv3Config, *, rngs):
        D = cfg.embed_dim
        self.cls_token = nnx.Param(jnp.zeros((1, 1, D), jnp.float32))
        self.mask_token = nnx.Param(
            jnp.zeros((1, 1, D), jnp.float32)
        )  # unused; consumes the ckpt key
        self.register_tokens = nnx.Param(jnp.zeros((1, cfg.num_registers, D), jnp.float32))
        self.patch_embeddings = nnx.Conv(
            cfg.in_ch,
            D,
            kernel_size=(cfg.patch, cfg.patch),
            strides=(cfg.patch, cfg.patch),
            padding="VALID",
            rngs=rngs,
        )
        self.num_registers = cfg.num_registers

    def __call__(self, x):  # x: (B,H,W,in_ch) float
        p = self.patch_embeddings(x)  # (B,h,w,D)
        b, h, w, d = p.shape
        p = p.reshape(b, h * w, d)
        cls = jnp.broadcast_to(self.cls_token[...], (b, 1, d))
        reg = jnp.broadcast_to(self.register_tokens[...], (b, self.num_registers, d))
        return jnp.concatenate([cls, reg, p], axis=1), (h, w)


class _Attention(nnx.Module):
    def __init__(self, cfg: DINOv3Config, *, rngs):
        D = cfg.embed_dim
        self.q_proj = nnx.Linear(D, D, use_bias=True, rngs=rngs)
        self.k_proj = nnx.Linear(D, D, use_bias=False, rngs=rngs)  # DINOv3: k bias is masked to 0
        self.v_proj = nnx.Linear(D, D, use_bias=True, rngs=rngs)
        self.o_proj = nnx.Linear(D, D, use_bias=True, rngs=rngs)
        self.num_heads, self.head_dim = cfg.num_heads, cfg.head_dim
        self.rope_dtype = jnp.dtype(cfg.rope_dtype)
        self.attn_impl = cfg.attn_impl

    def __call__(self, x, cos, sin, n_prefix):
        b, n, d = x.shape

        def split(t):
            return t.reshape(b, n, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        q, k, v = split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x))
        q, k = _apply_rope(q, k, cos, sin, n_prefix, self.rope_dtype)
        if self.attn_impl == "cudnn":
            from tracking.detector.mvq.attention import flash_attention

            def bthd(t):  # (B,heads,N,hd) -> (B,N,heads,hd)
                return t.transpose(0, 2, 1, 3)

            out = flash_attention(bthd(q), bthd(k), bthd(v)).reshape(b, n, d)
        else:
            att = jax.nn.softmax((q @ k.transpose(0, 1, 3, 2)) * (self.head_dim**-0.5), axis=-1)
            out = (att @ v).transpose(0, 2, 1, 3).reshape(b, n, d)
        return self.o_proj(out)


class _LayerScale(nnx.Module):
    def __init__(self, dim, *, init=1.0):
        self.lambda1 = nnx.Param(jnp.full((dim,), init, jnp.float32))

    def __call__(self, x):
        return x * self.lambda1[...]


class _MLP(nnx.Module):
    def __init__(self, cfg: DINOv3Config, *, rngs):
        D, H = cfg.embed_dim, int(round(cfg.embed_dim * cfg.mlp_ratio))
        self.up_proj = nnx.Linear(D, H, rngs=rngs)
        self.down_proj = nnx.Linear(H, D, rngs=rngs)

    def __call__(self, x):
        return self.down_proj(jax.nn.gelu(self.up_proj(x), approximate=False))


class _Layer(nnx.Module):
    def __init__(self, cfg: DINOv3Config, *, rngs):
        D = cfg.embed_dim
        self.norm1 = nnx.LayerNorm(D, epsilon=cfg.layer_norm_eps, rngs=rngs)
        self.attention = _Attention(cfg, rngs=rngs)
        self.layer_scale1 = _LayerScale(D)
        self.norm2 = nnx.LayerNorm(D, epsilon=cfg.layer_norm_eps, rngs=rngs)
        self.mlp = _MLP(cfg, rngs=rngs)
        self.layer_scale2 = _LayerScale(D)

    def __call__(self, x, cos, sin, n_prefix):
        x = x + self.layer_scale1(self.attention(self.norm1(x), cos, sin, n_prefix))
        return x + self.layer_scale2(self.mlp(self.norm2(x)))


class DINOv3(nnx.Module):
    """Input (B,H,W,in_ch) float32, ImageNet-normalised. Extra input channels
    beyond `in_ch` are sliced off (lets the 4-channel ViTPose path use it)."""

    def __init__(self, cfg: DINOv3Config, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.embeddings = _Embeddings(cfg, rngs=rngs)
        self.layer = nnx.List([_Layer(cfg, rngs=rngs) for _ in range(cfg.depth)])
        self.norm = nnx.LayerNorm(cfg.embed_dim, epsilon=cfg.layer_norm_eps, rngs=rngs)

    def forward_all(self, x, *, remat: bool = False):
        x = x[..., : self.cfg.in_ch]
        toks, (h, w) = self.embeddings(x)
        cos, sin = rope_cos_sin(h, w, self.cfg.head_dim, self.cfg.rope_base)
        n_prefix = 1 + self.cfg.num_registers
        for lyr in self.layer:
            fn = (
                nnx.remat(lambda m, t: m(t, cos, sin, n_prefix))
                if remat
                else (lambda m, t: m(t, cos, sin, n_prefix))
            )
            toks = fn(lyr, toks)
        return self.norm(toks)

    def __call__(self, x, *, remat: bool = False):
        return self.forward_all(x, remat=remat)[:, 1 + self.cfg.num_registers :]


# ---------------------------------------------------------------- loading
def dinov3_snapshot(repo_id: str) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=repo_id, allow_patterns=["*.safetensors", "config.json"])


def _assign(state: dict, path: list, value):
    node = state
    for k in path[:-1]:
        node = node[k]
    if path[-1] not in node:
        raise KeyError(".".join(map(str, path)))
    tgt = node[path[-1]]
    if tuple(tgt.shape) != tuple(value.shape):
        raise ValueError(f"shape {value.shape} vs {tgt.shape} at {'.'.join(map(str, path))}")
    node[path[-1]] = jnp.asarray(value, dtype=tgt.dtype)


def load_dinov3_safetensors(model: DINOv3, ckpt_dir: str) -> DINOv3:
    """Fill `model` from HF-layout safetensors in `ckpt_dir`. Every checkpoint
    key must map onto a model parameter and every model parameter must be
    filled, else RuntimeError -- a partially loaded backbone is worse than
    a loud failure."""
    from safetensors.numpy import load_file

    files = sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no *.safetensors under {ckpt_dir}")
    gdef, st = nnx.split(model)
    state = nnx.to_pure_dict(st)
    filled, unconsumed = set(), []
    for f in files:
        for key, val in load_file(f).items():
            parts = key.split(".")
            # Some `transformers` releases wrap the encoder ModuleList in an
            # internal `self.model` attribute, so `nn.Module.state_dict()`
            # emits `model.layer.N....` even though the *published* safetensors
            # checkpoint (verified via `huggingface_hub.get_safetensors_metadata`
            # on facebook/dinov3-vitb16-pretrain-lvd1689m) has no such prefix and
            # our module tree has no top-level `model` attribute either. Strip
            # it so both the real checkpoint and this in-memory state_dict load.
            if parts[0] == "model" and len(parts) > 1:
                parts = parts[1:]
            leaf = parts[-1]
            is_ln = parts[-2].startswith("norm") or parts[-2] == "norm"
            if leaf == "weight" and val.ndim == 4:  # conv (O,I,kh,kw)->(kh,kw,I,O)
                parts[-1], val = "kernel", val.transpose(2, 3, 1, 0)
            elif leaf == "weight" and val.ndim == 2:  # linear (out,in)->(in,out)
                parts[-1], val = "kernel", val.T
            elif leaf == "weight" and is_ln:
                parts[-1] = "scale"
            path = [int(p) if p.isdigit() else p for p in parts]
            try:
                _assign(state, path, val)
            except KeyError:
                unconsumed.append(key)
                continue
            filled.add(".".join(map(str, path)))
    if unconsumed:
        raise RuntimeError(f"{len(unconsumed)} unconsumed checkpoint keys, e.g. {unconsumed[:5]}")
    try:
        all_leaves = {
            "/".join(map(str, jax.tree_util.keystr(p, simple=True, separator="/").split("/")))
            for p, _ in jax.tree_util.tree_leaves_with_path(state)
        }
    except TypeError:
        all_leaves = {
            "".join(str(k) for k in p)
            .replace("[", "/")
            .replace("]", "")
            .replace("'", "")
            .lstrip("/")
            for p, _ in jax.tree_util.tree_leaves_with_path(state)
        }
    missing = sorted(leaf for leaf in all_leaves if leaf.replace("/", ".") not in filled)
    if missing:
        raise RuntimeError(f"{len(missing)} model params not filled, e.g. {missing[:5]}")
    return nnx.merge(gdef, state)


def load_pretrained(name: str = "dinov3_b16", *, rngs=None) -> DINOv3:
    cfg = DINOv3Config.vitb16() if name == "dinov3_b16" else DINOv3Config.vitl16()
    model = DINOv3(cfg, rngs=rngs or nnx.Rngs(0))
    return load_dinov3_safetensors(model, dinov3_snapshot(HF_REPOS[name]))
