"""Plain ViT backbone (patch embed + pre-norm blocks)."""

import jax
import jax.numpy as jnp
from flax import nnx


class PatchEmbed(nnx.Module):
    def __init__(self, in_ch: int, embed_dim: int, patch: int, *, rngs: nnx.Rngs):
        # NHWC conv; non-overlapping patches.
        self.proj = nnx.Conv(
            in_ch,
            embed_dim,
            kernel_size=(patch, patch),
            strides=(patch, patch),
            padding="VALID",
            rngs=rngs,
        )

    def __call__(self, x):  # x: (B,H,W,in_ch)
        x = self.proj(x)  # (B, H/16, W/16, embed_dim)
        b, h, w, d = x.shape
        return x.reshape(b, h * w, d)  # (B, N, D)


class Attention(nnx.Module):
    def __init__(self, dim: int, num_heads: int, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nnx.Linear(dim, dim * 3, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, rngs=rngs)

    def __call__(self, x):  # (B,N,D)
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # (3,B,heads,N,hd)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = jax.nn.softmax(attn, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(b, n, d)
        return self.proj(out)


class MLP(nnx.Module):
    def __init__(self, dim: int, mlp_ratio: int, *, rngs: nnx.Rngs):
        hidden = dim * mlp_ratio
        self.fc1 = nnx.Linear(dim, hidden, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, dim, rngs=rngs)

    def __call__(self, x):
        return self.fc2(jax.nn.gelu(self.fc1(x), approximate=False))


class Block(nnx.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int, *, rngs: nnx.Rngs):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = Attention(dim, num_heads, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp = MLP(dim, mlp_ratio, rngs=rngs)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nnx.Module):
    def __init__(self, cfg, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg.in_ch, cfg.embed_dim, cfg.patch, rngs=rngs)
        self.cls_token = nnx.Param(jnp.zeros((1, 1, cfg.embed_dim)))
        self.pos_embed = nnx.Param(jnp.zeros((1, cfg.num_tokens + 1, cfg.embed_dim)))
        self.blocks = nnx.List(
            [
                Block(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, rngs=rngs)
                for _ in range(cfg.depth)
            ]
        )
        self.norm = nnx.LayerNorm(cfg.embed_dim, rngs=rngs)

    def __call__(self, x):
        x = self.patch_embed(x)  # (B,N,D)
        b = x.shape[0]
        cls = jnp.broadcast_to(self.cls_token.value, (b, 1, x.shape[-1]))
        x = jnp.concatenate([cls, x], axis=1) + self.pos_embed.value
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 1:]  # drop cls
