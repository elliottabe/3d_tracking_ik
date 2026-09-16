"""D4RT-style query decoder: heavy 3D path (query self-attn + cross-attn),
light 2D path (cross-attn only), shared heads, refinement context gather."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from tracking.detector.mvq.fusion import MLP, Attn


def fourier(x, bands):
    """(...,d) in ~[-1,1] -> (..., d*(2*bands+1))."""
    freqs = (2.0 ** jnp.arange(bands)) * jnp.pi
    ang = x[..., None] * freqs
    return jnp.concatenate([x[..., None], jnp.sin(ang), jnp.cos(ang)], -1).reshape(
        x.shape[:-1] + (-1,)
    )


class CrossBlock(nnx.Module):
    def __init__(self, D, heads, ratio, *, rngs, self_attn: bool):
        self.self_attn = self_attn
        if self_attn:
            self.ns = nnx.LayerNorm(D, rngs=rngs)
            self.sa = Attn(D, heads, rngs=rngs)
        self.nq = nnx.LayerNorm(D, rngs=rngs)
        self.nk = nnx.LayerNorm(D, rngs=rngs)
        self.ca = Attn(D, heads, rngs=rngs)
        self.nm = nnx.LayerNorm(D, rngs=rngs)
        self.mlp = MLP(D, ratio, rngs=rngs)

    def __call__(self, q, bank, bank_valid, q_chunk: int | None = 512, impl: str = "xla"):
        if self.self_attn:
            h = self.ns(q)
            q = q + self.sa(h, h, None, q_chunk, impl)
        q = q + self.ca(self.nq(q), self.nk(bank), bank_valid, q_chunk, impl)
        return q + self.mlp(self.nm(q))


class Heads(nnx.Module):
    def __init__(self, D, *, rngs):
        self.xyz = nnx.Linear(D, 3, rngs=rngs, kernel_init=nnx.initializers.normal(1e-3))
        self.conf = nnx.Linear(D, 1, rngs=rngs)
        self.exist = nnx.Linear(D, 1, rngs=rngs)
        self.sex = nnx.Linear(D, 1, rngs=rngs)
        self.uv = nnx.Linear(D, 2, rngs=rngs, kernel_init=nnx.initializers.zeros)
        self.vis = nnx.Linear(D, 1, rngs=rngs)


class QueryDecoder(nnx.Module):
    def __init__(self, cfg, *, rngs):
        D, H, R = cfg.embed_dim, cfg.dec_heads, cfg.mlp_ratio
        self.cfg = cfg
        self.e_inst = nnx.Param(jax.random.normal(rngs.params(), (cfg.n_instances, D)) * 0.02)
        self.e_kp = nnx.Param(jax.random.normal(rngs.params(), (cfg.num_keypoints, D)) * 0.02)
        self.e_pass = nnx.Param(jnp.zeros((2, D)))
        self.blocks3d = nnx.List(
            [CrossBlock(D, H, R, rngs=rngs, self_attn=True) for _ in range(cfg.dec_layers_3d)]
        )
        self.blocks2d = nnx.List(
            [CrossBlock(D, H, R, rngs=rngs, self_attn=False) for _ in range(cfg.dec_layers_2d)]
        )
        self.to_view = nnx.Linear(D, D, rngs=rngs)
        nf = 2 * cfg.fourier_bands + 1
        self.geom_cam = nnx.Linear(
            8 * nf, D, rngs=rngs
        )  # Fourier(M.ravel()/10 (6), t_local/crop (2))
        self.refine_in = nnx.Linear(
            D + 3 * cfg.patch_rgb**2 + 2 * nf, D, rngs=rngs, kernel_init=nnx.initializers.zeros
        )
        self.prompt_proj = nnx.Linear(D, D, rngs=rngs)
        self.heads = Heads(D, rngs=rngs)

    # ------------------------------------------------------------ readouts
    def _read3d(self, h, n_inst, T, K):
        B = h.shape[0]
        h4 = h.reshape(B, n_inst, T, K, -1)
        return {
            "xyz": self.cfg.roi_scale * self.heads.xyz(h4),
            "conf_logit": self.heads.conf(h4)[..., 0],
            "exist_logit": self.heads.exist(h4.mean(axis=(2, 3)))[..., 0],
            "sex_logit": self.heads.sex(h4.mean(axis=(2, 3)))[..., 0],
        }

    def _path2d(self, h3d, bank, bank_valid, gcam, femb, n_inst, T, K):
        B, C = h3d.shape[0], gcam.shape[1]
        base = self.to_view(h3d).reshape(B, n_inst, T, 1, K, -1)
        q = base + gcam[:, None, None, :, None, :] + femb[None, None, :, None, None, :]
        q = q.reshape(B, n_inst * T * C * K, -1)
        qc, imp = self.cfg.q_chunk, self.cfg.attn_impl
        for blk in self.blocks2d:
            fn = (
                nnx.remat(lambda m, qq: m(qq, bank, bank_valid, qc, imp))
                if self.cfg.remat
                else (lambda m, qq: m(qq, bank, bank_valid, qc, imp))
            )
            q = fn(blk, q)
        q = q.reshape(B, n_inst, T, C, K, -1)
        return {
            "uv": self.cfg.crop * jax.nn.sigmoid(self.heads.uv(q)),
            "vis_logit": self.heads.vis(q)[..., 0],
        }

    def __call__(
        self,
        bank,
        bank_valid,
        M,
        t_local,
        femb,
        prompt_tok=None,
        prompt_on=None,
        refine_ctx=None,
        pass_idx=0,
    ):
        """bank (B,L,D); M (B,C,2,3); t_local (B,T,C,2); femb (T,D)."""
        cfg = self.cfg
        B = bank.shape[0]
        n_inst, K = cfg.n_instances, cfg.num_keypoints
        T, C = t_local.shape[1], t_local.shape[2]
        q = (
            self.e_inst[...][None, :, None, None, :]
            + self.e_kp[...][None, None, None, :, :]
            + femb[None, None, :, None, :]
        )  # (B?,I,T,K,D) broadcast
        q = jnp.broadcast_to(q, (B, n_inst, T, K, q.shape[-1]))
        if prompt_tok is not None:
            on = prompt_on if prompt_on is not None else jnp.ones((B,), bool)
            add = self.prompt_proj(prompt_tok) * on[:, None]
            q = q.at[:, 0].add(add[:, None, None, :])
        q = q + self.e_pass[...][pass_idx]
        if refine_ctx is not None:
            q = q + self.refine_in(refine_ctx)
        q = q.reshape(B, n_inst * T * K, -1)
        per_layer = []
        qc, imp = cfg.q_chunk, cfg.attn_impl
        for li, blk in enumerate(self.blocks3d):
            fn = (
                nnx.remat(lambda m, qq: m(qq, bank, bank_valid, qc, imp))
                if cfg.remat
                else (lambda m, qq: m(qq, bank, bank_valid, qc, imp))
            )
            q = fn(blk, q)
            if li % 2 == 1 and li != len(self.blocks3d) - 1:
                per_layer.append(self._read3d(q, n_inst, T, K))
        out = self._read3d(q, n_inst, T, K)
        gfeat = jnp.concatenate(
            [M.reshape(B, C, 6) / 10.0, t_local.mean(axis=1) / cfg.crop], -1
        )  # (B,C,8)
        gcam = self.geom_cam(fourier(gfeat, cfg.fourier_bands))
        out.update(self._path2d(q, bank, bank_valid, gcam, femb, n_inst, T, K))
        return out, per_layer, q


def gather_refine_context(xyz, M, t_local, cam_valid, grid_tokens, crops, patch, bands, crop):
    """xyz (B,I,T,K,3) -> context (B,I,T,K, D + 3*patch^2 + 2*(2*bands+1))."""
    from tracking.detector.mvq.geometry import project_local

    B, n_inst, T, K, _ = xyz.shape
    C = M.shape[1]
    g = grid_tokens.shape[3]
    H = crops.shape[3]
    uv = jax.vmap(
        lambda X, m, tl: jax.vmap(
            lambda Xt, tlt: project_local(Xt, m, tlt), in_axes=(1, 0), out_axes=1
        )(X, tl)
    )(xyz, M, t_local)  # (B,I,T,K,C,2)
    uv = jnp.moveaxis(uv, 4, 3)  # (B,I,T,C,K,2)
    scale = g / crop
    gx = jnp.clip(uv[..., 0] * scale - 0.5, 0, g - 1)
    gy = jnp.clip(uv[..., 1] * scale - 0.5, 0, g - 1)
    x0 = jnp.floor(gx).astype(jnp.int32)
    y0 = jnp.floor(gy).astype(jnp.int32)
    x1 = jnp.minimum(x0 + 1, g - 1)
    y1 = jnp.minimum(y0 + 1, g - 1)
    wx, wy = gx - x0, gy - y0

    def take(ix, iy):  # (B,I,T,C,K) idx -> (B,I,T,C,K,D)
        flat = grid_tokens.reshape(B, T, C, g * g, -1)
        idx = iy * g + ix  # (B,I,T,C,K)
        idx_b = jnp.moveaxis(idx, 1, -1)  # (B,T,C,K,I)
        got = jnp.take_along_axis(
            flat[:, :, :, :, None, :], idx_b.reshape(B, T, C, K * n_inst, 1, 1), axis=3
        )
        return jnp.moveaxis(got.reshape(B, T, C, K, n_inst, -1), 4, 1)  # (B,I,T,C,K,D)

    tok = (
        ((1 - wx) * (1 - wy))[..., None] * take(x0, y0)
        + (wx * (1 - wy))[..., None] * take(x1, y0)
        + ((1 - wx) * wy)[..., None] * take(x0, y1)
        + (wx * wy)[..., None] * take(x1, y1)
    )
    # RGB patch: nearest centre, offsets -r..r
    r = patch // 2
    cu = jnp.clip(jnp.round(uv[..., 0]).astype(jnp.int32), r, H - 1 - r)
    cv = jnp.clip(jnp.round(uv[..., 1]).astype(jnp.int32), r, H - 1 - r)
    offs = jnp.arange(-r, r + 1)
    pix = crops.reshape(B, T, C, H * H, 3)
    pidx = (cv[..., None, None] + offs[None, None, None, None, None, :, None]) * H + (
        cu[..., None, None] + offs[None, None, None, None, None, None, :]
    )  # (B,I,T,C,K,p,p)
    pidx = jnp.moveaxis(pidx.reshape(B, n_inst, T, C, K * patch * patch), 1, -1).reshape(
        B, T, C, -1
    )  # (B,T,C,K*p*p*I)
    rgb = jnp.take_along_axis(pix, pidx[..., None], axis=3)  # (B,T,C,K*p*p*I,3)
    rgb = rgb.reshape(B, T, C, K, patch * patch, n_inst, 3)
    rgb = jnp.moveaxis(rgb, 5, 1).reshape(B, n_inst, T, C, K, 3 * patch * patch)
    vw = cam_valid[:, None, :, :, None, None].astype(jnp.float32)  # (B,1,T,C,1,1)
    den = jnp.maximum(vw.sum(3), 1.0)
    tok_m = (tok * vw).sum(3) / den
    rgb_m = (rgb * vw).sum(3) / den  # (B,I,T,K,·)
    fuv = (fourier((uv / crop) * 2 - 1, bands) * vw).sum(3) / den
    return jnp.concatenate([tok_m, rgb_m, fuv], -1)
