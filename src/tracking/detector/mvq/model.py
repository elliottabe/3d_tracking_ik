from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from tracking.detector.backbones.dinov3 import DINOv3, DINOv3Config
from tracking.detector.mvq.decoder import QueryDecoder, fourier, gather_refine_context
from tracking.detector.mvq.fusion import FusionStack
from tracking.detector.mvq.geometry import ray_from_pixel, token_pixel_centres

# Real DINOv3 backbone presets, keyed by MVQConfig.backbone. When `backbone`
# names one of these, the preset's embed_dim/depth/num_heads are authoritative
# for the BACKBONE -- see MVQConfig.__post_init__ and MVQModel.__init__.
_BACKBONE_PRESETS = {"dinov3_b16": DINOv3Config.vitb16(), "dinov3_l16": DINOv3Config.vitl16()}


@dataclasses.dataclass(frozen=True)
class MVQConfig:
    crop: int = 448
    patch: int = 16
    embed_dim: int = 768
    num_keypoints: int = 50
    num_cameras: int = 7
    max_frames: int = 8
    n_instances: int = 4
    n_local: int = 2
    n_global: int = 2
    global_pool: int = 2
    dec_layers_3d: int = 8
    dec_layers_2d: int = 4
    dec_heads: int = 12
    mlp_ratio: float = 4.0
    refine_passes: int = 1
    patch_rgb: int = 9
    fourier_bands: int = 8
    roi_scale: float = 24.0
    camera_slot_embed: bool = True
    # A _BACKBONE_PRESETS key ("dinov3_b16", "dinov3_l16") makes the preset's
    # embed_dim/depth/num_heads authoritative: embed_dim must equal it, and
    # backbone_depth/backbone_heads are checked rather than applied. "tiny" is
    # a test-only escape hatch where all three apply verbatim, with no preset
    # and no pretrained weights.
    backbone: str = "dinov3_b16"
    # None = use the preset's own depth/heads (for a _BACKBONE_PRESETS name)
    # or vitb16's (12/12) otherwise. Set explicitly only to shrink the
    # backbone for a test (with backbone="tiny") or to pin/confirm a preset's
    # own value; a value that conflicts with a chosen preset raises.
    backbone_depth: int | None = None
    backbone_heads: int | None = None
    remat: bool = True
    # Query-chunk size for masked_attention's chunked path (None = unchunked),
    # honoured by every masked_attention call in the model, not just the 2D
    # path. Defaults off: chunking at q_chunk=512 cost +13% step time to save
    # 2.2GB, a bad trade once the attention-chunk remat fix (fusion.py) made
    # the unchunked path fit at the 4-8 samples/GPU this model trains at. Kept
    # configurable because a larger n_instances/num_cameras could make Nq big
    # enough to be worth it again. Applies ONLY to attn_impl="xla"; the cudnn
    # path never materialises logits to chunk.
    q_chunk: int | None = None
    # "xla" (CPU-safe explicit softmax) or "cudnn" (flash attention, GPU
    # only), threaded like q_chunk. configs/model/mvq.yaml ships "cudnn";
    # unit tests keep "xla" so they run on CPU.
    attn_impl: str = "xla"

    def __post_init__(self):
        if self.attn_impl not in ("xla", "cudnn"):
            raise ValueError(f"attn_impl must be 'xla' or 'cudnn', got {self.attn_impl!r}")
        preset = _BACKBONE_PRESETS.get(self.backbone)
        if preset is not None:
            if self.embed_dim != preset.embed_dim:
                raise ValueError(
                    f"backbone={self.backbone!r} is embed_dim={preset.embed_dim}, but "
                    f"MVQConfig.embed_dim={self.embed_dim} -- pass embed_dim={preset.embed_dim} "
                    f"(or backbone='tiny' for a from-scratch, test-only backbone shape)."
                )
            if self.backbone_depth is not None and self.backbone_depth != preset.depth:
                raise ValueError(
                    f"backbone={self.backbone!r} is depth={preset.depth}, but "
                    f"backbone_depth={self.backbone_depth} conflicts with it -- leave "
                    f"backbone_depth unset (None) to use the preset's own depth."
                )
            if self.backbone_heads is not None and self.backbone_heads != preset.num_heads:
                raise ValueError(
                    f"backbone={self.backbone!r} is num_heads={preset.num_heads}, but "
                    f"backbone_heads={self.backbone_heads} conflicts with it -- leave "
                    f"backbone_heads unset (None) to use the preset's own num_heads."
                )

    @property
    def grid(self):
        return self.crop // self.patch


class MVQModel(nnx.Module):
    def __init__(self, cfg: MVQConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        D = cfg.embed_dim
        preset = _BACKBONE_PRESETS.get(cfg.backbone)
        if preset is not None:
            depth = preset.depth if cfg.backbone_depth is None else cfg.backbone_depth
            heads = preset.num_heads if cfg.backbone_heads is None else cfg.backbone_heads
        else:
            default = DINOv3Config.vitb16()
            depth = default.depth if cfg.backbone_depth is None else cfg.backbone_depth
            heads = default.num_heads if cfg.backbone_heads is None else cfg.backbone_heads
        self.backbone = DINOv3(
            DINOv3Config(embed_dim=D, depth=depth, num_heads=heads, attn_impl=cfg.attn_impl),
            rngs=rngs,
        )
        nf = 2 * cfg.fourier_bands + 1
        self.geom = nnx.Linear(6 * nf, D, rngs=rngs)
        self.e_frame = nnx.Param(jax.random.normal(rngs.params(), (cfg.max_frames, D)) * 0.02)
        self.e_cam = nnx.Param(jnp.zeros((cfg.num_cameras, D)))
        self.fusion = FusionStack(cfg, rngs=rngs)
        self.decoder = QueryDecoder(cfg, rngs=rngs)
        self.prompt_ln = nnx.LayerNorm(D, rngs=rngs)

    def _bank(self, crops, cam_valid, M, t_local):
        cfg = self.cfg
        B, T, C, H, W, _ = crops.shape
        g = cfg.grid
        N = g * g
        toks = self.backbone(crops.reshape(B * T * C, H, W, 3), remat=cfg.remat).reshape(
            B, T, C, N, -1
        )
        centres = token_pixel_centres(g, g, cfg.patch)  # (N,2)

        def rays(Mb, tlb):  # per sample, per frame
            p0, d = ray_from_pixel(Mb, tlb, jnp.broadcast_to(centres, (C, N, 2)))
            return jnp.concatenate(
                [p0 / cfg.roi_scale, jnp.broadcast_to(d[:, None], p0.shape)], -1
            )  # (C,N,6)

        feats = jax.vmap(lambda Mb, tl: jax.vmap(lambda tlt: rays(Mb, tlt))(tl))(
            M, t_local
        )  # (B,T,C,N,6)
        toks = toks + self.geom(fourier(jnp.clip(feats, -3, 3) / 3.0, cfg.fourier_bands))
        toks = toks + self.e_frame[...][:T][None, :, None, None, :]
        if cfg.camera_slot_embed:
            toks = toks + self.e_cam[...][None, None, :, None, :]
        toks = toks * cam_valid[..., None, None]
        toks = self.fusion(toks.reshape(B, T * C, N, -1), cam_valid.reshape(B, T * C))
        return toks.reshape(B, T, C, N, -1)

    def _prompt(self, grid_tokens, prompt_mask, cam_valid):
        cfg = self.cfg
        B, T, C, N, D = grid_tokens.shape
        g = cfg.grid
        m = (
            prompt_mask.reshape(B, T, C, g, cfg.patch, g, cfg.patch).mean(axis=(4, 6)) > 0.5
        )  # (B,T,C,g,g)
        m = (m.reshape(B, T, C, N) & cam_valid[..., None]).astype(jnp.float32)
        num = (grid_tokens * m[..., None]).sum(axis=(1, 2, 3))
        den = jnp.maximum(m.sum(axis=(1, 2, 3)), 1.0)
        return self.prompt_ln(num / den[:, None])

    def __call__(self, crops, cam_valid, M, t_local, prompt_mask=None, *, prompt_on=None):
        cfg = self.cfg
        B, T, C = crops.shape[:3]
        N = cfg.grid**2
        grid_tokens = self._bank(crops, cam_valid, M, t_local)  # (B,T,C,N,D)
        bank = grid_tokens.reshape(B, T * C * N, -1)
        bank_valid = jnp.repeat(cam_valid.reshape(B, T * C), N, axis=1)
        femb = self.e_frame[...][:T]
        ptok = (
            self._prompt(grid_tokens, prompt_mask, cam_valid) if prompt_mask is not None else None
        )
        out, per_layer, _ = self.decoder(
            bank, bank_valid, M, t_local, femb, ptok, prompt_on, None, 0
        )
        aux_layers, aux_pass1 = list(per_layer), None
        for _p in range(cfg.refine_passes):
            aux_pass1 = out
            ctx = gather_refine_context(
                out["xyz"],
                M,
                t_local,
                cam_valid,
                grid_tokens.reshape(B, T, C, cfg.grid, cfg.grid, -1),
                crops,
                cfg.patch_rgb,
                cfg.fourier_bands,
                cfg.crop,
            )
            ctx = jax.lax.stop_gradient(ctx)
            out2, per2, _ = self.decoder(
                bank, bank_valid, M, t_local, femb, ptok, prompt_on, ctx, 1
            )
            out2["xyz"] = out["xyz"] + out2["xyz"]
            out2["uv"] = jnp.clip(out["uv"] + (out2["uv"] - cfg.crop / 2.0), 0.0, cfg.crop)
            aux_layers.extend(per2)
            out = out2
        out["aux_pass1"], out["aux_layers"] = aux_pass1, aux_layers
        return out


def assemble(out, center3D, crop_origin, exist_thresh=0.5, cam_valid=None):
    """numpy: model outputs -> world kp3d (NaN where absent), conf3d, full-frame kp2d, sex_prob.

    Two independent halves of spec Sec 4.6's NaN policy: an instance that
    doesn't exist (`exist_logit` below threshold) is NaN across every frame
    and keypoint; a FRAME with no valid camera at all (`cam_valid`, when
    given) is NaN across every instance and keypoint for that frame -- there
    is no observation to have triangulated a keypoint from. Per-keypoint
    visibility gating (a keypoint occluded in every view but the frame
    otherwise fine) is deferred to the P4 lifter, not this assembly step.

    `crop_origin` is `(B,C,2)`, not `(B,T,C,2)`: `v12_windows.py` crops every
    frame of a window at the SAME per-camera origin (the projection of the
    window's single center3D, computed once from frame 0), so there is only
    one origin per (sample, camera), shared across all T frames -- broadcast
    below, not indexed by T.

    Instance-slot meaning is fixed (P3a spec Sec 3): slot 0 = prompted,
    1 = female, 2 = male, 3 = other. `sex_prob` is `sigmoid(out["sex_logit"])`
    per slot, P(female) -- independent of `exist`/`cam_valid` gating above,
    since a slot's sex prediction is meaningful whether or not that slot's
    xyz/conf survived the NaN policy.
    """
    xyz, conf = np.asarray(out["xyz"]), 1 / (1 + np.exp(-np.asarray(out["conf_logit"])))
    exist = 1 / (1 + np.exp(-np.asarray(out["exist_logit"]))) >= exist_thresh  # (B,I)
    kp3d = xyz + np.asarray(center3D)[:, None, None, None, :]
    kp3d = np.where(exist[:, :, None, None, None], kp3d, np.nan)
    conf3d = np.where(exist[:, :, None, None], conf, 0.0)
    if cam_valid is not None:
        frame_ok = np.asarray(cam_valid).any(axis=2)  # (B,T)
        kp3d = np.where(frame_ok[:, None, :, None, None], kp3d, np.nan)
        conf3d = np.where(frame_ok[:, None, :, None], conf3d, 0.0)
    kp2d = np.asarray(out["uv"]) + np.asarray(crop_origin)[:, None, None, :, None, :]
    sex_prob = 1 / (1 + np.exp(-np.asarray(out["sex_logit"])))
    return (
        kp3d.astype(np.float32),
        conf3d.astype(np.float32),
        kp2d.astype(np.float32),
        sex_prob.astype(np.float32),
    )
