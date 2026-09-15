"""CenterDetect network: EfficientNet-b1 (InstanceNorm) backbone + BiFPN +
detection head, at JARVIS's `model_size="medium"` config
(`num_joints=1, in_channels=3`).

The backbone is the InstanceNorm variant, NOT the standard torchvision
BatchNorm EfficientNet-b3 used elsewhere in this repo's backbone comparisons
-- a different backbone for a different model. Scoped to exactly the
configuration the real checkpoint (`cd_focal_bg30/ckpt/epoch_004`) uses,
verified against its own Orbax metadata: 11 backbone blocks (stages 1-5 of
the b0 block-args table at width/depth coefficient 1.0, truncated at the P5
tap), `se_reduce`/`se_expand` naming, 4 BiFPN cells,
`conv_channel_coef=(24, 40, 112)`.

CHECKPOINT CONTRACT: this module is restored by Orbax, which matches
weights by pytree path. Do not rename any class, any `self.<attr> =
<Module>(...)` attribute, or any `nnx.Param` -- a rename silently leaves
that subtree on randomly-initialised weights.

Tensors are NHWC throughout (JAX convention).
"""

from __future__ import annotations

import dataclasses
import math

import flax.linen as linen_fn
import jax
import jax.numpy as jnp
from flax import nnx

CENTERDETECT_MODEL_SIZE = "medium"
CENTERDETECT_NUM_JOINTS = 1
CENTERDETECT_IN_CHANNELS = 3

# JARVIS's own (backbone_width, backbone_depth) for model_size="medium" --
# efficientnet-b1 in JARVIS's shifted `efficientnet_params()` table (see
# `_MODEL_SIZE_TABLE`).
_BACKBONE_WIDTH = 1.0
_BACKBONE_DEPTH = 1.0

# fpn_num_filters, fpn_cell_repeats, final_layer_sizes for "medium".
_FPN_NUM_FILTERS = 88
_FPN_CELL_REPEATS = 4
_FINAL_LAYER_SIZES = 88
# EfficientNetB3 (InstanceNorm) P3/P4/P5 channel dims at (1.0, 1.0).
_CONV_CHANNEL_COEF = (24, 40, 112)


def instance_norm(x: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    """Param-free InstanceNorm2d (affine=False, track_running_stats=False).

    x: NHWC. Mean/var computed per-sample per-channel over the H,W axes
    (biased variance, matching PyTorch's InstanceNorm2d).
    """
    mean = jnp.mean(x, axis=(1, 2), keepdims=True)
    var = jnp.var(x, axis=(1, 2), keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps)


def _pad_same(k: int) -> int:
    return (k - 1) // 2


def round_filters(
    filters: int, width_coefficient: float, depth_divisor: int = 8, min_depth: int | None = None
) -> int:
    multiplier = width_coefficient
    if not multiplier:
        return filters
    divisor = depth_divisor
    min_depth = min_depth or divisor
    filters *= multiplier
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats: int, depth_coefficient: float) -> int:
    multiplier = depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


@dataclasses.dataclass(frozen=True)
class _BaseStage:
    kernel_size: int
    num_repeat: int
    input_filters: int
    output_filters: int
    expand_ratio: int
    stride: int
    se_ratio: float


# b0 base block-args table (efficientnet() in JARVIS's utils.py).
_BASE_STAGES = [
    _BaseStage(
        kernel_size=3,
        num_repeat=1,
        input_filters=32,
        output_filters=16,
        expand_ratio=1,
        stride=1,
        se_ratio=0.25,
    ),
    _BaseStage(
        kernel_size=3,
        num_repeat=2,
        input_filters=16,
        output_filters=24,
        expand_ratio=6,
        stride=2,
        se_ratio=0.25,
    ),
    _BaseStage(
        kernel_size=5,
        num_repeat=2,
        input_filters=24,
        output_filters=40,
        expand_ratio=6,
        stride=2,
        se_ratio=0.25,
    ),
    _BaseStage(
        kernel_size=3,
        num_repeat=3,
        input_filters=40,
        output_filters=80,
        expand_ratio=6,
        stride=2,
        se_ratio=0.25,
    ),
    _BaseStage(
        kernel_size=5,
        num_repeat=3,
        input_filters=80,
        output_filters=112,
        expand_ratio=6,
        stride=1,
        se_ratio=0.25,
    ),
    _BaseStage(
        kernel_size=5,
        num_repeat=4,
        input_filters=112,
        output_filters=192,
        expand_ratio=6,
        stride=2,
        se_ratio=0.25,
    ),
    _BaseStage(
        kernel_size=3,
        num_repeat=1,
        input_filters=192,
        output_filters=320,
        expand_ratio=6,
        stride=1,
        se_ratio=0.25,
    ),
]


@dataclasses.dataclass(frozen=True)
class _BlockMeta:
    stage_idx: int
    kernel_size: int
    stride: int
    input_filters: int
    output_filters: int
    expand_ratio: int
    se_ratio: float
    id_skip: bool = True


def _build_all_block_meta(width_coefficient: float, depth_coefficient: float) -> list[_BlockMeta]:
    metas: list[_BlockMeta] = []
    for stage_idx, base in enumerate(_BASE_STAGES):
        inp = round_filters(base.input_filters, width_coefficient)
        oup = round_filters(base.output_filters, width_coefficient)
        num_repeat = round_repeats(base.num_repeat, depth_coefficient)

        metas.append(
            _BlockMeta(
                stage_idx=stage_idx,
                kernel_size=base.kernel_size,
                stride=base.stride,
                input_filters=inp,
                output_filters=oup,
                expand_ratio=base.expand_ratio,
                se_ratio=base.se_ratio,
            )
        )
        for _ in range(num_repeat - 1):
            metas.append(
                _BlockMeta(
                    stage_idx=stage_idx,
                    kernel_size=base.kernel_size,
                    stride=1,
                    input_filters=oup,
                    output_filters=oup,
                    expand_ratio=base.expand_ratio,
                    se_ratio=base.se_ratio,
                )
            )
    return metas


def _compute_save_idxs_and_truncate(metas: list[_BlockMeta]):
    """Transcribe JARVIS's `save_idxs`/`last_idx` loop: truncate the backbone
    at the block right before the THIRD stride-2 transition, and mark which
    block outputs are the P3/P4/P5 taps."""
    ignore_first = True
    last_idx = 0
    save_idxs: list[bool] = []
    for idx, meta in enumerate(metas):
        is_stride2 = meta.stride == 2
        if ignore_first and is_stride2:
            ignore_first = False
            save_idxs.append(False)
        else:
            save_idxs.append(is_stride2)
            if is_stride2:
                last_idx = idx - 1
    truncated = metas[: last_idx + 1]
    return truncated, save_idxs


class MBConvBlockJAX(nnx.Module):
    """JARVIS-custom MBConv block: for `stage_idx < 4`, the expand conv is
    never applied (the depthwise conv maps `input_filters ->
    input_filters*expand_ratio` directly, non-grouped); for `stage_idx >= 4`
    the standard expand -> grouped-depthwise path runs."""

    def __init__(self, meta: _BlockMeta, *, rngs: nnx.Rngs):
        self.stage_idx = meta.stage_idx
        self.stride = meta.stride
        self.input_filters = meta.input_filters
        self.output_filters = meta.output_filters
        self.expand_ratio = meta.expand_ratio
        self.id_skip = meta.id_skip
        self.has_se = meta.se_ratio is not None and 0 < meta.se_ratio <= 1

        inp = meta.input_filters
        oup = inp * meta.expand_ratio
        k = meta.kernel_size
        s = meta.stride
        pad = _pad_same(k)

        if meta.stage_idx < 4:
            self.expand_conv = None
            self.depthwise_conv = nnx.Conv(
                inp,
                oup,
                kernel_size=(k, k),
                strides=(s, s),
                padding=((pad, pad), (pad, pad)),
                use_bias=False,
                feature_group_count=1,
                rngs=rngs,
            )
        else:
            if meta.expand_ratio != 1:
                self.expand_conv = nnx.Conv(
                    inp,
                    oup,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    padding=((0, 0), (0, 0)),
                    use_bias=False,
                    rngs=rngs,
                )
            else:
                self.expand_conv = None
            self.depthwise_conv = nnx.Conv(
                oup,
                oup,
                kernel_size=(k, k),
                strides=(s, s),
                padding=((pad, pad), (pad, pad)),
                use_bias=False,
                feature_group_count=oup,
                rngs=rngs,
            )

        if self.has_se:
            num_squeezed = max(1, int(meta.input_filters * meta.se_ratio))
            self.se_reduce = nnx.Conv(
                oup,
                num_squeezed,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding=((0, 0), (0, 0)),
                use_bias=True,
                rngs=rngs,
            )
            self.se_expand = nnx.Conv(
                num_squeezed,
                oup,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding=((0, 0), (0, 0)),
                use_bias=True,
                rngs=rngs,
            )
        else:
            self.se_reduce = None
            self.se_expand = None

        self.project_conv = nnx.Conv(
            oup,
            meta.output_filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding=((0, 0), (0, 0)),
            use_bias=False,
            rngs=rngs,
        )

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = inputs
        if self.stage_idx < 4:
            x = self.depthwise_conv(x)
        else:
            if self.expand_conv is not None:
                x = self.expand_conv(inputs)
            x = self.depthwise_conv(x)

        x = instance_norm(x)
        x = nnx.silu(x)

        if self.has_se:
            x_sq = jnp.mean(x, axis=(1, 2), keepdims=True)
            x_sq = self.se_reduce(x_sq)
            x_sq = nnx.silu(x_sq)
            x_sq = self.se_expand(x_sq)
            x = nnx.sigmoid(x_sq) * x

        x = self.project_conv(x)
        x = instance_norm(x)

        if self.id_skip and self.stride == 1 and self.input_filters == self.output_filters:
            x = x + inputs
        return x


class EfficientNetB3(nnx.Module):
    """InstanceNorm EfficientNet backbone, truncated to (P3, P4, P5)."""

    def __init__(
        self,
        *,
        in_channels: int,
        width_coefficient: float,
        depth_coefficient: float,
        rngs: nnx.Rngs,
    ):
        self.in_channels = in_channels
        stem_out = round_filters(32, width_coefficient=width_coefficient)
        self.conv_stem = nnx.Conv(
            in_channels,
            stem_out,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=((1, 1), (1, 1)),
            use_bias=False,
            rngs=rngs,
        )

        all_metas = _build_all_block_meta(
            width_coefficient=width_coefficient, depth_coefficient=depth_coefficient
        )
        truncated_metas, save_idxs = _compute_save_idxs_and_truncate(all_metas)
        self._save_idxs = save_idxs

        self.blocks = nnx.List([MBConvBlockJAX(m, rngs=rngs) for m in truncated_metas])

    def __call__(self, x: jnp.ndarray):
        x = self.conv_stem(x)
        x = instance_norm(x)
        x = nnx.silu(x)

        feature_maps = []
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if self._save_idxs[idx + 1]:
                feature_maps.append(x)
        assert len(feature_maps) == 3, (
            f"expected 3 feature maps (P3,P4,P5), got {len(feature_maps)}"
        )
        return tuple(feature_maps)


def _relu_norm(w: jnp.ndarray, eps: float = 1e-4) -> jnp.ndarray:
    w = jax.nn.relu(w)
    return w / (jnp.sum(w) + eps)


def _softplus_norm(w: jnp.ndarray, eps: float = 1e-4) -> jnp.ndarray:
    w = jax.nn.softplus(w)
    return w / (jnp.sum(w) + eps)


def _upsample_nearest(x: jnp.ndarray, factor: int) -> jnp.ndarray:
    n, h, w, c = x.shape
    return jax.image.resize(x, (n, h * factor, w * factor, c), method="nearest")


def _upsample_to(x: jnp.ndarray, ref: jnp.ndarray) -> jnp.ndarray:
    n, h, w, c = ref.shape
    return jax.image.resize(x, (n, h, w, c), method="nearest")


def _maxpool2(x: jnp.ndarray) -> jnp.ndarray:
    return linen_fn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")


class SeparableConvBlockJAX(nnx.Module):
    """depthwise(k3,pad1,groups=in,bias=False) -> pointwise(1x1,bias=True) ->
    optional param-free InstanceNorm -> optional SiLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        norm: bool = True,
        activation: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        out_channels = out_channels if out_channels is not None else in_channels
        self.depthwise_conv = nnx.Conv(
            in_channels,
            in_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            use_bias=False,
            feature_group_count=in_channels,
            rngs=rngs,
        )
        self.pointwise_conv = nnx.Conv(
            in_channels,
            out_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding=((0, 0), (0, 0)),
            use_bias=True,
            rngs=rngs,
        )
        self.norm = norm
        self.activation = activation

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        if self.norm:
            x = instance_norm(x)
        if self.activation:
            x = nnx.silu(x)
        return x


class _ChannelProject(nnx.Module):
    """Conv2d(1x1, bias=True) -> param-free InstanceNorm2d."""

    def __init__(self, in_channels: int, out_channels: int, *, rngs: nnx.Rngs):
        self.conv = nnx.Conv(
            in_channels,
            out_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding=((0, 0), (0, 0)),
            use_bias=True,
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return instance_norm(self.conv(x))


_SEP_CONV_NAMES = (
    "conv6_up",
    "conv5_up",
    "conv4_up",
    "conv3_up",
    "conv4_down",
    "conv5_down",
    "conv6_down",
    "conv7_down",
)
_W1_NAMES = ("p6_w1", "p5_w1", "p4_w1", "p3_w1")
_W2_NAMES = ("p4_w2", "p5_w2", "p6_w2", "p7_w2")


def _w2_init_len(name: str) -> int:
    return 2 if name == "p7_w2" else 3


class BiFPNFirstJAX(nnx.Module):
    """First BiFPN cell: projects P3/P4/P5 to `num_channels` and derives P6/P7."""

    def __init__(self, num_channels: int, conv_channels: tuple[int, int, int], *, rngs: nnx.Rngs):
        c3, c4, c5 = conv_channels
        for name in _SEP_CONV_NAMES:
            setattr(self, name, SeparableConvBlockJAX(num_channels, rngs=rngs))

        self.p5_down_channel = _ChannelProject(c5, num_channels, rngs=rngs)
        self.p4_down_channel = _ChannelProject(c4, num_channels, rngs=rngs)
        self.p3_down_channel = _ChannelProject(c3, num_channels, rngs=rngs)
        self.p5_to_p6_conv = _ChannelProject(c5, num_channels, rngs=rngs)
        self.p4_down_channel_2 = _ChannelProject(c4, num_channels, rngs=rngs)
        self.p5_down_channel_2 = _ChannelProject(c5, num_channels, rngs=rngs)

        for name in _W1_NAMES:
            setattr(self, name, nnx.Param(jnp.ones((2,))))
        for name in _W2_NAMES:
            setattr(self, name, nnx.Param(jnp.ones((_w2_init_len(name),))))

    def __call__(self, inputs):
        p3, p4, p5 = inputs

        p6_in = _maxpool2(self.p5_to_p6_conv(p5))
        p7_in = _maxpool2(p6_in)
        p3_in = self.p3_down_channel(p3)
        p4_in = self.p4_down_channel(p4)
        p5_in = self.p5_down_channel(p5)

        w = _relu_norm(self.p6_w1.value)
        p6_up = self.conv6_up(nnx.silu(w[0] * p6_in + w[1] * _upsample_to(p7_in, p6_in)))

        w = _relu_norm(self.p5_w1.value)
        p5_up = self.conv5_up(nnx.silu(w[0] * p5_in + w[1] * _upsample_nearest(p6_up, 2)))

        w = _relu_norm(self.p4_w1.value)
        p4_up = self.conv4_up(nnx.silu(w[0] * p4_in + w[1] * _upsample_nearest(p5_up, 2)))

        w = _relu_norm(self.p3_w1.value)
        p3_out = self.conv3_up(nnx.silu(w[0] * p3_in + w[1] * _upsample_nearest(p4_up, 2)))

        p4_in2 = self.p4_down_channel_2(p4)
        p5_in2 = self.p5_down_channel_2(p5)

        w = _relu_norm(self.p4_w2.value)
        p4_out = self.conv4_down(nnx.silu(w[0] * p4_in2 + w[1] * p4_up + w[2] * _maxpool2(p3_out)))

        w = _relu_norm(self.p5_w2.value)
        p5_out = self.conv5_down(nnx.silu(w[0] * p5_in2 + w[1] * p5_up + w[2] * _maxpool2(p4_out)))

        w = _relu_norm(self.p6_w2.value)
        p6_out = self.conv6_down(nnx.silu(w[0] * p6_in + w[1] * p6_up + w[2] * _maxpool2(p5_out)))

        w = _relu_norm(self.p7_w2.value)
        p7_out = self.conv7_down(nnx.silu(w[0] * p7_in + w[1] * _maxpool2(p6_out)))

        return (p3_out, p4_out, p5_out, p6_out, p7_out)


class BiFPNJAX(nnx.Module):
    """Subsequent BiFPN cells: all 5 pyramid levels already at `num_channels`."""

    def __init__(self, num_channels: int, *, rngs: nnx.Rngs):
        for name in _SEP_CONV_NAMES:
            setattr(self, name, SeparableConvBlockJAX(num_channels, rngs=rngs))
        for name in _W1_NAMES:
            setattr(self, name, nnx.Param(jnp.ones((2,))))
        for name in _W2_NAMES:
            setattr(self, name, nnx.Param(jnp.ones((_w2_init_len(name),))))

    def __call__(self, inputs):
        p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        w = _relu_norm(self.p6_w1.value)
        p6_up = self.conv6_up(nnx.silu(w[0] * p6_in + w[1] * _upsample_to(p7_in, p6_in)))

        w = _relu_norm(self.p5_w1.value)
        p5_up = self.conv5_up(nnx.silu(w[0] * p5_in + w[1] * _upsample_nearest(p6_up, 2)))

        w = _relu_norm(self.p4_w1.value)
        p4_up = self.conv4_up(nnx.silu(w[0] * p4_in + w[1] * _upsample_nearest(p5_up, 2)))

        w = _relu_norm(self.p3_w1.value)
        p3_out = self.conv3_up(nnx.silu(w[0] * p3_in + w[1] * _upsample_nearest(p4_up, 2)))

        w = _relu_norm(self.p4_w2.value)
        p4_out = self.conv4_down(nnx.silu(w[0] * p4_in + w[1] * p4_up + w[2] * _maxpool2(p3_out)))

        w = _relu_norm(self.p5_w2.value)
        p5_out = self.conv5_down(nnx.silu(w[0] * p5_in + w[1] * p5_up + w[2] * _maxpool2(p4_out)))

        w = _relu_norm(self.p6_w2.value)
        p6_out = self.conv6_down(nnx.silu(w[0] * p6_in + w[1] * p6_up + w[2] * _maxpool2(p5_out)))

        w = _relu_norm(self.p7_w2.value)
        p7_out = self.conv7_down(nnx.silu(w[0] * p7_in + w[1] * _maxpool2(p6_out)))

        return (p3_out, p4_out, p5_out, p6_out, p7_out)


class EfficientTrack(nnx.Module):
    """CenterDetect network at `model_size="medium"`: EfficientNetB3
    (InstanceNorm) backbone -> 4 BiFPN cells -> first_conv -> deconv1
    (the full-resolution heatmap this module's callers use)."""

    def __init__(
        self,
        *,
        num_joints: int = CENTERDETECT_NUM_JOINTS,
        in_channels: int = CENTERDETECT_IN_CHANNELS,
        rngs: nnx.Rngs,
    ):
        self.num_joints = num_joints
        self.fpn_num_filters = _FPN_NUM_FILTERS
        self.fpn_cell_repeats = _FPN_CELL_REPEATS
        self.final_layer_sizes = _FINAL_LAYER_SIZES
        self.conv_channel_coef = _CONV_CHANNEL_COEF

        self.backbone = EfficientNetB3(
            in_channels=in_channels,
            width_coefficient=_BACKBONE_WIDTH,
            depth_coefficient=_BACKBONE_DEPTH,
            rngs=rngs,
        )

        cells = [BiFPNFirstJAX(self.fpn_num_filters, self.conv_channel_coef, rngs=rngs)]
        cells += [
            BiFPNJAX(self.fpn_num_filters, rngs=rngs) for _ in range(1, self.fpn_cell_repeats)
        ]
        self.bifpn = nnx.List(cells)

        self.weights_cat = nnx.Param(jnp.ones((3,)))
        self.first_conv = SeparableConvBlockJAX(
            self.fpn_num_filters, self.final_layer_sizes, norm=True, activation=False, rngs=rngs
        )

        # ConvTranspose2d(final_layer_sizes -> J, k4, s2, pad1, bias=False) in
        # PyTorch; hence padding=((2,2),(2,2)) for the 112 -> 224 doubling.
        self.deconv1 = nnx.ConvTranspose(
            self.final_layer_sizes,
            num_joints,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding=((2, 2), (2, 2)),
            use_bias=False,
            transpose_kernel=True,
            rngs=rngs,
        )
        self.final_conv1 = nnx.Conv(
            self.final_layer_sizes,
            num_joints,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            use_bias=False,
            rngs=rngs,
        )

    def forward_both(self, x: jnp.ndarray):
        features = self.backbone(x)
        for cell in self.bifpn:
            features = cell(features)

        x3 = _upsample_nearest(features[2], 4)
        x2 = _upsample_nearest(features[1], 2)
        w = _softplus_norm(self.weights_cat.value)
        x1 = w[0] * features[0] + w[1] * x2 + w[2] * x3

        pre = self.first_conv(x1)
        res2 = self.deconv1(pre)
        res1 = self.final_conv1(pre)
        return res1, res2

    def __call__(self, x: jnp.ndarray, *, use_running_average: bool = False) -> jnp.ndarray:
        """`use_running_average` is accepted (and ignored): EfficientTrack has
        no BatchNorm/running-stats layers (InstanceNorm is computed per-example),
        so there is nothing to switch. Kept for a uniform model call contract."""
        _, res2 = self.forward_both(x)
        return res2


def restore_centerdetect(ckpt_dir):
    """Restore a CenterDetect `EfficientTrack(num_joints=1, in_channels=3)`
    checkpoint. eval_shape -> replicated-sharding ShapeDtypeStruct targets ->
    Orbax StandardCheckpointer restore -> nnx.merge.

    Kept import-lazy (jax/orbax/flax) so this module imports on a CPU-only
    test box with no checkpoint present.
    """
    import orbax.checkpoint as ocp
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    def ctor():
        return EfficientTrack(
            num_joints=CENTERDETECT_NUM_JOINTS,
            in_channels=CENTERDETECT_IN_CHANNELS,
            rngs=nnx.Rngs(0),
        )

    gdef, abstract = nnx.split(nnx.eval_shape(ctor))
    repl = NamedSharding(Mesh(jax.devices(), axis_names=("data",)), P())
    target = jax.tree_util.tree_map(
        lambda v: jax.ShapeDtypeStruct(v.shape, v.dtype, sharding=repl), abstract
    )
    model = nnx.merge(gdef, ocp.StandardCheckpointer().restore(ckpt_dir, target=target))
    model.eval()
    return model


def jit_centerdetect_forward(model):
    """A restored CenterDetect nnx module -> a jitted `x -> heatmap` closure.

    `nnx.split`/`nnx.merge` makes an nnx module jit-compatible: the graphdef
    is a static Python object closed over, the state is the traced argument.
    `state` is bound with `functools.partial` (rather than closed over)
    solely so the compiled function stays inspectable; built ONCE per
    `CenterDetector` instance, never per call, so repeated calls hit the
    jax dispatch-cache instead of re-tracing.
    """
    import functools

    gdef, state = nnx.split(model)

    @jax.jit
    def _forward(state, x):
        return nnx.merge(gdef, state)(x, use_running_average=True)

    return functools.partial(_forward, state)
