"""Multi-view augmentation for mvq training windows. Every geometric op
updates the affine cameras so GT 3D still reprojects onto GT 2D exactly."""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import convolve2d

from tracking.detector.mvq.geometry import mirror_world, px_scale, rotate_world
from tracking.io.names import Order
from tracking.train.data.windows import CROP


@dataclasses.dataclass
class MVAugParams:
    enabled: bool = True
    rot_deg: float = 30.0
    scale_min: float = 0.8
    scale_max: float = 1.25
    translate_frac: float = 0.1
    world_yaw: bool = True
    world_tilt_deg: float = 30.0
    mirror_p: float = 0.5
    cam_drop_p: float = 0.1
    cam_drop_max: int = 2
    brightness: float = 0.2
    contrast: float = 0.2
    gamma: float = 0.2
    blur_max: float = 0.5
    noise_scale: float = 0.02
    pc_color: float = 0.2


def _mirror_name(name: str) -> str | None:
    """Left/right mirror of a keypoint name, or None if midline. The side
    token is the last char of the segment before the first '_'.

    >>> _mirror_name("EyeL"), _mirror_name("T1L_FeTi"), _mirror_name("Scutellum")
    ('EyeR', 'T1R_FeTi', None)
    """
    head = name.split("_", 1)[0]
    if head.endswith("L"):
        return name.replace(head, head[:-1] + "R", 1)
    if head.endswith("R"):
        return name.replace(head, head[:-1] + "L", 1)
    return None


def build_lr_swap(kp_names) -> np.ndarray:
    """int32 (K,) permutation mapping each keypoint to its L/R mirror (midline -> itself)."""
    order = Order(kp_names)
    swap = np.arange(len(order), dtype=np.int32)
    for i, name in enumerate(order):
        mirror = _mirror_name(name)
        if mirror is not None and mirror in order:
            swap[i] = order.index(mirror)
    if not np.array_equal(swap[swap], np.arange(len(order))):
        raise ValueError("lr_swap is not an involution -- check keypoint L/R naming")
    return swap


def assert_lr_swap_covers(kp_names, swap) -> None:
    """Raise unless every left/right keypoint in `kp_names` is paired to its
    mirror by `swap`. A silently unpaired keypoint makes the mirror
    augmentation relabel one side as the other -- a self-consistent, wrong
    pose no smoothness metric would catch."""
    order = Order(kp_names)
    missing = []
    for i, name in enumerate(order):
        mirror = _mirror_name(name)
        if mirror is None:
            continue
        if mirror not in order or int(swap[i]) != order.index(mirror):
            missing.append(name)
    if missing:
        raise ValueError(
            f"lr_swap does not pair {sorted(set(missing))}: the mirror augmentation "
            f"would leave those labels unmirrored"
        )


def _forward_affine(theta, s, tx, ty, c=(CROP - 1) / 2.0):
    """Forward map p' = A p + b: rotate by theta and scale by s about centre
    c, then translate by (tx, ty). theta/s/tx/ty are (C,); returns A (C,2,2), b (C,2)."""
    ct, st = np.cos(theta), np.sin(theta)
    A = s[:, None, None] * np.stack([np.stack([ct, -st], -1), np.stack([st, ct], -1)], -2)
    centre = np.array([c, c], np.float32)
    b = centre - np.einsum("cij,j->ci", A, centre) + np.stack([tx, ty], -1)
    return A.astype(np.float32), b.astype(np.float32)


def _warp_rgb(img, A, b):
    """Warp (H,W,ch) uint8 by the FORWARD map (A,b): output pixel q samples input A^-1 (q - b)."""
    H, W = img.shape[:2]
    Ai = np.linalg.inv(A)
    ys, xs = np.meshgrid(
        np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij"
    )
    q = np.stack([xs, ys], -1) - b
    src = q @ Ai.T
    coords = [src[..., 1], src[..., 0]]
    out = np.stack(
        [
            map_coordinates(
                img[..., ch].astype(np.float32), coords, order=1, mode="constant", cval=0.0
            )
            for ch in range(img.shape[-1])
        ],
        -1,
    )
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def _warp_mask(mask, A, b):
    m = mask[..., None].astype(np.uint8) * 255
    return _warp_rgb(m, A, b)[..., 0] > 127


def _per_view_affine(sample, params: MVAugParams, rng):
    T, C = sample["crops"].shape[:2]
    theta = np.deg2rad(rng.uniform(-params.rot_deg, params.rot_deg, size=C))
    s = rng.uniform(params.scale_min, params.scale_max, size=C)
    tx = rng.uniform(-params.translate_frac, params.translate_frac, size=C) * CROP
    ty = rng.uniform(-params.translate_frac, params.translate_frac, size=C) * CROP
    A, b = _forward_affine(theta, s, tx, ty)

    crops = np.stack(
        [
            np.stack([_warp_rgb(sample["crops"][t, c], A[c], b[c]) for c in range(C)])
            for t in range(T)
        ]
    )
    donor_mask = np.stack(
        [
            np.stack([_warp_mask(sample["donor_mask"][t, c], A[c], b[c]) for c in range(C)])
            for t in range(T)
        ]
    )
    M = np.einsum("cij,cjk->cik", A, sample["M"]).astype(np.float32)
    t_local = (np.einsum("cij,tcj->tci", A, sample["t_local"]) + b[None, :, :]).astype(np.float32)
    kp2d = (np.einsum("cij,ftckj->ftcki", A, sample["kp2d"]) + b[None, None, :, None, :]).astype(
        np.float32
    )
    inb = ((kp2d >= 0) & (kp2d <= CROP - 1)).all(-1)
    return {
        **sample,
        "crops": crops,
        "donor_mask": donor_mask,
        "M": M,
        "t_local": t_local,
        "kp2d": kp2d,
        "vis2d": sample["vis2d"] & inb,
        "px_scale": np.float32(px_scale(M)),
    }


def _rot_mat(yaw, tilt):
    cz, sz, cx, sx = np.cos(yaw), np.sin(yaw), np.cos(tilt), np.sin(tilt)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], np.float32)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], np.float32)
    return Rz @ Rx


def _world_rotation(sample, params: MVAugParams, rng):
    yaw = rng.uniform(0.0, 2 * np.pi) if params.world_yaw else 0.0
    tilt = np.deg2rad(rng.uniform(-params.world_tilt_deg, params.world_tilt_deg))
    R = _rot_mat(yaw, tilt)
    M = np.asarray(rotate_world(sample["M"], R), np.float32)
    X = np.einsum("ij,ftkj->ftki", R, sample["kp3d_local"]).astype(np.float32)
    return {**sample, "M": M, "kp3d_local": X}


def _mirror(sample, params: MVAugParams, rng, lr_swap):
    if rng.random() >= params.mirror_p:
        return sample
    M2, t2 = mirror_world(sample["M"], sample["t_local"][0], CROP)
    M2 = np.asarray(M2, np.float32)
    t_local = np.broadcast_to(np.asarray(t2, np.float32), sample["t_local"].shape).astype(
        np.float32
    )
    X = sample["kp3d_local"][..., lr_swap, :] * np.array([-1.0, 1.0, 1.0], np.float32)
    kp2d = sample["kp2d"][..., lr_swap, :].copy()
    kp2d[..., 0] = CROP - 1 - kp2d[..., 0]
    return {
        **sample,
        "crops": sample["crops"][..., ::-1, :],
        "donor_mask": sample["donor_mask"][..., ::-1],
        "M": M2,
        "t_local": t_local,
        "kp3d_local": X.astype(np.float32),
        "has3d": sample["has3d"][..., lr_swap],
        "kp2d": kp2d.astype(np.float32),
        "vis2d": sample["vis2d"][..., lr_swap],
    }


def _camera_dropout(sample, params: MVAugParams, rng):
    if rng.random() >= params.cam_drop_p:
        return sample
    n_drop = int(rng.integers(1, params.cam_drop_max + 1))
    ref = sample["cam_valid"].all(axis=0)
    score = rng.random(ref.shape[0]) + (~ref).astype(np.float32)
    order = np.argsort(score)
    rank = np.argsort(order)
    n_drop = min(n_drop, max(int(ref.sum()) - 3, 0))
    drop = rank < n_drop
    cam_valid = sample["cam_valid"] & ~drop[None, :]
    vis2d = sample["vis2d"] & cam_valid[None, :, :, None]
    return {**sample, "cam_valid": cam_valid, "vis2d": vis2d}


def _gauss_kernel2d(ksize, sigma):
    ax = np.arange(ksize, dtype=np.float32) - (ksize - 1) / 2.0
    g = np.exp(-(ax**2) / (2.0 * sigma**2))
    g = g / g.sum()
    return np.outer(g, g)


def _photometric(sample, params: MVAugParams, rng):
    """Independent brightness/contrast/gamma/blur/noise/per-channel colour
    jitter on every (T, C) crop -- each view gets its own random strength."""
    T, C, H, W, _ = sample["crops"].shape
    N = T * C
    flat = sample["crops"].reshape(N, H, W, 3).astype(np.float32) / 255.0

    b = rng.uniform(-params.brightness, params.brightness, size=(N, 1, 1, 1)).astype(np.float32)
    c = rng.uniform(1 - params.contrast, 1 + params.contrast, size=(N, 1, 1, 1)).astype(np.float32)
    g = rng.uniform(1 - params.gamma, 1 + params.gamma, size=(N, 1, 1, 1)).astype(np.float32)
    mean = flat.mean(axis=(1, 2, 3), keepdims=True)
    flat = np.clip((flat - mean) * c + mean + b, 0.0, 1.0) ** g

    if params.blur_max > 0:
        k = _gauss_kernel2d(5, 1.0)
        blurred = np.stack(
            [
                np.stack(
                    [
                        convolve2d(flat[n, ..., ch], k, mode="same", boundary="fill")
                        for ch in range(3)
                    ],
                    -1,
                )
                for n in range(N)
            ]
        )
        alpha = rng.uniform(0.0, params.blur_max, size=(N, 1, 1, 1)).astype(np.float32)
        flat = (1.0 - alpha) * flat + alpha * blurred

    if params.noise_scale > 0:
        scale = rng.uniform(0.0, params.noise_scale, size=(N, 1, 1, 1)).astype(np.float32)
        flat = np.clip(flat + rng.normal(size=flat.shape).astype(np.float32) * scale, 0.0, 1.0)

    if params.pc_color > 0:
        f = rng.uniform(1 - params.pc_color, 1 + params.pc_color, size=(N, 1, 1, 3)).astype(
            np.float32
        )
        flat = flat * f

    crops = np.clip(np.round(flat * 255.0), 0, 255).astype(np.uint8).reshape(sample["crops"].shape)
    return {**sample, "crops": crops}


def augment_window(sample, params: MVAugParams, rng: np.random.Generator, lr_swap) -> dict:
    if not params.enabled:
        return sample
    lr_swap = np.asarray(lr_swap)
    if (
        params.rot_deg > 0
        or params.scale_min != 1
        or params.scale_max != 1
        or params.translate_frac > 0
    ):
        sample = _per_view_affine(sample, params, rng)
    if params.world_yaw or params.world_tilt_deg > 0:
        sample = _world_rotation(sample, params, rng)
    if params.mirror_p > 0:
        sample = _mirror(sample, params, rng, lr_swap)
    if params.cam_drop_p > 0:
        sample = _camera_dropout(sample, params, rng)
    if any(
        v > 0
        for v in (
            params.brightness,
            params.contrast,
            params.gamma,
            params.blur_max,
            params.noise_scale,
            params.pc_color,
        )
    ):
        sample = _photometric(sample, params, rng)
    return sample
