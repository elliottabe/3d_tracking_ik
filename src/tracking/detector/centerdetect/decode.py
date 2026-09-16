"""Decode-side top-2 peak extraction and pixel-space bookkeeping for
CenterDetect (multi-animal, single output channel).
"""

from __future__ import annotations

import numpy as np

CENTERDETECT_INPUT_SIZE = 320
CENTERDETECT_HEATMAP_SIZE = 160

# ImageNet normalisation constants, matching training-time preprocessing.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def extract_top_k_peaks(heatmap, *, k: int = 2, suppression_radius: int = 15):
    """(B,H,W) or (B,H,W,1) heatmap -> (peaks_xy (B,k,2), conf (B,k))."""
    hm = np.asarray(heatmap)
    if hm.ndim == 4:
        if hm.shape[-1] != 1:
            raise ValueError(
                f"extract_top_k_peaks expects a single-channel heatmap, got shape {hm.shape}"
            )
        hm = hm[..., 0]
    b, h, w = hm.shape
    working = hm.copy()

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    peaks = np.zeros((b, k, 2), dtype=np.float32)
    conf = np.zeros((b, k), dtype=np.float32)
    for j in range(k):
        flat = working.reshape(b, h * w)
        m = flat.argmax(axis=1)  # (B,)
        py = (m // w).astype(np.float32)
        px = (m % w).astype(np.float32)
        peaks[:, j, 0] = px
        peaks[:, j, 1] = py
        conf[:, j] = flat[np.arange(b), m]
        for bi in range(b):
            mask = ((xx - px[bi]) ** 2 + (yy - py[bi]) ** 2) <= suppression_radius**2
            working[bi][mask] = -np.inf
    return peaks, conf


def peaks_to_full_image(peaks_xy, heatmap_size: int, img_w: int, img_h: int):
    """Rescale (..., 2) heatmap-space (x, y) peaks to ORIGINAL full-image
    pixel coordinates, undoing the anisotropic (aspect-distorting) resize
    CenterDetect preprocessing applies (separate x/y scale factors)."""
    peaks_xy = np.asarray(peaks_xy, dtype=np.float64)
    out = peaks_xy.copy()
    out[..., 0] *= img_w / float(heatmap_size)
    out[..., 1] *= img_h / float(heatmap_size)
    return out


def peaks_from_heatmap(hm, img_w: int, img_h: int, min_score: float = 0.2):
    """`(C,160,160)` or `(C,160,160,1)` CenterDetect heatmap -> full-image-px
    peaks.

    Args:
        hm: (C, H_hm, W_hm[, 1]) heatmap, single channel (num_joints=1).
        img_w, img_h: ORIGINAL full-image size (before the 320x320 squash);
            `peaks_to_full_image` applies the anisotropic x/y scale this
            implies, matching the training-side resize.
        min_score: a peak with score below this is reported as NaN (both
            coordinates) rather than a false-confident location -- so a
            camera with only one real animal doesn't hand a phantom second
            centre to `lift_peaks_to_centres`.

    Returns:
        peaks: (C, 2, 2) float32 full-image [x, y] px, NaN where score < min_score.
        scores: (C, 2) float32 RAW heatmap confidence, always returned
            unthresholded/un-NaN'd -- a peak being unusable is signalled by
            its COORDINATES being NaN, not by its score.
    """
    hm = np.asarray(hm)
    heatmap_size = hm.shape[1] if hm.ndim >= 3 else hm.shape[0]
    peaks_hm, conf = extract_top_k_peaks(hm, k=2, suppression_radius=15)
    full = peaks_to_full_image(peaks_hm, heatmap_size, img_w, img_h).astype(np.float32)
    conf = conf.astype(np.float32)
    below = conf < min_score
    full[below] = np.nan
    return full, conf


def preprocess(frame: np.ndarray) -> np.ndarray:
    """One full RGB frame -> the 320x320 CenterDetect model input.

    Args:
        frame: (H, W, 3) uint8 RGB.
    Returns:
        (320, 320, 3) float32, ImageNet-normalised.
    """
    from PIL import Image

    resized = np.asarray(
        Image.fromarray(frame).resize(
            (CENTERDETECT_INPUT_SIZE, CENTERDETECT_INPUT_SIZE), Image.BILINEAR
        )
    )
    return ((resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float32)
