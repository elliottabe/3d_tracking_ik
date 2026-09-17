"""Image transforms for MVQ: RGB normalization and crop positioning."""

import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def crop_origin(bbox, img_w, img_h, crop=448):
    """Top-left of a crop×crop window centered on bbox, clamped to image.

    Example: crop_origin([500, 200, 20, 20], 1936, 448, crop=100) -> (460, 160)
    """
    cx = bbox[0] + bbox[2] / 2.0
    cy = bbox[1] + bbox[3] / 2.0
    x0 = int(round(cx - crop / 2.0))
    y0 = int(round(cy - crop / 2.0))
    x0 = max(0, min(x0, img_w - crop))
    y0 = max(0, min(y0, img_h - crop))
    return x0, y0


def normalize_rgb(rgb01):
    """Normalize RGB ∈ [0,1] using ImageNet statistics.

    Example: normalize_rgb(np.ones((H, W, 3), dtype=np.float32) * 0.5)
    """
    return ((rgb01 - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float32)
