import numpy as np

from tracking.train.data.transforms import IMAGENET_MEAN, IMAGENET_STD, crop_origin, normalize_rgb


def test_imagenet_constants():
    np.testing.assert_allclose(IMAGENET_MEAN, [0.485, 0.456, 0.406], rtol=1e-6)
    np.testing.assert_allclose(IMAGENET_STD, [0.229, 0.224, 0.225], rtol=1e-6)


def test_crop_origin_centres_on_the_bbox_centre():
    x0, y0 = crop_origin([500, 200, 20, 20], 1936, 448, crop=100)
    assert (x0, y0) == (460, 160)


def test_crop_origin_clamps_at_the_left_and_top_edges():
    assert crop_origin([5, 5, 10, 10], 1936, 448, crop=100) == (0, 0)


def test_crop_origin_clamps_at_the_right_and_bottom_edges():
    x0, y0 = crop_origin([1930, 440, 10, 10], 1936, 448, crop=100)
    assert x0 == 1936 - 100
    assert y0 == 448 - 100


def test_crop_origin_handles_an_image_smaller_than_the_crop():
    x0, y0 = crop_origin([10, 10, 4, 4], 50, 40, crop=100)
    assert x0 <= 0 and y0 <= 0


def test_normalize_rgb_applies_imagenet_stats():
    x = np.full((2, 2, 3), 0.485, np.float32)
    out = normalize_rgb(x)
    np.testing.assert_allclose(out[..., 0], 0.0, atol=1e-6)
