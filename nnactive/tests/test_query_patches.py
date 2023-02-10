import numpy as np
import pytest

from nnactive.query_patches import get_top_n_non_overlapping_patches
from nnactive.uncertainty_aggregation.aggregate_uncertainties import (
    whole_patch_aggregation,
)


@pytest.fixture
def patch_size():
    return (10, 10, 10)


@pytest.fixture
def test_image():
    test_image = np.zeros((110, 10, 10))
    pixel_value = 1
    for start_idx in range(0, 110, 10):
        test_image[start_idx : start_idx + 10, 0:10, 0:10] = pixel_value
        pixel_value -= 0.1
    return test_image


def test_query_patches(test_image, patch_size):
    image_aggregated = whole_patch_aggregation(test_image, patch_size)
    selected_array = np.zeros_like(test_image)
    n = 3
    queried_patches = get_top_n_non_overlapping_patches(
        "test", n, image_aggregated, np.array(patch_size), selected_array
    )
    queried_patches = sorted(queried_patches, key=lambda d: d["score"], reverse=True)
    for index in range(n):
        assert queried_patches[index]["file"] == "test"
        assert queried_patches[index]["coords"] == (index * 10, 0, 0)
        np.testing.assert_array_almost_equal(
            queried_patches[index]["size"], np.array([10, 10, 10])
        )
        assert queried_patches[index]["score"] == pytest.approx(1 - index / 10)


def test_query_patches_with_annotated(test_image, patch_size):
    image_aggregated = whole_patch_aggregation(test_image, patch_size)
    selected_array = np.zeros_like(test_image)
    selected_array[0:10] = 1
    n = 3
    queried_patches = get_top_n_non_overlapping_patches(
        "test", n, image_aggregated, np.array(patch_size), selected_array
    )
    queried_patches = sorted(queried_patches, key=lambda d: d["score"], reverse=True)
    for index in range(1, n + 1):
        assert queried_patches[index - 1]["file"] == "test"
        assert queried_patches[index - 1]["coords"] == (index * 10, 0, 0)
        np.testing.assert_array_almost_equal(
            queried_patches[index - 1]["size"], np.array([10, 10, 10])
        )
        assert queried_patches[index - 1]["score"] == pytest.approx(1 - index / 10)
