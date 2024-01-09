import numpy as np
import pytest
import torch

from nnactive.aggregations.convolution import ConvolveAggScipy, ConvolveAggTorch
from nnactive.strategies.base_uncertainty import select_top_n_non_overlapping_patches


@pytest.fixture
def patch_size():
    return (10, 10, 10)


@pytest.fixture
def test_image():
    test_image = torch.zeros((110, 10, 10))
    pixel_value = 1
    for start_idx in range(0, 110, 10):
        test_image[start_idx : start_idx + 10, 0:10, 0:10] = pixel_value
        pixel_value -= 0.1
    return test_image


@pytest.fixture
def aggregations(patch_size):
    aggregations = [
        ConvolveAggTorch(patch_size, stride=1),
        ConvolveAggScipy(patch_size, stride=1),
    ]
    return aggregations


def test_query_patches(test_image, patch_size, aggregations):
    for aggregation in aggregations:
        image_aggregated, _ = aggregation.forward(test_image)
        selected_array = np.zeros_like(test_image)
        n = 3
        queried_patches = select_top_n_non_overlapping_patches(
            "test", n, image_aggregated, np.array(patch_size), selected_array
        )
        queried_patches = sorted(
            queried_patches, key=lambda d: d["score"], reverse=True
        )
        for index in range(n):
            assert queried_patches[index]["file"] == "test"
            assert queried_patches[index]["coords"] == (index * 10, 0, 0)
            np.testing.assert_array_almost_equal(
                queried_patches[index]["size"], np.array([10, 10, 10]), decimal=5
            )
            assert queried_patches[index]["score"] == pytest.approx(
                1 - index / 10, abs=2e-5
            )


def test_query_patches_with_annotated(test_image, patch_size, aggregations):
    for aggregation in aggregations:
        image_aggregated, _ = aggregation.forward(test_image)
        selected_array = np.zeros_like(test_image)
        selected_array[0:10] = 1
        n = 3
        queried_patches = select_top_n_non_overlapping_patches(
            "test", n, image_aggregated, np.array(patch_size), selected_array
        )
        queried_patches = sorted(
            queried_patches, key=lambda d: d["score"], reverse=True
        )
        for index in range(1, n + 1):
            assert queried_patches[index - 1]["file"] == "test"
            assert queried_patches[index - 1]["coords"] == (index * 10, 0, 0)
            np.testing.assert_array_almost_equal(
                queried_patches[index - 1]["size"], np.array([10, 10, 10]), decimal=5
            )
            assert queried_patches[index - 1]["score"] == pytest.approx(
                1 - index / 10, abs=2e-5
            )
