import numpy as np
import pytest
import torch
from pytest import approx

from nnactive.aggregations.convolution import ConvolveAggScipy, ConvolveAggTorch


@pytest.fixture
def patch_size():
    return (10, 10, 10)


@pytest.fixture
def test_image_0():
    return torch.zeros((64, 64, 64))


@pytest.fixture
def test_image_1():
    return torch.ones((64, 64, 64))


@pytest.fixture
def test_image_patch():
    test_image_patch = torch.zeros((64, 64, 64))
    test_image_patch[5:15, 10:20, 3:13] = 1
    return test_image_patch


@pytest.fixture
def test_image_2d():
    img = torch.zeros((64, 64), dtype=torch.float)
    img[4:6, 8:10] = 1
    return img


@pytest.fixture
def max_position_2d():
    max_position = (4, 8)
    return max_position


@pytest.fixture
def patch_size_2d():
    return [2, 2]


def test_aggregation_2d(patch_size_2d, test_image_2d, max_position_2d):
    aggregators = [ConvolveAggTorch(patch_size_2d, stride=i) for i in [1, 2, 4]]
    aggregators.append(ConvolveAggScipy(patch_size_2d))

    for aggregator in aggregators:
        aggregated, _ = aggregator.forward(test_image_2d)
        flat_aggregated_uncertainties = aggregated.flatten()

        sorted_uncertainty_indices = np.flip(np.argsort(flat_aggregated_uncertainties))
        max_position_flat = sorted_uncertainty_indices[0]
        max_value = flat_aggregated_uncertainties[max_position_flat]
        max_position = aggregator.backward_index(max_position_flat, aggregated.shape)
        assert max_value == approx(1.0, abs=1e-5)
        assert max_position == max_position_2d


def test_whole_patch_aggregation_zero_image(patch_size, test_image_0):
    aggregators = [ConvolveAggTorch(patch_size, stride=1), ConvolveAggScipy(patch_size)]

    for aggregator in aggregators:
        zero_aggregated, _ = aggregator.forward(test_image_0)
        expected_zero_aggregated = np.zeros(
            (np.array(test_image_0.shape) - (np.array(patch_size) - 1))
        )
        np.testing.assert_array_almost_equal(expected_zero_aggregated, zero_aggregated)


def test_whole_patch_aggregation_one_image(patch_size, test_image_1):
    aggregator = ConvolveAggScipy(patch_size)
    one_aggregated, _ = aggregator.forward(test_image_1)
    expected_one_aggregated = np.ones(
        (np.array(test_image_1.shape) - (np.array(patch_size) - 1))
    )
    np.testing.assert_array_almost_equal(
        expected_one_aggregated, one_aggregated, decimal=5
    )


def test_whole_patch_aggregation_patch_image(patch_size, test_image_patch):
    aggregator = ConvolveAggScipy(patch_size)
    patch_aggregated, _ = aggregator.forward(test_image_patch)
    max_value = patch_aggregated.max()
    max_position = aggregator.backward_index(
        patch_aggregated.argmax(), patch_aggregated.shape
    )
    assert max_value == approx(1.0, abs=1e-5)
    assert max_position == (5, 10, 3)
