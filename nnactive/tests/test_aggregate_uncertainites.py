import numpy as np
import pytest
import torch
from pytest import approx

from nnactive.aggregations.convolution import ConvolveAgg


@pytest.fixture
def patch_size():
    return (10, 10, 10)


@pytest.fixture
def test_image_0():
    return torch.zeros((64, 64, 64))


@pytest.fixture
def test_image_1():
    return torch.ones((64, 64, 64))


def convolve_agg(patch_size):
    return ConvolveAgg(patch_size)


@pytest.fixture
def test_image_patch():
    test_image_patch = torch.zeros((64, 64, 64))
    test_image_patch[5:15, 10:20, 3:13] = 1
    return test_image_patch


def test_whole_patch_aggregation_zero_image(patch_size, test_image_0):
    aggregator = ConvolveAgg(patch_size)
    zero_aggregated, _ = aggregator.forward(test_image_0)
    expected_zero_aggregated = np.zeros(
        (np.array(test_image_0.shape) - (np.array(patch_size) - 1))
    )
    np.testing.assert_array_almost_equal(expected_zero_aggregated, zero_aggregated)


def test_whole_patch_aggregation_one_image(patch_size, test_image_1):
    aggregator = ConvolveAgg(patch_size)
    one_aggregated, _ = aggregator.forward(test_image_1)
    expected_one_aggregated = np.ones(
        (np.array(test_image_1.shape) - (np.array(patch_size) - 1))
    )
    np.testing.assert_array_almost_equal(
        expected_one_aggregated, one_aggregated, decimal=5
    )


def test_whole_patch_aggregation_patch_image(patch_size, test_image_patch):
    aggregator = ConvolveAgg(patch_size)
    patch_aggregated, _ = aggregator.forward(test_image_patch)
    max_value = patch_aggregated.max()
    max_position = np.unravel_index(patch_aggregated.argmax(), patch_aggregated.shape)
    assert max_value == approx(1.0, abs=1e-5)
    assert max_position == (5, 10, 3)
