import numpy as np
import pytest
from pytest import approx

from nnactive_playground.uncertainty_aggregation.aggregate_uncertainties import (
    whole_patch_aggregation,
)


@pytest.fixture
def patch_size():
    return (10, 10, 10)


@pytest.fixture
def test_image_0():
    return np.zeros((64, 64, 64))


@pytest.fixture
def test_image_1():
    return np.ones((64, 64, 64))


@pytest.fixture
def test_image_patch():
    test_image_patch = np.zeros((64, 64, 64))
    test_image_patch[5:15, 10:20, 3:13] = 1
    return test_image_patch


def test_whole_patch_aggregation_zero_image(patch_size, test_image_0):
    zero_aggregated = whole_patch_aggregation(test_image_0, patch_size)
    expected_zero_aggregated = np.zeros(
        (np.array(test_image_0.shape) - (np.array(patch_size) - 1))
    )
    np.testing.assert_array_almost_equal(expected_zero_aggregated, zero_aggregated)


def test_whole_patch_aggregation_one_image(patch_size, test_image_1):
    one_aggregated = whole_patch_aggregation(test_image_1, patch_size)
    expected_one_aggregated = np.ones(
        (np.array(test_image_1.shape) - (np.array(patch_size) - 1))
    )
    np.testing.assert_array_almost_equal(expected_one_aggregated, one_aggregated)


def test_whole_patch_aggregation_patch_image(patch_size, test_image_patch):
    patch_aggregated = whole_patch_aggregation(test_image_patch, patch_size)
    max_value = patch_aggregated.max()
    max_position = np.unravel_index(patch_aggregated.argmax(), patch_aggregated.shape)
    assert max_value == approx(1.0)
    assert max_position == (5, 10, 3)
