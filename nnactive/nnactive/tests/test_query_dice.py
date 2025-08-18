import random

import numpy as np
import torch

from nnactive.strategies.dice_query import ExpectedPatchDiceScore


def test_get_coords():
    input_img = np.zeros([3, 8, 8, 8])
    patch_dice = ExpectedPatchDiceScore(patch_size=[2, 2, 2], stride=2)
    coords = patch_dice.get_coords_patches(input_img.shape[1:])
    expected_coords = []
    for i in range(0, 8, 2):
        for j in range(0, 8, 2):
            for k in range(0, 8, 2):
                expected_coords.append([j, i, k])
    expected_coords = np.array(expected_coords)
    np.testing.assert_array_equal(coords, expected_coords)


def test_get_coords_non_fitting_img_size():
    input_img = np.zeros([3, 9, 9, 9])
    patch_dice = ExpectedPatchDiceScore(patch_size=[2, 2, 2], stride=2)
    coords = patch_dice.get_coords_patches(input_img.shape[1:])
    expected_coords = []
    for i in range(0, 8, 2):
        for j in range(0, 8, 2):
            for k in range(0, 8, 2):
                expected_coords.append([j, i, k])
    expected_coords = np.array(expected_coords)
    np.testing.assert_array_equal(coords, expected_coords)


def test_get_coords_larger_patch():
    input_img = np.zeros([3, 8, 8, 8])
    patch_dice = ExpectedPatchDiceScore(patch_size=[4, 4, 4], stride=2)
    coords = patch_dice.get_coords_patches(input_img.shape[1:])
    expected_coords = []
    for i in range(0, 6, 2):
        for j in range(0, 6, 2):
            for k in range(0, 6, 2):
                expected_coords.append([j, i, k])
    expected_coords = np.array(expected_coords)
    np.testing.assert_array_equal(coords, expected_coords)


def test_get_coords_non_fitting_img_size_larger_patch():
    input_img = np.zeros([3, 9, 9, 9])
    patch_dice = ExpectedPatchDiceScore(patch_size=[4, 4, 4], stride=2)
    coords = patch_dice.get_coords_patches(input_img.shape[1:])
    expected_coords = []
    for i in range(0, 6, 2):
        for j in range(0, 6, 2):
            for k in range(0, 6, 2):
                expected_coords.append([j, i, k])
    expected_coords = np.array(expected_coords)
    np.testing.assert_array_equal(coords, expected_coords)


def get_testing_images():
    patch_size = [2, 2, 2]
    input_img = np.zeros([3, 8, 8, 8])
    input_img[-1] = 1
    input_img2 = input_img.copy()
    input_img3 = input_img.copy()
    patch_dice = ExpectedPatchDiceScore(patch_size=patch_size, stride=2)
    coords = patch_dice.get_coords_patches(input_img.shape[1:])
    random_locations = random.sample(coords.tolist(), 3)
    random_location_worst = random_locations[0]
    random_location_medium = random_locations[1]
    random_locations_best_agree = random_locations[2]

    # worst case: all predictions are different
    coords_end = np.array(random_location_worst) + np.array(patch_size)
    coord_slices_worst = tuple(
        slice(cs, ce, None) for cs, ce in zip(random_location_worst, coords_end)
    )
    input_img[0][coord_slices_worst] = 1
    input_img[-1][coord_slices_worst] = 0
    input_img2[1][coord_slices_worst] = 1
    input_img2[-1][coord_slices_worst] = 0
    # input_img3[2][coord_slices_worst] = 1

    # medium case: two predictions are same
    coords_end = np.array(random_location_medium) + np.array(patch_size)
    coord_slices_medium = tuple(
        slice(cs, ce, None) for cs, ce in zip(random_location_medium, coords_end)
    )
    input_img[0][coord_slices_medium] = 1
    input_img[-1][coord_slices_medium] = 0

    # best case: all predictions are same
    coords_end = np.array(random_locations_best_agree) + np.array(patch_size)
    coord_slices_best = tuple(
        slice(cs, ce, None) for cs, ce in zip(random_locations_best_agree, coords_end)
    )
    input_img[0][coord_slices_best] = 1
    input_img[-1][coord_slices_best] = 0
    input_img2[0][coord_slices_best] = 1
    input_img2[-1][coord_slices_best] = 0
    input_img3[0][coord_slices_best] = 1
    input_img3[-1][coord_slices_best] = 0
    return (
        input_img,
        input_img2,
        input_img3,
        random_location_worst,
        random_location_medium,
        random_locations_best_agree,
    )


def test_query_dice():
    for i in range(10):
        (
            input_img,
            input_img2,
            input_img3,
            random_location_worst,
            random_location_medium,
            random_locations_best_agree,
        ) = get_testing_images()
        dice = ExpectedPatchDiceScore(patch_size=[2, 2, 2], stride=2)
        probs = torch.from_numpy(np.stack([input_img, input_img2, input_img3], axis=0))
        sorted_dice_dict, _ = dice.forward(probs, device=torch.device("cpu"))
        np.testing.assert_array_equal(
            np.array(random_location_worst),
            np.array(list(sorted_dice_dict.keys())[0]),
        )
        np.testing.assert_array_equal(
            np.array(random_location_medium),
            np.array(list(sorted_dice_dict.keys())[1]),
        )
