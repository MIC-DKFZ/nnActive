from itertools import product

import pytest
import torch

from nnactive.strategies.utils import RepresentationHandler


def test_representation_basic():
    num_dims = 2
    num_img_channels = 1
    img_size = 10
    repr_scaling = 2
    repr_dim = 1

    img_shape = [num_img_channels] + [img_size] * num_dims

    input_shape = img_shape[1:]
    scaling_factors = [repr_scaling] * len(input_shape)

    representation = RepresentationHandler(
        input_shape, repr_dim, scaling_factor=scaling_factors
    )
    representation.init_representation()
    # ensure starting representation is empty
    assert representation.image.sum() == 0
    repr_shape = [repr_dim] + [2 for s in input_shape]
    input_representation = torch.ones(repr_shape)
    slicer = tuple(
        [
            slice(0, r_s * s_f)
            for r_s, s_f in zip(input_representation.shape[1:], scaling_factors)
        ]
    )
    for i in range(10):
        representation.update_que(input_representation)
        representation.update_representation(slicer)
        assert representation.image.sum() == input_representation.sum() * (i + 1)

    representation.build_representation()
    # ensure values are correct
    assert (
        representation.map_to_representation(slicer).sum() == input_representation.sum()
    )
    # ensure undefined values with no count are nan
    representation.image[
        representation.image_slice_to_representation_slice(slicer)
    ] = torch.nan
    assert torch.all(representation.image.isnan())


def create_representation():
    num_dims = 2
    num_img_channels = 1
    img_size = 10
    repr_scaling = 2
    repr_dim = 1
    repr_size = 1

    img_shape = [num_img_channels] + [img_size] * num_dims

    input_shape = img_shape[1:]
    scaling_factors = [repr_scaling] * len(input_shape)

    representation = RepresentationHandler(
        input_shape, repr_dim, scaling_factor=scaling_factors
    )
    representation.init_representation()
    repr_shape = [repr_dim] + [repr_size for s in input_shape]
    input_representation = torch.ones(repr_shape)
    slicers = []
    for i, indices in enumerate(
        product(*[range(s) for s in representation.image.shape[1:]])
    ):
        slicer = tuple(
            [
                slice(idx * s_f, idx * s_f + r_s * s_f)
                for idx, r_s, s_f in zip(
                    indices, input_representation.shape[1:], scaling_factors
                )
            ]
        )
        slicers.append(slicer)
        representation.update_que(input_representation * (i + 1))
        representation.update_representation(slicer)
    representation.build_representation()
    return representation, slicers


def test_representation_crop():
    num_dims = 2
    num_img_channels = 1
    img_size = 10
    repr_scaling = 2
    repr_dim = 1
    repr_size = 1

    img_shape = [num_img_channels] + [img_size] * num_dims

    input_shape = img_shape[1:]
    scaling_factors = [repr_scaling] * len(input_shape)

    representation = RepresentationHandler(
        input_shape, repr_dim, scaling_factor=scaling_factors
    )
    representation.init_representation()
    # ensure starting representation is empty
    assert representation.image.sum() == 0
    repr_shape = [repr_dim] + [repr_size for s in input_shape]
    input_representation = torch.ones(repr_shape)
    slicers = []
    for i, indices in enumerate(
        product(*[range(s) for s in representation.image.shape[1:]])
    ):
        slicer = tuple(
            [
                slice(idx * s_f, idx * s_f + r_s * s_f)
                for idx, r_s, s_f in zip(
                    indices, input_representation.shape[1:], scaling_factors
                )
            ]
        )
        slicers.append(slicer)
        representation.update_que(input_representation * (i + 1))
        representation.update_representation(slicer)

    before_build = representation.image.clone()
    assert torch.all(representation.n_predictions == 1)

    representation.build_representation()
    assert torch.all(before_build == representation.image)

    # now start testing cropping
    orig_shape = [r // 2 + 2 for r in input_shape]
    representation.set_orig_shape(orig_shape)
    representation.crop_repr_to_orig_shape()

    # ensure that each the cropped representation has values for all cropped pixels
    # As all internal representations are greater 0, the sum of cropped representations
    # should be greater 0 when the cropping is done correctly
    for i, indices in enumerate(product(*[range(s) for s in orig_shape])):
        slicer = tuple([slice(idx, idx + 1) for idx in indices])
        assert torch.sum(representation.map_to_representation(slicer)) > 0


def test_init_from_representation():
    repr = create_representation()[0]
    repr_from_rep = RepresentationHandler.init_from_representation(
        repr.image, repr.input_shape
    )

    assert all(
        s_r == s_i
        for s_r, s_i in zip(repr.scaling_factor, repr_from_rep.scaling_factor)
    )
