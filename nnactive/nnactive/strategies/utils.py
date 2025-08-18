from __future__ import annotations

import json
from pathlib import Path
from typing import Generator, Iterable, Union

import numpy as np
import torch
from loguru import logger
from skimage.morphology import ball
from torch.backends import cudnn

from nnactive.data import Patch
from nnactive.masking import does_overlap, percentage_overlap_array
from nnactive.utils.io import load_label_map
from nnactive.utils.padding import obtain_center_padding_slicers
from nnactive.utils.torchutils import maybe_gpu_binary_erosion


class RepresentationHandler:
    def __init__(
        self,
        input_shape: Iterable[int],
        repr_dim: int | None = None,
        scaling_factor: Iterable[int] | None = None,
        orig_shape: Iterable[int] | None = None,
        device=torch.device("cpu"),
    ):
        """Internal Representation Handler for Patch Selection.

        TODO: Add saving and loading of representations for large images on demand.

        Args:
            input_shape (Iterable[int]): Shape of image that representation is built on.
            repr_dim (int | None, optional): dimensionality of representation. Defaults to None.
            scaling_factor (Iterable[int] | None, optional): Factor how many pixels belong to one representation pixel. Defaults to None.
            orig_shape (Iterable[int] | None, optional): Shape of image before padding is applied. Defaults to None.
            device (Device, optional): Device on which internal representation is saved. Defaults to torch.device("cpu").
        """
        self.que: list[torch.Tensor] = []
        self.image: torch.Tensor = None
        self.n_predictions: torch.Tensor = None
        self.dtype = torch.float16
        self.input_shape = input_shape
        self.repr_dim = repr_dim
        self.scaling_factor = scaling_factor
        self.device = device
        self.built = False
        self.init = False
        self.orig_shape = orig_shape

    def init_representation(self):
        """Initialize the internal representation and counter."""
        representation_size = [
            i // s for i, s in zip(self.input_shape, self.scaling_factor)
        ]
        self.image = torch.zeros(
            [self.repr_dim] + representation_size, dtype=self.dtype, device=self.device
        )
        self.n_predictions = torch.zeros(
            representation_size, dtype=torch.uint8, device=self.device
        )
        self.init = True

    def set_orig_shape(self, orig_shape: Iterable[int]):
        self.orig_shape = orig_shape

    def crop_repr_to_orig_shape(self):
        """Crop internal representation to original shape."""
        if self.orig_shape is None:
            raise ValueError("Original Shape is not defined.")
        else:
            slicer = obtain_center_padding_slicers(
                old_shape=self.orig_shape, cur_shape=self.input_shape
            )
            slicer = self.image_slice_to_representation_slice(slicer)
            self.image = self.image[slicer]
            if self.n_predictions is not None:
                self.n_predictions = self.n_predictions[slicer]

    def update_que(self, tensor: torch.Tensor):
        """Add tensor to internal que.

        Args:
            tensor (torch.Tensor): expected shape is (repr_dim, *input_shape) or (N, repr_dim, *input_shape)
        """
        if len(tensor.shape) == len(self.input_shape) + 1:
            self.que.append(tensor)
        elif len(tensor.shape) == len(self.input_shape) + 2:
            for s_t in tensor:
                self.update_que(s_t)
        else:
            raise NotImplementedError(
                f"The size of input {tensor.shape} is not supported for this representation with input_shape {self.input_shape}"
            )

    def update_representation(self, slices: tuple[slice, ...], que_index: int = 0):
        """Update the internal representation and counters in image slice space in internal representation with que representation at index.

        Args:
            slices (tuple[slice, ...]): slices with image space correspondence for que_representation
            que_index (int, optional): index of internal que to get que_representation. Defaults to 0.
        """
        if not self.init:
            self.init_representation()
        repr_slice = self.image_slice_to_representation_slice(slices)
        data = self.que.pop(que_index)
        self.n_predictions[repr_slice[1:]] += 1
        self.image[repr_slice] += data

    def build_representation(self):
        """Build the internal representation from the que."""
        assert self.init
        assert len(self.que) == 0
        self.image /= self.n_predictions.to(self.dtype)[None]
        self.n_predictions = None
        self.built = True

    def map_to_representation(self, slices: tuple[slice, ...]) -> torch.Tensor:
        """Map image slice to representation slice."""
        repr_slice = self.image_slice_to_representation_slice(slices)
        return self.image[repr_slice]

    def image_slice_to_representation_slice(
        self, slices: tuple[slice, ...]
    ) -> tuple[slice, ...]:
        """Map image slice to representation slice. If slice is smaller than scaling factor, it is set to 1."""
        out_slices = [slice(None)]
        if len(slices) == len(self.input_shape):
            start_ind = 0
        elif len(slices) == len(self.input_shape) + 1:
            start_ind = 1

        for i in range(start_ind, len(slices)):
            start = slices[i].start // self.scaling_factor[i - start_ind]
            end = slices[i].stop // self.scaling_factor[i - start_ind]
            if end - start == 0:
                end += 1
            out_slices.append(
                slice(
                    start,
                    end,
                )
            )
        return out_slices

    @classmethod
    def init_from_representation(
        cls, image: torch.Tensor, input_shape: Iterable[int]
    ) -> RepresentationHandler:
        repr_dim = image.shape[0]
        scaling_factors = [i_s // r_s for i_s, r_s in zip(input_shape, image.shape[1:])]

        out = cls(input_shape, repr_dim, scaling_factors)
        out.init = True
        out.built = True
        out.image = image
        out.device = image.device
        return out


def power_noising(
    scores: np.ndarray | torch.Tensor,
    beta: float,
    rng: np.random.Generator = np.random.default_rng(),
) -> np.ndarray | torch.Tensor:
    """Perform power noising of samples with gumbel distribution.

    Args:
        scores (np.ndarray | torch.Tensor): scores #N
        beta (float): beta value for gumbel distribution

    Returns:
        np.ndarray | torch.Tensor: scores + epsilon #N
    """
    gumbel_samples = rng.gumbel(0, beta**-1, size=scores.shape)

    if isinstance(scores, np.ndarray):
        power_s = np.log(scores) + gumbel_samples
    else:
        power_s = scores.log() + torch.from_numpy(gumbel_samples).to(scores.device)
    return power_s


def query_starting_budget_all_classes(
    raw_labels_path: Path,
    file_ending: str,
    annotated_patches: list[Patch],
    patch_size: tuple[int],
    rng=np.random.default_rng(),
    trials_per_img: int = 600,
    verbose: bool = False,
    num_per_label: int = 2,
    additional_label_path: Path | None = None,
    additional_overlap: float = 0.8,
) -> list[Patch]:
    with (raw_labels_path.parent / "dataset.json").open() as file:
        dataset_json = json.load(file)
    label_dict_dataset_json = dataset_json["labels"]
    # Take first value if multi-region training e.g. BraTS
    for label in label_dict_dataset_json:
        if isinstance(label_dict_dataset_json[label], (list, tuple)):
            label_dict_dataset_json[label] = label_dict_dataset_json[label][0]
    label_dict_files = {k: [] for k in label_dict_dataset_json.keys() if k != "ignore"}
    filenames = [
        fp.name.removesuffix(file_ending)
        for fp in raw_labels_path.iterdir()
        if fp.name.endswith(file_ending)
    ]
    # create list of files for each label
    for fn in filenames:
        img = load_label_map(fn, raw_labels_path, file_ending)
        img_labels = set(np.unique(img))
        for label, files in label_dict_files.items():
            if label_dict_dataset_json[label] in img_labels:
                files.append(fn)
    # logger.debug(label_dict_files)
    labeled_patches = annotated_patches
    patches = []
    # Select a set amount of num_per_label samples for each class
    while len(label_dict_files) > 0:
        # select always label with least amount of examples!
        label_use = [(key, len(v)) for key, v in label_dict_files.items()]
        label_use.sort(key=lambda x: x[1])

        label = label_use[0][0]
        logger.debug(f"Label Use for Label {label}: {label_use}")
        if len(label_dict_files[label]) < num_per_label:
            raise RuntimeError(
                f'Label "{label}" does have less than {num_per_label} files. '
                f"This is not enough to ensure all classes are represented in all training folds."
            )
        else:
            samples: list[str] = label_dict_files.pop(label)
            rng.shuffle(samples)
        label_per_class_counter = 0

        # iteration over all samples that contain class and try to set patch inside
        for sample in samples:
            # if enough patches drawn per class, exit loop
            if label_per_class_counter == num_per_label:
                break

            labeled = False
            num_tries = 0
            additional_label = None
            if additional_label_path is not None:
                additional_label = load_label_map(
                    sample,
                    additional_label_path,
                    file_ending,
                )
                additional_label: np.ndarray = additional_label != 255
            if verbose:
                logger.debug(f"Loading Image: {sample} for label {label}")
            label_map = load_label_map(sample, raw_labels_path, file_ending)
            img_size = label_map.shape
            current_patch_list = labeled_patches + patches
            selected_patches_image = [
                patch
                for patch in current_patch_list
                if patch.file == sample + file_ending
            ]
            locs = np.argwhere(label_map == label_dict_dataset_json[label])

            # try drawing patches for sample until one fits or max_tries
            while not labeled:
                (
                    iter_patch_loc,
                    iter_patch_size,
                ) = generate_random_patch_from_locs(locs, img_size, patch_size, rng)
                patch = Patch(
                    file=sample + file_ending,
                    coords=iter_patch_loc,
                    size=iter_patch_size,
                )
                # check if patch is valid
                if not does_overlap(patch, selected_patches_image):
                    if additional_label is not None:
                        additional_overlap_patch = percentage_overlap_array(
                            patch, additional_label
                        )
                        if additional_overlap_patch <= additional_overlap:
                            for label_rm in label_dict_files:
                                if sample in label_dict_files[label_rm]:
                                    label_dict_files[label_rm].remove(sample)
                            # only temporary for patchsize = 1
                            labeled_val = label_map[patch.coords]
                            logger.debug(
                                f"Annotated Patch {patch} for Label {label_dict_dataset_json[label]} and it consists of label {labeled_val}"
                            )
                            patches.append(patch)
                            # print(f"Creating Patch with iteration: {num_tries}")
                            labeled = True
                            label_per_class_counter += 1

                    else:
                        for label_rm in label_dict_files:
                            if sample in label_dict_files[label_rm]:
                                label_dict_files[label_rm].remove(sample)
                        # only temporary for patchsize = 1
                        labeled_val = label_map[patch.coords]
                        logger.debug(
                            f"Annotated Patch {patch} for Label {label_dict_dataset_json[label]} and it consists of label {labeled_val}"
                        )
                        patches.append(patch)
                        labeled = True
                        label_per_class_counter += 1

                # if no new patch could fit inside of img do not consider again
                if num_tries == trials_per_img:
                    logger.info(f"Could not place patch in image {sample}")
                    logger.info(f"PatchCount: {len(patches)}")
                    logger.info(f"Num Tries: {num_tries}")
                    break
                num_tries += 1
        if label_per_class_counter < num_per_label:
            raise RuntimeError(
                f'Could not place {num_per_label} patches for class "{label}"'
            )
    # if verbose:
    #     logger.debug(patches)
    return patches


def obtain_random_patch_for_img(
    img_size: list, patch_size: list, rng: Generator
) -> tuple[list[int], list[int]]:
    """Generates a complete random patch fitting inside the image

    Args:
        img_size (list): size of image
        patch_size (list): size of patch
        rng (Generator): generator for seeding

    Returns:
        tuple[list[int], list[int]]: (location, patch_size)
    """
    patch_loc_ranges = []
    patch_real_size = []
    for dim_img, dim_patch in zip(img_size, patch_size):
        if dim_patch >= dim_img:
            patch_loc_ranges.append([0])
            patch_real_size.append(dim_img)
        else:
            patch_loc_ranges.append([i for i in range(dim_img - dim_patch)])
            patch_real_size.append(dim_patch)

    patch_loc = []
    for loc_range in patch_loc_ranges:
        patch_loc.append(rng.choice(loc_range))

    return (patch_loc, patch_real_size)


def get_infinte_iter(finite_list: Iterable):
    while True:
        for elt in finite_list:
            yield elt


def get_locs_from_segmentation(
    orig_seg: np.ndarray,
    area="seg",
    state: np.random.RandomState = np.random.default_rng(),
    background_cls: Union[int, None] = 0,
    verbose: bool = False,
):
    unique_cls = np.unique(orig_seg)
    delete_cls = [cl for cl in unique_cls if cl < 0]
    if len(delete_cls) > 0:
        logger.warning("Ignoring Cls < 0 for Patch Selection: {delete_cls}")

    if verbose:
        logger.debug(f"Ignoring Background Class for Selection: {background_cls}")
    unique_cls = np.array([cl for cl in unique_cls if cl not in delete_cls])
    counter = 0
    selected_cls = background_cls
    while selected_cls == background_cls:
        if counter == 200:
            raise RuntimeError("There is no non-background class in this image!")
        selected_cls = state.choice(unique_cls, 1).item()
        counter += 1
    if verbose:
        logger.debug(f"Select Area for Class {selected_cls}")
    use_seg = (orig_seg == selected_cls).astype(np.int8)

    cudnn.deterministic = False
    cudnn.benchmark = False
    if area == "border":
        use_seg_border = use_seg - maybe_gpu_binary_erosion(use_seg > 0, ball(1))
        return np.argwhere(use_seg_border > 0)
    elif area == "seg":
        return np.argwhere(use_seg > 0)
    else:
        raise NotImplementedError


def generate_random_patch_from_locs(
    locs: tuple | list | np.ndarray,
    img_size: list,
    patch_size: list,
    rng: Generator = np.random.default_rng(),
) -> tuple[list[int], list[int]]:
    """Locs describe the center of the area that should be cropped. Can be np.argwhere(img>0)"""
    patch_real_size = []
    # Get correct size of patch
    for dim_img, dim_patch in zip(img_size, patch_size):
        if dim_patch >= dim_img:
            patch_real_size.append(dim_img)
        else:
            patch_real_size.append(dim_patch)

    loc = locs[rng.choice(len(locs))]
    patch_loc = []

    for dim_loc, dim_img, dim_patch in zip(loc, img_size, patch_real_size):
        if dim_patch >= dim_img:
            patch_loc.append(0)
        else:
            # patch fits right into the image
            if dim_loc + dim_patch // 2 <= dim_img and dim_loc - dim_patch // 2 >= 0:
                patch_loc.append(dim_loc - dim_patch // 2)
            # patch overshoots, set to maximal possible value
            elif dim_loc + dim_patch // 2 > dim_img:
                patch_loc.append(dim_img - dim_patch)
            # patch undershoots, set to minimal possible value
            elif dim_loc - dim_patch // 2 < 0:
                patch_loc.append(0)
            else:
                raise NotImplementedError

    return (patch_loc, patch_real_size)
