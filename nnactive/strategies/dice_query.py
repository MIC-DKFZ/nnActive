from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import torch
from loguru import logger
from nnunetv2.training.loss.dice import SoftDiceLoss
from nnunetv2.utilities.file_path_utilities import get_output_folder

from nnactive.config import ActiveConfig
from nnactive.data import Patch
from nnactive.logger import monitor
from nnactive.nnunet.utils import get_raw_path
from nnactive.results.utils import get_results_folder as get_nnactive_results_folder
from nnactive.strategies.base import AbstractQueryMethod
from nnactive.strategies.base_uncertainty import nnActivePredictor
from nnactive.utils.io import load_label_map

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class PatchDice:
    def __init__(self, patch_size: list[int], stride: Union[int, list[int]] = 1):
        self.patch_size = patch_size
        self.stride = stride
        if isinstance(stride, int):
            self.stride = [stride] * len(self.patch_size)

    def get_dice_patch(self, mean_prob, prob_fold, kernel_size, dice_dict):
        dice_loss = SoftDiceLoss()
        mean_prob = torch.tensor(mean_prob).to(DEVICE)
        prob_fold = torch.tensor(prob_fold).to(DEVICE)
        for coord in dice_dict.keys():
            coords_end = np.array(coord) + kernel_size
            coord_slices = tuple(
                (slice(cs, ce, None) for cs, ce in zip(coord, coords_end))
            )
            coord_slices = (slice(None), *coord_slices)
            mean_patch = mean_prob[coord_slices]
            prob_patch = prob_fold[coord_slices]
            dice_dict[coord].append(dice_loss(mean_patch, prob_patch).item())
        return dice_dict

    def get_coords_patches(self, image_shape):
        kernel_size = [
            min(self.patch_size[i], image_shape[i]) for i in range(len(self.patch_size))
        ]
        image_shape = np.array(image_shape)
        kernel_size = np.array(kernel_size)
        stride = np.array(self.stride)
        max_pos = ((image_shape - kernel_size) // stride) * stride
        n_steps = (max_pos // stride) + 1

        coords_x = np.linspace(0, max_pos[0], n_steps[0]).astype(int)
        coords_y = np.linspace(0, max_pos[1], n_steps[1]).astype(int)
        coords_z = np.linspace(0, max_pos[2], n_steps[2]).astype(int)
        coords_x, coords_y, coords_z = np.meshgrid(coords_x, coords_y, coords_z)
        # Combine X, Y, and Z coordinates into tuples of 3D coordinates
        coordinates = np.stack((coords_x, coords_y, coords_z), axis=-1)
        # Flatten the coordinates to get a list of tuples
        coordinate_tuples = coordinates.reshape(-1, 3)
        return coordinate_tuples

    def forward(
        self,
        images: List[np.array] = None,
        dataset_id: int = None,
        num_folds: int = None,
    ):
        if images is None and dataset_id is None and num_folds is None:
            raise ValueError("Either images or dataset_id and num_folds must be given")

        if images is None:
            prob_path = get_nnactive_results_folder(dataset_id) / "temp"
            fold_paths = [prob_path / f"probs_fold{i}.npy" for i in range(0, num_folds)]
            mean_prob = np.mean([np.load(str(f)) for f in fold_paths], axis=0)
        else:
            mean_prob = np.mean(images, axis=0)

        # logger.info("Started forward")
        # due to softmax, prob shape is offset by one (class dimension)
        kernel_size = [
            min(self.patch_size[i], mean_prob.shape[i + 1])
            for i in range(len(self.patch_size))
        ]
        # breakpoint()
        coordinate_tuples = self.get_coords_patches(mean_prob.shape[1:])
        dice_dict = {
            tuple(coord.tolist()): []
            for coord in coordinate_tuples
            if not np.any(coord + kernel_size > mean_prob.shape[1:])
        }
        num_images = len(images) if images is not None else num_folds
        for i in range(num_images):
            prob_fold = images[i] if images is not None else np.load(str(fold_paths[i]))
            dice_dict = self.get_dice_patch(
                mean_prob, prob_fold, kernel_size, dice_dict
            )
        dice_dict = {k: np.nanmean(v) for k, v in dice_dict.items()}

        sorted_dice_dict = {
            k: v
            for k, v in sorted(
                dice_dict.items(), key=lambda item: item[1], reverse=True
            )
        }
        return sorted_dice_dict, kernel_size


class DiceQuery(AbstractQueryMethod):
    def __init__(
        self,
        dataset_id: int,
        query_size: int,
        patch_size: list[int],
        agg_stride: Union[int, list[int]],
        n_patch_per_image: int,
        file_ending: str = ".nii.gz",
        num_processes_preprocessing: int = 3,
        use_gaussian: bool = False,
        use_mirroring: bool = False,
        tile_step_size: float = 0.75,
        additional_label_path: Path | None = None,
        additional_overlap: float = 0.1,
        patch_overlap: float = 0,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(
            dataset_id,
            query_size,
            patch_size,
            file_ending,
            additional_label_path,
            additional_overlap,
            verbose=verbose,
        )
        self.config = ActiveConfig.get_from_id(dataset_id)
        self.num_processes_preprocessing = num_processes_preprocessing
        self.use_mirroring = use_mirroring
        self.use_gaussian = use_gaussian
        self.tile_step_size = tile_step_size
        self.agg_stride = agg_stride
        self.n_patch_per_image = n_patch_per_image

    def query(
        self, verbose: bool = False, already_annotated_patches=None
    ) -> list[Patch]:
        # Initialize Predictor
        predictor = nnActivePredictor(
            tile_step_size=self.tile_step_size,
            use_mirroring=self.use_mirroring,
            use_gaussian=self.use_gaussian,
            verbose=self.verbose,
            allow_tqdm=not self.verbose,
        )

        # Initialize Model for Predictor
        nnunet_plans_identifier = "nnUNetPlans"
        nnunet_trainer_name = self.config.trainer
        nnunet_config = self.config.model_config
        model_folder = get_output_folder(
            self.dataset_id, nnunet_trainer_name, nnunet_plans_identifier, nnunet_config
        )
        use_folds = tuple(range(self.config.working_folds))
        predictor.initialize_from_trained_model_folder(
            model_folder, use_folds=use_folds
        )

        source_folder = str(get_raw_path(self.dataset_id) / "imagesTr")
        output_folder = "/".join(model_folder.split("/")[:-1])

        data_iterator = predictor.get_data_iterator_from_folders(
            list_of_lists_or_source_folder=source_folder,
            output_folder_or_list_of_truncated_output_files=output_folder,
            num_processes_preprocessing=self.num_processes_preprocessing,
        )
        predictor.predict_from_data_iterator(data_iterator, self)
        return self.compose_query_of_patches()

    def query_from_probs(
        self, num_folds: int, image_shape: Iterable[int], label_file: str
    ):
        dice = PatchDice(patch_size=self.patch_size, stride=self.agg_stride)
        with monitor.timer("query_from_dice"):
            with torch.no_grad():
                logger.info("Compute pairwise dice...")
                sorted_dice_scores, kernel_size = dice.forward(
                    dataset_id=self.dataset_id, num_folds=num_folds
                )
                logger.info("Initialize selected array...")

                annotated_patches = [
                    patch
                    for patch in self.annotated_patches
                    if patch.file == label_file + ".nii.gz"
                ]
            logger.info("Select patches...")
            selected_patches = self.select_top_n_non_overlapping_patches(
                patch_size=kernel_size,
                sorted_dice_scores=sorted_dice_scores,
                annotated_patches=annotated_patches,
                label_file=label_file,
                n=self.n_patch_per_image,
            )
            logger.info("Finished patch selection.")
            self.top_patches += selected_patches

    def select_top_n_non_overlapping_patches(
        self,
        patch_size: list[int],
        sorted_dice_scores: Dict[Tuple[int], float],
        annotated_patches: list[Patch],
        label_file: str,
        n: int,
    ):
        selected_patches = []
        logger.info("Start finding non-overlapping patches.")
        additional_label = None
        if self.additional_label_path is not None:
            if self.verbose:
                logger.debug("Create additional label map.")
            additional_label = load_label_map(
                label_file,
                self.additional_label_path,
                self.file_ending,
            )
            additional_label: np.ndarray = additional_label != 255
        for coords, dice_score in sorted_dice_scores.items():
            # breakpoint()
            patch = Patch(
                file=label_file + ".nii.gz",
                coords=coords,
                size=patch_size,
            )
            if self.check_overlap(
                patch, annotated_patches, additional_label, verbose=self.verbose
            ):
                # If it is a non-overlapping region, append this patch to be queried
                selected_patches.append(
                    {
                        "file": label_file + ".nii.gz",
                        "coords": coords,
                        "size": patch_size,
                        "score": dice_score,
                    }
                )
                # Mark region as queried
                annotated_patches.append(patch)
            if n is not None and len(selected_patches) >= n:
                break

        logger.info(f"Finished patch selection for image {label_file}")
        return selected_patches

    def compose_query_of_patches(self):
        with monitor.timer("compose_query_of_patches"):
            sorted_top_patches = sorted(
                self.top_patches, key=lambda d: d["score"], reverse=True
            )[: self.query_size]
            patches = [
                {
                    "file": patch["file"],
                    "coords": patch["coords"],
                    "size": patch["size"],
                }
                for patch in sorted_top_patches
            ]
            patches = [Patch(**patch) for patch in patches]
            return patches
