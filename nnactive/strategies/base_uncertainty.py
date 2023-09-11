from __future__ import annotations

import os
import traceback
from abc import abstractmethod
from typing import Iterable, Tuple, Union

import numpy as np
import torch
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.export_prediction import (
    convert_predicted_logits_to_segmentation_with_correct_shape,
)
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.utilities.file_path_utilities import get_output_folder
from nnunetv2.utilities.helpers import empty_cache
from torch._dynamo import OptimizedModule

from nnactive.config.struct import ActiveConfig
from nnactive.data import Patch
from nnactive.masking import does_overlap, mark_selected
from nnactive.nnunet.utils import get_raw_path
from nnactive.strategies.base import AbstractQueryMethod


class AbstractUncertainQueryMethod(AbstractQueryMethod):
    def __init__(
        self,
        dataset_id: int,
        query_size: int,
        patch_size: list[int],
        file_ending: str = ".nii.gz",
        **kwargs,
    ):
        super().__init__(dataset_id, query_size, patch_size, file_ending)
        # self.top_patches = []  # carries the top patches of all images
        self.config = ActiveConfig.get_from_id(dataset_id)

    def query(self, verbose=False) -> list[Patch]:
        # Initialize Predictor
        predictor = nnActivePredictor()

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
            num_processes_preprocessing=3,
        )
        predictor.compute_from_data_iterator(data_iterator, self)
        return self.compose_query_of_patches()

    def query_from_probs(
        self, out_probs: torch.Tensor, image_shape: Iterable[int], label_file: str
    ):
        """Computes potential queries for a single input image and adds best queries to the internal list of queries.

        Args:
            out_probs (torch.Tensor): probability map for image
            image_shape (Iterable[int]): shape of image
            label_file (str): name of label file
        """

        uncertainty = self.get_uncertainty(out_probs)

        if torch.any(torch.isnan(uncertainty)):
            # unc_num_nan = torch.sum(torch.isnan(uncertainty))
            # unc_where_nan = torch.argwhere(torch.isnan(uncertainty))
            print(f" NAN values in uncertainties for image {label_file}")
        agg_uncertainty, kernel_size = self.aggregate(uncertainty)
        agg_uncertainty = agg_uncertainty.cpu().numpy()

        selected_array = self.initialize_selected_array(
            image_shape, label_file, self.annotated_patches
        )

        selected_patches = self.select_top_n_non_overlapping_patches(
            kernel_size, agg_uncertainty, selected_array, label_file, self.query_size
        )

        self.top_patches += selected_patches

    @abstractmethod
    def get_uncertainty(self, out_probs: torch.Tensor) -> torch.Tensor:
        """Compute uncertainty values from out_probs

        Args:
            out_probs (torch.Tensor): probability maps for image [M x C x XYZ]

        Returns:
            torch.Tensor: outputs
        """

    def select_top_n_non_overlapping_patches(
        self,
        patch_size: list[int],
        aggregated: np.ndarray,
        selected_array: np.ndarray,
        label_file: str,
        n: int,
    ) -> list[dict]:
        selected_patches = []
        # sort only once since this can take a significant amount of time

        flat_aggregated_uncertainties = aggregated.flatten()

        sorted_uncertainty_indices = np.flip(np.argsort(flat_aggregated_uncertainties))
        sorted_uncertainty_scores = np.take_along_axis(
            flat_aggregated_uncertainties, sorted_uncertainty_indices, axis=0
        )
        # Iterate over the sorted uncertainty scores and their indices to get the most uncertain
        for uncertainty_score, uncertainty_index in zip(
            sorted_uncertainty_scores, sorted_uncertainty_indices
        ):
            # get coordinates in image space from aggregated indices
            coords = self.transpose_aggregate(uncertainty_index, aggregated.shape)
            # Check if coordinated overlap with already queried region
            if not does_overlap(coords, patch_size, selected_array):
                # If it is a non-overlapping region, append this patch to be queried
                selected_patches.append(
                    {
                        "file": label_file + ".nii.gz",
                        "coords": coords,
                        "size": patch_size,
                        "score": uncertainty_score,
                    }
                )
                # selected += 1
                # Mark region as queried
                selected_array = mark_selected(selected_array, coords, patch_size)
                # Stop if we reach the maximum number of patches to be queried
            if n is not None and len(selected_patches) >= n:
                break
        return selected_patches

    def aggregate(self, uncertainty: torch.Tensor) -> Tuple[torch.Tensor, list[int]]:
        kernel_size = [
            min(self.patch_size[i], uncertainty.shape[i])
            for i in range(len(self.patch_size))
        ]
        kernel = torch.ones(size=[1, 1] + kernel_size, device=uncertainty.device)

        if len(uncertainty.shape) == 2:
            aggregated = torch.nn.functional.conv2d(
                uncertainty.unsqueeze(0).unsqueeze(0),
                weight=kernel,
            )
        elif len(uncertainty.shape) == 3:
            aggregated = torch.nn.functional.conv3d(
                uncertainty.unsqueeze(0).unsqueeze(0),
                weight=kernel,
            )
        else:
            raise NotImplementedError()
        return aggregated.squeeze(), kernel_size

    def transpose_aggregate(
        self, aggregated_index: np.ndarray, aggregated_shape: Iterable[int]
    ) -> np.ndarray:
        """Compute indices from the aggregated images into starting coordinates for patches in image space

        Args:
            aggregated_index (np.ndarray): index in aggregated space
            aggregated_shape (Iterable[int]): shape of aggregated image

        Returns:
            np.ndarray: corresponding index in image space
        """
        # Get the index as coordinates
        # this works after convolution index_img_start = index_conv_start, index_img_end = index_conv_start+ size_dim
        return np.unravel_index(aggregated_index, aggregated_shape)

    def compose_query_of_patches(self):
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


class nnActivePredictor(nnUNetPredictor):
    def predict_fold_logits_from_preprocessed_data(
        self, data: torch.TensorType
    ) -> torch.TensorType:
        """Computes the logits/probs for all folds.

        Args:
            data (torch.TensorType): Preprocessed Data

        Returns:
            torch.TensorType: num_folds x classes x ...
        """
        original_perform_everything_on_gpu = self.perform_everything_on_gpu
        with torch.no_grad():
            predictions = None
            if self.perform_everything_on_gpu:
                try:
                    for params in self.list_of_parameters:
                        # messing with state dict names...
                        if not isinstance(self.network, OptimizedModule):
                            self.network.load_state_dict(params)
                        else:
                            self.network._orig_mod.load_state_dict(params)

                        if predictions is None:
                            predictions = [
                                self.predict_sliding_window_return_logits(data)
                            ]
                        else:
                            predictions.append(
                                self.predict_sliding_window_return_logits(data)
                            )

                except RuntimeError:
                    print(
                        "Prediction with perform_everything_on_gpu=True failed due to insufficient GPU memory. "
                        "Falling back to perform_everything_on_gpu=False. Not a big deal, just slower..."
                    )
                    print("Error:")
                    traceback.print_exc()
                    predictions = None
                    self.perform_everything_on_gpu = False

            if predictions is None:
                for params in self.list_of_parameters:
                    # messing with state dict names...
                    if not isinstance(self.network, OptimizedModule):
                        self.network.load_state_dict(params)
                    else:
                        self.network._orig_mod.load_state_dict(params)

                    if predictions is None:
                        predictions = [self.predict_sliding_window_return_logits(data)]
                    else:
                        predictions.append(
                            self.predict_sliding_window_return_logits(data)
                        )
            predictions = torch.stack(predictions, 0)

            self.perform_everything_on_gpu = original_perform_everything_on_gpu
        return predictions

    def compute_from_data_iterator(
        self,
        data_iterator,
        query_method: AbstractUncertainQueryMethod,
        save_probabilities: bool = False,
        num_processes_segmentation_export: int = default_num_processes,
    ):
        """
        each element returned by data_iterator must be a dict with 'data', 'ofile' and 'data_properites' keys!
        If 'ofile' is None, the result will be returned instead of written to a file
        """
        # set up multiprocessing for spawning

        for preprocessed in data_iterator:
            data = preprocessed["data"]
            if isinstance(data, str):
                delfile = data
                data = torch.from_numpy(np.load(data))
                os.remove(delfile)

            ofile = preprocessed["ofile"]
            filename = os.path.basename(ofile)
            if ofile is not None:
                print(f"\nPredicting {os.path.basename(ofile)}:")
            else:
                print(f"\nPredicting image of shape {data.shape}:")

            print(f"perform_everything_on_gpu: {self.perform_everything_on_gpu}")

            properties = preprocessed["data_properites"]

            # check multiprocessing for saving files

            # we do not need to convert logits with softmax as convert_xxx already does so for us
            logits = self.predict_fold_logits_from_preprocessed_data(data).cpu()

            logits_nf = torch.isfinite(logits) == 0
            if torch.any(logits_nf):
                print(f"Replacing non finite values in logits for image {filename}")
                logits[logits == torch.inf] = 100
                logits[logits == -torch.inf] = -100
                logits_nf = torch.isfinite(logits) == 0
                if torch.any(logits_nf):
                    raise RuntimeError(f"NAN values in logits for image {filename}")
            del logits_nf

            out_probs = []

            for prediction in logits:
                (
                    _,
                    out_prob,
                ) = convert_predicted_logits_to_segmentation_with_correct_shape(
                    prediction,
                    self.plans_manager,
                    self.configuration_manager,
                    self.label_manager,
                    properties,
                    return_probabilities=True,
                )
                out_probs.append(out_prob)
            out_probs = torch.from_numpy(np.stack(out_probs, axis=0)).to(
                device=self.device
            )  # Shape: M x C x ...

            if torch.any(torch.isnan(out_probs)):
                # probs_num_nan = torch.sum(torch.isnan(out_probs))
                # probs_where_nan = torch.argwhere(torch.isnan(out_probs))
                print(f" NAN values in probablities for image {filename}")

            query_method.query_from_probs(
                out_probs, properties["shape_before_cropping"], filename
            )
            # TODO: possibly add some multiprocessing as in nnUNet_predictor

        if isinstance(data_iterator, MultiThreadedAugmenter):
            data_iterator._finish()

        # clear lru cache
        compute_gaussian.cache_clear()
        # clear device cache
        empty_cache(self.device)

    def get_data_iterator_from_folders(
        self,
        list_of_lists_or_source_folder: Union[str, list[list[str]]],
        output_folder_or_list_of_truncated_output_files: Union[str, None, list[str]],
        num_processes_preprocessing: int = 3,
        num_parts: int = 1,
        part_id: int = 0,
        save_probabilities: bool = False,
    ):
        # sort out input and output filenames
        (
            list_of_lists_or_source_folder,
            output_filename_truncated,
            seg_from_prev_stage_files,
        ) = self._manage_input_and_output_lists(
            list_of_lists_or_source_folder,
            output_folder_or_list_of_truncated_output_files,
            None,
            True,
            part_id,
            num_parts,
            save_probabilities,
        )
        if len(list_of_lists_or_source_folder) == 0:
            return

        data_iterator = self._internal_get_data_iterator_from_lists_of_filenames(
            list_of_lists_or_source_folder,
            seg_from_prev_stage_files,
            output_filename_truncated,
            num_processes_preprocessing,
        )
        return data_iterator
