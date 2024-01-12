from __future__ import annotations

import os
import shutil
import traceback
from abc import abstractmethod
from typing import Dict, Iterable, Union

import numpy as np
import psutil
import torch
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.export_prediction import (
    convert_predicted_logits_to_segmentation_with_correct_shape,
)
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.utilities.dataset_name_id_conversion import convert_dataset_name_to_id
from nnunetv2.utilities.file_path_utilities import get_output_folder
from nnunetv2.utilities.helpers import empty_cache
from torch._dynamo import OptimizedModule
from tqdm import tqdm

from nnactive.aggregations.convolution import ConvolveAggScipy, ConvolveAggTorch
from nnactive.config.struct import ActiveConfig
from nnactive.data import Patch
from nnactive.masking import does_overlap, mark_selected
from nnactive.nnunet.utils import get_raw_path
from nnactive.results.utils import get_results_folder as get_nnactive_results_folder
from nnactive.strategies.base import AbstractQueryMethod

# TODO: replace this with a variable which is easier to access!
NPP = 1


class AbstractUncertainQueryMethod(AbstractQueryMethod):
    def __init__(
        self,
        dataset_id: int,
        query_size: int,
        patch_size: list[int],
        agg_stride: Union[int, list[int]],
        file_ending: str = ".nii.gz",
        **kwargs,
    ):
        super().__init__(dataset_id, query_size, patch_size, file_ending)
        self.config = ActiveConfig.get_from_id(dataset_id)
        if (
            agg_stride == 1
        ):  # TODO: for strides < 8 for large images scipy is still faster. This can be implemented better
            self.aggregation = ConvolveAggScipy(patch_size, stride=agg_stride)
        else:
            self.aggregation = ConvolveAggTorch(patch_size, stride=agg_stride)

        print(
            f"Aggregation is performed using: {self.aggregation.__class__.__name__} with stride {agg_stride}"
        )

    def query(self, verbose=False) -> list[Patch]:
        # Initialize Predictor
        predictor = nnActivePredictor(
            tile_step_size=0.75,
            use_mirroring=False,
            verbose=verbose,
            allow_tqdm=not verbose,
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
            num_processes_preprocessing=NPP,
        )
        predictor.predict_from_data_iterator(data_iterator, self)
        return self.compose_query_of_patches()

    def query_from_probs(
        self, num_folds: int, image_shape: Iterable[int], label_file: str
    ):
        """Computes potential queries for a single input image and adds best queries to the internal list of queries.

        Args:
            out_probs (torch.Tensor): probability map for image
            image_shape (Iterable[int]): shape of image
            label_file (str): name of label file
        """
        with torch.no_grad():
            print("Compute uncertaintes...")
            uncertainty = self.get_uncertainty(num_folds)

            if torch.any(torch.isnan(uncertainty)):
                # unc_num_nan = torch.sum(torch.isnan(uncertainty))
                # unc_where_nan = torch.argwhere(torch.isnan(uncertainty))
                raise ValueError(f" NAN values in uncertainties for image {label_file}")
            print("Aggregate uncertainties...")
            agg_uncertainty, kernel_size = self.aggregation.forward(uncertainty)

        print("Initialize selected array...")
        selected_array = self.initialize_selected_array(
            image_shape, label_file, self.annotated_patches
        )

        print("Select patches...")
        selected_patches = self.select_top_n_non_overlapping_patches(
            kernel_size, agg_uncertainty, selected_array, label_file, self.query_size
        )
        print("Finished patch selection.")
        self.top_patches += selected_patches

    @abstractmethod
    def get_uncertainty(self, num_folds: int) -> torch.Tensor:
        """Compute uncertainty values from out_probs

        Args:
            out_probs (torch.Tensor): probability maps for image [M x C x XYZ]

        Returns:
            torch.Tensor: outputs [M x C xXYZ]
        """

    def select_top_n_non_overlapping_patches(
        self,
        patch_size: list[int],
        aggregated: np.ndarray,
        selected_array: np.ndarray,
        label_file: str,
        n: int,
        verbose: bool = False,
    ) -> list[dict]:
        selected_patches = []
        # sort only once since this can take a significant amount of time
        print("Sort potential queries")
        flat_aggregated_uncertainties = aggregated.flatten()

        sorted_uncertainty_indices = np.flip(np.argsort(flat_aggregated_uncertainties))
        sorted_uncertainty_scores: list[float] = np.take_along_axis(
            flat_aggregated_uncertainties, sorted_uncertainty_indices, axis=0
        ).tolist()
        print("Start finding non-overlapping patches.")
        # Iterate over the sorted uncertainty scores and their indices to get the most uncertain

        iterator = zip(sorted_uncertainty_scores, sorted_uncertainty_indices)
        pbar0 = tqdm(total=n, position=0, desc="Patch Selection", disable=verbose)
        pbar1 = tqdm(
            total=len(sorted_uncertainty_scores),
            position=1,
            desc="Possible Patch Search",
            disable=verbose,
        )

        for i, (uncertainty_score, uncertainty_index) in enumerate(iterator):
            pbar1.update()
            # get coordinates in image space from aggregated indices
            coords = self.aggregation.backward_index(
                uncertainty_index, aggregated.shape
            )
            # coords = np.unravel_index(uncertainty_index, aggregated.shape)
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
                pbar0.update()
            if n is not None and len(selected_patches) >= n:
                break
        pbar1.close()
        pbar0.close()
        print("Selected patches")
        return selected_patches

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
    def postprocess_logits(
        self, logits: np.ndarray | torch.Tensor, properties: Dict
    ) -> np.ndarray:
        """Postprocess logits to return probs in the end
        Args:
            logits: logits to postprocess
            properties: image properties

        Returns:
            np.ndarray: output probabilities

        """
        process = psutil.Process()
        print(f"RAM used:~{process.memory_info().rss * 1e-9}GB")
        logits_nf = torch.isfinite(logits) == 0
        if torch.any(logits_nf):
            raise RuntimeError(f"NAN values in logits")
        del logits_nf

        (
            _,
            out_prob,
        ) = convert_predicted_logits_to_segmentation_with_correct_shape(
            logits,
            self.plans_manager,
            self.configuration_manager,
            self.label_manager,
            properties,
            return_probabilities=True,
        )

        # fastest way to check if nan in np array
        # according to https://stackoverflow.com/questions/6736590/fast-check-for-nan-in-numpy
        if np.isnan(np.sum(out_prob)):
            raise ValueError(f"NAN values in probablities in image!")

        return out_prob

    def save_out_probs_temp(self, out_probs: np.ndarray, fold: int):
        """Save the predicted probabilities as temporary files on disk to use later

        Args:
            out_probs (np.ndarray): Predicted probabilities
            fold (int): current predicted fold
        """
        dataset_id = convert_dataset_name_to_id(self.plans_manager.dataset_name)
        temp_path = get_nnactive_results_folder(dataset_id) / "temp"
        os.makedirs(temp_path, exist_ok=True)
        np.save(str(temp_path / f"probs_fold{fold}"), out_probs)

    def delete_temp_path(self):
        """Delete temp files to not mess up in subsequent query steps"""
        dataset_id = convert_dataset_name_to_id(self.plans_manager.dataset_name)
        temp_path = get_nnactive_results_folder(dataset_id) / "temp"
        shutil.rmtree(temp_path)

    def predict_fold_logits_from_preprocessed_data(
        self, data: torch.TensorType, properties
    ):
        """Computes the logits/probs for all folds.

        Args:
            data (torch.TensorType): Preprocessed Data
        """
        original_perform_everything_on_device = self.perform_everything_on_device
        with torch.no_grad():
            if self.perform_everything_on_device:
                try:
                    for fold, params in enumerate(self.list_of_parameters):
                        # messing with state dict names...
                        if not isinstance(self.network, OptimizedModule):
                            self.network.load_state_dict(params)
                        else:
                            self.network._orig_mod.load_state_dict(params)

                        process = psutil.Process()
                        print(f"RAM used:~{process.memory_info().rss * 1e-9}GB")
                        logits = self.predict_sliding_window_return_logits(data)
                        out_probs = self.postprocess_logits(logits, properties)
                        self.save_out_probs_temp(out_probs, fold)

                except RuntimeError:
                    print(
                        "Prediction with perform_everything_on_gpu=True failed due to insufficient GPU memory. "
                        "Falling back to perform_everything_on_gpu=False. Not a big deal, just slower..."
                    )
                    print("Error:")
                    traceback.print_exc()
                    self.perform_everything_on_device = False

            if not self.perform_everything_on_device:
                # TODO: probably do not predict everything from scratch again but only from fold where gpu prediciton is canceled
                for fold, params in enumerate(self.list_of_parameters):
                    # messing with state dict names...
                    if not isinstance(self.network, OptimizedModule):
                        self.network.load_state_dict(params)
                    else:
                        self.network._orig_mod.load_state_dict(params)
                    logits = self.predict_sliding_window_return_logits(data)
                    out_probs = self.postprocess_logits(logits, properties)
                    self.save_out_probs_temp(out_probs, fold)

            self.perform_everything_on_device = original_perform_everything_on_device

    def predict_from_data_iterator(
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

            print(f"perform_everything_on_gpu: {self.perform_everything_on_device}")

            properties = preprocessed["data_properties"]
            self.predict_fold_logits_from_preprocessed_data(data, properties)

            print("Start Query")
            query_method.query_from_probs(
                len(self.list_of_parameters),
                properties["shape_before_cropping"],
                filename,
            )

            self.delete_temp_path()
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


def select_top_n_non_overlapping_patches(
    image_name: str,
    n: int,
    uncertainty_scores: np.ndarray,
    patch_size: np.ndarray,
    selected_array: np.ndarray,
) -> list[dict]:
    """
    Get the most n uncertain non-overlapping patches for one image based on the aggregated uncertainty map

    Args:
        image_name (str): the name of the aggregated uncertainty map (npz file)
        n (int): number of non-overlapping patches that should be queried at most
        uncertainty_scores (np.ndarray): the aggregated uncertainty map
        patch_size (np.ndarray): patch size that was used to aggregate the uncertainties
        selected_array (np.ndarray): array with already labeled patches

    Returns:
        list[dict]: the most n uncertain non-overlapping patches for one image
    """
    selected_patches = []
    sorted_uncertainty_scores = np.flip(np.sort(uncertainty_scores.flatten()))
    sorted_uncertainty_indices = np.flip(np.argsort(uncertainty_scores.flatten()))
    # This was just for visualization purposes in MITK
    # selected = 0

    # Iterate over the sorted uncertainty scores and their indices to get the most uncertain
    for uncertainty_score, uncertainty_index in zip(
        sorted_uncertainty_scores, sorted_uncertainty_indices
    ):
        # Get the index as coordinates
        coords = np.unravel_index(uncertainty_index, uncertainty_scores.shape)
        # Check if coordinated overlap with already queried region
        if not does_overlap(coords, patch_size, selected_array):
            # If it is a non-overlapping region, append this patch to be queried
            selected_patches.append(
                {
                    "file": image_name,
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
