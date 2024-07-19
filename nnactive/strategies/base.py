from __future__ import annotations

import multiprocessing as mp
import os
import shutil
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, Type, Union

import numpy as np
import psutil
import torch
import wandb
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from loguru import logger
from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.export_probs import (
    convert_predicted_logits_to_probs_with_correct_shape,
)
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.utilities.file_path_utilities import get_output_folder
from nnunetv2.utilities.helpers import empty_cache
from torch._dynamo import OptimizedModule
from torch.backends import cudnn
from tqdm import tqdm

from nnactive.config import ActiveConfig
from nnactive.config.struct import ActiveConfig
from nnactive.data import Patch
from nnactive.logger import monitor
from nnactive.loops.loading import get_patches_from_loop_files
from nnactive.masking import does_overlap, percentage_overlap, percentage_overlap_array
from nnactive.nnunet.utils import get_raw_path
from nnactive.utils.io import load_label_map
from nnactive.utils.patches import get_slices_for_file_from_patch
from nnactive.utils.timer import CudaTimer, Timer


class AbstractQueryMethod(ABC):
    def __init__(
        self,
        config: ActiveConfig,
        dataset_id: int,
        file_ending: str = ".nii.gz",
        additional_label_path: Path | None = None,
        verbose: bool = False,
        loop_val: int = 0,
        seed: int | None = None,
        **kwargs,
    ):
        logger.info(f"Initializeing Query Method for loop {loop_val}")
        self.dataset_id = dataset_id
        self.additional_label_path = additional_label_path
        self.file_ending = file_ending
        self.top_patches: list[dict] = []
        self.verbose = verbose
        self.config = config
        self.annotated_patches = get_patches_from_loop_files(
            get_raw_path(self.dataset_id), loop_val
        )
        self.seed = seed if seed is not None else self.config.seed + loop_val
        self.rng = np.random.default_rng(self.seed)
        self.__post_init__()

    def __post_init__(self):
        pass

    @abstractmethod
    def query(
        self,
        verbose=False,
        n_gpus: int = 0,
    ) -> list[Patch]:
        pass

    # @property
    # def annotated_patches(self) -> list[Patch]:
    #     return get_patches_from_loop_files(get_raw_path(self.dataset_id))

    def check_overlap(
        self,
        ipatch: Patch,
        patches: list[Patch],
        additional_label: None | np.ndarray = None,
        verbose: bool = False,
    ) -> bool:
        # start with checking overlap compared to other patches
        allow_patch = False
        if self.config.patch_overlap > 0:
            patch_overlap = percentage_overlap(ipatch, patches)
            allow_patch = patch_overlap <= self.config.patch_overlap
            if verbose and allow_patch:
                logger.debug(
                    f"Patch creation succesful with patch overlap: {patch_overlap} <= {self.config.patch_overlap} overlap with additional labels."
                )
        else:
            allow_patch = not does_overlap(ipatch, patches)

        # check overlap with additional labels
        if additional_label is not None and allow_patch:
            additional_overlap = percentage_overlap_array(ipatch, additional_label)
            if additional_overlap <= self.config.additional_overlap:
                if verbose:
                    logger.debug(
                        f"Patch creation succesful with additional labels overlap: {additional_overlap} <= {self.config.additional_overlap} overlap with additional labels."
                    )
                return True
            else:
                allow_patch = False
        return allow_patch

    def initialize_selected_array(
        self,
        image_shape: Iterable[int],
        label_file: str,
        annotated_patches: list[Patch],
    ) -> np.ndarray:
        """Initializes the array which simulates which are already selected.

        Args:
            image_shape (Iterable[int]): shape of initial image
            label_file (str): name of label file (with ending e.g. .nii.gz)
            annotated_patches (list[Patch]): list of already annotated patches

        Returns:
            np.ndarray: boolean array with annotated areas having value True
        """
        selected_array = np.zeros(image_shape, dtype=bool)

        # mark patches as selected
        patch_access = get_slices_for_file_from_patch(annotated_patches, label_file)

        for slices in patch_access:
            selected_array[slices] = 1
        return selected_array

    @classmethod
    def init_from_dataset_id(
        cls: Type[AbstractQueryMethod],
        config: ActiveConfig,
        dataset_id: int,
        loop_val: int,
        seed: int,
        additional_label_path: Path | None = None,
        **kwargs,
    ):
        additional_label_path: Path = (
            get_raw_path(dataset_id) / "addTr"
            if additional_label_path is None
            else additional_label_path
        )
        if not additional_label_path.is_dir():
            additional_label_path = None
        strategy = cls(
            dataset_id=dataset_id,
            additional_label_path=additional_label_path,
            config=config,
            loop_val=loop_val,
            seed=seed,
            **kwargs,
        )
        return strategy


class BasePredictionQuery(AbstractQueryMethod):
    def __init__(
        self,
        dataset_id: int,
        config: ActiveConfig,
        loop_val: int,
        seed: int | None,
        file_ending: str = ".nii.gz",
        num_processes_preprocessing: int = 3,
        additional_label_path: Path | None = None,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(
            dataset_id=dataset_id,
            config=config,
            loop_val=loop_val,
            file_ending=file_ending,
            additional_label_path=additional_label_path,
            verbose=verbose,
            seed=seed,
        )

        self.num_processes_preprocessing = num_processes_preprocessing

    def get_n_patch_per_image(self):
        return self.config.n_patch_per_image

    def query_file_from_dict(
        self,
        query_dicts: list[dict[str, Any]],
        file_id: str,
        device: torch.device = torch.device("cuda:0"),
    ) -> list[dict[str, Any]]:
        """Computes potential queries for a single input image and adds best queries to the internal list of queries.

        Args:
            query_dicts (list[dict[str, Any]]): each element in the list stands for one fold.
            file_id (str): name of label file without suffix
            device (_type_, optional): _description_. Defaults to torch.device("cuda:0").

        Returns:
            list[dict[str, Any]]: selection of potential queries for current file
        """
        with (
            monitor.timer("query_from_probs") if monitor.is_active() else nullcontext()
        ):
            value_dicts = self.strategy(query_dicts, device)

            logger.info("Initialize selected array...")
            annotated_patches = [
                patch
                for patch in self.annotated_patches
                if patch.file == file_id + ".nii.gz"
            ]

            logger.info("Select patches...")
            selected_patches: list[dict] = self.select_file_patches(
                value_dicts,
                annotated_patches=annotated_patches,
                label_file=file_id,
                n=self.get_n_patch_per_image(),
            )
            logger.info("Finished patch selection.")
            self.top_patches += selected_patches
        return value_dicts

    @abstractmethod
    def strategy(
        self,
        query_dict: list[dict[str, Any]],
        device: torch.device = torch.device("cuda:0"),
    ) -> list[dict[str, Any]]:
        pass

    def select_file_patches(
        self,
        value_dicts: list[dict[str, Any]],
        annotated_patches: list[Patch],
        label_file: str,
        n: int,
    ) -> list[dict[str, Any]]:
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

        selected_patches = []
        pbar0 = tqdm(total=n, position=0, desc="Patch Selection", disable=self.verbose)
        pbar1 = tqdm(
            total=len(value_dicts),
            position=1,
            desc="Possible Patch Search",
            disable=self.verbose,
        )
        while len(value_dicts) > 0:
            index = self.next_best_vd_index(value_dicts)
            value_dict = value_dicts.pop(index)
            value_dict["file"] = label_file + ".nii.gz"

            pbar1.update()
            patch = Patch(
                file=value_dict["file"],
                coords=value_dict["coords"],
                size=value_dict["size"],
            )

            # Check if coordinated overlap with already queried region
            if self.check_overlap(
                patch, annotated_patches, additional_label, verbose=self.verbose
            ):
                # If it is a non-overlapping region, append this patch to be queried
                selected_patches.append(value_dict)
                # Mark region as queried
                annotated_patches.append(patch)
                # Stop if we reach the maximum number of patches to be queried
                pbar0.update()
            if n is not None and len(selected_patches) >= n:
                break
        pbar1.close()
        pbar0.close()
        logger.info(f"Finished patch selection for image {label_file}")
        return selected_patches

    def next_best_vd_index(self, value_dicts: list[dict]) -> int:
        """Assume that value_dicts are already order that first one has highest score etc."""
        return 0

    def get_file_handler(self, temp_path: Path):
        return TemporaryFileHandler(temp_path)

    def cleanup_prediction():
        """Performed after prediction of each single image and subsequent query_file_from_dict"""
        pass

    def cleanup_query():
        """Performed after query"""
        pass

    def query_part(
        self,
        part_id: int = 0,
        num_parts: int = 1,
        device: torch.device = torch.device("cuda:0"),
    ) -> list[dict]:
        temp_file_handler = self.get_file_handler(
            get_raw_path(self.dataset_id) / f"temp_probs_part{part_id}"
        )

        torch.cuda.set_device(device)
        # Initialize Predictor
        predictor = self.build_query_predictor(device)
        # Initialize Model for Predictor
        nnunet_plans_identifier = self.config.model_plans
        nnunet_trainer_name = self.config.trainer
        nnunet_config = self.config.model_config
        model_folder = get_output_folder(
            self.dataset_id, nnunet_trainer_name, nnunet_plans_identifier, nnunet_config
        )
        use_folds = tuple(range(self.config.train_folds))
        predictor.initialize_from_trained_model_folder(
            model_folder, use_folds=use_folds
        )

        source_folder = str(get_raw_path(self.dataset_id) / "imagesTr")
        output_folder = "/".join(model_folder.split("/")[:-1])

        data_iterator = predictor.get_data_iterator_from_folders(
            list_of_lists_or_source_folder=source_folder,
            output_folder_or_list_of_truncated_output_files=output_folder,
            num_processes_preprocessing=self.num_processes_preprocessing,
            part_id=part_id,
            num_parts=num_parts,
        )
        predictor.predict_from_data_iterator(data_iterator, self, temp_file_handler)
        return self.top_patches

    def build_query_predictor(self, device: torch.device):
        predictor = BaseQueryPredictor(
            tile_step_size=self.config.tile_step_size,
            use_mirroring=self.config.use_mirroring,
            use_gaussian=self.config.use_gaussian,
            verbose=self.verbose,
            allow_tqdm=not self.verbose,
            device=device,
        )

        return predictor

    def query(self, n_gpus: int = 0, verbose: bool = False) -> list[Patch]:
        if n_gpus == 0:
            device = torch.device("cuda:0")
            self.query_part(part_id=0, num_parts=1, device=device)
        else:
            devices = [torch.device(f"cuda:{i}") for i in range(n_gpus)]
            num_parts = [n_gpus] * n_gpus
            parts = [i for i in range(n_gpus)]
            try:
                with ProcessPoolExecutor(
                    max_workers=n_gpus, mp_context=mp.get_context("spawn")
                ) as executor:
                    for top_patch_part in executor.map(
                        self.wrap_query_part,
                        parts,
                        num_parts,
                        devices,
                        [wandb.run.group] * n_gpus,
                    ):
                        self.top_patches.extend(top_patch_part)

            except BrokenProcessPool as exc:
                raise MemoryError(
                    "One of the worker processes died. "
                    "This usually happens because you run out of memory. "
                    "Try running with less processes."
                ) from exc

        return self.compose_query_of_patches()

    # TODO: Rewrite this function to allow diveristy methods!
    def compose_query_of_patches(self):
        with (
            monitor.timer("compose_query_of_patches")
            if monitor.is_active()
            else nullcontext()
        ):
            sorted_top_patches = sorted(
                self.top_patches, key=lambda d: d["score"], reverse=True
            )[: self.config.query_size]
            patches = [
                {
                    "file": patch["file"],
                    "coords": patch["coords"],
                    "size": patch["size"],
                }
                for patch in sorted_top_patches
            ]
            patches = [Patch(**patch) for patch in patches]
            if len(patches) < self.config.query_size:
                raise RuntimeError(
                    f"Not enough patches could be queried, {len(patches)} instead of {self.config.query_size}"
                )
            return patches


class TemporaryFileHandler:
    def __init__(self, temp_path: Path, save_keys: tuple[str, ...] = ("probs",)):
        """Class to handle saving of temporary files right after prediction step.

        Args:
            temp_path (Path): path to save files to
        """
        self.temp_path = temp_path
        self.default_filenames = {key: key + "_fold" for key in save_keys}

    def save_temporary_files(
        self,
        temporary_dict: dict,
        filename: str | None = None,
        fold: int | str | None = None,
    ) -> dict[str, Path]:
        """Save temporary files in temporary dict and returns handles to obtain them again.

        Args:
            temporary_dict (dict): _description_
            filename (str) : if specified is expected to carry suffix .xyz
        """
        if filename is None:
            filename = ""
        save_timer = Timer()
        save_timer.start()
        save_files = dict()
        os.makedirs(self.temp_path, exist_ok=True)
        for key in self.default_filenames:
            save_file = self.temp_path / (
                filename + self.default_filenames[key] + self.build_suffix(fold)
            )

            np.save(save_file, temporary_dict[key])
            save_files[key] = save_file
        logger.debug(f"Time for saving: {save_timer.stop()/1000}s")
        return save_files

    def build_suffix(self, fold: int | str | None):
        return f"{fold}.npy"

    def clean_up(self):
        if self.temp_path.is_dir():
            shutil.rmtree(self.temp_path)


class BaseQueryPredictor(nnUNetPredictor):
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
        logger.trace(
            f"RAM used before conversion of logits to probs:~{psutil.Process().memory_info().rss * 1e-9}GB"
        )
        # NAN Checking is now handled by nnU-Net
        # logits_nf = torch.isfinite(logits) == 0
        # if torch.any(logits_nf):
        #     raise RuntimeError(f"NAN values in logits")
        # del logits_nf

        conversion_timer = CudaTimer()
        conversion_timer.start()
        logger.debug(f"Shape before postprocessing: {logits.shape}")
        out_prob = convert_predicted_logits_to_probs_with_correct_shape(
            logits.cpu(),
            self.plans_manager,
            self.configuration_manager,
            self.label_manager,
            properties,
        )
        logger.debug(f"Shape after postprocessing: {out_prob.shape}")
        logger.debug(f"Time for conversion: {conversion_timer.stop()/1000}s")

        # fastest way to check if nan in np array
        # according to https://stackoverflow.com/questions/6736590/fast-check-for-nan-in-numpy
        # NAN Checking is now handled by nnU-Net
        # if np.isnan(np.sum(out_prob)):
        #     raise ValueError(f"NAN values in probablities in image!")

        return out_prob

    def predict_fold_logits_from_preprocessed_data(
        self,
        data: torch.TensorType,
        properties: dict,
        temp_file_handler: TemporaryFileHandler,
    ) -> list[dict]:
        """Computes the logits/probs for all folds.

        Args:
            data (torch.TensorType): Preprocessed Data
        """
        original_perform_everything_on_device = self.perform_everything_on_device
        out: list[dict] = [None] * len(self.list_of_parameters)

        with torch.no_grad():
            if self.perform_everything_on_device:
                try:
                    for fold, params in enumerate(self.list_of_parameters):
                        # messing with state dict names...
                        if not isinstance(self.network, OptimizedModule):
                            self.network.load_state_dict(params)
                        else:
                            self.network._orig_mod.load_state_dict(params)

                        logger.info(
                            f"RAM used before sliding window prediction:~{psutil.Process().memory_info().rss * 1e-9}GB"
                        )
                        logits = self.predict_sliding_window_return_logits(data)

                        out_probs = self.postprocess_logits(logits, properties)
                        out[fold] = temp_file_handler.save_temporary_files(
                            {"probs": out_probs}
                        )

                except RuntimeError:
                    logger.exception(
                        "Prediction with perform_everything_on_gpu=True failed due to insufficient GPU memory. "
                        "Falling back to perform_everything_on_gpu=False. Not a big deal, just slower..."
                    )
                    # print("Error:")
                    # traceback.print_exc()
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
                    out[fold] = temp_file_handler.save_temporary_files(
                        {"probs": out_probs}
                    )

            self.perform_everything_on_device = original_perform_everything_on_device
        return out

    def predict_from_data_iterator(
        self,
        data_iterator,
        query_method: BasePredictionQuery,
        temp_file_handler: TemporaryFileHandler,
        save_probabilities: bool = False,
        num_processes_segmentation_export: int = default_num_processes,
    ):
        """
        This function is executed by query_method inside of query_method.query().
        each element returned by data_iterator must be a dict with 'data', 'ofile' and 'data_properites' keys!
        If 'ofile' is None, the result will be returned instead of written to a file
        """
        # set up multiprocessing for spawning

        for preprocessed in (
            monitor.itertimer(data_iterator, name="get_queries_per_image")
            if monitor.is_active()
            else data_iterator
        ):
            # Reminder: GPU issues can be nicely evaluated using case_00223 on KiTS21_small/KiTS21...
            # if os.path.basename(preprocessed["ofile"]) not in [
            #     "case_00223",
            #     "case_00210",
            # ]:
            #     print("Skipping file:", preprocessed["ofile"])
            #     continue
            data = preprocessed["data"]
            if isinstance(data, str):
                delfile = data
                data = torch.from_numpy(np.load(data))
                os.remove(delfile)

            ofile = preprocessed["ofile"]
            if ofile is not None:
                filename: str = os.path.basename(ofile)
                logger.info(f"\nPredicting {filename}:")
            else:
                logger.info(f"\nPredicting image of shape {data.shape}:")

            logger.info(
                f"perform_everything_on_gpu: {self.perform_everything_on_device}"
            )

            properties = preprocessed["data_properties"]

            if torch.cuda.is_available():
                cudnn.benchmark = True

            query_dicts: list[dict[str, Any]] = (
                self.predict_fold_logits_from_preprocessed_data(
                    data, properties, temp_file_handler=temp_file_handler
                )
            )

            logger.info("Start Query")
            # Benchmark = True can lead to problems during inference with convolutions
            # illegal memory access
            if torch.cuda.is_available():
                cudnn.benchmark = False
            query_method.query_file_from_dict(
                query_dicts,
                filename,
                device=self.device,
            )

            temp_file_handler.clean_up()
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
