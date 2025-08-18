import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import torch
from loguru import logger
from nnunetv2.utilities.file_path_utilities import get_output_folder

import wandb
from nnactive.aggregations.convolution import ConvolveAggScipy, ConvolveAggTorch
from nnactive.config import ActiveConfig
from nnactive.data import Patch
from nnactive.logger import monitor
from nnactive.nnunet.utils import get_raw_path
from nnactive.results.utils import get_results_folder as get_nnactive_results_folder
from nnactive.strategies.base import AbstractQueryMethod, BaseQueryPredictor
from nnactive.strategies.registry import register_strategy
from nnactive.utils.io import load_label_map
from nnactive.utils.torchutils import estimate_free_cuda_memory, get_tensor_memory_usage

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class ExpectedPatchDiceScore:
    def __init__(self, patch_size: list[int], stride: Union[int, list[int]] = 1):
        """Class containing the code which returns code to compute negative expected dice (expected dice score) and returns a sorted list giving the coords and negative dice scores.

        Args:
            patch_size (list[int]): _description_
            stride (Union[int, list[int]], optional): _description_. Defaults to 1.
        """
        self.patch_size = patch_size
        self.stride = stride
        if isinstance(stride, int):
            self.stride = [stride] * len(self.patch_size)
        if (
            stride == 1
        ):  # TODO: for strides < 8 for large images scipy is still faster. This can be implemented better
            self.aggregation = ConvolveAggScipy(patch_size, stride=stride)
        else:
            self.aggregation = ConvolveAggTorch(patch_size, stride=stride)

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
        probs: list[Path] | torch.Tensor,
        device: torch.device = DEVICE,
    ):
        overall_time_start = time.perf_counter()
        fold = 0

        mean_prob = compute_mean_probs(probs, device).to(torch.float)

        num_images = len(probs)
        dice_dict: dict[tuple, list[float]] = {}
        mean_device = mean_prob.device
        logger.info(f"Mean prob on device: {mean_prob.device}")
        # iterate over M
        for fold in range(num_images):
            img_start = time.perf_counter()
            # Perform this multiplication on CPU so as to not run OOM
            if mean_prob.device != "cpu":
                logger.info("Putting mean prob on CPU to avoid Cuda OOM error")
                mean_prob = mean_prob.to("cpu")
            if isinstance(probs, list):
                prob_fold = torch.from_numpy(np.load(probs[fold]))
            else:
                # no deepcopy required as we perform no inplace computations
                prob_fold = probs[fold].to(torch.float)

            TP = 2 * mean_prob * prob_fold
            Div = mean_prob + prob_fold

            # Perform time consuming aggregation on GPU
            TP = TP.to(mean_device)
            Div = Div.to(mean_device)
            logger.info(f"TP and Div on device: {TP.device}")
            class_dice = None
            if device.type == "cuda" and get_tensor_memory_usage(
                TP[0]
            ) * 10 > estimate_free_cuda_memory(device):
                TP = TP.to("cpu")
                Div = Div.to("cpu")
            # iterate over classes
            for c in range(TP.shape[0]):
                try:
                    conv, kernel_size = self.aggregation.forward(TP[c])
                    conv /= self.aggregation.forward(Div[c])[0]
                except RuntimeError as e:
                    logger.debug(
                        "Possibly CUDA OOM error, try to obtain compute_val on CPU."
                    )
                    TP = TP.to("cpu")
                    Div = Div.to("cpu")
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    conv, kernel_size = self.aggregation.forward(TP[c])
                    conv /= self.aggregation.forward(Div[c])[0]
                # obtain results
                if class_dice is None:
                    class_dice = np.zeros((TP.shape[0], *conv.shape))  # C x XYZ
                class_dice[c] = conv
            # get mean dice
            # perhaps sum would make more sense but mean and sum max is identical.
            # Just be careful about nanmean!
            ######### rank according to negative dice #####
            neg_dice: np.ndarray = -1 * np.nanmean(class_dice, axis=0)
            # get for each coordinate in image space the corresponding dice values
            for elt in range(neg_dice.size):
                # image space coordinate
                coords = self.aggregation.backward_index(elt, neg_dice.shape)
                # aggregated dice space coordinate
                coords_dice = tuple(
                    [t.item() for t in np.unravel_index(elt, neg_dice.shape)]
                )
                if coords not in dice_dict:
                    dice_dict[coords] = [neg_dice[coords_dice]]
                else:
                    dice_dict[coords].append(neg_dice[coords_dice])
            img_end = time.perf_counter()
            logger.info(f"Finished image in {img_end - img_start:.4f}sec")
        dice_dict: dict[tuple, float] = {k: np.nanmean(v) for k, v in dice_dict.items()}
        dice_dict = {k: v for k, v in dice_dict.items() if not np.isnan(v)}
        # get list containing negative scores ranked in descending order
        sorted_dice_dict = {
            k: v
            for k, v in sorted(
                dice_dict.items(), key=lambda item: item[1], reverse=True
            )
        }
        overall_time_end = time.perf_counter()
        logger.info(
            f"Finished all images in {overall_time_end - overall_time_start:.4f}sec"
        )
        return sorted_dice_dict, kernel_size


@register_strategy("expected_dice")
class ExpectedDiceQuery(AbstractQueryMethod):
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
        config: ActiveConfig | None = None,
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
            config=config,
        )
        self.num_processes_preprocessing = num_processes_preprocessing
        self.use_mirroring = use_mirroring
        self.use_gaussian = use_gaussian
        self.tile_step_size = tile_step_size
        self.agg_stride = agg_stride
        self.n_patch_per_image = n_patch_per_image
        self.strategy = ExpectedPatchDiceScore(self.config.patch_size, self.agg_stride)

    def wrap_query_part(
        self,
        part_id: int = 0,
        num_parts: int = 1,
        device: torch.device = torch.device("cuda:0"),
        wandb_group: str = "Test",
    ) -> list[dict]:
        self.config.set_nnunet_env()
        with monitor.active_run(group=wandb_group):
            top_patches = self.query_part(part_id, num_parts, device)
        return top_patches

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

    def query_part(
        self,
        part_id: int = 0,
        num_parts: int = 1,
        device: torch.device = torch.device("cuda:0"),
    ) -> list[dict]:
        temp_path = get_raw_path(self.dataset_id) / f"temp_probs_part{part_id}"

        torch.cuda.set_device(device)
        # Initialize Predictor
        predictor = BaseQueryPredictor(
            tile_step_size=self.tile_step_size,
            use_mirroring=self.use_mirroring,
            use_gaussian=self.use_gaussian,
            verbose=self.verbose,
            allow_tqdm=not self.verbose,
            device=device,
        )

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
        predictor.predict_from_data_iterator(data_iterator, self, temp_path=temp_path)
        return self.compose_query_of_patches()

    def query_from_probs(
        self,
        probs: list[Path] | np.ndarray,
        image_shape: Iterable[int],
        label_file: str,
        device: torch.device = DEVICE,
    ):
        with monitor.timer("query_from_probs"):
            with torch.no_grad():
                logger.info("Compute pairwise dice...")
                sorted_dice_scores, kernel_size = self.strategy.forward(
                    probs, device=device
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

    def compose_query_of_patches(self) -> list[Patch]:
        """Returns the final query based on all patches and respective scores
        in self.top_patches.

        Returns:
            list[Patch]: Query of patches with highest scores.
        """
        with monitor.timer("compose_query_of_patches"):
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
            return patches


def compute_mean_probs(
    probs: list[Path] | torch.Tensor, device: torch.device = DEVICE
) -> torch.Tensor:
    """Compute predictive entropyon list of paths saving npy arrays or a tensor.

    Args:
        probs (list[Path] | torch.Tensor): paths to probability maps for image
            [C x XYZ] per item in list or [M x C x XYZ]
        device (str, optional): preferred device for computation. Defaults to DEVICE.

    Returns:
        torch.Tensor: Mean Probs C x H x W x D (on device)
    """
    logger.info("Compute mean probabilities")

    def _compute_mean_prob(mean_prob: torch.Tensor, probs: list[Path] | torch.Tensor):
        for fold in range(1, len(probs)):
            if isinstance(probs, list):
                temp_val = torch.from_numpy(np.load(probs[fold])).to(mean_prob.device)
            else:
                temp_val = deepcopy(probs[fold]).to(mean_prob.device)
            mean_prob += temp_val
            del temp_val
        mean_prob /= len(probs)
        return mean_prob

    fold = 0
    if isinstance(probs, list):
        compute_val = torch.from_numpy(np.load(probs[fold])).to(device)
    else:
        compute_val = deepcopy(probs[fold]).to(device)
    # check if it will fit into GPU
    if device.type == "cuda":
        if (get_tensor_memory_usage(compute_val) * 2) * 1.1 < estimate_free_cuda_memory(
            device
        ):
            use_device = device
        else:
            use_device = torch.device("cpu")
            logger.debug(
                f"Computation on {device} not feasible due to VRAM, falling back to {use_device} for computation and then move to {device}"
            )
    else:
        # CPU case
        use_device = device

    try:
        logger.debug(f"Compute Mean Prob on device: {use_device}")
        compute_val = compute_val.to(use_device)
        compute_val = _compute_mean_prob(compute_val, probs)
    except RuntimeError as e:
        logger.debug("Possibly CUDA OOM error, try to obtain compute_val on CPU.")
        del compute_val
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        compute_val = compute_mean_probs(probs, torch.device("cpu"))
    return compute_val.to(device)
