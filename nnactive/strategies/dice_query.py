import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple, Union

import numpy as np
import torch
from loguru import logger
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from nnunetv2.utilities.file_path_utilities import get_output_folder
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

from nnactive.aggregations.convolution import ConvolveAggScipy, ConvolveAggTorch
from nnactive.config import ActiveConfig
from nnactive.data import Patch
from nnactive.logger import monitor
from nnactive.nnunet.utils import get_raw_path
from nnactive.results.utils import get_results_folder as get_nnactive_results_folder
from nnactive.strategies.base import AbstractQueryMethod
from nnactive.strategies.base_uncertainty import nnActivePredictor
from nnactive.utils.io import load_label_map
from nnactive.utils.torchutils import estimate_free_cuda_memory, get_tensor_memory_usage

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class SoftDiceLoss(nn.Module):
    def __init__(
        self,
        apply_nonlin: Callable = None,
        batch_dice: bool = False,
        do_bg: bool = True,
        smooth: float = 1.0,
        ddp: bool = True,
        clip_tp: float = None,
    ):
        """ """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.clip_tp = clip_tp
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        if self.ddp and self.batch_dice:
            tp = AllGatherGrad.apply(tp).sum(0)
            fp = AllGatherGrad.apply(fp).sum(0)
            fn = AllGatherGrad.apply(fn).sum(0)

        if self.clip_tp is not None:
            tp = torch.clip(tp, min=self.clip_tp, max=None)

        nominator = 2 * tp
        denominator = 2 * tp + fp + fn

        dc = (nominator + self.smooth) / (torch.clip(denominator + self.smooth, 1e-8))

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        # TODO this is changed compared to nnU-Net code
        dc = dc.mean(dim=1)

        return -dc


class PatchDataset(Dataset):
    def __init__(self, mean_prob, prob_fold, coords, kernel_size):
        self.coords = coords
        self.kernel_size = kernel_size
        self.mean_prob = torch.tensor(mean_prob)
        self.prob_fold = torch.tensor(prob_fold)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coords_end = np.array(self.coords[idx]) + self.kernel_size
        coord_slices = tuple(
            (slice(cs, ce, None) for cs, ce in zip(self.coords[idx], coords_end))
        )
        coord_slices = (slice(None), *coord_slices)
        mean_patch = self.mean_prob[coord_slices]
        prob_patch = self.prob_fold[coord_slices]
        return mean_patch, prob_patch, (self.coords[idx])


class PatchDice:
    def __init__(self, patch_size: list[int], stride: Union[int, list[int]] = 1):
        self.patch_size = patch_size
        self.stride = stride
        if isinstance(stride, int):
            self.stride = [stride] * len(self.patch_size)

    def get_dice_patch(
        self, mean_prob, prob_fold, kernel_size, dice_dict, batch_size=16, num_workers=8
    ):
        dice_loss = SoftDiceLoss(ddp=False)
        patch_dataset = PatchDataset(
            mean_prob, prob_fold, list(dice_dict.keys()), kernel_size
        )

        patch_dataloader = torch.utils.data.DataLoader(
            patch_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        for batch in tqdm(patch_dataloader):
            dice_loss_batch = dice_loss(batch[0].to(DEVICE), batch[1].to(DEVICE))
            for i, dice in enumerate(dice_loss_batch):
                coords = batch[2]
                coord = tuple(
                    [coords[0][i].item(), coords[1][i].item(), coords[2][i].item()]
                )
                dice_dict[coord].append(dice.item())
        # for mean_patch, prob_patch, coord in patch_dataloader:
        #     mean_patch = mean_patch
        #     prob_patch = prob_patch
        #     coord = tuple([c.item() for c in coord])
        #     dice_dict[coord].append(dice_loss(mean_patch, prob_patch).item())
        # mean_prob = torch.tensor(mean_prob).to(DEVICE)
        # prob_fold = torch.tensor(prob_fold).to(DEVICE)
        # for coord in dice_dict.keys():
        #     coords_end = np.array(coord) + kernel_size
        #     coord_slices = tuple(
        #         (slice(cs, ce, None) for cs, ce in zip(coord, coords_end))
        #     )
        #     coord_slices = (slice(None), *coord_slices)
        #     mean_patch = mean_prob[coord_slices]
        #     prob_patch = prob_fold[coord_slices]
        #     dice_dict[coord].append(dice_loss(mean_patch, prob_patch).item())
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
        batch_size=16,
        num_workers=8,
    ):
        overall_time_start = time.perf_counter()
        if images is None and dataset_id is None and num_folds is None:
            raise ValueError("Either images or dataset_id and num_folds must be given")

        if images is None:
            prob_path = get_nnactive_results_folder(dataset_id) / "temp"
            fold_paths = [prob_path / f"probs_fold{i}.npy" for i in range(0, num_folds)]
            mean_prob = get_mean_prob(dataset_id, num_folds, device=DEVICE)
        else:
            mean_prob = torch.from_numpy(np.mean(images, axis=0)).to(DEVICE)

        # due to softmax, prob shape is offset by one (class dimension)
        kernel_size = [
            min(self.patch_size[i], mean_prob.shape[i + 1])
            for i in range(len(self.patch_size))
        ]
        # breakpoint()
        # coordinate_tuples = self.get_coords_patches(mean_prob.shape[1:])
        # dice_dict = {
        #     tuple(coord.tolist()): []
        #     for coord in coordinate_tuples
        #     if not np.any(coord + kernel_size > mean_prob.shape[1:])
        # }
        num_images = len(images) if images is not None else num_folds
        dice_dict = None
        mean_device = mean_prob.device
        logger.info(f"Mean prob on device: {mean_prob.device}")
        for i in range(num_images):
            img_start = time.perf_counter()
            if mean_prob.device != "cpu":
                logger.info("Putting mean prob on CPU to avoid Cuda OOM error")
                mean_prob = mean_prob.to("cpu")
            prob_fold = images[i] if images is not None else np.load(str(fold_paths[i]))
            prob_fold = torch.from_numpy(prob_fold)
            # dice_dict = self.get_dice_patch(
            #     mean_prob, prob_fold, kernel_size, dice_dict, batch_size, num_workers
            # )
            TP = 2 * mean_prob * prob_fold
            Div = mean_prob + prob_fold
            if (
                self.stride == 1
            ):  # TODO: for strides < 8 for large images scipy is still faster. This can be implemented better
                agg = ConvolveAggScipy(kernel_size, stride=self.stride)
            else:
                TP = TP.type(torch.float32).to(mean_device)
                Div = Div.type(torch.float32).to(mean_device)
                logger.info(f"TP and Div on device: {TP.device}")
                agg = ConvolveAggTorch(kernel_size, stride=self.stride)
            class_dice = None
            for i in range(TP.shape[0]):
                conv = agg.forward(TP[i])[0] / agg.forward(Div[i])[0]
                if class_dice is None:
                    class_dice = np.zeros((TP.shape[0], *conv.shape))
                class_dice[i] = conv
            dice = np.nanmean(class_dice, axis=0)
            for i in range(dice.size):
                coords = agg.backward_index(i, dice.shape)
                coords_dice = (
                    coords
                    if self.stride == 1
                    else tuple([t.item() for t in np.unravel_index(i, dice.shape)])
                )
                if dice_dict is None:
                    dice_dict = {coords: [dice[coords_dice]]}
                elif coords not in dice_dict:
                    dice_dict[coords] = [dice[coords_dice]]
                else:
                    dice_dict[coords].append(dice[coords_dice])
            img_end = time.perf_counter()
            logger.info(f"Finished image in {img_end - img_start:.4f}sec")
        dice_dict = {k: np.nanmean(v) for k, v in dice_dict.items()}
        dice_dict = {k: v for k, v in dice_dict.items() if not np.isnan(v)}
        sorted_dice_dict = {
            k: v
            for k, v in sorted(
                dice_dict.items(), key=lambda item: item[1]  # , reverse=True
            )
        }
        overall_time_end = time.perf_counter()
        logger.info(
            f"Finished all images in {overall_time_end - overall_time_start:.4f}sec"
        )
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


def get_mean_prob(dataset_id, num_folds, device: str = DEVICE):
    fold = 0
    prob_path = str(
        get_nnactive_results_folder(dataset_id) / "temp" / f"probs_fold{fold}.npy"
    )
    compute_val = torch.from_numpy(np.load(prob_path))
    # check if it will fit into GPU
    if get_tensor_memory_usage(compute_val) * 2.1 < estimate_free_cuda_memory():
        try:
            logger.info(f"Compute entropy on device: {device}")
            compute_val = compute_val.to(device)
            mean_p = _compute_mean_prob(compute_val, num_folds, dataset_id)
        except RuntimeError as e:
            logger.info("Possibly CUDA OOM error, try to obtain compute_val on CPU.")
            del compute_val
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            mean_p = get_mean_prob(num_folds, dataset_id, "cpu").to(device)
        return mean_p
    else:
        logger.info(f"Compute entropy on CPU instead of {device}")
        mean_p = _compute_mean_prob(compute_val, num_folds, dataset_id)
        return mean_p.to(device)


def _compute_mean_prob(mean_prob: torch.Tensor, num_folds: int, dataset_id: int):
    for fold in range(1, num_folds):
        prob_path = str(
            get_nnactive_results_folder(dataset_id) / "temp" / f"probs_fold{fold}.npy"
        )
        cur_prob = torch.from_numpy(np.load(prob_path)).to(mean_prob.device)
        if mean_prob is None:
            mean_prob = cur_prob
        else:
            mean_prob += cur_prob
        del cur_prob
    mean_prob /= num_folds
    return mean_prob
