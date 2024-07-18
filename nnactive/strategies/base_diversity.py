from __future__ import annotations

import itertools
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import nullcontext
from itertools import accumulate
from pathlib import Path
from typing import Callable, Iterable, Union

import numpy as np
import torch
import wandb
from dynamic_network_architectures.architectures import unet
from loguru import logger
from nnunetv2.utilities.file_path_utilities import get_output_folder
from tqdm import tqdm

from nnactive.aggregations.convolution import ConvolveAggScipy, ConvolveAggTorch
from nnactive.config.struct import ActiveConfig
from nnactive.data import Patch
from nnactive.logger import monitor
from nnactive.nnunet.utils import get_raw_path
from nnactive.strategies.base import AbstractQueryMethod
from nnactive.strategies.base_uncertainty import nnActivePredictor
from nnactive.strategies.utils import RepresentationHandler
from nnactive.utils.io import load_label_map


class DiversityQueryMethod(AbstractQueryMethod):
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
        self.n_patch_per_image = n_patch_per_image
        self.use_mirroring = use_mirroring
        self.use_gaussian = use_gaussian
        self.tile_step_size = tile_step_size
        if (
            agg_stride == 1
        ):  # TODO: for strides < 8 for large images scipy is still faster. This can be implemented better
            self.aggregation = ConvolveAggScipy(patch_size, stride=agg_stride)
        else:
            self.aggregation = ConvolveAggTorch(patch_size, stride=agg_stride)

        logger.info(
            f"Aggregation is performed using: {self.aggregation.__class__.__name__} with stride {agg_stride}"
        )

    def query_part(
        self,
        part_id: int = 0,
        num_parts: int = 1,
        device: torch.device = torch.device("cuda:0"),
    ) -> list[dict]:
        temp_path = get_raw_path(self.dataset_id) / f"temp_probs_part{part_id}"

        torch.cuda.set_device(device)
        # Initialize Predictor
        predictor = DiversityPredictor(
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
        predictor.setup_representations()

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
        return self.top_patches

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

    def query_from_probs(
        self,
        probs: list[Path] | np.ndarray,
        image_shape: Iterable[int],
        label_file: str,
        device: torch.device = torch.device("cuda:0"),
    ) -> tuple[torch.Tensor, np.ndarray]:
        """Computes potential queries for a single input image and adds best queries to the internal list of queries.

        Args:
            out_probs (torch.Tensor): probability map for image
            image_shape (Iterable[int]): shape of image
            label_file (str): name of label file
        """
        with (
            monitor.timer("query_from_probs") if monitor.is_active() else nullcontext()
        ):
            with torch.no_grad():
                logger.info("Compute uncertaintes...")
                uncertainty = self.get_uncertainty(probs, device=device)

                if torch.any(torch.isnan(uncertainty)):
                    # unc_num_nan = torch.sum(torch.isnan(uncertainty))
                    # unc_where_nan = torch.argwhere(torch.isnan(uncertainty))
                    raise ValueError(
                        f" NAN values in uncertainties for image {label_file}"
                    )
                logger.info("Aggregate uncertainties...")
                agg_uncertainty, kernel_size = self.aggregation.forward(uncertainty)

            logger.info("Initialize selected array...")
            annotated_patches = [
                patch
                for patch in self.annotated_patches
                if patch.file == label_file + ".nii.gz"
            ]

            logger.info("Select patches...")
            selected_patches = self.select_top_n_non_overlapping_patches(
                patch_size=kernel_size,
                aggregated=agg_uncertainty,
                annotated_patches=annotated_patches,
                label_file=label_file,
                n=self.n_patch_per_image,
            )
            logger.info("Finished patch selection.")
            self.top_patches += selected_patches
        return uncertainty, agg_uncertainty

    def get_uncertainty(
        self,
        probs: list[Path] | torch.Tensor,
        device: torch.device = torch.device("cuda:0"),
    ) -> torch.Tensor:
        """Compute uncertainty values from out_probs

        Args:
            probs (list[Path] | torch.Tensor): paths to probability maps for image
            [1 x C x XYZ] per item in list or [M x C x XYZ]

        Returns:
            torch.Tensor: outputs [XYZ] on device
        """
        out = torch.from_numpy(np.load(probs[0]))
        return out[0].to(device)

    def select_top_n_non_overlapping_patches(
        self,
        patch_size: list[int],
        aggregated: np.ndarray,
        annotated_patches: list[Patch],
        label_file: str,
        n: int,
    ) -> list[dict]:
        """Select top-n non overlapping patches in image.

        Args:
            patch_size (list[int]): size of patch
            aggregated (np.ndarray): aggregated uncertainty
            annotated_patches (list[Patch]): list of annotated patches per image
            label_file (str): name of label_file without ending (.nii.gz)
            n (int): number of patches to select

        Returns:
            list[dict]: dict contains all values required to build a Patch
        """
        selected_patches = []
        # sort only once since this can take a significant amount of time
        logger.info("Sort potential queries")
        flat_aggregated_uncertainties = aggregated.flatten()

        sorted_uncertainty_indices = np.flip(np.argsort(flat_aggregated_uncertainties))
        sorted_uncertainty_scores: list[float] = np.take_along_axis(
            flat_aggregated_uncertainties, sorted_uncertainty_indices, axis=0
        ).tolist()
        logger.info("Start finding non-overlapping patches.")
        # Iterate over the sorted uncertainty scores and their indices to get the most uncertain

        iterator = zip(sorted_uncertainty_scores, sorted_uncertainty_indices)
        pbar0 = tqdm(total=n, position=0, desc="Patch Selection", disable=self.verbose)
        pbar1 = tqdm(
            total=len(sorted_uncertainty_scores),
            position=1,
            desc="Possible Patch Search",
            disable=self.verbose,
        )

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

        for i, (uncertainty_score, uncertainty_index) in enumerate(iterator):
            pbar1.update()
            # get coordinates in image space from aggregated indices
            coords = self.aggregation.backward_index(
                uncertainty_index, aggregated.shape
            )
            patch = Patch(
                file=label_file + ".nii.gz",
                coords=coords,
                size=patch_size,
            )

            # Check if coordinated overlap with already queried region
            if self.check_overlap(
                patch, annotated_patches, additional_label, verbose=self.verbose
            ):
                # If it is a non-overlapping region, append this patch to be queried
                selected_patches.append(
                    {
                        "file": label_file + ".nii.gz",
                        "coords": coords,
                        "size": patch_size,
                        "score": uncertainty_score,
                    }
                )
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


class DiversityPredictor(nnActivePredictor):
    def setup_representations(self):
        self.network: unet.PlainConvUNet = self.network
        self.forward_representations: dict[str, list[torch.Tensor]] = {}
        downsamplig_stages = np.cumprod(
            self.network.encoder.strides, axis=0, dtype=int
        ).tolist()

        features_stages: list[int] = (
            self.configuration_manager.network_arch_init_kwargs["features_per_stage"]
        )
        # get encoder parameters
        self.stages = {
            f"encoder.stages.{i}": {
                "ds": downsamplig_stages[i],
                "feat": features_stages[i],
            }
            for i in range(len(self.network.encoder.stages))
        }

        for name, submodule in self.network.named_modules():
            if name in self.stages:
                submodule.register_forward_hook(self.hook_creator(name))

        print(self.forward_representations.keys())
        self.img_representations: dict[str, RepresentationHandler] = {}

    def _internal_predict_sliding_window_return_logits(
        self,
        data: torch.Tensor,
        slicers: Iterable[tuple[slice, ...]],
        do_on_device: bool = True,
    ):
        # get clean forward representations again for current image / fold
        self.forward_representations = {}
        if len(slicers) == 2:
            print(slicers)
        for key in self.stages:
            self.img_representations[key] = RepresentationHandler(
                input_shape=list(data.shape[1:]),
                scaling_factor=self.stages[key]["ds"],
                repr_dim=self.stages[key]["feat"],
            )
        out = super()._internal_predict_sliding_window_return_logits(
            data, slicers, do_on_device
        )

        for key in self.forward_representations:
            self.img_representations[key].init_representation()
            if len(self.img_representations[key].que) == len(slicers):
                for sl in slicers:
                    self.img_representations[key].update_representation(sl, 0)
            else:
                raise NotImplementedError(
                    "Length of representations greater than lenght of slices."
                )

        # free up space
        self.forward_representations = {}

        return out

    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        prediction = self.network(x)

        if mirror_axes is not None:
            # check for invalid numbers in mirror_axes
            # x should be 5d for 3d images and 4d for 2d. so the max value of mirror_axes cannot exceed len(x.shape) - 3
            assert (
                max(mirror_axes) <= x.ndim - 3
            ), "mirror_axes does not match the dimension of the input!"

            mirror_axes = [m + 2 for m in mirror_axes]
            axes_combinations = [
                c
                for i in range(len(mirror_axes))
                for c in itertools.combinations(mirror_axes, i + 1)
            ]
            for axes in axes_combinations:
                prediction += torch.flip(self.network(torch.flip(x, axes)), axes)
            prediction /= len(axes_combinations) + 1

        return prediction

    def hook_creator(
        self, name: str
    ) -> Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], None]:

        def hook_fn(m: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
            if self.forward_representations.get(name) is None:
                self.forward_representations[name] = []
            self.forward_representations[name].append(output.to("cpu"))

        return hook_fn

    def setup_badge(self):
        self.network: unet.PlainConvUNet = self.network
        self.forward_representations: dict[str, list[torch.Tensor]] = {}
        # get encoder parameters
        # stage_names = [
        #     f"encoder.stages.{i}" for i in range(len(self.network.encoder.stages))
        # ]
        final_representation_output_name = [
            f"decoder.seg_layers.{len(self.network.decoder.seg_layers)-2}"
        ]

        for name, submodule in self.network.named_modules():
            if name in final_representation_output_name:
                submodule.register_forward_hook(self.hook_creator(name))
        # last layer
        # get gradients for
        # forward
        # out = self.network.decoder.seg_layers[-1](
        #     self.representations[final_representation_output_name]
        # )
        # parameters: weight and bias
        # how to do this: save save representations ing image space

        # get all weights and biases for each model
        # compute final layer forward pass
        # compute gradient


# if __name__ == "__main__":
#     nnactive_results_folder = Path(
#         "/home/c817h/Documents/projects/nnactive_project/nnActive_data/Dataset004_Hippocampus/nnActive_results/Dataset021_Hippocampus__patch-20_20_20__qs-20__unc-random-label2__seed-12345"
#     )
#     analysis = AnalyzeQueries.initialize_from_config_path(
#         nnactive_results_folder, loop_val=0
#     )
#     analysis.initialize_querymethods([DiversityQueryMethod])
#     analysis.predict_training_set_fold(0)
