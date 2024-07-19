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
from nnactive.strategies.base import BasePredictionQuery, BaseQueryPredictor
from nnactive.strategies.utils import RepresentationHandler
from nnactive.utils.io import load_label_map


class DiversityQueryMethod(BasePredictionQuery):

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


class DiversityPredictor(BaseQueryPredictor):
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
        self.

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
