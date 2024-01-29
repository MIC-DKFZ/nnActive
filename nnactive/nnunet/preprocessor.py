import multiprocessing
import shutil
from pathlib import Path
from time import sleep
from typing import Union

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from loguru import logger
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.preprocessing.preprocessors.default_preprocessor import (
    DefaultPreprocessor,
)
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder
from tqdm import tqdm

from nnactive.loops.loading import get_loop_patches, get_patches_from_loop_files


class nnActivePreprocessor(DefaultPreprocessor):
    """Preprocessor for nnUNet which works identical to nnUNet DefaultPreprocessor.
    Only change is that only the cases for which annotations are in loop files noted are actually preprocessed.
    Further, it also allows to partially only preprocess the data which has been added in the newest loop_file.
    """

    def run(
        self,
        dataset_name_or_id: Union[int, str],
        configuration_name: str,
        plans_identifier: str,
        num_processes: int,
        do_all: bool = False,
    ):
        """
        data identifier = configuration name in plans. EZ.
        """
        dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)

        assert isdir(
            join(nnUNet_raw, dataset_name)
        ), "The requested dataset could not be found in nnUNet_raw"

        plans_file = join(nnUNet_preprocessed, dataset_name, plans_identifier + ".json")
        assert isfile(plans_file), (
            "Expected plans file (%s) not found. Run corresponding nnUNet_plan_experiment "
            "first." % plans_file
        )
        plans = load_json(plans_file)
        plans_manager = PlansManager(plans)
        configuration_manager = plans_manager.get_configuration(configuration_name)

        if self.verbose:
            logger.debug(
                f"Preprocessing the following configuration: {configuration_name}"
            )
        if self.verbose:
            logger.debug(configuration_manager)

        dataset_json_file = join(nnUNet_preprocessed, dataset_name, "dataset.json")
        dataset_json = load_json(dataset_json_file)

        loop_path = Path(nnUNet_raw) / dataset_name
        if do_all:
            patches = get_patches_from_loop_files(loop_path)
        else:
            patches = get_loop_patches(loop_path, None)
        identifiers = [
            patch.file.replace(dataset_json["file_ending"], "") for patch in patches
        ]
        output_directory = join(
            nnUNet_preprocessed, dataset_name, configuration_manager.data_identifier
        )

        if isdir(output_directory) and do_all:
            shutil.rmtree(output_directory)

        maybe_mkdir_p(output_directory)

        output_filenames_truncated = [join(output_directory, i) for i in identifiers]

        file_ending = dataset_json["file_ending"]
        # list of lists with image filenames
        image_fnames = create_lists_from_splitted_dataset_folder(
            join(nnUNet_raw, dataset_name, "imagesTr"), file_ending, identifiers
        )
        # list of segmentation filenames
        seg_fnames = [
            join(nnUNet_raw, dataset_name, "labelsTr", i + file_ending)
            for i in identifiers
        ]

        # multiprocessing magic.
        r = []
        with multiprocessing.get_context("spawn").Pool(num_processes) as p:
            for outfile, infiles, segfiles in zip(
                output_filenames_truncated, image_fnames, seg_fnames
            ):
                r.append(
                    p.starmap_async(
                        self.run_case_save,
                        (
                            (
                                outfile,
                                infiles,
                                segfiles,
                                plans_manager,
                                configuration_manager,
                                dataset_json,
                            ),
                        ),
                    )
                )
            remaining = list(range(len(output_filenames_truncated)))
            # p is pretty nifti. If we kill workers they just respawn but don't do any work.
            # So we need to store the original pool of workers.
            workers = [j for j in p._pool]
            with tqdm(
                desc=None, total=len(output_filenames_truncated), disable=self.verbose
            ) as pbar:
                while len(remaining) > 0:
                    all_alive = all([j.is_alive() for j in workers])
                    if not all_alive:
                        raise RuntimeError(
                            "Some background worker is 6 feet under. Yuck."
                        )
                    done = [i for i in remaining if r[i].ready()]
                    for _ in done:
                        pbar.update()
                    remaining = [i for i in remaining if i not in done]
                    sleep(0.1)
