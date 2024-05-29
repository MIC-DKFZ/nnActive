import shutil
from typing import List, Tuple, Union

import nnunetv2.paths
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    maybe_mkdir_p,
    subfiles,
)
from loguru import logger
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from nnactive.cli.registry import register_subcommand
from nnactive.config.struct import ActiveConfig, RuntimeConfig
from nnactive.nnunet.preprocessor import nnActivePreprocessor
from nnactive.results.state import State
from nnactive.results.utils import get_results_folder


@register_subcommand("nnunet_preprocess")
def preprocess(
    config: ActiveConfig,
    runtime_config: RuntimeConfig,
    continue_id: int | None = None,
    verbose: bool = False,
    do_all: bool = False,
    force: bool = False,
) -> None:
    config.set_nnunet_env()

    if continue_id is None:
        state = State.latest(config)
    else:
        state = State.get_id_state(continue_id)

    num_processes = [runtime_config.num_processes]
    configurations = [config.model_config]
    if not isinstance(num_processes, list):
        num_processes = list(num_processes)
    if len(num_processes) == 1:
        num_processes = num_processes * len(configurations)
    if len(num_processes) != len(configurations):
        raise RuntimeError(
            f"The list provided with num_processes must either have len 1 or as many elements as there are "
            f"configurations (see --help). Number of configurations: {len(configurations)}, length "
            f"of num_processes: "
            f"{len(num_processes)}"
        )

    dataset_name = convert_id_to_dataset_name(state.dataset_id)
    logger.info(f"Preprocessing dataset {dataset_name}")
    plans_file = join(
        nnunetv2.paths.nnUNet_preprocessed, dataset_name, config.model_plans + ".json"
    )
    plans_manager = PlansManager(plans_file)
    for n, c in zip(num_processes, configurations):
        logger.info(f"Configuration: {c}...")
        if c not in plans_manager.available_configurations:
            raise FileNotFoundError(
                f"INFO: Configuration {c} not found in plans file {config.model_plans + '.json'} of "
                f"dataset {dataset_name}. Skipping."
            )
            continue
        configuration_manager = plans_manager.get_configuration(c)
        preprocessor = nnActivePreprocessor(verbose=verbose)
        preprocessor.run(
            state.dataset_id, c, config.model_plans, num_processes=n, do_all=do_all
        )
    maybe_mkdir_p(
        join(nnunetv2.paths.nnUNet_preprocessed, dataset_name, "gt_segmentations")
    )
    [
        shutil.copy(
            i,
            join(
                join(
                    nnunetv2.paths.nnUNet_preprocessed, dataset_name, "gt_segmentations"
                )
            ),
        )
        for i in subfiles(join(nnunetv2.paths.nnUNet_raw, dataset_name, "labelsTr"))
    ]

    if not force:
        state.preprocess = True
        state.save_state()
