import os
import shutil
import subprocess
from pathlib import Path

from nnactive.cli.registry import register_subcommand
from nnactive.cli.subcommands.convert_to_partannotated import (
    convert_dataset_to_partannotated,
)
from nnactive.config.struct import ActiveConfig, RuntimeConfig
from nnactive.paths import nnActive_data
from nnactive.results.state import State


def base_dataset_name(dataset_id: int):
    for folder in data_path().iterdir():
        if folder.name.startswith(f"Dataset{dataset_id:03d}"):
            return folder.name
    raise RuntimeError(f"No Dataset with corresponding base id: {dataset_id}")


def data_path():
    data_path = os.getenv("nnActive_raw")
    if data_path is None:
        raise ValueError("OS variable nnUNet_raw is not set.")
    return Path(data_path)


def existing_dsets():
    existing_dsets = [
        folder.name
        for folder in nnActive_data.iterdir()
        if folder.is_dir() and folder.name.startswith("Dataset")
    ]
    return existing_dsets


def check_dataset_id(output_id: int, force_override: bool):
    dset_name = f"Dataset{output_id:03d}"
    if any([dset.startswith(dset_name) for dset in existing_dsets()]):
        print(f"Dataset beginning with '{dset_name}' already exists in {data_path()}.")
        if not force_override:
            return False
        else:
            os_variables = [
                "nnUNet_results",
                "nnUNet_raw",
                "nnUNet_preprocessed",
                "nnActive_results",
            ]
            for os_variable in os_variables:
                base_path = Path(os.getenv(os_variable))
                rm_dirs = [
                    folder
                    for folder in base_path.iterdir()
                    if folder.name.startswith(f"Dataset{output_id:03d}")
                ]
                for rm_dir in rm_dirs:
                    print(f"Deleting folder: {rm_dir}")
                    shutil.rmtree(rm_dir)
    return True


def convert_dset(config: ActiveConfig, state: State):
    print("Converting Dataset")

    past_suffix = f"__unc-{config.uncertainty}__seed-{config.seed}"
    name_suffix = config.pre_suffix + past_suffix

    convert_dataset_to_partannotated(
        config.base_id,
        state.dataset_id,
        0,
        config.starting_budget_size,
        config.patch_size,
        name_suffix,
        {},
        config.starting_budget,
        config.seed,
        config.additional_overlap,
    )


def prepare_dset(config: RuntimeConfig, state: State):
    subprocess.run(
        f"nnUNetv2_extract_fingerprint -d {state.dataset_id}  -np {config.num_processes}",
        shell=True,
        check=True,
    )
    subprocess.run(
        f"nnUNetv2_plan_experiment -d {state.dataset_id} -np {config.num_processes}",
        shell=True,
        check=True,
    )


def setup_al(config: ActiveConfig):
    config.group_dir().mkdir(exist_ok=True)
    state = State.next_free_state(config)
    state.save_state()
    config.save_id(state.dataset_id)
    return state


@register_subcommand("setup")
def main(
    config: ActiveConfig,
    runtime_config: RuntimeConfig,
    debug: bool = False,
    force: bool = False,
):
    config.set_nnunet_env()
    state = setup_al(config)
    convert_dset(config, state)
    prepare_dset(runtime_config, state)
