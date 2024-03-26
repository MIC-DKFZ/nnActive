import os
import shutil
import subprocess
from pathlib import Path

from nnactive.cli.registry import register_subcommand
from nnactive.cli.subcommands.convert_to_partannotated import (
    convert_dataset_to_partannotated,
)
from nnactive.config.struct import ActiveConfig
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


def convert_dset(config: ActiveConfig):
    print("Converting Dataset")

    past_suffix = f"__unc-{config.uncertainty}__seed-{config.seed}"
    name_suffix = config.pre_suffix + past_suffix

    convert_dataset_to_partannotated(
        config.base_id,
        config.id,
        0,
        config.query_size,
        config.patch_size,
        name_suffix,
        {},
        config.starting_budget,
        config.seed,
        config.additional_overlap,
    )


def prepare_dset(config: ActiveConfig):
    subprocess.run(
        f"nnUNetv2_extract_fingerprint -d {config.id}  -np {config.num_processes}",
        shell=True,
        check=True,
    )
    subprocess.run(
        f"nnUNetv2_plan_experiment -d {config.id} -np {config.num_processes}",
        shell=True,
        check=True,
    )


def setup_al(config: ActiveConfig):
    state = State(dataset_id=config.id)
    state.save_state()


@register_subcommand("setup")
def main(config: ActiveConfig, debug: bool = False, force: bool = False):
    config.set_nnunet_env()
    if check_dataset_id(config.id, force):
        if debug:
            print(f"Creating Dataset{config.id:3d}....")
            return
        convert_dset(config)
        prepare_dset(config)
        config.save_id(config.id)
        setup_al(config)
