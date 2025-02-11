import shutil
import subprocess
from pathlib import Path

import nnunetv2.paths
from loguru import logger

from nnactive.cli.registry import register_subcommand
from nnactive.config.struct import ActiveConfig, RuntimeConfig
from nnactive.data.conversion import convert_dataset_to_partannotated
from nnactive.nnunet.utils import get_raw_path
from nnactive.results.state import State
from nnactive.results.utils import get_results_folder

__standard_suffix_format = "__unc-{uncertainty}__seed-{seed}"


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


def setup_al(config: ActiveConfig, exp_id: int = None):
    config.group_dir().mkdir(exist_ok=True)
    if exp_id is None:
        state = State.next_free_state(config)
    else:
        state = State(name=config.name(), dataset_id=exp_id)
    state.save_state()
    config.save_id(state.dataset_id)
    return state


@register_subcommand("setup_experiment")
def main(
    config: ActiveConfig,
    runtime_config: RuntimeConfig,
    exp_id: int = None,
):
    config.set_nnunet_env()
    # Prepare new experiment state
    state = setup_al(config, exp_id)
    # Create partly annotated dataset
    convert_dataset_to_partannotated(
        base_id=config.base_id,
        target_id=state.dataset_id,
        full_images=0,
        num_patches=config.starting_budget_size,
        patch_size=config.patch_size,
        name_suffix=(
            config.pre_suffix + __standard_suffix_format.format(**config.to_str_dict())
        ),
        patch_kwargs={},
        strategy=config.starting_budget,
        seed=config.seed,
        additional_overlap=config.additional_overlap,
    )
    # Prepare partly annotated dataset for nnUNet training
    prepare_dset(runtime_config, state)

    # If experiment is given in 'queries_from_experiment' config entry:
    # - get corresponding nnUNet_raw directory
    # - copy the loop files to current nnUNet_raw/PrecomputedLoops
    # - replace the current loop_000.json file
    # - reset the loop (for the case that loop_000.json was different)
    if config.queries_from_experiment is not None:
        command = f"""
            nnactive util_get_experiment_dirs --experiment \
            {config.queries_from_experiment} | grep "nnUNet_raw: " | \
            grep -o "'[^']*'" \
        """
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            raise ValueError(result.stderr)
        else:
            path = Path(result.stdout.strip().strip("'"))
            loop_files = [f for f in path.iterdir() if f.name.startswith("loop_")]
            raw_data_path = get_raw_path(state.dataset_id)
            target_path = raw_data_path / "PrecomputedLoops"
            target_path.mkdir(exist_ok=True)
            logger.info(
                f"Copying the following loop files to {str(target_path)}: "
                f"{', '.join([str(f.name) for f in loop_files])}"
            )
            for loop_file in loop_files:
                shutil.copy(loop_file, target_path)

        (raw_data_path / "loop_000.json").unlink()
        shutil.copy(target_path / "loop_000.json", raw_data_path)
        subprocess.run(
            f"nnactive util_reset_loops --nnActive_results_folder={str(get_results_folder(state.dataset_id))} --loop=0",
            shell=True,
        )
        if result.returncode != 0:
            raise ValueError(result.stderr)
