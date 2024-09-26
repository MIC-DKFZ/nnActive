import subprocess

import nnunetv2.paths

from nnactive.cli.registry import register_subcommand
from nnactive.config.struct import ActiveConfig, RuntimeConfig
from nnactive.data.conversion import convert_dataset_to_partannotated
from nnactive.results.state import State

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


def setup_al(config: ActiveConfig):
    config.group_dir().mkdir(exist_ok=True)
    state = State.next_free_state(config)
    state.save_state()
    config.save_id(state.dataset_id)
    return state


@register_subcommand("setup_experiment")
def main(
    config: ActiveConfig,
    runtime_config: RuntimeConfig,
):
    config.set_nnunet_env()
    # Prepare new experiment state
    state = setup_al(config)
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
