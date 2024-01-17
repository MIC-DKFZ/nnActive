import shutil
from typing import List, Tuple, Union

from batchgenerators.utilities.file_and_folder_operations import (
    join,
    maybe_mkdir_p,
    subfiles,
)
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from nnactive.cli.registry import register_subcommand
from nnactive.nnunet.preprocessor import nnActivePreprocessor
from nnactive.results.state import State


def preprocess_dataset(
    dataset_id: int,
    plans_identifier: str = "nnUNetPlans",
    configurations: Union[Tuple[str], List[str]] = ("2d", "3d_fullres", "3d_lowres"),
    num_processes: Union[int, Tuple[int, ...], List[int]] = (8, 4, 8),
    verbose: bool = False,
    do_all: bool = False,
    force: bool = False,
) -> None:
    try:
        state = State.get_id_state(dataset_id, verify=not force)
    except FileNotFoundError:
        state = State(dataset_id=dataset_id, loop=0)

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

    dataset_name = convert_id_to_dataset_name(dataset_id)
    print(f"Preprocessing dataset {dataset_name}")
    plans_file = join(nnUNet_preprocessed, dataset_name, plans_identifier + ".json")
    plans_manager = PlansManager(plans_file)
    for n, c in zip(num_processes, configurations):
        print(f"Configuration: {c}...")
        if c not in plans_manager.available_configurations:
            raise FileNotFoundError(
                f"INFO: Configuration {c} not found in plans file {plans_identifier + '.json'} of "
                f"dataset {dataset_name}. Skipping."
            )
            continue
        configuration_manager = plans_manager.get_configuration(c)
        preprocessor = nnActivePreprocessor(verbose=verbose)
        preprocessor.run(
            dataset_id, c, plans_identifier, num_processes=n, do_all=do_all
        )
    maybe_mkdir_p(join(nnUNet_preprocessed, dataset_name, "gt_segmentations"))
    [
        shutil.copy(
            i, join(join(nnUNet_preprocessed, dataset_name, "gt_segmentations"))
        )
        for i in subfiles(join(nnUNet_raw, dataset_name, "labelsTr"))
    ]

    if not force:
        state.preprocess = True
        state.save_state()


def preprocess(
    dataset_ids: List[int],
    plans_identifier: str = "nnUNetPlans",
    configurations: Union[Tuple[str], List[str]] = ("2d", "3d_fullres", "3d_lowres"),
    num_processes: Union[int, Tuple[int, ...], List[int]] = (8, 4, 8),
    verbose: bool = False,
    do_all: bool = False,
    force: bool = False,
):
    for d in dataset_ids:
        preprocess_dataset(
            d, plans_identifier, configurations, num_processes, verbose, do_all, force
        )


@register_subcommand(
    "nnunet_preprocess",
    [
        (
            ("-d"),
            {
                "nargs": "+",
                "type": int,
                "required": True,
                "help": "[REQUIRED] List of dataset IDs. Example: 2 4 5. This will run fingerprint extraction, experiment "
                "planning and preprocessing for these datasets. Can of course also be just one dataset",
            },
        ),
        (
            ("-plans_name"),
            {
                "default": "nnUNetPlans",
                "type": str,
                "required": False,
                "help": "[OPTIONAL] You can use this to specify a custom plans file that you may have generated",
            },
        ),
        (
            ("-c"),
            {
                "default": ["2d", "3d_fullres", "3d_lowres"],
                "nargs": "+",
                "type": str,
                "required": False,
                "help": "[OPTIONAL] Configurations for which the preprocessing should be run. Default: 2d 3f_fullres "
                "3d_lowres. 3d_cascade_fullres does not need to be specified because it uses the data "
                "from 3f_fullres. Configurations that do not exist for some dataset will be skipped.",
            },
        ),
        (
            ("-np"),
            {
                "default": [8, 4, 8],
                "nargs": "+",
                "type": int,
                "required": False,
                "help": "[OPTIONAL] Use this to define how many processes are to be used. If this is just one number then "
                "this number of processes is used for all configurations specified with -c. If it's a "
                "list of numbers this list must have as many elements as there are configurations. We "
                "then iterate over zip(configs, num_processes) to determine then umber of processes "
                "used for each configuration. More processes is always faster (up to the number of "
                "threads your PC can support, so 8 for a 4 core CPU with hyperthreading. If you don't "
                "know what that is then dont touch it, or at least don't increase it!). DANGER: More "
                "often than not the number of processes that can be used is limited by the amount of "
                "RAM available. Image resampling takes up a lot of RAM. MONITOR RAM USAGE AND "
                "DECREASE -np IF YOUR RAM FILLS UP TOO MUCH!. Default: 8 processes for 2d, 4 "
                "for 3d_fullres, 8 for 3d_lowres and 4 for everything else",
            },
        ),
        (
            ("--do_all"),
            {
                "action": "store_true",
                "required": False,
                "help": "Set this Flag to True if the data for all loops instead of just most recent oneshould be preprocessed.",
            },
        ),
        (
            ("--verbose"),
            {
                "action": "store_true",
                "required": False,
                "help": "Set this to print a lot of stuff. Useful for debugging. Will disable progrewss bar! "
                "Recommended for cluster environments",
            },
        ),
        (
            ("--force"),
            {
                "action": "store_true",
                "required": False,
                "help": "Set this to force using this command without taking the state.json of the dataset into account! "
                "Be Careufl with this one!",
            },
        ),
    ],
)
def main(args):
    if args.np is None:
        default_np = {"2d": 4, "3d_lowres": 8, "3d_fullres": 4}
        np = {default_np[c] if c in default_np.keys() else 4 for c in args.c}
    else:
        np = args.np
    preprocess(
        args.d,
        args.plans_name,
        configurations=args.c,
        num_processes=np,
        verbose=args.verbose,
        do_all=args.do_all,
        force=args.force,
    )
