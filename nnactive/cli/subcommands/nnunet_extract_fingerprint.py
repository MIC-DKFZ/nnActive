import shutil
from argparse import Namespace
from typing import List, Optional, Tuple, Type, Union

import nnunetv2
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.configuration import default_num_processes
from nnunetv2.experiment_planning.dataset_fingerprint.fingerprint_extractor import (
    DatasetFingerprintExtractor,
)
from nnunetv2.experiment_planning.verify_dataset_integrity import (
    verify_dataset_integrity,
)
from nnunetv2.paths import nnUNet_raw
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name

from nnactive.cli.registry import register_subcommand
from nnactive.data.resampling import resample_dataset
from nnactive.nnunet.fingerprint_extractor import NNActiveDatasetFingerprintExtractor
from nnactive.nnunet.utils import get_preprocessed_path, get_raw_path, read_dataset_json


def extract_fingerprint_dataset(
    dataset_id: int,
    fingerprint_extractor_class: Type[
        DatasetFingerprintExtractor
    ] = NNActiveDatasetFingerprintExtractor,
    num_processes: int = default_num_processes,
    check_dataset_integrity: bool = False,
    clean: bool = True,
    verbose: bool = True,
):
    """
    Returns the fingerprint as a dictionary (additionally to saving it)
    """
    dataset_name = convert_id_to_dataset_name(dataset_id)
    print(dataset_name)

    if check_dataset_integrity:
        verify_dataset_integrity(join(nnUNet_raw, dataset_name), num_processes)

    fpe = fingerprint_extractor_class(dataset_id, num_processes, verbose=verbose)

    return fpe.run(overwrite_existing=clean)


@register_subcommand(
    "nnunet_extract_fingerprint",
    [
        (("-d", "--dataset_id"), {"type": int}),
        (
            ("-np"),
            {
                "type": int,
                "default": default_num_processes,
                "required": False,
                "help": f"[OPTIONAL] Number of processes used for fingerprint extraction. "
                f"Default: {default_num_processes}",
            },
        ),
        (
            ("--verify_dataset_integrity"),
            {
                "required": False,
                "default": False,
                "action": "store_true",
                "help": "[RECOMMENDED] set this flag to check the dataset integrity. This is useful and should be done once for "
                "each dataset!",
            },
        ),
        (
            ("--clean"),
            {
                "required": False,
                "default": False,
                "action": "store_true",
                "help": "[OPTIONAL] Set this flag to overwrite existing fingerprints. If not set and a fingerprint exists, the extractor won't run.",
            },
        ),
        (
            ("--verbose"),
            {
                "required": False,
                "action": "store_true",
                "help": "Set this to print a lot of stuff. Useful for debugging. Disables the progress bar! Recommended for clusters.",
            },
        ),
    ],
)
def nnunet_extract_fingerprint(args: Namespace) -> None:
    dataset_id: int = args.dataset_id
    np: int = args.np
    verify_dataset_integrity: bool = args.verify_dataset_integrity
    clean: bool = args.clean
    verbose: bool = args.verbose

    extract_fingerprint_dataset(
        dataset_id,
        fingerprint_extractor_class=NNActiveDatasetFingerprintExtractor,
        num_processes=np,
        check_dataset_integrity=verify_dataset_integrity,
        clean=clean,
        verbose=verbose,
    )
