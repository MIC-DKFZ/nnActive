# TODO: Split this into two different files, s.a. in other folders
import json
import os
import shutil
import subprocess
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw, nnUNet_results
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name

from nnactive.data.annotate import create_labels_from_patches
from nnactive.data.create_empty_masks import (
    add_ignore_label_to_dataset_json,
    read_dataset_json,
)
from nnactive.loops.loading import save_loop
from nnactive.nnunet.io import generate_custom_splits_file
from nnactive.nnunet.utils import get_patch_size
from nnactive.results.utils import (
    convert_id_to_dataset_name as nnactive_id_to_dataset_name,
)
from nnactive.strategies import init_strategy
from nnactive.utils.hostutils import get_verbose

NNUNET_RAW = Path(nnUNet_raw) if nnUNet_raw is not None else None
NNUNET_PREPROCESSED = (
    Path(nnUNet_preprocessed) if nnUNet_preprocessed is not None else None
)
NNUNET_RESULTS = Path(nnUNet_results) if nnUNet_results is not None else None


def main():
    fingerprint_call = "nnUNetv2_extract_fingerprint -d {} --verify_dataset_integrity"
    plan_call = "nnUNetv2_plan_experiment -d {}"
    dataset_id = 138
    subprocess.run(fingerprint_call.format(dataset_id), shell=True, check=True)
    subprocess.run(plan_call.format(dataset_id), shell=True, check=True)

    # load here the plans and fingerprint


    

    
    
    resampled_raw_dir = NNUNET_RAW/
    


if __name__ == "__main__":
    main()
