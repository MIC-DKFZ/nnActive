import inspect
import itertools
import multiprocessing
import os
import traceback
from copy import deepcopy
from time import sleep
from typing import List, Optional, Tuple, Union

import nnunetv2
import numpy as np
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import (
    isdir,
    isfile,
    join,
    load_json,
    maybe_mkdir_p,
    save_json,
    subdirs,
)
from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.data_iterators import (
    PreprocessAdapterFromNpy,
    preprocessing_iterator_fromfiles,
    preprocessing_iterator_fromnpy,
)
from nnunetv2.inference.export_prediction import (
    convert_predicted_logits_to_segmentation_with_correct_shape,
    export_prediction_from_logits,
)
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import (
    compute_gaussian,
    compute_steps_for_sliding_window,
)
from nnunetv2.utilities.file_path_utilities import (
    check_workers_alive_and_busy,
    get_output_folder,
)
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.helpers import dummy_context, empty_cache
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.label_handling.label_handling import (
    determine_num_input_channels,
)
from nnunetv2.utilities.plans_handling.plans_handler import (
    ConfigurationManager,
    PlansManager,
)
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder


def predict_entry_point():
    import argparse

    parser = argparse.ArgumentParser(
        description="Use this to run inference with nnU-Net. This function is used when "
        "you want to manually specify a folder containing a trained nnU-Net "
        "model. This is useful when the nnunet environment variables "
        "(nnUNet_results) are not set."
    )
    parser.add_argument(
        "-i",
        type=str,
        required=True,
        help="input folder. Remember to use the correct channel numberings for your files (_0000 etc). "
        "File endings must be the same as the training dataset!",
    )
    parser.add_argument(
        "-o",
        type=str,
        required=True,
        help="Output folder. If it does not exist it will be created. Predicted segmentations will "
        "have the same name as their source images.",
    )
    parser.add_argument(
        "-d",
        type=str,
        required=True,
        help="Dataset with which you would like to predict. You can specify either dataset name or id",
    )
    parser.add_argument(
        "-p",
        type=str,
        required=False,
        default="nnUNetPlans",
        help="Plans identifier. Specify the plans in which the desired configuration is located. "
        "Default: nnUNetPlans",
    )
    parser.add_argument(
        "-tr",
        type=str,
        required=False,
        default="nnUNetTrainer",
        help="What nnU-Net trainer class was used for training? Default: nnUNetTrainer",
    )
    parser.add_argument(
        "-c",
        type=str,
        required=True,
        help="nnU-Net configuration that should be used for prediction. Config must be located "
        "in the plans specified with -p",
    )
    parser.add_argument(
        "-f",
        nargs="+",
        type=str,
        required=False,
        default=(0, 1, 2, 3, 4),
        help="Specify the folds of the trained model that should be used for prediction. "
        "Default: (0, 1, 2, 3, 4)",
    )
    parser.add_argument(
        "-step_size",
        type=float,
        required=False,
        default=0.5,
        help="Step size for sliding window prediction. The larger it is the faster but less accurate "
        "the prediction. Default: 0.5. Cannot be larger than 1. We recommend the default.",
    )
    parser.add_argument(
        "--disable_tta",
        action="store_true",
        required=False,
        default=False,
        help="Set this flag to disable test time data augmentation in the form of mirroring. Faster, "
        "but less accurate inference. Not recommended.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Set this if you like being talked to. You will have "
        "to be a good listener/reader.",
    )
    parser.add_argument(
        "--save_probabilities",
        action="store_true",
        help='Set this to export predicted class "probabilities". Required if you want to ensemble '
        "multiple configurations.",
    )
    parser.add_argument(
        "--continue_prediction",
        action="store_true",
        help="Continue an aborted previous prediction (will not overwrite existing files)",
    )
    parser.add_argument(
        "-chk",
        type=str,
        required=False,
        default="checkpoint_final.pth",
        help="Name of the checkpoint you want to use. Default: checkpoint_final.pth",
    )
    parser.add_argument(
        "-npp",
        type=int,
        required=False,
        default=3,
        help="Number of processes used for preprocessing. More is not always better. Beware of "
        "out-of-RAM issues. Default: 3",
    )
    parser.add_argument(
        "-nps",
        type=int,
        required=False,
        default=3,
        help="Number of processes used for segmentation export. More is not always better. Beware of "
        "out-of-RAM issues. Default: 3",
    )
    parser.add_argument(
        "-prev_stage_predictions",
        type=str,
        required=False,
        default=None,
        help="Folder containing the predictions of the previous stage. Required for cascaded models.",
    )
    parser.add_argument(
        "-num_parts",
        type=int,
        required=False,
        default=1,
        help="Number of separate nnUNetv2_predict call that you will be making. Default: 1 (= this one "
        "call predicts everything)",
    )
    parser.add_argument(
        "-part_id",
        type=int,
        required=False,
        default=0,
        help="If multiple nnUNetv2_predict exist, which one is this? IDs start with 0 can end with "
        "num_parts - 1. So when you submit 5 nnUNetv2_predict calls you need to set -num_parts "
        "5 and use -part_id 0, 1, 2, 3 and 4. Simple, right? Note: You are yourself responsible "
        "to make these run on separate GPUs! Use CUDA_VISIBLE_DEVICES (google, yo!)",
    )
    parser.add_argument(
        "-device",
        type=str,
        default="cuda",
        required=False,
        help="Use this to set the device the inference should run with. Available options are 'cuda' "
        "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
        "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_predict [...] instead!",
    )
    parser.add_argument(
        "--disable_progress_bar",
        action="store_true",
        required=False,
        default=False,
        help="Set this flag to disable progress bar. Recommended for HPC environments (non interactive "
        "jobs)",
    )

    print(
        "\n#######################################################################\nPlease cite the following paper "
        "when using nnU-Net:\n"
        "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
        "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
        "Nature methods, 18(2), 203-211.\n#######################################################################\n"
    )

    args = parser.parse_args()
    args.f = [i if i == "all" else int(i) for i in args.f]

    model_folder = get_output_folder(args.d, args.tr, args.p, args.c)

    if not isdir(args.o):
        maybe_mkdir_p(args.o)

    # slightly passive aggressive haha
    assert (
        args.part_id < args.num_parts
    ), "Do you even read the documentation? See nnUNetv2_predict -h."

    assert args.device in [
        "cpu",
        "cuda",
        "mps",
    ], f"-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}."
    if args.device == "cpu":
        # let's allow torch to use hella threads
        import multiprocessing

        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device("cpu")
    elif args.device == "cuda":
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device("cuda")
    else:
        device = torch.device("mps")

    predictor = nnUNetPredictor(
        tile_step_size=args.step_size,
        use_gaussian=True,
        use_mirroring=not args.disable_tta,
        perform_everything_on_device=True,
        device=device,
        verbose=args.verbose,
        verbose_preprocessing=args.verbose,
        allow_tqdm=not args.disable_progress_bar,
    )
    predictor.initialize_from_trained_model_folder(
        model_folder, args.f, checkpoint_name=args.chk
    )
    predictor.predict_from_files(
        args.i,
        args.o,
        save_probabilities=args.save_probabilities,
        overwrite=not args.continue_prediction,
        num_processes_preprocessing=args.npp,
        num_processes_segmentation_export=args.nps,
        folder_with_segs_from_prev_stage=args.prev_stage_predictions,
        num_parts=args.num_parts,
        part_id=args.part_id,
    )
    # r = predict_from_raw_data(args.i,
    #                           args.o,
    #                           model_folder,
    #                           args.f,
    #                           args.step_size,
    #                           use_gaussian=True,
    #                           use_mirroring=not args.disable_tta,
    #                           perform_everything_on_device=True,
    #                           verbose=args.verbose,
    #                           save_probabilities=args.save_probabilities,
    #                           overwrite=not args.continue_prediction,
    #                           checkpoint_name=args.chk,
    #                           num_processes_preprocessing=args.npp,
    #                           num_processes_segmentation_export=args.nps,
    #                           folder_with_segs_from_prev_stage=args.prev_stage_predictions,
    #                           num_parts=args.num_parts,
    #                           part_id=args.part_id,
    #                           device=device)
