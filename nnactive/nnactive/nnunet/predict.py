import os
from typing import Union

import torch
from batchgenerators.utilities.file_and_folder_operations import isdir, maybe_mkdir_p
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.file_path_utilities import get_output_folder

from nnactive.logger import monitor


def predict_entry_point(
    input_folder: str,
    output_folder: str,
    dataset_id: int,
    train_identifier: str,
    configuration_identifier: str,
    plans_identifier: str = "nnUNetPlans",
    folds: Union[str, list[int], tuple[int]] = (0, 1, 2, 3, 4),
    step_size: float = 0.5,
    disable_tta: bool = False,
    verbose: bool = False,
    save_probabilities: bool = False,
    continue_prediction: bool = False,
    checkpoint: str = "checkpoint_final.pth",
    npp: int = 3,
    nps: int = 3,
    prev_stage_predictions: str = None,
    num_parts: int = 1,
    part_id: int = 0,
    device: Union[torch.device, str, int] = "cuda",
    disable_progress_bar: bool = False,
):
    folds = [i if i == "all" else int(i) for i in folds]

    model_folder = get_output_folder(
        dataset_id, train_identifier, plans_identifier, configuration_identifier
    )

    predict_from_model_folder(
        input_folder,
        output_folder,
        model_folder,
        folds,
        step_size,
        disable_tta,
        verbose,
        save_probabilities,
        continue_prediction,
        checkpoint,
        npp,
        nps,
        prev_stage_predictions,
        num_parts,
        part_id,
        device,
        disable_progress_bar,
    )


def predict_from_model_folder(
    input_folder: str,
    output_folder: str,
    model_folder: str,
    folds: Union[str, list[int], tuple[int]] = (0, 1, 2, 3, 4),
    step_size: float = 0.5,
    disable_tta: bool = False,
    verbose: bool = False,
    save_probabilities: bool = False,
    continue_prediction: bool = False,
    checkpoint: str = "checkpoint_final.pth",
    npp: int = 3,
    nps: int = 3,
    prev_stage_predictions: str = None,
    num_parts: int = 1,
    part_id: int = 0,
    device: Union[torch.device, str, int] = "cuda",
    disable_progress_bar: bool = False,
):
    if not isdir(output_folder):
        maybe_mkdir_p(output_folder)

    # slightly passive aggressive haha
    assert (
        part_id < num_parts
    ), "Do you even read the documentation? See nnUNetv2_predict -h."

    if isinstance(device, torch.device):
        if device.type == "cuda":
            torch.set_default_device(device)
    else:
        assert device in [
            "cpu",
            "cuda",
            "mps",
        ], f"-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {device}."
        if device == "cpu":
            # let's allow torch to use hella threads
            import multiprocessing

            torch.set_num_threads(multiprocessing.cpu_count())
            device = torch.device("cpu")
        elif device == "cuda":
            # multithreading in torch doesn't help nnU-Net if run on GPU
            try:
                os.environ["torchset"]
            except KeyError:
                torch.set_num_threads(1)
                torch.set_num_interop_threads(1)
                os.environ["torchset"] = "True"
            device = torch.device("cuda")
        else:
            device = torch.device("mps")

    predictor = nnUNetPredictor(
        tile_step_size=step_size,
        use_gaussian=True,
        use_mirroring=not disable_tta,
        perform_everything_on_device=True,
        device=device,
        verbose=verbose,
        verbose_preprocessing=verbose,
        allow_tqdm=not disable_progress_bar,
        log_times=monitor if monitor.is_active() else None,
    )
    predictor.initialize_from_trained_model_folder(
        model_folder, folds, checkpoint_name=checkpoint
    )
    predictor.predict_from_files(
        input_folder,
        output_folder,
        save_probabilities=save_probabilities,
        overwrite=not continue_prediction,
        num_processes_preprocessing=npp,
        num_processes_segmentation_export=nps,
        folder_with_segs_from_prev_stage=prev_stage_predictions,
        num_parts=num_parts,
        part_id=part_id,
    )
