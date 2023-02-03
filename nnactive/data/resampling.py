from __future__ import annotations

import functools
import json
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
from nnunetv2.experiment_planning.plan_and_preprocess import PlansManager
from nnunetv2.preprocessing.preprocessors.default_preprocessor import (
    ConfigurationManager,
    DefaultPreprocessor,
)
from rich.progress import track


def resample_to_target_spacing(
    name: str,
    dataset_cfg: dict[str, Any],
    rs_img_path: Path,
    rs_gt_path: Path,
    preprocessor: DefaultPreprocessor,
    plans_manager: PlansManager,
    config_manager: ConfigurationManager,
    img_path: Path,
    gt_path: Path,
) -> None:
    """Convert an image to target spacing

    Args:
        name: name of the image file
        dataset_cfg: nnUNet dataset json
        rs_img_path: resampled images path
        rs_gt_path: resampled gt labels path
        img_path: images path
        gt_path: gt labels path
        preprocessor: nnUNet preprocessor object
        plans_manager: nnUNet PlansManager object
        config_manager: nnUNet ConfigurationManager object
    """
    input_images = [str(rs_img_path / f"{name}_0000{dataset_cfg['file_ending']}")]

    data, seg, properties = preprocessor.run_case(
        input_images,
        seg_file=str(rs_gt_path / f"{name}{dataset_cfg['file_ending']}"),
        plans_manager=plans_manager,
        configuration_manager=config_manager,
        dataset_json=dataset_cfg,
    )
    data = data.transpose(
        [0, *[i + 1 for i in plans_manager.transpose_backward]]
    ).squeeze()
    seg = seg.transpose(
        [0, *[i + 1 for i in plans_manager.transpose_backward]]
    ).squeeze()

    img_itk_new = sitk.GetImageFromArray(data)
    img_itk_new.SetSpacing(
        [config_manager.spacing[i] for i in plans_manager.transpose_backward]
    )
    img_itk_new.SetOrigin(properties["sitk_stuff"]["origin"])
    img_itk_new.SetDirection(np.array(properties["sitk_stuff"]["direction"]))
    sitk.WriteImage(
        img_itk_new,
        (img_path / name).with_suffix(dataset_cfg["file_ending"]),
    )

    img_itk_new = sitk.GetImageFromArray(seg)
    img_itk_new.SetSpacing(
        [config_manager.spacing[i] for i in plans_manager.transpose_backward]
    )
    img_itk_new.SetOrigin(properties["sitk_stuff"]["origin"])
    img_itk_new.SetDirection(np.array(properties["sitk_stuff"]["direction"]))
    sitk.WriteImage(
        img_itk_new,
        (gt_path / name).with_suffix(dataset_cfg["file_ending"]),
    )


def resample_dataset(
    dataset_cfg: dict[str, Any],
    rs_img_path: Path,
    rs_gt_path: Path,
    raw_path: Path,
    img_path: Path,
    gt_path: Path,
    preprocessed_path: Path,
    n_workers: int,
) -> None:
    """Convert all images and labels to target spacing

    Args:
        dataset_cfg: nnUNet dataset json
        rs_img_path: resampled images path
        rs_gt_path: resampled gt labels path
        raw_path: nnUNet raw path
        img_path: images path
        gt_path: gt labels path
        preprocessed_path: nnUNet preprocessed path
        n_workers: number of parallel processes
    """

    with open(raw_path / "dataset.json") as file:
        dataset_cfg = json.load(file)

    configuration = "3d_fullres"
    plans_file = preprocessed_path / "nnUNetPlans.json"

    with plans_file.open() as file:
        plans_cfg = json.load(file)

    preprocessor = DefaultPreprocessor(False)
    plans_manager = PlansManager(plans_cfg)
    config_manager = plans_manager.get_configuration(configuration)

    gt_path.mkdir(exist_ok=True)
    img_path.mkdir(exist_ok=True)
    imgs = list((raw_path / "rs_gt_labelsTr").glob(f"**/*{dataset_cfg['file_ending']}"))

    names = [path.name.replace(dataset_cfg["file_ending"], "") for path in imgs]

    worker_fn = functools.partial(
        resample_to_target_spacing,
        dataset_cfg=dataset_cfg,
        rs_img_path=rs_img_path,
        rs_gt_path=rs_gt_path,
        preprocessor=preprocessor,
        plans_manager=plans_manager,
        config_manager=config_manager,
        img_path=img_path,
        gt_path=gt_path,
    )
    try:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for _ in track(executor.map(worker_fn, names), total=len(names)):
                pass
    except BrokenProcessPool as exc:
        raise MemoryError(
            "One of the worker processes died. "
            "This usually happens because you run out of memory. "
            "Try running with less processes."
        ) from exc
