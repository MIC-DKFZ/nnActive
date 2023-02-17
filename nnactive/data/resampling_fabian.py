# Code by Fabian Isensee
from multiprocessing import Pool

import numpy as np
import SimpleITK as sitk
import torch
from acvl_utils.array_manipulation.resampling import maybe_resample_on_gpu
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape
from torch.nn import functional as F


def resample_save(
    source_image: str,
    source_label: str,
    target_image: str,
    target_label: str,
    target_spacing: tuple[float, ...] = (0.3, 0.3, 0.3),
    skip_existing: bool = True,
    export_pool: Pool = None,
):
    print(f"{os.path.basename(source_image)}")
    if skip_existing and isfile(target_label) and isfile(target_image):
        return None, None

    seg_source = sitk.GetArrayFromImage(sitk.ReadImage(source_label)).astype(np.uint8)
    im_source = sitk.ReadImage(source_image)

    source_spacing = im_source.GetSpacing()
    source_origin = im_source.GetOrigin()
    source_direction = im_source.GetDirection()

    im_source = sitk.GetArrayFromImage(im_source).astype(np.float32)
    source_shape = im_source.shape

    # resample image
    target_shape = compute_new_shape(
        source_shape, list(source_spacing)[::-1], target_spacing
    )

    print(f"source shape: {source_shape}, target shape {target_shape}")

    # one hot generation is slow af. Let's do it this way:
    seg_source = torch.from_numpy(seg_source)
    seg_onehot_target_shape = None
    seg_source_gpu = None
    try:
        torch.cuda.empty_cache()
        device = "cuda:0"
        # having the target array on device will blow up, so we need to have this on CPU
        with torch.no_grad():
            seg_source_gpu = seg_source.to(device)
            seg_onehot_target_shape = F.interpolate(
                seg_source_gpu.half()[None, None], tuple(target_shape), mode="trilinear"
            )[0, 0].cpu()
        del seg_source_gpu
    except RuntimeError:
        print(
            "GPU wasnt happy with this resampling. Lets give the CPU a chance to sort it out"
        )
        print(f"source shape {source_shape}, target shape {target_shape}")
        del seg_source_gpu
        device = "cpu"
        with torch.no_grad():
            seg_onehot_target_shape = F.interpolate(
                seg_source.to(device).float()[None, None],
                tuple(target_shape),
                mode="trilinear",
            )[0, 0].cpu()
    finally:
        torch.cuda.empty_cache()

    seg_onehot_target_shape = (seg_onehot_target_shape > 0.5).numpy().astype(np.uint8)

    seg_target_itk = sitk.GetImageFromArray(seg_onehot_target_shape)
    seg_target_itk.SetSpacing(tuple(list(target_spacing)[::-1]))
    seg_target_itk.SetOrigin(source_origin)
    seg_target_itk.SetDirection(source_direction)

    # now resample images. For simplicity, just make this linear
    im_source = (
        maybe_resample_on_gpu(
            torch.from_numpy(im_source[None]),
            tuple(target_shape),
            return_type=torch.float,
            compute_precision=torch.float,
            fallback_compute_precision=float,
        )[0]
        .cpu()
        .numpy()
    )

    # export image
    im_target = sitk.GetImageFromArray(im_source)
    im_target.SetSpacing(tuple(list(target_spacing)[::-1]))
    im_target.SetOrigin(source_origin)
    im_target.SetDirection(source_direction)

    if export_pool is None:
        sitk.WriteImage(im_target, target_image)
        sitk.WriteImage(seg_target_itk, target_label)
        return None, None
    else:
        r1 = export_pool.starmap_async(sitk.WriteImage, ((im_target, target_image),))
        r2 = export_pool.starmap_async(
            sitk.WriteImage, ((seg_target_itk, target_label),)
        )
        return r1, r2
