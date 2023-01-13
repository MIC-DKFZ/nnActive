import json
import os
from multiprocessing import Pool, set_start_method
from pathlib import Path

import numpy as np
import numpy.typing as npt
import SimpleITK as sitk
from nnunetv2.experiment_planning.plan_and_preprocess import PlansManager
from nnunetv2.preprocessing.preprocessors.default_preprocessor import \
    DefaultPreprocessor
from rich.pretty import pprint
from rich.progress import Progress, track

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

raw_path = Path("~/Data/nnUNet/raw/Dataset135_KiTS2021/").expanduser()
preprocessed_path = Path("~/Data/nnUNet/preprocessed/Dataset135_KiTS2021/").expanduser()
gt_path = raw_path / "gt_labelsTr"
img_path = raw_path / "imagesTr"
rs_gt_path = raw_path / "rs_gt_labelsTr"
rs_img_path = raw_path / "rs_imagesTr"

with open(raw_path / "dataset.json") as f:
    dataset_cfg = json.load(f)
configuration = "3d_fullres"
plans_file = preprocessed_path / "nnUNetPlans.json"
with plans_file.open() as f:
    plans_cfg = json.load(f)

preprocessor = DefaultPreprocessor(False)

plans_manager = PlansManager(plans_cfg)

ignore_label = 4

patch_size = 128
n_samples = 32

crop_min_size = 0.4
crop_max_size = 0.9


config_manager = plans_manager.get_configuration(configuration)


def resample_to_target_spacing(name: str):
    """Convert an image to target spacing"""
    input_images = [str(rs_img_path / f"{name}_0000{dataset_cfg['file_ending']}")]
    input_seg = str(rs_gt_path / f"{name}{dataset_cfg['file_ending']}")

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


def resample_all():
    """Convert all images and labels to target spacing"""
    gt_path.mkdir(exist_ok=True)
    img_path.mkdir(exist_ok=True)
    imgs = list((raw_path / "rs_gt_labelsTr").glob(f"**/*{dataset_cfg['file_ending']}"))

    names = [path.name.replace(dataset_cfg["file_ending"], "") for path in imgs]
    with Pool(processes=6, maxtasksperchild=1) as pool:
        for _ in track(
            pool.imap(resample_to_target_spacing, names, chunksize=8),
            total=len(names),
        ):
            pass


def patch_ids_to_image_coords(patch_ids: list[int] | npt.NDArray, bins, files, sizes):
    # patch_id = x + y * xs + z * xs * ys
    # => z = patch_id // (xs * ys)
    # => temp = x + y * xs = patch_id % (xs * ys)
    # => y = temp // xs
    # => x = temp % xs

    img_ids = np.digitize(patch_ids, bins)
    coords = []
    for patch_id, img_id in zip(patch_ids, img_ids):

        patch_id = patch_id - bins[img_id - 1]

        x, y, _ = sizes[img_id]
        xs = x // patch_size
        ys = y // patch_size

        pz = patch_id // (xs * ys)
        tmp = patch_id % (xs * ys)
        py = tmp // xs
        px = tmp % xs

        coords.append(
            {
                "file": files[img_id],
                "coords": [px * patch_size, py * patch_size, pz * patch_size],
            }
        )

    return coords


def compute_patch_mapping():
    img_sizes = {}

    if not (raw_path / "img_sizes.json").is_file():
        imgs = list(
            (raw_path / "gt_labelsTr").glob(f"**/*{dataset_cfg['file_ending']}")
        )

        for path in track(imgs):
            img = sitk.ReadImage(path)
            name = path.name.replace(dataset_cfg["file_ending"], "")
            size = img.GetSize()

            print(f"{name}: {size}")
            img_sizes[name] = size

        with open(raw_path / "img_sizes.json", "w") as f:
            json.dump(img_sizes, f)
    else:
        with open(raw_path / "img_sizes.json") as f:
            img_sizes = json.load(f)

    img_sizes = dict(sorted(img_sizes.items()))
    return img_sizes


def starting_budget_random_grid():
    img_sizes = compute_patch_mapping()
    for k, (x, y, z) in img_sizes.items():
        img_sizes[k] = {
            "img_size": [x, y, z],
            "n_patches": (x // patch_size) * (y // patch_size) * (z // patch_size),
        }

    n_patches = np.array(list(v["n_patches"] for v in img_sizes.values()))
    bins = np.cumsum(n_patches)
    files = list(img_sizes.keys())
    sizes = np.array(list(v["img_size"] for v in img_sizes.values()))

    n_patches_total = np.sum(n_patches)
    patch_ids = np.random.choice(n_patches_total, n_samples, replace=False)

    patches = patch_ids_to_image_coords(patch_ids, bins, files, sizes)

    return patches


def random_crop(xs, ys, zs):
    xt = np.random.randint(crop_min_size * xs, crop_max_size * xs)
    yt = np.random.randint(crop_min_size * ys, crop_max_size * ys)
    zt = np.random.randint(crop_min_size * zs, crop_max_size * zs)

    i = np.random.randint(0, xs - xt + 1)
    j = np.random.randint(0, ys - yt + 1)
    k = np.random.randint(0, zs - zt + 1)

    return i, j, k, xt, yt, zt


def starting_budget_random_crop():
    img_sizes = compute_patch_mapping()

    crops = []
    for file, (x, y, z) in img_sizes.items():
        i, j, k, xt, yt, zt = random_crop(x, y, z)
        crops.append(
            {
                "file": file,
                "coords": [i, j, k],
                "size": [xt, yt, zt],
            }
        )

    return crops


def make_folder(patches):
    output_dir = raw_path / "labelsTr"
    output_dir.mkdir(exist_ok=True)

    for patch in track(patches):
        img_itk = sitk.ReadImage(
            (gt_path / patch["file"]).with_suffix(dataset_cfg["file_ending"])
        )
        img_npy = sitk.GetArrayFromImage(img_itk)
        spacing = img_itk.GetSpacing()
        origin = img_itk.GetOrigin()
        direction = np.array(img_itk.GetDirection())

        img_new = np.full_like(img_npy, ignore_label)

        xs, xe = patch["coords"][0], patch["coords"][0] + patch["size"][0]
        ys, ye = patch["coords"][1], patch["coords"][1] + patch["size"][1]
        zs, ze = patch["coords"][2], patch["coords"][2] + patch["size"][2]

        img_new[xs:xe, ys:ye, zs:ze] = img_npy[xs:xe, ys:ye, zs:ze]

        img_itk_new = sitk.GetImageFromArray(img_new)
        img_itk_new.SetSpacing(spacing)
        img_itk_new.SetOrigin(origin)
        img_itk_new.SetDirection(direction)
        sitk.WriteImage(
            img_itk_new,
            (output_dir / patch["file"]).with_suffix(dataset_cfg["file_ending"]),
        )


def main():
    resample_all()
    patches = starting_budget_random_crop()
    pprint(patches)
    make_folder(patches)


if __name__ == "__main__":
    main()
