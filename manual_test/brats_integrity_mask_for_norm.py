import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw, nnUNet_results
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name
from tqdm import tqdm

from nnactive.cli.subcommands.convert_to_partannotated import (
    convert_dataset_to_partannotated,
)
from nnactive.cli.subcommands.nnunet_extract_fingerprint import (
    extract_fingerprint_dataset,
)
from nnactive.cli.subcommands.resample_nnunet_dataset import resample_nnunet_dataset
from nnactive.utils.io import load_json

NNUNET_RAW = Path(nnUNet_raw) if nnUNet_raw is not None else None
NNUNET_PREPROCESSED = (
    Path(nnUNet_preprocessed) if nnUNet_preprocessed is not None else None
)
NNUNET_RESULTS = Path(nnUNet_results) if nnUNet_results is not None else None

VERIFY_DATA = True


def test_potential_iterable(foo, bar, max_depth: int = 100, depth: int = 0) -> bool:
    if hasattr(foo, "__iter__") and depth < max_depth:
        assert len(foo) == len(bar)
        identical = all(
            [
                test_potential_iterable(s_v, c_v, max_depth=max_depth, depth=depth + 1)
                for s_v, c_v in zip(foo, bar, strict=True)
            ]
        )
    else:
        identical = foo == bar
    return identical


def compute_file_hash(file_path):
    hash_md5 = hashlib.md5()

    with open(file_path, "rb") as file:
        # Read the file in chunks to handle large files
        for chunk in iter(lambda: file.read(4096), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


def compute_npz_hash(file_path):
    np_file = np.load(file_path)
    hash_dict = {}
    for key, val in np_file.items():
        hasher = hashlib.md5()
        hasher.update(val)
        hash_dict[key] = hasher.hexdigest()
    return hash_dict


def compute_npz_key_hash(file_path: Path, key: str):
    np_file = np.load(file_path)
    hasher = hashlib.md5()
    hasher.update(np_file[key])
    return hasher.hexdigest()


def main():
    fingerprint_call = (
        "nnUNetv2_extract_fingerprint -d {} --verify_dataset_integrity --clean -np 12"
    )
    plan_call = "nnUNetv2_plan_experiment -d {}"
    src_id = 138
    resample_id = src_id + 1
    sparse_id = src_id + 2
    dataset_ids = [src_id, resample_id, sparse_id]
    src_name = convert_id_to_dataset_name(src_id)
    resample_name = "Dataset139_BraTSTempResample"

    subprocess.run(fingerprint_call.format(src_id), shell=True, check=True)
    subprocess.run(plan_call.format(src_id), shell=True, check=True)
    src_raw_dir = NNUNET_RAW / src_name

    resample_name = "Dataset139_BraTSTempResample"
    try:
        convert_id_to_dataset_name(resample_id)
        shutil.rmtree(NNUNET_RAW / resample_name)
    except:
        pass

    shutil.copytree(src_raw_dir, NNUNET_RAW / resample_name)
    resampled_raw_dir = NNUNET_RAW / convert_id_to_dataset_name(resample_id)
    subprocess.run(fingerprint_call.format(resample_id), shell=True, check=True)
    subprocess.run(plan_call.format(resample_id), shell=True, check=True)
    resample_nnunet_dataset(resample_id, 6)
    extract_fingerprint_dataset(
        resample_id,
        num_processes=12,
        check_dataset_integrity=True,
        clean=True,
        verbose=False,
    )
    subprocess.run(plan_call.format(resample_id), shell=True, check=True)

    convert_dataset_to_partannotated(
        resample_id,
        sparse_id,
        full_images=0,
        num_patches=25,
        patch_size=[64, 64, 64],
        name_suffix="Convert",
        strategy="random-label",
        force=True,
    )
    subprocess.run(fingerprint_call.format(sparse_id), shell=True, check=True)
    subprocess.run(plan_call.format(sparse_id), shell=True, check=True)

    sparse_name = convert_id_to_dataset_name(sparse_id)

    prep_dirs = {
        src_name: src_id,
        resample_name: resample_id,
        sparse_name: sparse_id,
    }
    for key in prep_dirs:
        prep_dirs[key] = NNUNET_PREPROCESSED / convert_id_to_dataset_name(
            prep_dirs[key]
        )

    fingerprints = {}
    plans = {}
    for key in prep_dirs:
        plans[key] = load_json(prep_dirs[key] / "nnUNetPlans.json")
        fingerprints[key] = load_json(prep_dirs[key] / "dataset_fingerprint.json")

    src_plans = plans[src_name]
    src_fp = fingerprints[src_name]

    test_plans_functions = {
        "mask_for_norm": lambda x: x.get("configurations")
        .get("3d_fullres")
        .get("use_mask_for_norm"),
        "spacing": lambda x: x.get("configurations").get("3d_fullres").get("spacing"),
        "median_image_size_in_voxels": lambda x: x.get("configurations")
        .get("3d_fullres")
        .get("median_image_size_in_voxels"),
        "patch_size": lambda x: x.get("configurations")
        .get("3d_fullres")
        .get("median_image_size_in_voxels"),
        "patch_size": lambda x: x.get("configurations")
        .get("3d_fullres")
        .get("median_image_size_in_voxels"),
    }

    test_fps_functions = {"shapes_after_crop": lambda x: x.get("shapes_after_crop")}

    for fp_name, fp in fingerprints.items():
        for test_name, test_function in test_fps_functions.items():
            src_val = test_function(src_fp)
            comp_val = test_function(fp)

            identical = test_potential_iterable(src_val, comp_val)
            if identical == False:
                raise RuntimeError(
                    f"The nnU-Net Fingerprints changed from parent to child datset.\n"
                    f"Parent: {src_name}\n"
                    f"Child: {plans_name}\n"
                    f"Value: {fp_name}\n"
                )

    for plans_name, plan in plans.items():
        for test_name, test_function in test_plans_functions.items():
            src_val = test_function(src_plans)
            comp_val = test_function(plan)

            identical = test_potential_iterable(src_val, comp_val)
            if identical == False:
                raise RuntimeError(
                    f"The nnU-Net Plans changed from parent to child datset.\n"
                    f"Parent: {src_name}\n"
                    f"Child: {plans_name}\n"
                    f"Value: {test_name}\n"
                )

    if VERIFY_DATA:
        for d_id in dataset_ids:
            subprocess.run(
                "nnUNetv2_preprocess -d {} -np 12 -c 3d_fullres".format(d_id),
                shell=True,
                check=True,
            )

        subset = 100
        npz_files_lists = {}
        hash_files_lists = {}
        data_hash_files_lists = {}
        for name, prep_dir in prep_dirs.items():
            prep_dir_folder = prep_dir / "nnUNetPlans_3d_fullres"
            npz_files_lists[name] = [
                fn for fn in os.listdir(prep_dir_folder) if fn.endswith(".npz")
            ]
            npz_files_lists[name].sort()
            subset_temp = len(npz_files_lists[name]) if subset is None else subset
            print(prep_dir_folder)
            hash_files_lists[name] = [
                compute_file_hash(prep_dir_folder / npz_file)
                for npz_file in tqdm(
                    npz_files_lists[name][:subset_temp], desc=".npz hashes"
                )
            ]
            data_hash_files_lists[name] = [
                compute_npz_key_hash(prep_dir_folder / npz_file, "data")
                for npz_file in tqdm(npz_files_lists[name][:subset_temp], "data hashes")
            ]

        src_data_hash_files_list = data_hash_files_lists[src_name]
        src_npz_files_list = npz_files_lists[src_name]
        src_hash_files_lists = hash_files_lists[src_name]

        for plans_name, npz_files_list in npz_files_lists.items():
            print(f"Verification of: {plans_name}")
            print(f"Number of files: {len(npz_files_list)}")
            identical = test_potential_iterable(
                src_npz_files_list, npz_files_list, max_depth=1
            )

            if identical == False:
                raise RuntimeWarning(
                    f"The nnU-Net Preprocess files changed from parent to child datset.\n"
                    f"Parent: {src_name}\n"
                    f"Child: {plans_name}\n"
                    f"Exact filenames and order in preprocessed!\n"
                )

            identical = test_potential_iterable(
                src_hash_files_lists, hash_files_lists[plans_name], max_depth=1
            )
            if identical == False:
                if not plans_name == sparse_name:
                    raise RuntimeWarning(
                        f"The nnU-Net Preprocess files changed from parent to child datset.\n"
                        f"Parent: {src_name}\n"
                        f"Child: {plans_name}\n"
                        f"Hashes of .npz files are different!!\n"
                    )

            identical = test_potential_iterable(
                src_data_hash_files_list,
                data_hash_files_lists[plans_name],
                max_depth=1,
            )
            if identical == False:
                raise RuntimeWarning(
                    f"The nnU-Net Preprocess files changed from parent to child datset.\n"
                    f"Parent: {src_name}\n"
                    f"Child: {plans_name}\n"
                    f"Hashes of 'data' in .npz files are different!!\n"
                )

    print("Data verification succesful!")


if __name__ == "__main__":
    main()
