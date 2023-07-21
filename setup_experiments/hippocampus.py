import os
import shutil
import subprocess
from argparse import ArgumentParser
from itertools import product
from pathlib import Path

parser = ArgumentParser()
parser.add_argument(
    "-f",
    "--force_override",
    action="store_true",
    help="Overrides Dataset and clears all previous experiments.\n Be careful with this one!",
)

DATA_PATH = os.getenv("nnUNet_raw")
if DATA_PATH is None:
    raise ValueError("OS variable nnUNet_raw is not set.")
DATA_PATH = Path(DATA_PATH)

if __name__ == "__main__":
    args = parser.parse_args()
    force_override = args.force_override
    exisisting_dsets = [
        folder.name for folder in DATA_PATH.iterdir() if folder.is_dir()
    ]
    seeds = [12345, 12346, 12347]
    uncertainties = ["random", "mutual_information", "pred_entropy"]
    dataset_id = 4
    first_d_set = 500
    query_size = 5
    query_steps = 20
    trainer = "nnActiveTrainer_100epochs"
    starting_budget = "random"
    num_processes = 4

    vals = [val for val in product(uncertainties, seeds)]

    for i, (unc, seed) in enumerate(vals):
        ex_command = "nnactive convert"
        output_id = first_d_set + i
        dset_name = f"Dataset{output_id:03d}"
        if any([dset.startswith(dset_name) for dset in exisisting_dsets]):
            print(
                f"Dataset beginning with '{dset_name}' already exists in {DATA_PATH}."
            )
            if not force_override:
                continue
            else:
                os_variables = [
                    "nnUNet_results",
                    "nnUNet_raw",
                    "nnUNet_preprocessed",
                    "nnActive_results",
                ]
                for os_variable in os_variables:
                    base_path = Path(os.getenv(os_variable))
                    rm_dirs = [
                        folder
                        for folder in base_path.iterdir()
                        if folder.name.startswith(f"Dataset{output_id:03d}")
                    ]
                    for rm_dir in rm_dirs:
                        print(f"Deleting folder: {rm_dir}")
                        shutil.rmtree(rm_dir)

        name_suffix = f"patch-full_patch-unc-{unc}-seed-{seed}"
        ex_call = f"{ex_command} -d {dataset_id} -o {output_id} --strategy {starting_budget} --seed {seed} --num-patches {query_size} --name-suffix {name_suffix}"

        print(ex_call)
        subprocess.run(ex_call, shell=True, check=True)

        subprocess.run(
            f"nnUNetv2_extract_fingerprint -d {output_id}  -np 4",
            shell=True,
            check=True,
        )
        subprocess.run(
            f"nnUNetv2_plan_experiment -d {output_id} -np 4", shell=True, check=True
        )

        ex_command = "nnactive setup_al_experiment"
        subprocess.run(
            f"{ex_command} -d {output_id} --trainer {trainer} --base_id {dataset_id} --query-steps {query_steps} --query-size {query_size} --uncertainty {unc} --starting-budget {starting_budget} --seed {seed}",
            shell=True,
            check=True,
        )

    first_d_set = first_d_set + len(vals)
    patch_size = "20 20 20"
    query_steps = 20
    query_size = 10

    for i, (unc, seed) in enumerate(vals):
        ex_command = "nnactive convert"
        output_id = first_d_set + i
        name_suffix = f"patch-patch20-unc-{unc}-seed-{seed}"
        dset_name = f"Dataset{output_id:03d}"
        if any([dset.startswith(dset_name) for dset in exisisting_dsets]):
            print(
                f"Dataset beginning with '{dset_name}' already exists in {DATA_PATH}."
            )
            if not force_override:
                continue
            else:
                os_variables = [
                    "nnUNet_results",
                    "nnUNet_raw",
                    "nnUNet_preprocessed",
                    "nnActive_results",
                ]
                for os_variable in os_variables:
                    base_path = os.getenv(os_variable)
                    if base_path is None:
                        raise ValueError(f"OS variable '{os_variable}' is not set.")
                    base_path = Path(base_path)
                    rm_dirs = [
                        folder
                        for folder in base_path.iterdir()
                        if folder.name.startswith(f"Dataset{output_id:03d}")
                    ]
                    for rm_dir in rm_dirs:
                        print(f"Deleting folder: {rm_dir}")
                        shutil.rmtree(rm_dir)

        subprocess.run(
            f"{ex_command} -d {dataset_id} -o {output_id} --strategy {starting_budget} --seed {seed} --patch-size {patch_size} --num-patches {query_size} --name-suffix {name_suffix}",
            shell=True,
            check=True,
        )

        subprocess.run(
            f"nnUNetv2_extract_fingerprint -d {output_id} -np 4", shell=True, check=True
        )
        subprocess.run(
            f"nnUNetv2_plan_experiment -d {output_id}  -np 4", shell=True, check=True
        )

        ex_command = "nnactive setup_al_experiment"
        subprocess.run(
            f"{ex_command} -d {output_id} --trainer {trainer} --base_id {dataset_id} --query-steps {query_steps} --query-size {query_size} --uncertainty {unc} --patch-size {patch_size} --starting-budget {starting_budget} --seed {seed}",
            shell=True,
            check=True,
        )
