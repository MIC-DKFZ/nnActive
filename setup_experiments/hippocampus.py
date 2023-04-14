import subprocess
from itertools import product
from pathlib import Path

# SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts"

if __name__ == "__main__":
    seeds = [12345, 12346, 12347]
    uncertainties = ["random", "mutual_information", "pred_entropy"]
    dataset_id = 4
    first_d_set = 500
    # patch_size = []
    query_size = 5
    query_steps = 20
    trainer = "nnActiveTrainer_100epochs"
    starting_budget = "random"
    num_processes = 4

    vals = [val for val in product(uncertainties, seeds)]

    for i, (unc, seed) in enumerate(vals):
        ex_command = "nnactive convert"
        output_id = first_d_set + i
        name_suffix = f"patch-full_patch-unc-{unc}-seed-{seed}"
        ex_call = f"{ex_command} -d {dataset_id} -o {output_id} --strategy {starting_budget} --seed {seed} --num-patches {query_size} --name-suffix {name_suffix}"

        print(ex_call)
        subprocess.run(ex_call, shell=True)

        subprocess.run(
            f"nnUNetv2_extract_fingerprint -d {output_id}  -np 4",
            shell=True,
        )
        subprocess.run(
            f"nnUNetv2_plan_experiment -d {output_id} -np 4",
            shell=True,
        )

        ex_command = "nnactive setup_al_experiment"
        subprocess.run(
            f"{ex_command} -d {output_id} --trainer {trainer} --base_id {dataset_id} --query-steps {query_steps} --query-size {query_size} --uncertainty {unc} --starting-budget {starting_budget} --seed {seed}",
            shell=True,
        )

    first_d_set = first_d_set + len(vals)
    patch_size = "20 20 20"
    query_steps = 20
    query_size = 10

    for i, (unc, seed) in enumerate(vals):
        ex_command = "nnactive convert"
        output_id = first_d_set + i
        name_suffix = f"patch-patch20-unc-{unc}-seed-{seed}"
        subprocess.run(
            f"{ex_command} -d {dataset_id} -o {output_id} --strategy {starting_budget} --seed {seed} --patch-size {patch_size} --num-patches {query_size} --name-suffix {name_suffix}",
            shell=True,
        )

        subprocess.run(
            f"nnUNetv2_extract_fingerprint -d {output_id} -np 4",
            shell=True,
        )
        subprocess.run(
            f"nnUNetv2_plan_experiment -d {output_id}  -np 4",
            shell=True,
        )

        ex_command = "nnactive setup_al_experiment"
        subprocess.run(
            f"{ex_command} -d {output_id} --trainer {trainer} --base_id {dataset_id} --query-steps {query_steps} --query-size {query_size} --uncertainty {unc} --patch-size {patch_size} --starting-budget {starting_budget} --seed {seed}",
            shell=True,
        )
