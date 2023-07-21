import subprocess
from itertools import product

if __name__ == "__main__":
    seeds = [12345, 12346, 12347]
    uncertainties = ["pred_entropy", "mutual_information", "random"]
    dataset_id = 983
    first_d_set = 993
    query_size = 25
    query_steps = 3
    trainer = "nnActiveTrainer_airway_5epochs"
    starting_budget = "random"
    num_processes = 4
    format_suffix = "__patch-full_patch__sb-{starting_budget}__unc-{unc}__seed-{seed}"

    vals = [val for val in product(uncertainties, seeds)]

    for i, (unc, seed) in enumerate(vals):
        ex_command = "nnactive convert"
        output_id = first_d_set + i
        name_suffix = format_suffix.format(
            starting_budget=starting_budget, unc=unc, seed=seed
        )
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
        if i == 0:
            break

    # first_d_set += len(vals)
    # starting_budget = "random-label"

    # for i, (unc, seed) in enumerate(vals):
    #     ex_command = "nnactive convert"
    #     output_id = first_d_set + i
    #     name_suffix = format_suffix.format(
    #         starting_budget=starting_budget, unc=unc, seed=seed
    #     )
    #     ex_call = f"{ex_command} -d {dataset_id} -o {output_id} --strategy {starting_budget} --seed {seed} --num-patches {query_size} --name-suffix {name_suffix}"

    #     print(ex_call)
    #     subprocess.run(ex_call, shell=True, check=True)

    #     subprocess.run(
    #         f"nnUNetv2_extract_fingerprint -d {output_id}  -np 4",
    #         shell=True,
    #         check=True,
    #     )
    #     subprocess.run(
    #         f"nnUNetv2_plan_experiment -d {output_id} -np 4", shell=True, check=True
    #     )

    #     ex_command = "nnactive setup_al_experiment"
    #     subprocess.run(
    #         f"{ex_command} -d {output_id} --trainer {trainer} --base_id {dataset_id} --query-steps {query_steps} --query-size {query_size} --uncertainty {unc} --starting-budget {starting_budget} --seed {seed}",
    #         shell=True,
    #         check=True,
    #     )
    #     if i == 0:
    #         break
