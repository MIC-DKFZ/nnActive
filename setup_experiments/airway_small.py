import subprocess
from itertools import product

from experiment_setup import DatasetSetup

if __name__ == "__main__":
    seeds = [12345, 12346, 12347]
    uncertainties = ["pred_entropy", "mutual_information", "random"]
    dataset_id = 983
    first_d_set = 993
    query_size = 33
    query_steps = 3
    trainer = "nnActiveTrainer_airway_5epochs"
    starting_budget = "pred_entropy"
    num_processes = 4
    train_folds = 1
    starting_budget = "random-label"
    force_override = False

    pre_suffix = f"__patch-std_patch__sb-{starting_budget}"
    num_experiments = 1

    # Experiments with whole Images as Patches
    setter = DatasetSetup(
        base_id=dataset_id,
        query_steps=query_steps,
        uncertainties=uncertainties,
        query_size=query_size,
        trainer=trainer,
        starting_budget=starting_budget,
        num_processes=num_processes,
        train_folds=train_folds,
        force_override=force_override,
        pre_suffix=pre_suffix,
    )
    setter.rollout(first_d_set, num_experiments)
