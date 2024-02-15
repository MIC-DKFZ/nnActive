from argparse import ArgumentParser

from experiment_setup import DatasetSetup

dataset_id = 4
train_folds = 5
num_processes = 10
query_size = 5
query_steps = 10
uncertainties = ["random", "mutual_information", "pred_entropy"]
trainer = "nnActiveTrainer_100epochs"
starting_budget = "random"


parser = ArgumentParser()
parser = DatasetSetup.add_args(parser)

if __name__ == "__main__":
    args = parser.parse_args()

    seeds = [12345, 12346, 12347]
    uncertainties = ["pred_entropy", "mutual_information", "random"]
    dataset_id = 4
    first_d_set = 500
    query_size = 5
    query_steps = 20
    trainer = "nnActiveTrainer_5epochs"
    starting_budget = "random-label-all-classes"
    num_processes = 10
    train_folds = 5
    pre_suffix = "__patch-full_patch"

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
        force_override=args.force_override,
        pre_suffix=pre_suffix,
    )
    setter.rollout(first_d_set, num_experiments=args.num_experiments, debug=args.debug)

    # Experiments with smaller Patch Size
    first_d_set = first_d_set + len(setter.vals)
    uncertainties = ["pred_entropy", "mutual_information", "random-label", "random"]
    # uncertainties = ["mutual_information"]
    patch_size = [20, 20, 20]
    query_steps = 20
    query_size = 10

    pre_suffix = "__patch-20"

    setter = DatasetSetup(
        base_id=dataset_id,
        query_steps=query_steps,
        query_size=query_size,
        trainer=trainer,
        starting_budget=starting_budget,
        patch_size=patch_size,
        num_processes=num_processes,
        train_folds=train_folds,
        force_override=args.force_override,
        pre_suffix=pre_suffix,
    )
    setter.rollout(first_d_set, num_experiments=args.num_experiments, debug=args.debug)

    # Experiments with smaller Patch Size & bigger Query Size

    first_d_set = first_d_set + len(setter.vals)
    query_steps = 10
    query_size = 20

    pre_suffix = "__patch-20__qs20"
    setter = DatasetSetup(
        base_id=dataset_id,
        query_steps=query_steps,
        query_size=query_size,
        trainer=trainer,
        starting_budget=starting_budget,
        patch_size=patch_size,
        num_processes=num_processes,
        train_folds=train_folds,
        force_override=args.force_override,
        pre_suffix=pre_suffix,
    )
    setter.rollout(first_d_set, num_experiments=args.num_experiments, debug=args.debug)

    # Experiments with smaller Patch Size & much bigger Query Size

    first_d_set = first_d_set + len(setter.vals)
    query_steps = 5
    query_size = 40
    pre_suffix = "__patch-20__qs40"
    setter = DatasetSetup(
        base_id=dataset_id,
        query_steps=query_steps,
        query_size=query_size,
        trainer=trainer,
        starting_budget=starting_budget,
        patch_size=patch_size,
        num_processes=num_processes,
        train_folds=train_folds,
        force_override=args.force_override,
        pre_suffix=pre_suffix,
    )
    setter.rollout(first_d_set, num_experiments=args.num_experiments, debug=args.debug)
