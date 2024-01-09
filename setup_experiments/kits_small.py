from argparse import ArgumentParser

from experiment_setup import DatasetSetup

parser = ArgumentParser()
parser = DatasetSetup.add_args(parser)

if __name__ == "__main__":
    args = parser.parse_args()
    seeds = [12345, 12346, 12347]
    uncertainties = ["pred_entropy", "mutual_information", "random"]
    dataset_id = 982
    first_d_set = 992
    # patch_size = []
    query_size = 10
    query_steps = 3
    trainer = "nnActiveTrainer_5epochs"
    starting_budget = "random-label"
    num_processes = 4
    train_folds = 2
    pre_suffix = "__patch-full_patch"
    add_validation = "--disable_tta"
    add_uncertainty = "--diable_tta"
    agg_stride = 8
    if args.num_experiments is None:
        args.num_experiments = 1

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
        agg_stride=agg_stride,
        add_validation=add_validation,
        add_uncertainty=add_uncertainty,
    )
    setter.rollout(first_d_set, num_experiments=args.num_experiments, debug=args.debug)
