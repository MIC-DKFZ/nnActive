from argparse import ArgumentParser

from experiment_setup import DatasetSetup

parser = ArgumentParser()
parser = DatasetSetup.add_args(parser)

if __name__ == "__main__":
    args = parser.parse_args()
    seeds = [12345]  # , 12346, 12347]
    # uncertainties = ["mutual_information"]
    uncertainties = ["pred_entropy"]
    # uncertainties = ["pred_entropy", "random", "mutual_information", "random-label"]
    dataset_id = 216
    first_d_set = 800
    # patch_size = []
    query_size = 5
    query_steps = 20
    agg_stride = 8
    trainer = "nnActiveTrainer_5epochs"
    starting_budget = "random-label"
    pre_suffix = "__patch-full_patch"
    train_folds = 5
    num_processes = 4
    add_validation = ""
    add_uncertainty = ""

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
        agg_stride=agg_stride,
        pre_suffix=pre_suffix,
        add_validation=add_validation,
        add_uncertainty=add_uncertainty,
    )
    setter.rollout(first_d_set, num_experiments=args.num_experiments, debug=args.debug)
