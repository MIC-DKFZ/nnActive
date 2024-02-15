from argparse import ArgumentParser

from experiment_setup import DatasetSetup

parser = ArgumentParser()
parser = DatasetSetup.add_args(parser)

if __name__ == "__main__":
    args = parser.parse_args()
    seeds = [12345, 12346, 12347]
    uncertainties = ["pred_entropy", "mutual_information", "random-label", "random"]
    dataset_id = 985
    first_d_set = 995
    query_size = 25
    query_steps = 5
    trainer = "nnActiveTrainer_5epochs"
    starting_budget = "random-label-all-classes"
    num_processes = 8
    train_folds = 5
    patch_size = None
    pre_suffix = "__patch-full"
    add_validation = ""
    add_uncertainty = ""

    if args.num_experiments is None:
        args.num_experiments = 1

    setter = DatasetSetup(
        base_id=dataset_id,
        query_steps=query_steps,
        uncertainties=uncertainties,
        query_size=query_size,
        trainer=trainer,
        patch_size=patch_size,
        starting_budget=starting_budget,
        num_processes=num_processes,
        train_folds=train_folds,
        force_override=args.force_override,
        pre_suffix=pre_suffix,
        add_validation=add_validation,
        add_uncertainty=add_uncertainty,
    )

    setter.rollout(first_d_set, num_experiments=args.num_experiments, debug=args.debug)
