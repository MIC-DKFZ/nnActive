from argparse import ArgumentParser

from experiment_setup import DatasetSetup

parser = ArgumentParser()
parser = DatasetSetup.add_args(parser)

# Important Note: AMOS small Needs to be created with 0.3 relative size
if __name__ == "__main__":
    args = parser.parse_args()
    force_override = args.force_override

    seeds = [12345]  # , 12346, 12347]
    uncertainties = ["mutual_information", "random", "pred_entropy"]
    dataset_id = 984
    first_d_set = 994
    query_size = 60
    query_steps = 3
    agg_stride = 8
    trainer = "nnActiveTrainer_5epochs"
    starting_budget = "random-label-all-classes"
    num_processes = 4
    train_folds = 5
    if args.num_experiments is None:
        args.num_experiments = 1

    # Experiments with whole Images as Patches
    setter = DatasetSetup(
        base_id=dataset_id,
        query_steps=query_steps,
        uncertainties=uncertainties,
        query_size=query_size,
        trainer=trainer,
        starting_budget=starting_budget,
        num_processes=num_processes,
        agg_stride=agg_stride,
        train_folds=train_folds,
        force_override=force_override,
    )
    setter.rollout(first_d_set, num_experiments=args.num_experiments)
