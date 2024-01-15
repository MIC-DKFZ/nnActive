from argparse import ArgumentParser

from experiment_setup import DatasetSetup

parser = ArgumentParser()
parser = DatasetSetup.add_args(parser)


if __name__ == "__main__":
    args = parser.parse_args()
    force_override = args.force_override

    seeds = [12345]  # , 12346, 12347]
    uncertainties = ["mutual_information", "random", "pred_entropy"]
    dataset_id = 984
    first_d_set = 990
    query_size = 60
    query_steps = 3
    trainer = "nnActiveTrainer_5epochs"
    starting_budget = "random-label"
    num_processes = 4
    train_folds = 1

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
    )
    setter.rollout(first_d_set, num_experiments=args.num_experiments)
