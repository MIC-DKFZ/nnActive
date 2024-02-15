from argparse import ArgumentParser

from experiment_setup import DatasetSetup

parser = ArgumentParser()
parser = DatasetSetup.add_args(parser)

if __name__ == "__main__":
    # Ready for initial run!
    args = parser.parse_args()
    seeds = [12345, 12346, 12347]
    uncertainties = ["pred_entropy", "mutual_information", "random-label", "random"]
    dataset_id = 216
    first_d_set = 800
    query_size = 32
    query_steps = 10
    agg_stride = 8
    trainer = "nnActiveTrainer_200epochs"
    starting_budget = "random-label-all-classes"
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
