from argparse import ArgumentParser

from experiment_setup import DatasetSetup

parser = ArgumentParser()
parser = DatasetSetup.add_args(parser)

if __name__ == "__main__":
    # ready for initial run!
    args = parser.parse_args()
    force_override = args.force_override
    seeds = [12345, 12346, 12347]
    uncertainties = ["pred_entropy", "mutual_information", "random-label", "random"]
    dataset_id = 980
    first_d_set = 900
    query_size = 10
    query_steps = 10
    trainer = "nnActiveTrainer_airway_200epochs"
    starting_budget = "random-label-all-classes"
    train_folds = 5
    agg_stride = 8
    num_processes = 4
    patch_size = None
    pre_suffix = f"__patch-full_patch"

    setter = DatasetSetup(
        base_id=dataset_id,
        patch_size=patch_size,
        query_steps=query_steps,
        uncertainties=uncertainties,
        query_size=query_size,
        trainer=trainer,
        starting_budget=starting_budget,
        num_processes=num_processes,
        train_folds=train_folds,
        agg_stride=agg_stride,
        force_override=force_override,
        pre_suffix=pre_suffix,
    )
    setter.rollout(first_d_set, num_experiments=args.num_experiments, debug=args.debug)
