import os
from argparse import ArgumentParser, Namespace

from nnactive.cli.registry import register_subcommand
from nnactive.config import ActiveConfig
from nnactive.nnunet.utils import convert_id_to_dataset_name, get_patch_size
from nnactive.paths import get_nnActive_results
from nnactive.results.state import State


@register_subcommand(
    "setup_al_experiment",
    [
        (("-d", "--dataset"), {"type": int, "required": True, "help": "Dataset ID"}),
        ("--trainer", {"type": str, "default": "nnActiveTrainer_5epochs"}),
        (
            ("-p", "--patch-size"),
            {"nargs": "+", "type": int, "default": None, "help": "Patch Size"},
        ),
        (
            "--base_id",
            {
                "type": int,
                "default": None,
                "help": "Dataset from which patch size is taken",
            },
        ),
        (("-qs", "--query-size"), {"type": int, "default": 10}),
        ("--uncertainty", {"type": str, "default": "random"}),
        ("--query-steps", {"type": int, "default": 10}),
        ("--starting-budget", {"type": str, "default": "standard"}),
        ("--seed", {"type": int, "default": 12345}),
        (("-np", "--num-processes"), {"type": int, "default": 4}),
    ],
)
def main(args: Namespace) -> None:
    trainer = args.trainer
    query_size = args.query_size
    uncertainty = args.uncertainty
    query_steps = args.query_steps
    starting_budget = args.starting_budget
    seed = args.seed
    num_processes = args.num_processes

    if args.patch_size is None and args.base_id is None:
        raise ValueError("Either patch_size or base_id have to be set")
    patch_size = (
        args.patch_size if args.patch_size is not None else get_patch_size(args.base_id)
    )

    dataset_id: int = args.dataset

    dataset_name: str = convert_id_to_dataset_name(dataset_id)

    results_path = get_nnActive_results()

    save_path = results_path / dataset_name

    config = ActiveConfig(
        trainer=trainer,
        patch_size=patch_size,
        uncertainty=uncertainty,
        query_size=query_size,
        query_steps=query_steps,
        starting_budget=starting_budget,
        seed=seed,
        num_processes=num_processes,
    )

    os.makedirs(save_path, exist_ok=True)

    config.save_id(dataset_id)
    state = State(dataset_id=dataset_id)
    state.save_state()
