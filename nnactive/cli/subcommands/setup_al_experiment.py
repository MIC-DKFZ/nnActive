import json
import os
from argparse import Namespace
from typing import List

import numpy as np

from nnactive.cli.registry import register_subcommand
from nnactive.config import ActiveConfig
from nnactive.nnunet.paths import NNUNET_RAW
from nnactive.nnunet.utils import (
    convert_id_to_dataset_name,
    get_patch_size,
    get_preprocessed_path,
    read_dataset_json,
)
from nnactive.paths import get_nnActive_results
from nnactive.results.state import State
from nnactive.utils import create_mitk_geometry_patch


def create_mitk_geometry_file(dataset_id: int, target_id: int, patch_size: List[int]):
    preprocessed_path = get_preprocessed_path(dataset_id)
    if not preprocessed_path.exists():
        assert "Please preprocess the dataset before"
    nnunet_plans_path = preprocessed_path / "nnUNetPlans.json"
    with open(nnunet_plans_path, "r") as f:
        nnunet_plans = json.load(f)
    scale_factor = nnunet_plans["original_median_spacing_after_transp"]
    if len(scale_factor) == 3:
        scale_factor.reverse()
    scale_factor = np.array(scale_factor)
    if len(patch_size) == 3:
        patch_size.reverse()
    patch_size = np.array(patch_size)
    target_dir = NNUNET_RAW / convert_id_to_dataset_name(target_id)
    create_mitk_geometry_patch.main(
        target_dir / "patch.mitkgeometry",
        tuple(np.multiply(scale_factor, patch_size)),
    )


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

    # get base dataset name
    base_dataset_key = "annotated_id"
    dataset_json = read_dataset_json(dataset_id)

    if base_dataset_key in dataset_json:
        base_dataset_ident = dataset_json[base_dataset_key]
        if isinstance(base_dataset_ident, int):
            base_dataset_name: str = convert_id_to_dataset_name(base_dataset_ident)
        else:
            raise ValueError(f"dataset.json['{base_dataset_key}'] is not of type int.")
    else:
        base_dataset_name = dataset_name

    config = ActiveConfig(
        trainer=trainer,
        patch_size=patch_size,
        uncertainty=uncertainty,
        query_size=query_size,
        query_steps=query_steps,
        starting_budget=starting_budget,
        seed=seed,
        num_processes=num_processes,
        dataset=base_dataset_name,
    )

    os.makedirs(save_path, exist_ok=True)

    config.save_id(dataset_id)
    state = State(dataset_id=dataset_id)
    state.save_state()
    if args.base_id:
        plans_dataset_id = args.base_id
    else:
        plans_dataset_id = dataset_id
    create_mitk_geometry_file(
        dataset_id=plans_dataset_id, target_id=dataset_id, patch_size=patch_size
    )
