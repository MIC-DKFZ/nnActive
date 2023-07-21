from argparse import Namespace
from pathlib import Path

from nnactive.analysis import compare_multi_experiment_results
from nnactive.cli.registry import register_subcommand
from nnactive.paths import get_nnActive_results


@register_subcommand(
    "analyze_experiments",
    [
        (
            ("--base_path"),
            {
                "type": str,
                "default": get_nnActive_results(),
                "help": "Path to nnActive results for later use",
            },
        ),
        (
            ("--base_dataset_id"),
            {
                "type": int,
                "default": None,
                "help": "Dataset ID from which the childern are used for visualization",
            },
        ),
    ],
)
def main(args: Namespace):
    base_path = Path(args.base_path)
    base_dataset_id = args.base_dataset_id
    compare_multi_experiment_results(base_path, base_dataset_id)
