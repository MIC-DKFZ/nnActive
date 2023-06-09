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
    ],
)
def main(args: Namespace):
    base_path = Path(args.base_path)
    compare_multi_experiment_results(base_path)
