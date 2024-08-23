import shtab
from jsonargparse import ActionConfigFile, ArgumentParser, Namespace

# from nnactive.cli import subcommands
from nnactive.cli.registry import add_subcommands, run_subcommand

from .subcommands import (
    analyze_experiments,
    human_al_workflow,
    init_data,
    init_resampling,
    manual_workflow,
    run_al_loops,
    setup,
    utilities,
)


def main() -> None:
    """Entry point for the command line interface

    This gets installed as a script named `nnactive` by pip.
    """
    parser = ArgumentParser()
    shtab.add_argument_to(parser, ["-s", "--print-completion"])

    add_subcommands(parser)
    args = parser.parse_args()

    import multiprocessing

    multiprocessing.set_start_method("spawn")

    run_subcommand(args)
