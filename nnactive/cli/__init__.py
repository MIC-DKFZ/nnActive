import shtab
from jsonargparse import ActionConfigFile, ArgumentParser, Namespace

from nnactive.cli.registry import add_subcommands, run_subcommand

from .subcommands import (
    analyze_experiments,
    manual_crop_pred,
    nnunet_preprocess,
    run_al_loops,
    setup,
    train_nnUNet_ensemble,
)


def main() -> None:
    """Entry point for the command line interface

    This gets installed as a script named `nnactive` by pip.
    """
    parser = ArgumentParser()
    shtab.add_argument_to(parser, ["-s", "--print-completion"])

    add_subcommands(parser)

    args = parser.parse_args()
    run_subcommand(args)
