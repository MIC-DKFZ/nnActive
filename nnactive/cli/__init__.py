from jsonargparse import ActionConfigFile, ArgumentParser, Namespace

from nnactive.cli.registry import add_subcommands, run_subcommand

from .subcommands import setup, nnunet_preprocess, train_nnUNet_ensemble, run_al_loops


def main() -> None:
    """Entry point for the command line interface

    This gets installed as a script named `nnactive` by pip.
    """
    parser = ArgumentParser()

    add_subcommands(parser)

    args = parser.parse_args()
    run_subcommand(args)
