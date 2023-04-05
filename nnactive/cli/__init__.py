import argparse

from .subcommands import *
from .registry import _add_to_parser


def main() -> None:
    """Entry point for the command line interface

    This gets installed as a script named `nnactive` by pip.
    """
    parser = argparse.ArgumentParser()
    parser.set_defaults(command=lambda _: parser.print_help())

    _add_to_parser(parser)

    args = parser.parse_args()
    args.command(args)
