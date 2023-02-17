import argparse

def main() -> None:
    """Entry point for the command line interface

    This gets installed as a script named `nnactive` by pip.
    """
    parser = argparse.ArgumentParser()
    # subparsers = parser.add_subparsers(title="commands")
    parser.set_defaults(command=lambda _: parser.print_help())

    args = parser.parse_args()
    args.command(args)
