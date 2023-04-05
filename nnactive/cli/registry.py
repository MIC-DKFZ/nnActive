import functools
from argparse import ArgumentParser, Namespace
from typing import Any, Callable

__subcommands = {}
__args = {}


ArgType = str | tuple[str, ...] | tuple[(str | tuple[str, ...]), dict[str, Any]]
ArgsType = list[ArgType] | Callable[[ArgumentParser], None]


def _add_args(parser: ArgumentParser, args: list[ArgType]) -> None:
    for arg in args:
        match arg:
            case str(name):
                parser.add_argument(name, action="store_true")
            case [str(name), dict(kwargs)]:
                parser.add_argument(name, **kwargs)
            case tuple((tuple(names), dict(kwargs))):
                parser.add_argument(*names, **kwargs)
            case tuple(names) if all(map(lambda n: isinstance(n, str), names)):
                parser.add_argument(*names, action="store_true")
            case _:
                raise ValueError(f'Cannot add arguments from "{arg}"')


def register_subcommand(
    name: str, args: ArgsType
) -> Callable[[Callable[[Namespace], None]], Callable[[Namespace], None]]:
    """Register a function as a CLI subcommand

    Args:
        name: name of the subcommand
        args: list of arguments the command should accept

    Returns:
        registerd fundtion
    """
    def _inner_wrapper(
        func: Callable[[Namespace], None]
    ) -> Callable[[Namespace], None]:
        __subcommands[name] = func
        match args:
            case f if callable(f):
                __args[name] = f
            case [*_]:
                __args[name] = functools.partial(_add_args, args=args)
            case _:
                raise ValueError(f'Cannot add arguments from "{args}"')

        return func

    return _inner_wrapper


def _add_to_parser(parser: ArgumentParser) -> None:
    subparsers = parser.add_subparsers(title="commands")
    for name, func in __subcommands.items():
        parser = subparsers.add_parser(name)
        if argsfunc := __args.get(name):
            argsfunc(parser)
        parser.set_defaults(command=func)
