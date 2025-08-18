import argparse
import types
import typing
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Callable, ParamSpec, TypeVar

from jsonargparse import ArgumentParser, Namespace
from jsonargparse._common import Action

from nnactive.config.struct import ActiveConfig, RuntimeConfig
from nnactive.experiments import get_experiment, list_experiments

previous_config: ContextVar = ContextVar("previous_config", default=None)


@contextmanager
def previous_config_context(cfg):
    token = previous_config.set(cfg)
    try:
        yield
    finally:
        previous_config.reset(token)


class ActionExperiment(Action):
    """Action to indicate that an argument is an experiment name."""

    def __init__(self, **kwargs):
        """Initializer for ActionExperiment instance."""
        if "default" in kwargs:
            raise ValueError("ActionExperiment does not accept a default.")
        opt_name = kwargs["option_strings"]
        opt_name = (
            opt_name[0]
            if len(opt_name) == 1
            else [x for x in opt_name if x[0:2] == "--"][0]
        )
        if "." in opt_name:
            raise ValueError("ActionExperiment must be a top level option.")
        if "help" not in kwargs:
            # TODO: hint to list-experiments
            kwargs["help"] = "Name of an experiment."
        super().__init__(**kwargs)

    def __call__(self, parser, cfg, values, option_string=None):
        """Parses the given experiment configuration and adds all the corresponding keys to the namespace.

        Raises:
            TypeError: If there are problems parsing the configuration.
        """
        self.apply_experiment_config(parser, cfg, self.dest, values)

    @staticmethod
    def apply_experiment_config(parser: ArgumentParser, cfg, dest, value) -> None:
        with previous_config_context(cfg):
            experiment_cfg = get_experiment(value)
            tcfg = parser.parse_object(
                {"config": asdict(experiment_cfg)},
                env=False,
                defaults=False,
            )
            cfg_merged = parser.merge_config(tcfg, cfg)
            cfg.__dict__.update(cfg_merged.__dict__)
            cfg[dest] = value


__subcommands = {}

P = ParamSpec("P")


T = TypeVar("T", bound=type)


def _dict_to_dataclass(cfg: dict, cls: T) -> T:
    def __dict_to_dataclass(cfg, cls: type, key: str):  # noqa: ANN202,ANN001
        try:
            if is_dataclass(cls):
                fieldtypes = typing.get_type_hints(cls)
                return cls(
                    **{
                        k: __dict_to_dataclass(v, fieldtypes[k], k)
                        for k, v in cfg.items()
                    }
                )
            if (
                isinstance(cls, types.UnionType)
                and len(cls.__args__) == 2
                and cls.__args__[1] == type(None)
                and is_dataclass(cls.__args__[0])
                and isinstance(cfg, dict)
            ):
                fieldtypes = typing.get_type_hints(cls.__args__[0])
                return cls.__args__[0](
                    **{
                        k: __dict_to_dataclass(v, fieldtypes[k], k)
                        for k, v in cfg.items()
                    }
                )
            if typing.get_origin(cls) == list:
                return [
                    __dict_to_dataclass(v, typing.get_args(cls)[0], key) for v in cfg
                ]
            if cls == Path or (
                isinstance(cls, types.UnionType)
                and Path in cls.__args__
                and cfg is not None
            ):
                return Path(cfg)
        except:
            print(key)
            raise
        return cfg

    return __dict_to_dataclass(cfg, cls, "")  # pyright: ignore [reportReturnType]


def register_subcommand(name: str) -> Callable[[Callable[P, None]], Callable[P, None]]:
    def decorator(fn: Callable[P, None]) -> Callable[P, None]:
        __subcommands[name] = fn
        return fn

    return decorator


def add_subcommands(parser: ArgumentParser):
    subcommands = parser.add_subcommands(dest="command")
    for name, fn in __subcommands.items():
        subparser = ArgumentParser()

        signature = typing.get_type_hints(fn)
        if "config" in signature and signature["config"] == ActiveConfig:
            subparser.add_argument(
                "--experiment", action=ActionExperiment, choices=list_experiments()
            )

        subparser.add_function_arguments(fn)

        subcommands.add_subcommand(name, subparser)


def run_subcommand(args: Namespace):
    kwargs = args[args.command].as_dict()
    if "config" in kwargs:
        kwargs["config"] = _dict_to_dataclass(args[args.command].config, ActiveConfig)

    if "runtime_config" in kwargs:
        kwargs["runtime_config"] = _dict_to_dataclass(
            args[args.command].runtime_config, RuntimeConfig
        )

    if "experiment" in kwargs:
        del kwargs["experiment"]
    __subcommands[args.command](**kwargs)
