from argparse import ArgumentParser, Namespace
from pathlib import Path

from nnactive.analysis import compare_multi_experiment_results
from nnactive.cli.registry import register_subcommand
from nnactive.config.struct import ActiveConfig, RuntimeConfig
from nnactive.paths import get_nnActive_results


@register_subcommand("analyze_experiments")
def analyze_experiments(
    config: ActiveConfig = ActiveConfig([0, 0, 0]),
    runtime_config: RuntimeConfig = RuntimeConfig(),
    base_path: str = get_nnActive_results(),
    base_dataset_id: int | None = None,
):
    base_path = Path(base_path)
    compare_multi_experiment_results(base_path, base_dataset_id)
