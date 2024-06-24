from pathlib import Path

from nnactive.analysis import analyze_multi_experiment_results
from nnactive.cli.registry import register_subcommand
from nnactive.config.struct import ActiveConfig, RuntimeConfig
from nnactive.paths import get_nnActive_results


@register_subcommand("analyze_experiments")
def analyze_experiments(
    config: ActiveConfig = ActiveConfig([0, 0, 0]),
    runtime_config: RuntimeConfig = RuntimeConfig(),
    base_path: str | Path = get_nnActive_results(),
    raw_path: str | Path | None = None,
    output_path: str | Path = Path("."),
    all_plots: bool = True,
):
    if raw_path is not None:
        raw_path = Path(raw_path)
    base_path = Path(base_path)
    output_path = Path(output_path)
    analyze_multi_experiment_results(
        base_path,
        base_raw_path=raw_path,
        filter_final=True,
        output_dir=output_path,
        all_plots=all_plots,
    )
