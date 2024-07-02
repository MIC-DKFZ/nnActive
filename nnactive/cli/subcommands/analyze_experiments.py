from pathlib import Path

from nnactive.analyze.analyze_queries import analyze_queries_from_probs
from nnactive.analyze.analyze_results import analyze_multi_experiment_results
from nnactive.analyze.qualitative_loops import visualize_query_trajectory
from nnactive.cli.registry import register_subcommand
from nnactive.paths import get_nnActive_data, get_nnActive_results


@register_subcommand("analyze_experiments")
def analyze_experiments(
    base_path: str | Path = get_nnActive_results(),
    raw_path: str | Path = get_nnActive_data(),
    output_path: str | Path = Path("."),
    all_plots: bool = True,
):
    """Analyze Experiment Results. Requires nnUNet_raw data to be set up properly.

    Args:
        base_path (str | Path, optional): path with nnActive_results folder structure. Defaults to get_nnActive_results().
        raw_path (str | Path | None, optional): path with nnActive_data folder structure. get_nnActive_data().
        output_path (str | Path, optional): Path where to output results. Defaults to Path(".").
        all_plots (bool, optional): Whether to create all possible plots. Defaults to True.
    """
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


@register_subcommand("visualize_queries_from_results_path")
def entry_visualize_queries_from_probs_from_results_folder(
    results_folder: str, loop_val: int | None = None
):
    """Simulate and visualize queries from probs.
    Probs are expected to be stored in structure:
    {experiment_raw_folder}/analysis/loop_{loop_val}/predTr_{fold}


    Args:
        results_folder (str): Path to exact experiment results.
        loop_val (int | None, optional): loop value. Defaults to None.
    """
    results_folder = Path(results_folder)
    analyze_queries_from_probs(results_folder, loop_val)


@register_subcommand("visualize_query_trajectory")
def entry_visualize_query_trajectory(raw_folder: str, output_folder: str | None = None):
    """Visualize queries for an experiment.
    If output_folder is None they are saved in raw_folder/query_anlaysis/loop_{id}/...

    Args:
        raw_folder (str): folder to one nnUNet_raw folder in experiment.
        output_folder (str | None, optional): see description. Defaults to None.
    """
    raw_folder = Path(raw_folder)
    if output_folder is None:
        output_folder = raw_folder / "query__analysis"
    output_folder = Path(output_folder)
    visualize_query_trajectory(raw_folder, output_folder)
