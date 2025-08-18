from pathlib import Path

from tqdm import tqdm

from nnactive.analyze.analyze_queries import (
    analyze_queries_from_probs,
    predict_trainingset_model,
)
from nnactive.analyze.analyze_results import analyze_multi_experiment_results
from nnactive.analyze.qualitative_loops import (
    plot_query_trajectory,
    plot_region_predictions_across_loops,
    visualize_query_trajectory,
)
from nnactive.cli.registry import register_subcommand
from nnactive.paths import get_nnActive_data, get_nnActive_results


@register_subcommand("analyze_experiments")
def analyze_experiments(
    base_path: str | Path = get_nnActive_results(),
    raw_path: str | Path = get_nnActive_data(),
    output_path: str | Path = Path("."),
):
    """Analyze Experiment Results. Requires nnUNet_raw data to be set up properly.

    Args:
        base_path (str | Path, optional): path with nnActive_results folder structure. Defaults to get_nnActive_results().
        raw_path (str | Path | None, optional): path with nnActive_data folder structure. get_nnActive_data().
        output_path (str | Path, optional): Path where to output results. Defaults to Path(".").
    """
    raw_path = Path(raw_path)
    base_path = Path(base_path)
    output_path = Path(output_path)
    analyze_multi_experiment_results(
        base_path,
        base_raw_path=raw_path,
        filter_final=True,
        output_dir=output_path,
    )


@register_subcommand("visualize_queries_from_probs")
def visualize_queries_from_probs(results_folder: str, loop_val: int | None = None):
    """Simulate and visualize queries from probs.
    Probs are expected to be stored in structure:
    {experiment_raw_folder}/analysis/loop_{loop_val}/predTr_{fold}


    Args:
        results_folder (str): Path to exact experiment results with config.json.
        loop_val (int | None, optional): loop value. Defaults to None.
    """
    results_folder = Path(results_folder)
    analyze_queries_from_probs(results_folder, loop_val)


@register_subcommand("analyze_predict_trainingset_model")
def analyze_predict_trainingset_model(
    results_folder: Path,
    folds: int | list[int],
    loop_val: int | None = None,
    npp: int = 3,
    nps: int = 3,
    disable_progress_bar: bool = False,
    num_parts: int = 1,
    part_id: int = 0,
    verbose: bool = False,
):
    """Predict fold models on training set for specified loop and saves predictions and outputs.
    Outputs will be stored in structure:
    {experiment_raw_folder}/analysis/loop_{loop_val}/predTr_{fold}

    Args:
        results_folder (Path): Path to exact experiment results with config.json.
        folds (int | list[int]): folds for prediction (if list give all)
        loop_val (int | None, optional): loop value. Defaults to None.
        npp (int, optional): num processes preprocessing. Defaults to 3.
        nps (int, optional): num processes postprocessing. Defaults to 3.
        disable_progress_bar (bool, optional): useful on cluster. Defaults to False.
        num_parts (int, optional): splits prediction into multiple parts (filewise). Defaults to 1.
        part_id (int, optional): which part is executed (starts with 0). Defaults to 0.
        verbose (bool, optional): read a lot. Defaults to False.
    """
    results_folder = Path(results_folder)
    predict_trainingset_model(
        results_folder,
        folds,
        loop_val,
        npp,
        nps,
        disable_progress_bar,
        num_parts,
        part_id,
        verbose,
    )


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


@register_subcommand("visualize_queries_images")
def entry_visualize_queries_iamges(
    raw_folder: str, output_folder: str | None = None, img_folder: str | None = None
):
    raw_folder = Path(raw_folder)
    if output_folder is None:
        output_folder = raw_folder / "query__images"
    else:
        output_folder = Path(output_folder) / raw_folder.name

    if img_folder is None:
        img_folder = raw_folder / "imagesTr"
    img_folder = Path(img_folder)

    plot_query_trajectory(
        raw_folder=raw_folder, img_folder=img_folder, save_folder=output_folder
    )


@register_subcommand("visualize_region_predictions")
def entry_visualize_region_predictions(
    img_folder: str,
    gt_folder: str,
    raw_folder: str | None = None,
    raw_folders_from_file: str | None = None,
    image_name: str | None = None,
    slice_axis: int = 0,
    results_folder: str | None = None,
    output_folder: str | None = None,
    max_loops: int | None = None,
):
    img_folder = Path(img_folder)
    gt_folder = Path(gt_folder)

    if image_name is not None:
        image_names = [image_name]
    else:
        image_names = [f.name for f in gt_folder.iterdir() if f.is_file()]

    for name in tqdm(sorted(image_names), total=len(image_names)):
        plot_region_predictions_across_loops(
            raw_folder=raw_folder,
            raw_folders_from_file=raw_folders_from_file,
            results_folder=results_folder,
            image_name=name,
            slice_axis=slice_axis,
            img_folder=img_folder,
            gt_folder=gt_folder,
            save_folder=output_folder,
            max_loops=max_loops,
        )
