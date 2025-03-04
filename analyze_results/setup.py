import os
from pathlib import Path

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import kendalltau

from nnactive.analyze.aggregate_results import pretty_auc
from nnactive.analyze.analysis import SettingAnalysis
from nnactive.utils.io import save_df_to_txt

CUSTOM_ORDER = [
    "mutual_information",
    "power_bald",
    "softrank_bald",
    "pred_entropy",
    "power_pe",
    "random",
    "random-label",
    "random-label2",
]

small_dict = {
    "mutual information": "BALD",
    "power bald": "PowerBALD",
    "softrank bald": "SoftrankBALD",
    "pred entropy": "Predictive Entropy",
    "power pe": "PowerPE",
    "random": "Random",
    "random-label": "Random 66% FG",
    "random-label2": "Random 33% FG",
}

RENAMING_DICT = {}
keys = list(small_dict.keys())
for key in keys:
    RENAMING_DICT[key] = small_dict[key]
    RENAMING_DICT[key.replace(" ", "_")] = small_dict[key]

VALUE_TO_COLOR_MAP = {
    -2: "#FF0000",  # red
    -1: "#F08080",  # lightcoral
    0: "#FFFFFF",  # white
    1: "#90EE90",  # light green
    2: "#008000",  # green
}

QM_TO_COLOR = {
    "BALD": "#bcbd22",  # Yellow-green
    "PowerBALD": "#ff7f0e",  # Orange
    "SoftrankBALD": "#7f7f7f",  # Gray
    "PowerPE": "#2ca02c",  # Green
    "Predictive Entropy": "#1f77b4",  # Blue
    "Random": "#9467bd",  # Purple
    "Random 66% FG": "#e377c2",  # Light Red
    "Random 33% FG": "#d62728",  # Red
}

BASEPATH = Path(
    "/home/c817h/Documents/projects/nnactive_project/nnactive/results/horeka_rsync_final/"
)

SAVEPATH = Path(
    "/home/c817h/Documents/projects/nnactive_project/nnactive/results/horeka_rsync_eval/"
)
if not SAVEPATH.exists():
    print("Creating Savepath: ", SAVEPATH)
    os.makedirs(SAVEPATH)


def df_to_multicol(df):
    """Inplace conversion of DataFrame columns to MultiIndex"""
    column_map = {}
    for col in df.columns:
        s_col = col.split(" ")
        column_map[col] = (s_col[0], " ".join(s_col[1:2]))
    df.columns = pd.MultiIndex.from_tuples([column_map[col] for col in df.columns])


def html_to_latex_color(hex_color):
    rgb = mcolors.hex2color(hex_color)  # Convert to (R, G, B) tuple (0-1 scale)
    return f"{{rgb,1:red,{rgb[0]:.2f};green,{rgb[1]:.2f};blue,{rgb[2]:.2f}}}"


def get_ranking_cmap(
    values: np.ndarray,
    significances: np.ndarray,
    colormapping: dict[int, str] = VALUE_TO_COLOR_MAP,
):
    vmap = np.zeros(values.shape, dtype=np.int8)
    vmap[values > 0] = 1
    vmap[values < 0] = -1
    vmap[significances] = vmap[significances] * 2
    cmap = np.array([[colormapping[v] for v in row] for row in vmap])
    return cmap


def compute_column_normalized_gmap(data: pd.DataFrame, invert: bool):

    # NOTE: Manually compute gradient map because Normalize returns 0 if vmax - vmin == 0, but we
    # NOTE:   want it to be 1 in that case

    gmap = data.to_numpy(float)
    gmap_min = np.nanmin(gmap, axis=0)
    gmap_max = np.nanmax(gmap, axis=0)

    for col in range(gmap.shape[1]):
        vmin = gmap_min[col] - (0.0001 if invert else 0)
        vmax = gmap_max[col] + (0 if invert else 0.0001)
        gmap_use = gmap
        if invert:
            vmin_0 = vmin
            vmin = -vmax
            vmax = -vmin_0
            gmap_use = -gmap

        gmap[:, col] = mcolors.Normalize(vmin, vmax)(gmap_use[:, col])

    return gmap


def load_setting_data_to_df(
    CUSTOM_ORDER: list[str],
    FINAL_COLUMNS: list[dict],
    setting_paths: dict[str, dict[str, dict[str, Path]]],
    setting_analyses: dict[str, dict[str, dict[str, SettingAnalysis]]],
) -> dict[str, dict[str, dict[str, pd.DataFrame]]]:
    """Load the data from the setting paths and analyses into a dictionary of DataFrames

    Args:
        CUSTOM_ORDER (list[str]): List of query methods to load in the order they should be displayed
        FINAL_COLUMNS (list[dict]): List of dictionaries with the columns to load and their names
        setting_paths (dict[str, dict[str, dict[str, Path]]]): from evalutor
        setting_analyses (dict[str, dict[str, dict[str, SettingAnalysis]]]): from evalutor

    Returns:
        dict[str, dict[str, dict[str, pd.DataFrame]]]: combined pandas DataFrames in nested dict.
    """
    data_dict = {}
    # Loading all data and preprocessing
    for dataset in setting_analyses:
        data_dict[dataset] = {}
        for budget in setting_analyses[dataset]:
            data_dict[dataset][budget] = {}
            for sett in setting_analyses[dataset][budget]:
                analysis = setting_analyses[dataset][budget][sett]
                auc = pd.read_json(setting_paths[dataset][budget][sett] / "auc.json")
                auc.index.name = "Query Method"
                auc = auc.map(lambda x: np.round(x * 100, 2))
                beta = pd.read_json(setting_paths[dataset][budget][sett] / "beta.json")
                beta.set_index("Query Method", inplace=True)
                beta = beta.round(2)
                combined_df = pd.concat([auc, beta], axis=1)
                combined_df = combined_df.reindex(CUSTOM_ORDER, level=0)
                combined_df = combined_df.rename(RENAMING_DICT, axis=0)
                combined_df = combined_df[[c["ReadCol"] for c in FINAL_COLUMNS]]
                combined_df.rename(
                    columns={c["ReadCol"]: c["PrintCol"] for c in FINAL_COLUMNS},
                    inplace=True,
                )

                data_dict[dataset][budget][sett] = combined_df
    return data_dict


def calculate_difference_with_std(
    df_pos: pd.DataFrame,
    df_neg: pd.DataFrame,
    mean_key: tuple[str, str],
    std_key: tuple[str, str],
) -> pd.DataFrame:
    df_diff = df_pos[mean_key] - df_neg[mean_key]
    df_std = np.sqrt((df_pos[std_key] ** 2 + df_neg[std_key] ** 2))
    df_diff = pd.concat(
        [df_diff, df_std],
        axis=1,
        keys=[(mean_key[0], "mean"), (std_key[0], "mean std")],
    )
    return df_diff


def compute_kendalltau_correlation_from_dfs(
    dfs: list[pd.DataFrame], qms: list[str], metric: str, significance: float = 0.10
):
    results = []
    for qm in qms:
        values = []
        compare_ranking = []
        for i, df in enumerate(dfs):
            df_sub = df[df["Query Method"] == qm]
            values.extend([v for v in df_sub[metric]])
            compare_ranking.extend([i] * len(df_sub[metric]))
        corr, pval = kendalltau(values, compare_ranking)
        results.append(
            {
                "Query Method": qm,
                "corr": corr,
                "pval": pval,
                "significance": pval < significance,
            }
        )
    return pd.DataFrame(results).set_index("Query Method")


def apply_latex_coloring(df: pd.DataFrame, color_array: np.ndarray) -> pd.DataFrame:
    styled = df.copy().astype(str)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            color = html_to_latex_color(color_array[i, j])
            styled.iloc[i, j] = f"\\cellcolor{color} {df.iloc[i, j]}"
    return styled


def save_styled_to_latex(df: pd.DataFrame, save_path: Path) -> pd.DataFrame:
    df.to_latex(save_path, escape=False, multicolumn_format="c", multirow=True)


#################### Fixing Hippocampus Experiments automatically ####################
def shorten_hippocampus(BASEPATH):
    hippcompaus_path = BASEPATH / "Dataset004_Hippocampus"
    datapath = (
        hippcompaus_path / "patch-20_20_20__sb-random-label2-all-classes__sbs-20__qs-20"
    )
    save_path = hippcompaus_path / f"{datapath.name}__5loops"
    os.makedirs(save_path, exist_ok=True)

    analysis = SettingAnalysis.load(datapath / "analysis.pkl")

    df_filter = analysis.df[analysis.df["Loop"] <= 4]
    df_filter[analysis.max_loops_key] = 5
    df_filter = df_filter.reset_index()

    analysis.df = df_filter

    analysis.save(save_path / "analysis.pkl")

    # overview metrics
    auc_df = analysis.compute_auc_df()

    analysis.save(save_path=save_path / "analysis.pkl")

    # overview metrics
    auc_df = analysis.compute_auc_df()
    # pprint(auc_df)
    auc_df.to_json(save_path / "auc.json")
    save_df_to_txt(auc_df, save_path / "auc.txt")
    save_df_to_txt(
        pretty_auc(pd.read_json(save_path / "auc.json"), seeds=True),
        save_path / "auc_pretty.txt",
    )

    ppm = analysis.compute_pairwise_penalty("Mean Dice")
    ppm.plot_pairwise_matrix(ppm.matrix, savepath=save_path / "ppm.png")
    ppm.save(save_path / "ppm.json")

    trainer = str(analysis.df["trainer"].unique()[0])
    trainer_use = "nnUNetTrainer"
    if len(trainer.split("_")) > 1:
        epochs = trainer.split("_")
        trainer_use = f"{trainer_use}_{epochs[-1]}"
        logger.info(f"Using Full Performance Trainer: {trainer_use}")
    trainers = [
        f.label for f in analysis.full_performance_dict[analysis.main_performance_key]
    ]
    compute_beta = True
    if trainer_use not in trainers:
        if len(trainers) > 0:
            trainer_use = trainers[0]
            logger.info(
                f"Using substitute Full Performance Trainer {trainer_use} from {trainers}"
            )
        else:
            compute_beta = False
    if compute_beta:
        betas = analysis.compute_beta_curve(
            trainer_use,
            "percentage_of_voxels_foreground",
        )
        betas_df = betas.to_beta_df()
        betas_df.to_json(save_path / "beta.json")
        save_df_to_txt(betas_df, save_path / "beta.txt")

    # overview plots
    selected_classes = None
    n_performance_cols = 3
    if selected_classes is None:
        selected_classes = [
            int(i.split(" ")[1]) for i in analysis.df.columns if i.startswith("Class")
        ][:3]
        while len(selected_classes) < n_performance_cols:
            selected_classes.append(None)


shorten_hippocampus(BASEPATH)
