from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from evaluator import (
    get_settings_for_combination,
    load_settings,
    rename_settings_in_analysis,
)
from setup import CUSTOM_ORDER, QM_TO_COLOR, RENAMING_DICT, SAVEPATH

from nnactive.analyze.metrics import PairwiseMatrix, PairwisePenaltyMatrix
from nnactive.utils.io import save_df_to_txt

# For final version in paper
# When using increase dpi and save to .pdf as vector graphic!
# Increase font size or reduce figure size to make text readable in paper!
# matplotlib.rcParams["text.usetex"] = True
# matplotlib.rcParams["font.family"] = "Computer Modern"

MAIN_ORDER = CUSTOM_ORDER
LEGEND = False

NORANDOM_ORDER = MAIN_ORDER.copy()
NORANDOM_ORDER.remove("random")
# NORANDOM_ORDER.remove("class_pe33")
MINIORDER = [
    "class_pe66",
    "class_power_pe66_exp",
    "random-label",
    "random-label2",
]
savepath = SAVEPATH / "figures"
IMGTYPE = "png"

COMPARATIVE = False

USE_SETTINGS_LIST = [
    # {"setting_names": ["Main"], "custom_order": MAIN_ORDER},
    # {"setting_names": ["500 Epochs"], "custom_order": NORANDOM_ORDER},
    # {"setting_names": ["Precomputed"], "custom_order": NORANDOM_ORDER},
    # {"setting_names": ["Precomputed"], "custom_order": MAIN_ORDER},
    # {"setting_names": ["Patchx1/2"], "custom_order": MAIN_ORDER},
    # Used for nnActive_v2
    {"setting_names": ["500 Epochs"], "custom_order": MINIORDER},
    {"setting_names": ["Main"], "custom_order": MINIORDER},
]

RENAME_SETTINGS = None

RANDOM_BASELINES = [
    "Random 33% FG",
    "Random 66% FG",
    # "Random",  # 500 Epochs setting does not run with this baseline
]
SORT_BY_PERFORMANCE = False
MIRROR_BAR_PLOTS = False


def pairwisematrix_to_df(matrix: PairwiseMatrix, name_dict=None):
    norm_val = merged_matrix.max_pos_ent

    # Convert matrix to DataFrame
    matrix = matrix.matrix
    df_matrix = PairwiseMatrix.creat_vis_df(matrix, round=norm_val is not None)

    # Rename columns and index if name_dict is provided
    if name_dict:
        df_matrix.rename(columns=name_dict, index=name_dict, inplace=True)

    if norm_val:
        df_matrix = df_matrix / norm_val * 100
        df_matrix = df_matrix.round(1)

    for i in range(df_matrix.shape[1]):
        df_matrix.iloc[i, i] = np.nan
    order = list(df_matrix.index)
    df_matrix.loc["Delete"] = np.nan
    df_matrix = df_matrix.reindex(order[:-1] + ["Delete"] + order[-1:])
    return df_matrix


def rel_improvement_barplot(
    data: pd.DataFrame,
    out_path: Path | str,
    sort_by_performance: bool = False,
    mirrored: bool = False,
    value_text: bool = True,
    baseline: str = "Random 66% FG",
    add_legend: bool = True,
):
    to_drop = ["Mean", "Random", "Random 33% FG", "Random 66% FG"]
    to_drop.remove(baseline)
    data = data.drop(to_drop, axis=0, errors="ignore")

    # Extract column (*, Random 66%)
    df_col = data[[baseline]].dropna().reset_index()
    df_col.columns = ["Method", "Value"]

    # Extract row (Random 66%, *)
    df_row = data.loc[baseline].dropna().reset_index()
    df_row.columns = ["Method", "Value"]

    if sort_by_performance:
        # Sort by the upward values (*, Random 66%) in descending order
        df_col = df_col.sort_values(by="Value", ascending=True)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6, 3))

    bar_width = 0.8 if mirrored else 0.4
    x = np.arange(len(df_col))

    # Plot bars with consistent colors and different textures
    for i, method in enumerate(df_col["Method"]):
        if method == "Mean":
            continue

        color = QM_TO_COLOR[method]

        # Upward bars (*, Random 66%)
        ax.bar(
            x[i] if mirrored else (x[i] - bar_width / 2),
            df_col.loc[df_col["Method"] == method, "Value"],
            color=color,
            width=bar_width,
        )
        # Show thick line at 0 so that it's not empty
        if not mirrored and df_col.loc[df_col["Method"] == method, "Value"].item() == 0:
            plt.hlines(0, xmin=x[i] - bar_width, xmax=x[i], color=color, linewidth=3)
        # Downward bars (Random 66%, *)
        ax.bar(
            x[i] if mirrored else (x[i] + bar_width / 2),
            df_row.loc[df_row["Method"] == method, "Value"] * (-1 if mirrored else 1),
            color=color,
            width=bar_width,
            hatch="//",
            alpha=0.5,
        )
        # Add values as text
        if value_text:
            if mirrored:
                plt.text(
                    x[i],
                    df_col.loc[df_col["Method"] == method, "Value"].item() + 0.5,
                    s=f"{df_col.loc[df_col['Method'] == method, 'Value'].item():.0f}",
                    color=color,
                    horizontalalignment="center",
                )
                plt.text(
                    x[i],
                    -df_row.loc[df_row["Method"] == method, "Value"].item() - 0.5,
                    s=f"{df_row.loc[df_row['Method'] == method, 'Value'].item():.0f}",
                    color=color,
                    horizontalalignment="center",
                    verticalalignment="top",
                )
            else:
                plt.text(
                    x[i] - bar_width / 2,
                    df_col.loc[df_col["Method"] == method, "Value"].item() + 0.5,
                    s=f"{df_col.loc[df_col['Method'] == method, 'Value'].item():.0f}",
                    color=color,
                    horizontalalignment="center",
                )
                plt.text(
                    x[i] + bar_width / 2,
                    df_row.loc[df_row["Method"] == method, "Value"].item() + 0.5,
                    s=f"{df_row.loc[df_row['Method'] == method, 'Value'].item():.0f}",
                    color=color,
                    horizontalalignment="center",
                )

    if add_legend:
        # Create dummy legend handles
        # upper_bar_legend = mpatches.Patch(
        #     color="black", alpha=0.7, label="Outperforming Random (66% FG)"
        # )
        upper_bar_legend = mpatches.Patch(color="black", alpha=0.7, label="Wins")
        lower_bar_legend = mpatches.Patch(
            color="black",
            hatch="//",
            alpha=0.3,
            label="Losses",
        )

        # Add the legend
        ax.legend(handles=[upper_bar_legend, lower_bar_legend])
    ax.grid(axis="x")
    ax.set_axisbelow(True)

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(
        [x if i % 2 == 0 else f"\n{x}" for i, x in enumerate(df_col["Method"])]
    )
    # ax.set_xticklabels(df_col["Method"], rotation=45, ha="center")
    # plt.ylabel("Fraction of main study experiments [%]")
    plt.ylabel("Fraction of experiments [%]")
    plt.axhline(0, color="black", linewidth=1)

    # Make all yticklabels positive
    if mirrored:
        ax.set_yticklabels([f"{abs(int(tick))}" for tick in ax.get_yticks()])

    print(f"Saving to {out_path}")
    plt.savefig(out_path, bbox_inches="tight")


if __name__ == "__main__":

    for setting in USE_SETTINGS_LIST:
        setting_names = setting["setting_names"]
        custom_order = setting["custom_order"]
        setting_paths = get_settings_for_combination(setting_names)
        setting_data = load_settings(setting_paths)
        if RENAME_SETTINGS is not None:
            rename_settings_in_analysis(setting_data)
        save_setting = (
            "_".join(setting_names).replace(" ", "").replace("/", "-").lower()
        )
        print_setting = " & ".join(setting_names)

        all_matrices = {}
        for dataset in setting_data:
            all_matrices[dataset] = {}
            for budget in setting_data[dataset]:
                all_matrices[dataset][budget] = {}
                for subsetting in setting_data[dataset][budget]:
                    analysis = setting_data[dataset][budget][subsetting]
                    matrix = analysis.compute_pairwise_penalty()
                    del_algs = [a for a in matrix.algs if a not in custom_order]
                    for a in del_algs:
                        matrix.delete_alg(a)
                    matrix = matrix.custom_order_matrix(custom_order)
                    matrix.rename_algs(RENAMING_DICT)
                    all_matrices[dataset][budget][subsetting] = matrix

        for RANDOM_BASELINE in RANDOM_BASELINES:
            if RANDOM_BASELINE not in RENAMING_DICT.values():
                continue
            for dataset in all_matrices:
                merged_matrix = PairwisePenaltyMatrix.create_merged_matrix(
                    [
                        mat[subsetting]
                        for subsetting in setting_names
                        for mat in all_matrices[dataset].values()
                    ]
                )
                df_matrix = pairwisematrix_to_df(merged_matrix)
                fname = f"rel_improvement_{save_setting}_{RANDOM_BASELINE.lower().replace(' ', '').replace('%', '')}_{dataset}{'_mirrored' if MIRROR_BAR_PLOTS else ''}.{IMGTYPE}"
                rel_improvement_barplot(
                    df_matrix,
                    out_path=savepath / fname,
                    mirrored=MIRROR_BAR_PLOTS,
                    value_text=True,
                    sort_by_performance=SORT_BY_PERFORMANCE,
                    baseline=RANDOM_BASELINE,
                    add_legend=LEGEND,
                )

            merged_matrix = PairwisePenaltyMatrix.create_merged_matrix(
                [
                    all_matrices[d][b][s]
                    for d in all_matrices
                    for b in all_matrices[d]
                    for s in all_matrices[d][b]
                ]
            )
            df_matrix = pairwisematrix_to_df(merged_matrix)
            fname = f"rel_improvement_{save_setting}_{RANDOM_BASELINE.lower().replace(' ', '').replace('%', '')}{'_mirrored' if MIRROR_BAR_PLOTS else ''}.{IMGTYPE}"
            rel_improvement_barplot(
                df_matrix,
                out_path=savepath / fname,
                mirrored=MIRROR_BAR_PLOTS,
                value_text=True,
                sort_by_performance=SORT_BY_PERFORMANCE,
                baseline=RANDOM_BASELINE,
                add_legend=LEGEND,
            )
