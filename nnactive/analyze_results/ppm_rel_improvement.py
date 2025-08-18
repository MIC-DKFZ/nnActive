from collections import defaultdict
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
from setup import CUSTOM_ORDER, QM_TO_COLOR, RENAMING_DICT, SAVEPATH, SAVETYPE

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
MINIORDER = [
    "random-label",
    "random-label2",
]
savepath = SAVEPATH / "figures"

COMPARATIVE = False

USE_SETTINGS_LIST = [
    {"setting_names": ["Main"], "custom_order": MAIN_ORDER},
    {"setting_names": ["500 Epochs"], "custom_order": NORANDOM_ORDER},
    # {"setting_names": ["Precomputed"], "custom_order": NORANDOM_ORDER},
    # {"setting_names": ["Precomputed"], "custom_order": MAIN_ORDER},
    # {"setting_names": ["Patchx1/2"], "custom_order": MAIN_ORDER},
    # Used for nnActive_v2
    # {"setting_names": ["500 Epochs"], "custom_order": MINIORDER},
    # {"setting_names": ["Main"], "custom_order": MINIORDER},
    {
        "setting_names": ["Main", "500 Epochs"],
        "custom_order": MINIORDER,
        "print_name_settings_dict": {"Main": "200 Epochs"},
    },
]

for setting in USE_SETTINGS_LIST:
    if "print_name_settings_dict" not in setting:
        setting["print_name_settings_dict"] = {}

RENAME_SETTINGS = None

RANDOM_BASELINES = [
    "Random 33% FG",
    "Random 66% FG",
    # "Random",  # 500 Epochs setting does not run with this baseline
]
SORT_BY_PERFORMANCE = False
MIRROR_BAR_PLOTS = False


def pairwisematrix_to_df(matrix: PairwiseMatrix, name_dict=None):
    norm_val = matrix.max_pos_ent

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
        print_name_settings_dict = setting["print_name_settings_dict"]
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

        for random_baseline in RANDOM_BASELINES:
            if (
                random_baseline not in RENAMING_DICT.values()
                or random_baseline not in custom_order
            ):
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
                fname = f"rel_improvement_{save_setting}_{random_baseline.lower().replace(' ', '').replace('%', '')}_{dataset}{'_mirrored' if MIRROR_BAR_PLOTS else ''}.{SAVETYPE}"
                rel_improvement_barplot(
                    df_matrix,
                    out_path=savepath / fname,
                    mirrored=MIRROR_BAR_PLOTS,
                    value_text=True,
                    sort_by_performance=SORT_BY_PERFORMANCE,
                    baseline=random_baseline,
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
            fname = f"rel_improvement_{save_setting}_{random_baseline.lower().replace(' ', '').replace('%', '')}{'_mirrored' if MIRROR_BAR_PLOTS else ''}.{SAVETYPE}"
            rel_improvement_barplot(
                df_matrix,
                out_path=savepath / fname,
                mirrored=MIRROR_BAR_PLOTS,
                value_text=True,
                sort_by_performance=SORT_BY_PERFORMANCE,
                baseline=random_baseline,
                add_legend=LEGEND,
            )
        ######################## Win Loss single method Barplots ########################
        for method in custom_order:
            method_name = RENAMING_DICT.get(method, method)
            if method_name in RANDOM_BASELINES:
                continue
            for dataset, budget_matrices in all_matrices.items():
                x = np.arange(len(RANDOM_BASELINES))
                bar_width = 0.4
                df_dict = {}
                colors = ["red", "orange", "purple"]
                labels = ["Low", "Medium", "High"]

                setting_wins = defaultdict(list)
                setting_losses = defaultdict(list)
                setting_max_val = defaultdict(lambda: 0)

                for budget, subsetting_matrices in budget_matrices.items():
                    df_dict[budget] = {}
                    for subsetting, matrix in subsetting_matrices.items():
                        if method_name not in matrix.algs:
                            continue
                        df_matrix = pairwisematrix_to_df(matrix)
                        df_dict[budget][subsetting] = df_matrix
                        wins = df_matrix.loc[method_name, RANDOM_BASELINES]
                        losses = df_matrix.loc[RANDOM_BASELINES, method_name]
                        setting_wins[subsetting].append(wins)
                        setting_losses[subsetting].append(losses)
                        setting_max_val[subsetting] += matrix.max_pos_ent

                fig, axs = plt.subplots(
                    1,
                    len(setting_wins),
                    figsize=(6 * len(setting_wins), 3),
                    sharey=True,
                )
                if not isinstance(axs, np.ndarray):
                    axs = [axs]

                for i_s, (s) in enumerate(setting_wins):
                    wins_arr = setting_wins[s]
                    losses_arr = setting_losses[s]
                    ax = axs[i_s]
                    np_wins = np.array(wins_arr) / setting_max_val[s]
                    np_losses = np.array(losses_arr) / setting_max_val[s]
                    cumsum_wins = np.cumsum(np_wins, axis=0)
                    cumsum_losses = np.cumsum(np_losses, axis=0)

                    for i in range(len(wins_arr)):
                        ax.bar(
                            x - bar_width / 2,
                            np_wins[i],
                            width=bar_width,
                            color=colors[i],
                            bottom=cumsum_wins[i - 1] if i > 0 else 0,
                        )
                        ax.bar(
                            x + bar_width / 2,
                            np_losses[i],
                            width=bar_width,
                            color=colors[i],
                            bottom=cumsum_losses[i - 1] if i > 0 else 0,
                            hatch="//",
                            # alpha=0.5,
                            alpha=1,
                        )
                    ax.grid(axis="x")
                    xticks = np.concatenate(
                        [x - bar_width / 2, x + bar_width / 2, x], axis=0
                    )
                    xlabels = (
                        ["Wins"] * len(x)
                        + ["Losses"] * len(x)
                        + [f"\n{method_name} vs. {r}" for r in RANDOM_BASELINES]
                        # + [f"\n{method_name} vs. \n {r}" for r in RANDOM_BASELINES]
                    )
                    ax.set_xticks(xticks, labels=xlabels)
                    for l in ax.get_xticklabels()[-len(x) :]:
                        l.set_fontweight("bold")
                    ax.set_title(print_name_settings_dict.get(s, s))
                    if i_s == 0:
                        print("Setting Label")
                        ax.set_ylabel("Fraction of experiments [%]")
                        pass

                    if len(setting_wins) == 1:
                        legends = []
                        for i, label in enumerate(labels):
                            legends.append(mpatches.Patch(color=colors[i], label=label))
                        # Add the legend
                        ax.legend(handles=legends)
                legends = (
                    [mpatches.Patch(color="None", label="Label Regime:")]
                    + [
                        mpatches.Patch(color=colors[i], label=labels[i])
                        for i in range(len(labels))
                    ]
                    + [mpatches.Patch(color="None", label="Outcome:")]
                    + [
                        mpatches.Patch(color="black", alpha=1, label="Wins"),
                        mpatches.Patch(
                            facecolor="white",
                            # color="black",
                            edgecolor="black",
                            hatch="//",
                            alpha=1,
                            label="Losses",
                            # edgecolor="white",
                        ),
                    ]
                )
                fig.tight_layout()
                fig.legend(
                    loc="lower center",
                    bbox_to_anchor=(0.5, -0.1),
                    ncols=len(legends),
                    handles=legends,
                )
                if not savepath.is_dir():
                    savepath.mkdir(parents=True, exist_ok=True)
                plt.savefig(
                    savepath / "ppm_rel_improvement_stacked_"
                    f"{save_setting}_{method_name.lower().replace(' ', '').replace('%', '')}_{dataset}{'_mirrored' if MIRROR_BAR_PLOTS else ''}.{SAVETYPE}",
                    bbox_inches="tight",
                )
                plt.close()

        ######################## Win Neutral Loss single method Barplots ########################
        for method in custom_order:
            method_name = RENAMING_DICT.get(method, method)
            if method_name in RANDOM_BASELINES:
                continue
            for dataset, budget_matrices in all_matrices.items():
                x = np.arange(len(RANDOM_BASELINES))
                bar_width = 0.3
                df_dict = {}
                colors = ["red", "orange", "purple"]
                labels = ["Low", "Medium", "High"]

                setting_wins = defaultdict(list)
                setting_losses = defaultdict(list)

                setting_max_val = defaultdict(lambda: 0)

                for budget, subsetting_matrices in budget_matrices.items():
                    df_dict[budget] = {}
                    for subsetting, matrix in subsetting_matrices.items():
                        if method_name not in matrix.algs:
                            continue
                        df_matrix = pairwisematrix_to_df(matrix)
                        df_dict[budget][subsetting] = df_matrix
                        wins = df_matrix.loc[method_name, RANDOM_BASELINES]
                        losses = df_matrix.loc[RANDOM_BASELINES, method_name]
                        setting_wins[subsetting].append(wins)
                        setting_losses[subsetting].append(losses)
                        setting_max_val[subsetting] += matrix.max_pos_ent

                fig, axs = plt.subplots(
                    1,
                    len(setting_wins),
                    figsize=(6 * len(setting_wins), 3),
                    sharey=True,
                )
                if not isinstance(axs, np.ndarray):
                    axs = [axs]
                for i_s, (s) in enumerate(setting_wins):
                    wins_arr = setting_wins[s]
                    losses_arr = setting_losses[s]
                    ax = axs[i_s]
                    np_wins = np.array(wins_arr) / setting_max_val[s]
                    np_losses = np.array(losses_arr) / setting_max_val[s]
                    np_neutral = 100 / setting_max_val[s] - (np_wins + np_losses)
                    cumsum_neutral = np.cumsum(np_neutral, axis=0)
                    cumsum_wins = np.cumsum(np_wins, axis=0)
                    cumsum_losses = np.cumsum(np_losses, axis=0)
                    for i in range(len(wins_arr)):
                        ax.bar(
                            x - bar_width,
                            np_wins[i],
                            width=bar_width,
                            color=colors[i],
                            bottom=cumsum_wins[i - 1] if i > 0 else 0,
                        )
                        ax.bar(
                            x,
                            np_neutral[i],
                            width=bar_width,
                            color=colors[i],
                            # hatch="/",
                            alpha=0.6,
                            bottom=cumsum_neutral[i - 1] if i > 0 else 0,
                        )
                        ax.bar(
                            x + bar_width,
                            np_losses[i],
                            width=bar_width,
                            color=colors[i],
                            bottom=cumsum_losses[i - 1] if i > 0 else 0,
                            hatch="\\",
                            # alpha=0.5,
                            alpha=1,
                        )
                    ax.grid(axis="x")
                    xticks = np.concatenate(
                        [
                            x - bar_width,
                            x + bar_width,
                            x,
                            x + 1e-10,
                        ],
                        axis=0,
                    )
                    xlabels = (
                        ["Win"] * len(x)
                        + ["Loss"] * len(x)
                        + ["Tie"] * len(x)
                        + [f"\n{method_name} vs. {r}" for r in RANDOM_BASELINES]
                        # + [f"Tie\n{method_name} vs. {r}" for r in RANDOM_BASELINES]
                    )
                    ax.set_xticks(xticks, labels=xlabels)
                    for l in ax.get_xticklabels()[-len(x) :]:
                        l.set_fontweight("bold")
                    ax.set_title(print_name_settings_dict.get(s, s))
                    if i_s == 0:
                        print("Setting Label")
                        ax.set_ylabel("Fraction of experiments [%]")
                        pass

                    if len(setting_wins) == 1:
                        legends = []
                        for i, label in enumerate(labels):
                            legends.append(mpatches.Patch(color=colors[i], label=label))
                        # Add the legend
                        ax.legend(handles=legends)
                legends = (
                    [mpatches.Patch(color="None", label="Label Regime:")]
                    + [
                        mpatches.Patch(color=colors[i], label=labels[i])
                        for i in range(len(labels))
                    ]
                    + [mpatches.Patch(color="None", label="Outcome:")]
                    + [
                        mpatches.Patch(
                            facecolor="black",
                            edgecolor="black",
                            alpha=1,
                            label="Win",
                        ),
                        mpatches.Patch(
                            facecolor="black",
                            edgecolor="black",
                            # hatch="//",
                            alpha=0.5,
                            label="Tie",
                        ),
                        mpatches.Patch(
                            facecolor="black",
                            edgecolor="white",
                            hatch="\\\\",
                            alpha=1,
                            label="Loss",
                        ),
                    ]
                )
                fig.tight_layout()
                fig.legend(
                    loc="lower center",
                    bbox_to_anchor=(0.5, -0.1),
                    ncols=len(legends),
                    handles=legends,
                )
                plt.savefig(
                    savepath / "ppm_rel_improvement_stacked_wnl_"
                    f"{save_setting}_{method_name.lower().replace(' ', '').replace('%', '')}_{dataset}{'_mirrored' if MIRROR_BAR_PLOTS else ''}.{SAVETYPE}",
                    bbox_inches="tight",
                )
                plt.close()
