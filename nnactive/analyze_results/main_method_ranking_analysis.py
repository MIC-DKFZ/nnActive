from pathlib import Path

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
import seaborn as sns
from evaluator import (
    get_settings_for_combination,
    load_settings,
    rename_settings_in_analysis,
)
from matplotlib.axes import Axes
from scipy.stats import friedmanchisquare
from setup import (
    BASEPATH,
    CUSTOM_ORDER,
    FINAL_COLUMNS,
    MAIN_ORDER,
    QM_TO_COLOR,
    SAVEPATH,
    SAVETYPE,
    load_setting_data_to_df,
)

USETEX = False
if USETEX:
    matplotlib.rcParams["text.usetex"] = True
    matplotlib.rcParams["font.family"] = "Computer Modern"

savepath = SAVEPATH / "figures"
savepath.mkdir(exist_ok=True, parents=True)

NAME = "main_method_ranking"
COLLEVELANMES = ["Dataset", "Label Regime", "Metric"]
SCORES = ["AUBC", "Final Dice", "beta"]


COMPARATIVE = False

ADD_LEGEND = False

PLOT_CONFIGS = [
    {
        "settings": ["Main", "Patchx1/2"],
        "comparative": False,
        "rename": None,
        "copy": None,
        "order": MAIN_ORDER,
        "name": NAME + "_joined",
    },
    {
        "settings": ["Main"],
        "comparative": False,
        "rename": None,
        "copy": None,
        "order": MAIN_ORDER,
        "name": NAME,
    },
    {
        "settings": ["Patchx1/2"],
        "comparative": False,
        "rename": None,
        "copy": None,
        "order": MAIN_ORDER,
        "name": NAME,
    },
    {
        "settings": ["Main"],
        "comparative": False,
        "rename": None,
        "copy": None,
        "order": CUSTOM_ORDER,
        "name": NAME + "_ablation",
    },
]


# Sort by first and second levels, using QS numeric values for the second level
def sort_key(col):
    # Extract the numeric part of the second-level column (e.g., 'QS 20' -> 20)
    first_level, second_level = col
    second_level_numeric = int(second_level.split(" ")[-1])
    return (first_level, second_level_numeric)


def plot_row(whole_data: pd.DataFrame, ax1: Axes, ax2: Axes, metric: str):
    # Compute method rankings
    ranks: pd.DataFrame = (
        whole_data.loc[:, (slice(None), slice(None), metric)]
        .droplevel(level=2, axis=1)
        .rank(ascending=False, method="min")
    )

    # Create ranking line plot
    for method_name in ranks.index:
        if "Random" in method_name:
            markerfacecolor = "white"
            markerfacecolor = "black"
        else:
            markerfacecolor = QM_TO_COLOR[method_name]
        ax1.plot(
            ranks.loc[method_name, :].values,
            marker="o",
            label=method_name,
            color=QM_TO_COLOR[method_name],
            ls="--" if "Random" in method_name else "-",
            lw=2.5,
            markerfacecolor=markerfacecolor,
        )

    upper_limit = len(ranks) + 0.5
    ax1.add_patch(
        patches.Rectangle(
            (-0.5, 0), 3, upper_limit, linewidth=1, facecolor="k", alpha=0.1
        )
    )
    ax1.add_patch(
        patches.Rectangle(
            (2.5, 0), 3, upper_limit, linewidth=1, facecolor="k", alpha=0.03
        )
    )
    ax1.add_patch(
        patches.Rectangle(
            (5.5, 0), 3, upper_limit, linewidth=1, facecolor="k", alpha=0.1
        )
    )
    ax1.add_patch(
        patches.Rectangle(
            (8.5, 0), 3, upper_limit, linewidth=1, facecolor="k", alpha=0.03
        )
    )

    ax1.set_xlim(-0.5, ranks.shape[-1] - 0.5)
    ax1.set_ylim(0.5, ranks.shape[0] + 0.5)
    ax1.grid(axis="y")
    num_cols_per_level_0 = np.array(
        [len(ranks[level].columns) for level in ranks.columns.levels[0]]
    )

    if all(num_cols_per_level_0 == 1):
        labels = list(ranks.columns.levels[0])
    else:
        labels = []
        count = 0
        for num_col in num_cols_per_level_0:
            loc_level_0 = num_col // 2
            use_cols = ranks.columns[count : count + num_col]
            if USETEX:
                labels.extend(
                    [
                        (
                            f"{c[1]}\n\n" + rf"\textbf{{{c[0]}}}"
                            if i % num_col == loc_level_0
                            else c[1]
                        )
                        for i, c in enumerate(use_cols)
                    ]
                )
            else:
                labels.extend(
                    [
                        (f"{c[1]}\n\n{c[0]}" if i % num_col == loc_level_0 else c[1])
                        for i, c in enumerate(use_cols)
                    ]
                )
    ax1.set_xticks(ticks=np.arange(ranks.shape[-1]), labels=labels)
    ax1.set_yticks(ticks=np.arange(ranks.shape[0]) + 1)
    ax1.set_ylabel(f"Method Rank ({metric})")

    # Create mean rank plot
    avg_ranks = ranks.mean(axis=1)
    std_ranks = ranks.std(axis=1)
    ranks_sorted = avg_ranks.rank(ascending=True, method="first")
    for i, method_name in enumerate(ranks.index):
        if "Random" in method_name:
            markerfacecolor = "white"
        else:
            markerfacecolor = QM_TO_COLOR[method_name]
        ax2.errorbar(
            -0.5 + ranks_sorted[method_name] / (len(ranks_sorted) + 1),
            [avg_ranks[method_name]],
            yerr=[std_ranks[method_name]],
            fmt=".",
            markersize=8,
            color=QM_TO_COLOR[method_name],
            markerfacecolor=markerfacecolor,
            lw=2,
            # markerfacecolor=QM_TO_COLOR[method_name],
        )

    ax2.set_xlim(-0.5, 0.5)
    ax2.set_ylim(0.5, ranks.shape[0] + 0.5)
    ax2.set_xticks(ticks=[0], labels=["Mean Rank"])
    ax2.set_yticks(ticks=np.arange(ranks.shape[0]) + 1)
    ax2.tick_params("x", length=0, pad=7)
    ax2.grid(axis="x")


if __name__ == "__main__":
    for config in PLOT_CONFIGS:
        setting = config["settings"]
        rename_setting = config["rename"]
        custom_order = config["order"]
        plot_prefix = config["name"]
        print(setting)
        print_setting = "_".join(setting).replace(" ", "").replace("/", "-")
        setting_paths = get_settings_for_combination(setting)
        setting_analyses = load_settings(setting_paths, comparative=COMPARATIVE)
        if rename_setting is not None:
            rename_settings_in_analysis(setting_analyses, rename_setting)
            rename_settings_in_analysis(setting_paths, rename_setting)

        data_dict = load_setting_data_to_df(
            custom_order, FINAL_COLUMNS, setting_paths, setting_analyses
        )
        plot_metrics = [c["PrintCol"] for c in FINAL_COLUMNS if c["better"] == "higher"]

        if len(setting) == 1:
            name = setting[0].lower().replace(" ", "")
            whole_data: dict[str, dict[str, pd.DataFrame]] = {}
            for dataset in data_dict:
                whole_data[dataset] = {}
                for budget in data_dict[dataset]:
                    whole_data[dataset][budget] = data_dict[dataset][budget][setting[0]]
                whole_data[dataset] = pd.concat(
                    whole_data[dataset].values(),
                    axis=1,
                    keys=whole_data[dataset].keys(),
                    names=COLLEVELANMES[1:],
                )
            whole_data = pd.concat(
                whole_data,
                axis=1,
                keys=whole_data.keys(),
                names=COLLEVELANMES,
            )

            for metric in plot_metrics:
                fig, (ax1, ax2) = plt.subplots(
                    1,
                    2,
                    # sharey=True,
                    figsize=(14, 4),
                    width_ratios=(12, 1),
                    gridspec_kw={
                        "wspace": 0.05,  # 0.03
                    },
                )

                plot_row(whole_data, ax1, ax2, metric)

                for use_legend in [False, True]:
                    if use_legend:
                        move_down = 0.04
                        ncols = 5
                        y_offset = -(
                            0.1 + move_down * np.ceil(len(whole_data.index) / ncols)
                        )
                        # ax1.legend(loc=(0.025, -0.35), handlelength=4, ncols=5)
                        fig.legend(
                            loc="lower center",
                            bbox_to_anchor=(0.5, y_offset),
                            handlelength=4,
                            ncols=ncols,
                        )

                    metric_name = metric.lower().replace(" ", "")
                    plt.savefig(
                        savepath
                        / f"{plot_prefix}--{print_setting}--{metric_name}--l-{use_legend}.{SAVETYPE}",
                        bbox_inches="tight",
                    )

                # if ADD_LEGEND:
                #     ax1.legend(loc=(0.1, -0.25), handlelength=4, ncols=4)

                # # Save figure
                # # plt.tight_layout()
                # metric_name = metric.lower().replace(" ", "")
                # plt.savefig(
                #     savepath
                #     / f"{plot_prefix}--{print_setting}--{metric_name}.{SAVETYPE}",
                #     bbox_inches="tight",
                # )

            fig, axs = plt.subplots(
                len(plot_metrics),
                2,
                figsize=(14, 4 * len(plot_metrics)),
                # sharey=True,
                # sharex=True,
                width_ratios=(12, 1),
                gridspec_kw={"wspace": 0.05, "hspace": 0.1},
            )
            for metric, ax_row in zip(plot_metrics, axs):
                plot_row(whole_data, ax_row[0], ax_row[1], metric)
            for use_legend in [False, True]:
                if use_legend:
                    move_down = 0.04
                    ncols = 5
                    y_offset = -(
                        0.1 + move_down * np.ceil(len(whole_data.index) / ncols)
                    )
                    fig.legend(
                        loc="lower center",
                        handles=axs[-1][0].get_legend_handles_labels()[0],
                        labels=axs[-1][0].get_legend_handles_labels()[1],
                        bbox_to_anchor=(0.5, 0),
                        handlelength=4,
                        ncols=ncols,
                    )
                plt.savefig(
                    savepath
                    / f"{plot_prefix}--{print_setting}--l-{use_legend}.{SAVETYPE}",
                    bbox_inches="tight",
                )
            # Statistical test showing differences between methods
            for metric in plot_metrics:
                print(metric)
                ranks: pd.DataFrame = (
                    whole_data.loc[:, (slice(None), slice(None), metric)]
                    .droplevel(level=2, axis=1)
                    .rank(ascending=False, method="min")
                )
                stats, p = friedmanchisquare(
                    *[method_ranks[1].values for method_ranks in ranks.iterrows()]
                )
                # print(f"Friedman test for {metric}: {stats}, p-value: {p}")
                nemenyi = sp.posthoc_nemenyi_friedman(ranks.T.values)
                nemenyi.columns = nemenyi.index = ranks.index
                # print(nemenyi)
                plt.subplots(figsize=(10, 8))
                import seaborn as sns

                sns.heatmap(
                    nemenyi,
                    annot=True,
                    fmt=".2f",
                    cmap="Blues",
                    cbar=False,
                    square=True,
                    linewidths=0.5,
                    linecolor="black",
                )
                plt.title(
                    f"Post-Hoc Nemenyi test for {metric} with Frieman test (p={p*100:.2f}%)"
                )
                nemenyi_path = savepath / "nemenyi"
                nemenyi_path.mkdir(exist_ok=True, parents=True)
                plt.savefig(
                    nemenyi_path
                    / f"{plot_prefix}--{print_setting}--{metric}--nemenyi.{SAVETYPE}",
                    bbox_inches="tight",
                )
                plt.close()

            # Plot mean ranks
            for use_mean_std in [True, False]:
                fig, axs = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
                # set y axis limits

                axs[0].set_ylim(0.5, ranks.shape[0] + 0.5)
                axs[0].set_ylabel(f"Avg. Method Rank")
                axs[0].set_yticks(ticks=np.arange(ranks.shape[0]) + 1)
                for i, metric in enumerate(plot_metrics):
                    axs[i].set_title(metric)
                    ranks: pd.DataFrame = (
                        whole_data.loc[:, (slice(None), slice(None), metric)]
                        .droplevel(level=2, axis=1)
                        .rank(ascending=False, method="min")
                    )

                    mean_ranks = ranks.mean(axis=1)
                    std_ranks = ranks.std(axis=1)
                    if use_mean_std:
                        std_ranks = std_ranks / np.sqrt(len(ranks.columns))
                    ranks_sorted = mean_ranks.rank(ascending=True, method="first")
                    for method_name in mean_ranks.index:
                        if "Random" in method_name:
                            markerfacecolor = "white"
                        else:
                            markerfacecolor = QM_TO_COLOR[method_name]
                        axs[i].errorbar(
                            -0.5 + ranks_sorted[method_name] / (len(ranks_sorted) + 1),
                            y=mean_ranks[method_name],
                            yerr=std_ranks[method_name],
                            fmt=".",
                            markersize=8,
                            color=QM_TO_COLOR[method_name],
                            markerfacecolor=markerfacecolor,
                            label=method_name,
                            lw=2,
                        )
                    # remove xticks
                    axs[i].set_xticks(ticks=[])
                mean_plot_path = savepath / "mean_ranks"
                mean_plot_path.mkdir(exist_ok=True, parents=True)
                for use_legend in [False, True]:
                    if use_legend:
                        fig.legend(
                            loc="lower center",
                            handles=axs[0].get_legend_handles_labels()[0],
                            labels=axs[0].get_legend_handles_labels()[1],
                            # bbox_to_anchor=(0.5, -0.2),
                            bbox_to_anchor=(0.5, -0.1),
                            handlelength=4,
                            ncols=5,
                        )
                    plt.savefig(
                        mean_plot_path
                        / f"{plot_prefix}--{print_setting}--mean_ranks__mstd-{use_mean_std}__l-{use_legend}.{SAVETYPE}",
                        bbox_inches="tight",
                    )
                plt.close()

        else:
            name = "--".join(
                [s.lower().replace(" ", "").replace("/", "-") for s in setting]
            )
            out_data = {}
            whole_data: dict[str, dict[str, pd.DataFrame]] = {}
            for dataset in data_dict:
                whole_data[dataset] = {}
                for budget in data_dict[dataset]:
                    whole_data[dataset][budget] = {
                        s: data_dict[dataset][budget][s] for s in setting
                    }

            for s in setting:
                out_data[s] = {}
                for dataset in whole_data:
                    out_data[s][dataset] = pd.concat(
                        [whole_data[dataset][b][s] for b in whole_data[dataset]],
                        axis=1,
                        keys=whole_data[dataset].keys(),
                        names=COLLEVELANMES[1:],
                    )
                out_data[s] = pd.concat(
                    out_data[s].values(),
                    axis=1,
                    keys=out_data[s].keys(),
                    names=COLLEVELANMES,
                )

            for metric in plot_metrics:
                print(metric)
                ranks = {
                    s: (
                        out_data[s]
                        .loc[:, (slice(None), slice(None), metric)]
                        .droplevel(level=2, axis=1)
                        .rank(ascending=False, method="min")
                    )
                    for s in out_data
                }

                agg_ranks = np.concatenate([ranks[s].values for s in ranks], axis=1)

                stats, p = friedmanchisquare(*agg_ranks)
                # print(f"Friedman test for {metric}: {stats}, p-value: {p}")
                nemenyi = sp.posthoc_nemenyi_friedman(agg_ranks.T)
                nemenyi.columns = nemenyi.index = ranks[s].index

                fig, axs = plt.subplots(figsize=(6, 6))
                sns.heatmap(
                    nemenyi,
                    ax=axs,
                    annot=True,
                    fmt=".2f",
                    cmap="Blues",
                    cbar=False,
                    square=True,
                    linewidths=0.5,
                    linecolor="black",
                )
                plt.title(
                    f"Post-Hoc Nemenyi test for {metric} with Friedman test (p={p*100:.2f}%)"
                )
                nemenyi_path = savepath / "nemenyi"
                nemenyi_path.mkdir(exist_ok=True, parents=True)
                plt.savefig(
                    nemenyi_path
                    / f"{plot_prefix}--{print_setting}--{metric}--nemenyi.{SAVETYPE}",
                    bbox_inches="tight",
                )

            for mean_std in [True, False]:
                fig, axs = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
                # set y axis limits

                axs[0].set_ylim(0.5, ranks[s].shape[0] + 0.5)
                axs[0].set_ylabel(f"Avg. Method Rank")
                axs[0].set_yticks(ticks=np.arange(ranks[s].shape[0]) + 1)
                for i, metric in enumerate(plot_metrics):
                    axs[i].set_title(metric)
                    ranks = {
                        s: (
                            out_data[s]
                            .loc[:, (slice(None), slice(None), metric)]
                            .droplevel(level=2, axis=1)
                            .rank(ascending=False, method="min")
                        )
                        for s in out_data
                    }
                    ranks = pd.concat(
                        [ranks[s] for s in ranks],
                        axis=1,
                        keys=out_data.keys(),
                        # names=["Setting"] + COLLEVELANMES,
                    )
                    print(ranks)

                    mean_ranks = ranks.mean(axis=1)
                    std_ranks = ranks.std(axis=1)
                    if mean_std:
                        std_ranks = std_ranks / np.sqrt(len(ranks.columns))
                    ranks_sorted = mean_ranks.rank(ascending=True, method="first")
                    for method_name in mean_ranks.index:
                        markerfacecolor = QM_TO_COLOR[method_name]
                        if "Random" in method_name:
                            markerfacecolor = "white"
                        axs[i].errorbar(
                            -0.5 + ranks_sorted[method_name] / (len(ranks_sorted) + 1),
                            y=mean_ranks[method_name],
                            yerr=std_ranks[method_name],
                            fmt=".",
                            markersize=12,
                            color=QM_TO_COLOR[method_name],
                            markerfacecolor=markerfacecolor,
                            lw=3,
                            label=method_name,
                        )
                    # remove xticks
                    axs[i].set_xticks(ticks=[])
                mean_plot_path = savepath / "mean_ranks"
                mean_plot_path.mkdir(exist_ok=True, parents=True)
                for use_legend in [False, True]:
                    if use_legend:
                        fig.legend(
                            loc="lower center",
                            handles=axs[0].get_legend_handles_labels()[0],
                            labels=axs[0].get_legend_handles_labels()[1],
                            bbox_to_anchor=(0.5, -0.1),
                            handlelength=4,
                            ncols=5,
                        )
                    plt.savefig(
                        mean_plot_path
                        / f"{plot_prefix}--{print_setting}--mean_ranks__meanstd-{mean_std}--l-{use_legend}.{SAVETYPE}",
                        bbox_inches="tight",
                    )
                plt.close()
