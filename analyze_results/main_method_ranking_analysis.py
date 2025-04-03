from pathlib import Path

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from evaluator import (
    get_settings_for_combination,
    load_settings,
    rename_settings_in_analysis,
)
from matplotlib.axes import Axes
from setup import (
    BASEPATH,
    CUSTOM_ORDER,
    QM_TO_COLOR,
    RENAMING_DICT,
    SAVEPATH,
    load_setting_data_to_df,
)

from nnactive.utils.io import save_df_to_txt

USETEX = False
if USETEX:
    matplotlib.rcParams["text.usetex"] = True
    matplotlib.rcParams["font.family"] = "Computer Modern"

savepath = SAVEPATH / "figures"
savepath.mkdir(exist_ok=True, parents=True)
filetype = "pdf"

NAME = "main_method_ranking"
COLLEVELANMES = ["Dataset", "Budget", "Metric"]

USE_SETTINGS_LIST = [
    ["Main"],
]
SCORES = ["AUBC", "Final Dice", "beta"]

RENAME_SETTINGS_LIST = [
    None,
]

COMPARATIVE = False
FINAL_COLUMNS = [
    {"ReadCol": "('Mean Dice AUBC', 'mean')", "PrintCol": "AUBC", "better": "higher"},
    {"ReadCol": "('Mean Dice AUBC', 'std')", "PrintCol": "AUBC std", "better": None},
    {
        "ReadCol": "('Mean Dice Final', 'mean')",
        "PrintCol": "Final Dice",
        "better": "higher",
    },
    {
        "ReadCol": "('Mean Dice Final', 'std')",
        "PrintCol": "Final Dice std",
        "better": None,
    },
    {"ReadCol": "beta", "PrintCol": "FG-Eff", "better": "higher"},
    {"ReadCol": "beta_std", "PrintCol": "FG-Eff std", "better": None},
]

ADD_LEGEND = False


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
    for method_name in QM_TO_COLOR:
        if method_name not in ranks.index:
            continue
        ax1.plot(
            ranks.loc[method_name, :].values,
            marker="o",
            label=method_name,
            color=QM_TO_COLOR[method_name],
            ls="--" if "Random" in method_name else "-",
            lw=2.5,
            markerfacecolor=(
                "white" if "Random" in method_name else QM_TO_COLOR[method_name]
            ),
            # markerfacecolor=QM_TO_COLOR[method_name],
        )

    ax1.add_patch(
        patches.Rectangle((-0.5, 0), 3, 8.5, linewidth=1, facecolor="k", alpha=0.1)
    )
    ax1.add_patch(
        patches.Rectangle((2.5, 0), 3, 8.5, linewidth=1, facecolor="k", alpha=0.03)
    )
    ax1.add_patch(
        patches.Rectangle((5.5, 0), 3, 8.5, linewidth=1, facecolor="k", alpha=0.1)
    )
    ax1.add_patch(
        patches.Rectangle((8.5, 0), 3, 8.5, linewidth=1, facecolor="k", alpha=0.03)
    )

    ax1.set_xlim(-0.5, 11.5)
    ax1.set_ylim(0.5, ranks.shape[0] + 0.5)
    ax1.grid(axis="y")

    if USETEX:
        ax1.set_xticks(
            ticks=np.arange(ranks.shape[-1]),
            labels=[
                (f"{c[1]}\n\n" + rf"\textbf{{{c[0]}}}" if i % 3 == 1 else c[1])
                for i, c in enumerate(ranks.columns)
            ],
        )
    else:
        ax1.set_xticks(
            ticks=np.arange(ranks.shape[-1]),
            labels=[
                (f"{c[1]}\n\n{c[0]}" if i % 3 == 1 else c[1])
                for i, c in enumerate(ranks.columns)
            ],
        )
    ax1.set_yticks(ticks=np.arange(ranks.shape[0]) + 1)
    ax1.set_ylabel(f"Method Rank ({metric})")

    # Create mean rank plot
    avg_ranks = ranks.mean(axis=1)
    std_ranks = ranks.std(axis=1)
    ranks_sorted = avg_ranks.rank(ascending=True, method="first")
    for i, method_name in enumerate(QM_TO_COLOR):
        if method_name not in ranks.index:
            continue
        ax2.errorbar(
            -0.5 + ranks_sorted[method_name] / (len(ranks_sorted) + 1),
            [avg_ranks[method_name]],
            yerr=[std_ranks[method_name]],
            fmt=".",
            markersize=8,
            color=QM_TO_COLOR[method_name],
            markerfacecolor=(
                "white" if "Random" in method_name else QM_TO_COLOR[method_name]
            ),
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
    for setting, rename_setting in zip(USE_SETTINGS_LIST, RENAME_SETTINGS_LIST):
        print(setting)
        print_setting = "_".join(setting).replace(" ", "").replace("/", "-")
        setting_paths = get_settings_for_combination(setting)
        setting_analyses = load_settings(setting_paths, comparative=COMPARATIVE)
        if rename_setting is not None:
            rename_settings_in_analysis(setting_analyses, rename_setting)
            rename_settings_in_analysis(setting_paths, rename_setting)

        data_dict = load_setting_data_to_df(
            CUSTOM_ORDER, FINAL_COLUMNS, setting_paths, setting_analyses
        )

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

            plot_metrics = [
                c["PrintCol"] for c in FINAL_COLUMNS if c["better"] == "higher"
            ]
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

                if ADD_LEGEND:
                    ax1.legend(loc=(0.1, -0.25), handlelength=4, ncols=4)

                # Save figure
                # plt.tight_layout()
                metric_name = metric.lower().replace(" ", "")
                plt.savefig(
                    savepath / f"method_ranking--{name}--{metric_name}.{filetype}",
                    bbox_inches="tight",
                )

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
            axs[-1][0].legend(loc=(0.1, -0.375), handlelength=4, ncols=4)
            plt.savefig(
                savepath / f"method_ranking--{name}.{filetype}",
                bbox_inches="tight",
            )

        else:
            raise NotImplementedError
