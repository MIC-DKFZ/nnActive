from copy import deepcopy
from pathlib import Path
from typing import Iterable

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from evaluator import (
    get_settings_for_combination,
    load_settings,
    rename_settings_in_analysis,
)
from scipy.stats import kendalltau
from setup import (
    BASEPATH,
    QM_TO_COLOR,
    RENAMING_DICT,
    SAVEPATH,
    SAVETYPE,
    apply_latex_coloring,
    get_ranking_cmap,
    save_styled_to_latex,
)

from nnactive.analyze.analysis import SettingAnalysis

plt.style.use("default")

SAVENAME = "bootstrap_ranking"
STANDARD_COLNAMES = ["Low", "Medium", "High"]


COMPARATIVE = False
COLLEVELNAMES = ["Dataset", "Label Regime", "Setting", "Metric"]

USE_SETTINGS_LIST = [
    ["Main"],  # enable for mean_rank in appendix
    # ["500 Epochs"],
    # ["Precomputed"],
    # ["QSx2"],
    # ["QSx1/2"],
    ["Patchx1/2"],  # enable for mean_rank in appendix
]
RENAME_SETTINGS_LIST = [None] * len(USE_SETTINGS_LIST)

C_METRICS = ["AUBC", "Final"]
# C_METRICS = ["AUBC"]


savepath = SAVEPATH / "figures"
texpath = SAVEPATH / "tex" / "bootstrap_ranking"
texpath.mkdir(exist_ok=True, parents=True)
savepath.mkdir(exist_ok=True, parents=True)


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


def compute_rankings(
    df: pd.DataFrame, metrics: Iterable[str] | str = "Mean Dice AUBC"
) -> pd.DataFrame:
    rankings = []
    if not isinstance(metrics, (list, tuple)):
        metrics = [metrics]

    for seed in df["seed"].unique():
        df_subset = df[df["seed"] != seed]  # Leave one seed out
        grouped = df_subset.groupby("Query Method").mean()
        for metric in metrics:
            rank_name = f"Rank {metric}"
            grouped[rank_name] = grouped[metric].rank(ascending=False, method="min")

        for method, row in grouped.iterrows():
            ranking_dict = {
                "Query Method": method,
                "Left Out Seed": seed,
            }
            for metric in metrics:
                ranking_dict[f"Rank {metric}"] = row[f"Rank {metric}"]
            rankings.append(ranking_dict)

    return pd.DataFrame(rankings)


def add_subplot_labels(fig, axs, entire_data, n_rows):
    row_start = 0
    for name, settings in entire_data.items():
        row_end = row_start + len(settings) - 1
        mid_row = (row_start + row_end) / 2
        mid_position = (
            axs[int(mid_row)].get_position().bounds[1]
            # + axs[int(row_start)].get_position().bounds[1]
            + axs[int(mid_row)].get_position().bounds[3] / 2
        )
        fig.text(
            0,  # Position near the y-axis
            mid_position,  # Align with middle of axes
            name,
            ha="center",
            va="center",
            rotation=90,
            fontsize=12,
            fontweight="bold",
        )
        row_start = row_end + 1


def nested_dict_of_df_to_df(
    nested_bootstrap_ranking_dict: dict[str, dict[str, pd.DataFrame]],
) -> pd.DataFrame:
    bootstrap_rankings_dict = deepcopy(nested_bootstrap_ranking_dict)
    for dset_name in bootstrap_rankings_dict:
        for budget in bootstrap_rankings_dict[dset_name]:
            bootstrap_rankings_dict[dset_name][budget]["Dataset"] = dset_name
            bootstrap_rankings_dict[dset_name][budget]["Label Regime"] = budget
    bootstrap_rankings_df = pd.concat(
        [v for d in bootstrap_rankings_dict.values() for v in d.values()]
    )

    return bootstrap_rankings_df


def plot_bootstrap_ranking_overview(
    IMGTYPE,
    savepath,
    print_setting,
    nested_bootstrap_ranking_dict,
    c_metric,
):
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        # sharey=True,
        figsize=(14, 6),
        width_ratios=(12, 1),
        gridspec_kw={
            "wspace": 0.05,  # 0.03
        },
        sharey=True,
    )
    bootstrap_rankings_df = nested_dict_of_df_to_df(nested_bootstrap_ranking_dict)
    mean_ranks = bootstrap_rankings_df.groupby(
        ["Query Method", "Dataset", "Label Regime"]
    ).mean(numeric_only=True)
    locations = [
        pd.IndexSlice[:, [d], [b]]
        for d in nested_bootstrap_ranking_dict
        for b in nested_bootstrap_ranking_dict[d]
    ]
    mean_ranks = pd.concat(
        [mean_ranks.loc[loc] for loc in locations],
        axis=0,
    )
    plot_key = f"Rank Mean Dice {c_metric}"

    # Create ranking line plot
    methods = []
    for method_name in QM_TO_COLOR:
        if method_name not in mean_ranks.index.get_level_values("Query Method"):
            continue
        methods.append(method_name)
        ax1.plot(
            mean_ranks.loc[method_name, plot_key].values,
            marker="o",
            label=method_name,
            color=QM_TO_COLOR[method_name],
            ls="--" if "Random" in method_name else "-",
            lw=2,
            markerfacecolor=(
                "white" if "Random" in method_name else QM_TO_COLOR[method_name]
            ),
        )
    ax1.set_ylabel(f"Mean Rank ({c_metric})")
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
    ax1.set_ylim(0.5, len(methods) + 0.5)
    ax1.grid(axis="y")

    ax1.set_xticks(
        ticks=np.arange(len(locations)),
        labels=[
            (f"{c[1]}\n\n{c[0]}" if i % 3 == 1 else c[1])
            for i, c in enumerate(
                [
                    (d, b)
                    for d in nested_bootstrap_ranking_dict
                    for b in nested_bootstrap_ranking_dict[d]
                ]
            )
        ],
    )
    ax1.set_yticks(ticks=np.arange(len(methods)) + 1)
    ax1.set_ylabel(f"Method Rank ({c_metric})")
    ax1.legend(loc=(0.1, -0.25), handlelength=4, ncols=4)

    avg_ranks = bootstrap_rankings_df.groupby("Query Method").mean(numeric_only=True)[
        plot_key
    ]
    std_ranks = bootstrap_rankings_df.groupby("Query Method").std(numeric_only=True)[
        plot_key
    ]

    ranks_sorted = avg_ranks.rank(ascending=True, method="first")
    for i, method_name in enumerate(methods):
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
        )
    ax2.set_xlim(-0.5, 0.5)
    ax2.set_xticks(ticks=[0], labels=["Mean Rank"])
    ax2.tick_params("x", length=0, pad=7)
    ax2.grid(axis="y")

    plt.savefig(
        savepath / f"bootstrap_ranking-overview-{print_setting}-{c_metric}.{IMGTYPE}"
    )


def plot_bootstrap_rankings(
    IMGTYPE,
    savepath,
    add_subplot_labels,
    print_setting,
    nested_bootstrap_ranking_dict,
    c_metric,
    add_mean_rankings,
):
    n_rows = sum([len(v) for v in nested_bootstrap_ranking_dict.values()])
    n_rows += 1 if add_mean_rankings else 0
    fig, axs = plt.subplots(n_rows, 1, figsize=(16, 2 * n_rows), sharex=True)
    row_count = 0
    key = f"Rank Mean Dice {c_metric}"

    for (
        i,
        name,
    ) in enumerate(nested_bootstrap_ranking_dict):
        for budget, setting in nested_bootstrap_ranking_dict[name].items():
            ax = axs[row_count]
            add_legend = row_count == n_rows - 1
            sns.histplot(
                data=setting,
                x=key,
                ax=ax,
                hue="Query Method",
                palette=QM_TO_COLOR,
                legend=add_legend,
                multiple="stack",
                discrete=True,
            )
            mean_setting = setting.groupby("Query Method").mean()
            for method, row in mean_setting.iterrows():
                ax.axvline(row[key], color=QM_TO_COLOR[method], lw=2)
            ax.set_ylabel(budget)
            ax.set_xlabel("Rank")
            row_count += 1
    if add_mean_rankings:
        ax = axs[row_count]
        bootstrap_rankings_df = nested_dict_of_df_to_df(nested_bootstrap_ranking_dict)
        bootstrap_rankings_df["Setting"] = print_setting
        mean_rankings_df = bootstrap_rankings_df.groupby("Query Method").mean(
            numeric_only=True
        )
        for method, row in mean_rankings_df.iterrows():
            ax.axvline(
                row[key],
                color=QM_TO_COLOR[method],
                label=method,
                lw=2,
            )
        ax.set_ylabel("Mean")
        # Only used on last axis
    ax.legend(loc=(0.2, -0.6), handlelength=4, ncols=4)

    fig.tight_layout()
    add_subplot_labels(fig, axs, nested_bootstrap_ranking_dict, n_rows)
    fig.tight_layout()

    plt.savefig(savepath / f"bootstrap_ranking-{print_setting}-{c_metric}.{IMGTYPE}")


def create_latex_from_ranked_data(
    SAVENAME: str,
    texpath: Path,
    CMAP,
    n: str,
    save_df: pd.DataFrame,
    order_index: Iterable[str] | None = None,
):
    styled = save_df.copy()
    if order_index is not None:
        styled = styled.reindex(order_index).dropna()
    styled.index = styled.index.str.replace("%", "\\%")
    cmap_vals = styled.rank(ascending=True, method="min")
    cmap_vals = cmap_vals / cmap_vals.max()
    styled = styled.round(2)
    styled = styled.map(lambda x: f"{x:.2f}")
    styled = styled.style.background_gradient(
        cmap=CMAP, axis=None, subset=styled.columns, gmap=cmap_vals
    )
    styled.to_latex(texpath / f"{SAVENAME}--{n}--mean_rankings.tex", convert_css=True)


if __name__ == "__main__":
    full_df = []

    for setting, rename_setting in zip(USE_SETTINGS_LIST, RENAME_SETTINGS_LIST):
        print(setting)
        print_setting = "_".join(setting).replace(" ", "").replace("/", "-")
        setting_paths = get_settings_for_combination(setting)
        setting_analyses = load_settings(setting_paths, comparative=COMPARATIVE)
        if rename_setting is not None:
            rename_settings_in_analysis(setting_analyses, rename_setting)
            rename_settings_in_analysis(setting_paths, rename_setting)
        nested_bootstrap_ranking_dict: dict[str, dict[str, pd.DataFrame]] = {}
        statistics = []
        for dataset in setting_analyses:
            print(dataset)
            dataset_settings = setting_analyses[dataset]
            bootstrap_ranking_dict = {}
            for budget in dataset_settings:
                budget_settings = dataset_settings[budget]
                for budget_setting in budget_settings:
                    data = budget_settings[budget_setting]
                    aucvals = pd.DataFrame(data._compute_auc_row_dicts([f"Mean Dice"]))
                    aucvals = aucvals[aucvals["Query Method"].isin(CUSTOM_ORDER)]
                    aucvals["Query Method"] = aucvals["Query Method"].replace(
                        RENAMING_DICT
                    )
                    bootstrap_rankings = compute_rankings(
                        aucvals, metrics=[f"Mean Dice {metric}" for metric in C_METRICS]
                    )
                    bootstrap_ranking_dict[budget] = bootstrap_rankings
                    seeds = bootstrap_rankings["Left Out Seed"].unique()
                    stat_dict = {
                        "Name": dataset,
                        "Label Regime": budget,
                    }
                    for c_metric in C_METRICS:
                        taus = []
                        pvals = []
                        for s1 in seeds:
                            for s2 in seeds:
                                if s1 != s2:
                                    tau, p = kendalltau(
                                        bootstrap_rankings[
                                            bootstrap_rankings["Left Out Seed"] == s1
                                        ][f"Rank Mean Dice {c_metric}"],
                                        bootstrap_rankings[
                                            bootstrap_rankings["Left Out Seed"] == s2
                                        ][f"Rank Mean Dice {c_metric}"],
                                    )
                                    taus.append(tau)
                                    pvals.append(p)
                        taus = np.array(taus)
                        pvals = np.array(pvals)
                        stat_dict[f"{c_metric} Taus"] = taus
                        stat_dict[f"{c_metric} Pvals"] = pvals
                    statistics.append(stat_dict)
            nested_bootstrap_ranking_dict[dataset] = bootstrap_ranking_dict
        statistics_df = pd.DataFrame(statistics)
        max_decimal = 3
        for c_metric in C_METRICS:
            statistics_df[f"Mean {c_metric} Tau"] = (
                statistics_df[f"{c_metric} Taus"].apply(np.mean).round(max_decimal)
            )
            statistics_df[f"Mean {c_metric} Pval"] = (
                statistics_df[f"{c_metric} Pvals"].apply(np.mean).round(max_decimal)
            )

        mlist = [
            f"Mean {c_metric} {suffix}"
            for c_metric in C_METRICS
            for suffix in ["Tau", "Pval"]
        ]
        print(statistics_df[["Name", "Label Regime"] + mlist])

        add_mean_rankings = True
        bootstrap_rankings_df = nested_dict_of_df_to_df(nested_bootstrap_ranking_dict)
        bootstrap_rankings_df["Setting"] = print_setting

        for c_metric in C_METRICS:
            plot_bootstrap_rankings(
                SAVETYPE,
                savepath,
                add_subplot_labels,
                print_setting,
                nested_bootstrap_ranking_dict,
                c_metric,
                add_mean_rankings,
            )
            plot_bootstrap_ranking_overview(
                SAVETYPE,
                savepath,
                print_setting,
                nested_bootstrap_ranking_dict,
                c_metric,
            )
        full_df.append(bootstrap_rankings_df)
    full_df = pd.concat(full_df)
