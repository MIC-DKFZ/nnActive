from pathlib import Path

import IPython
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau
from setup import BASEPATH, QM_TO_COLOR, RENAMING_DICT

from nnactive.analyze.analysis import SettingAnalysis

STANDARD_COLNAMES = ["Low", "Medium", "High"]

SETTINGS = {
    "AMOS": [
        "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-40__qs-40",
        "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-200__qs-200",
        "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-500__qs-500",
    ],
    "KiTS": [
        "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-40__qs-40",
        "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-200__qs-200",
        "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-500__qs-500",
    ],
    "ACDC": [
        "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-30__qs-30",
        "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-60__qs-60",
        "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-90__qs-90",
    ],
    "Hippocampus": [
        "Dataset004_Hippocampus/patch-20_20_20__sb-random-label2-all-classes__sbs-20__qs-20__5loops",
        "Dataset004_Hippocampus/patch-20_20_20__sb-random-label2-all-classes__sbs-40__qs-40",
        "Dataset004_Hippocampus/patch-20_20_20__sb-random-label2-all-classes__sbs-60__qs-60",
    ],
}

savepath = Path(
    "/home/c817h/Documents/projects/nnactive_project/nnactive/results/horeka_rsync_final/"
)

for name in SETTINGS:
    SETTINGS[name] = [BASEPATH / p for p in SETTINGS[name]]

CUSTOM_ORDER = [
    "mutual information",
    "power bald",
    "softrank bald",
    "pred entropy",
    "power pe",
    "random",
    "random-label",
    "random-label2",
]


def compute_rankings(df, metrics="Mean Dice AUBC"):
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


entire_data = {}
mean_rankings = []
statistics = []
for name, paths in SETTINGS.items():

    fn = name

    settings = []
    metric = ["Mean Dice"]
    for j, path in enumerate(paths):
        if path.is_dir():
            setting = SettingAnalysis.load(path / "analysis.pkl")
            aucvals = pd.DataFrame(setting._compute_auc_row_dicts(metric))
            aucvals = aucvals[aucvals["Query Method"].isin(RENAMING_DICT)]
            aucvals["Query Method"] = aucvals["Query Method"].replace(RENAMING_DICT)
            bootstrap_rankings = compute_rankings(
                aucvals, metrics=[f"Mean Dice {m}" for m in ["AUBC", "Final"]]
            )
            mean_bootstrap_rankings = bootstrap_rankings.groupby("Query Method").mean()
            mean_rankings.append(mean_bootstrap_rankings)
            settings.append(bootstrap_rankings)
            print(mean_bootstrap_rankings)
            seeds = bootstrap_rankings["Left Out Seed"].unique()
            stat_dict = {
                "Name": name,
                "Budget": STANDARD_COLNAMES[j],
            }
            for c_metric in ["AUBC", "Final"]:
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

            # print(f"{name} - {path}")
            # print(f"Mean Tau: {np.mean(taus)}")
            # print(f"Mean Pval: {np.mean(pvals)}")
            statistics.append(stat_dict)
    entire_data[name] = settings
entire_data["Mean"] = [pd.concat(mean_rankings)]
statistics_df = pd.DataFrame(statistics)
max_decimal = 3
for c_metric in ["AUBC", "Final"]:
    statistics_df[f"Mean {c_metric} Tau"] = (
        statistics_df[f"{c_metric} Taus"].apply(np.mean).round(max_decimal)
    )
    statistics_df[f"Mean {c_metric} Pval"] = (
        statistics_df[f"{c_metric} Pvals"].apply(np.mean).round(max_decimal)
    )
# statistics_df["Mean Tau"] = statistics_df["Taus"].apply(np.mean).round(3)
# statistics_df["Std Tau"] = (
#     statistics_df["Taus"].apply(lambda x: np.std(x, ddof=1)).round(3)
# )
# statistics_df["Mean Pval"] = statistics_df["Pvals"].apply(np.mean)
mlist = ["Mean AUBC Tau", "Mean AUBC Pval", "Mean Final Tau", "Mean Final Pval"]
print(statistics_df[["Name", "Budget"] + mlist])


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
        ax_mid = axs[int(mid_row)]
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


n_rows = sum([len(v) for v in entire_data.values()])
fig, axs = plt.subplots(n_rows, 1, figsize=(16, 2 * n_rows), sharex=True)
row_count = 0
for i, (name, settings) in enumerate(entire_data.items()):
    for j, setting in enumerate(settings):
        ax = axs[row_count]
        add_legend = row_count == n_rows - 1
        if name == "Mean":
            mean_setting = setting.groupby("Query Method").mean()
            for method, row in mean_setting.iterrows():
                ax.axvline(
                    row["Rank Mean Dice AUBC"],
                    color=QM_TO_COLOR[method],
                    label=method,
                    lw=2,
                )
            ax.legend(loc=(0.2, -0.6), handlelength=4, ncols=4)
        else:
            sns.histplot(
                data=setting,
                x="Rank Mean Dice AUBC",
                ax=ax,
                hue="Query Method",
                palette=QM_TO_COLOR,
                legend=add_legend,
                multiple="stack",
                discrete=True,
            )
            mean_setting = setting.groupby("Query Method").mean()
            for method, row in mean_setting.iterrows():
                ax.axvline(row["Rank Mean Dice AUBC"], color=QM_TO_COLOR[method], lw=2)
            # ax.set_title(f"{name} - {j}")
            # ax.set_ylabel(f"{name} - {j}")
            ax.set_ylabel(STANDARD_COLNAMES[j])
        ax.set_xlabel("Rank")
        row_count += 1
fig.tight_layout()
add_subplot_labels(fig, axs, entire_data, n_rows)
fig.tight_layout()

plt.savefig("test.png")
