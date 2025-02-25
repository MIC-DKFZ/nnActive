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


# order = ["Dataset", "Setting", "df"]

# datasets = set([data["Dataset"] for data in data_dicts])

# whole_data = {}
# for dataset in datasets:
#     whole_data[dataset] = {}
#     for data in data_dicts:
#         if data["Dataset"] == dataset:
#             whole_data[dataset][data["Setting"]] = data["df"]
#     whole_data[dataset] = pd.concat(
#         whole_data[dataset],
#         axis=1,
#         keys=whole_data[dataset].keys(),
#         names=["Setting"],
#     )
#     d_folder = dataset.replace(" ", "_")
#     save_df_to_txt(whole_data[dataset], savepath / d_folder / f"{fn}.txt")
# if len(whole_data) == 0:
#     print(f"Skipping {fn}")
#     continue
# whole_data = pd.concat(
#     whole_data, axis=1, keys=whole_data.keys(), names=["Dataset"]
# )
# whole_data = whole_data.reindex(CUSTOM_ORDER, level=0)
# whole_data = whole_data.rename(RENAMING_DICT, axis=0)
# save_df_to_txt(whole_data, savepath / f"{fn}.txt")

# with open(savepath / f"{fn}.md", "w") as f:
#     f.write(whole_data.to_markdown())

# cmap = "Oranges"
# higher_is_better = ["AUBC", "Final", "beta"]
# subset = [col for col in whole_data.columns if col[-1] in higher_is_better]

# print_data = whole_data.copy(deep=True)
# for n in print_data.index:
#     print_data.rename(index={n: n.replace("%", "\%")}, inplace=True)
# gmap = _compute_gmap(print_data[subset], invert=True)
# for col in subset:
#     std_col = tuple(list(col[:-1]) + [col[-1] + " std"])
#     print_data[col] = (
#         print_data[col].apply(lambda x: f"{x:.2f}")
#         + " ± "
#         + print_data[std_col].apply(lambda x: f"{x:.2f}")
#     )
#     del print_data[std_col]

# columns = ""
# levels = [whole_data.columns.levels]
# cur_col = None
# split_level = 2
# for col in print_data.columns:
#     if cur_col == col[:split_level]:
#         columns += "c"

#     else:
#         cur_col = col[:split_level]
#         columns += "|c"

# styled = print_data.style.background_gradient(
#     "Oranges", axis=None, subset=subset, gmap=gmap
# )
# tex_fn = savepath / f"{fn}.tex"
# styled.to_latex(
#     tex_fn,
#     convert_css=True,
#     hrules=True,
#     multicol_align="c|",
#     column_format="l" + columns + "|",
# )

# entire_data.append(whole_data)
# aubc_cols = [col for col in whole_data.columns if col[-1] == "AUBC"]
# aubc_std_cols = [col for col in whole_data.columns if col[-1] == "AUBC std"]
# aubc_vals = whole_data[aubc_cols]
# aubc_std_vals = whole_data[aubc_std_cols]
# for col in aubc_vals:
#     aubc_vals[(*col[:-1], "AUBC rank")] = aubc_vals[col].rank(ascending=False)

# entire_data.append(aubc_vals)

# whole_data = pd.concat(entire_data, axis=1)
# aubc_cols = [col for col in whole_data.columns if col[-1] == "AUBC"]
# aubc_std_cols = [col for col in whole_data.columns if col[-1] == "AUBC std"]
# aubc_vals = whole_data[aubc_cols]
# aubc_std_vals = whole_data[aubc_std_cols]
# rank_cols = [(*col[:-1], "AUBC rank") for col in aubc_vals]
# for col in aubc_vals:
#     aubc_vals[(*col[:-1], "AUBC rank")] = aubc_vals[col].rank(ascending=False)


# mean_rank = aubc_vals[rank_cols].mean(axis=1).sort_values()
# meadian_rank = aubc_vals[rank_cols].median(axis=1).sort_values()
# mean_aubc = whole_data[aubc_cols].mean(axis=1).sort_values()
# mean_aubc_rank = mean_aubc.rank(ascending=False)

# aggregated_rankings_aubc = pd.concat(
#     [mean_rank, meadian_rank, mean_aubc_rank, mean_aubc],
#     axis=1,
#     keys=["Mean Rank (AUBC)", "Median Rank (AUBC)", "Rank (Mean AUBC)", "Mean AUBC"],
# )
# aggregated_rankings_aubc = aggregated_rankings_aubc.reindex(CUSTOM_ORDER)

# print(aggregated_rankings_aubc)


# print("Analysing for best QM")
# # best_qm = "power bald"
# best_qm = "softrank bald"
# rbs = ["random-label", "random-label2", "random"]
# print(best_qm)
# print("\n")

# aubc_gain = aubc_vals.loc[best_qm] - aubc_vals.loc[rbs]
# aubc_gain = aubc_gain[aubc_cols]

# mean_gain = aubc_gain.mean(axis=1)
# print("Mean Gain")
# print(mean_gain)
# aubc_gain_pos = aubc_gain > 0
# percentage_gain = aubc_gain_pos.mean(axis=1)
# print("Gain Scenarios")
# print(percentage_gain)


# # print("Only Medium and Large Label Regimes")
# # aubc_gain = aubc_vals.loc[best_qm] - aubc_vals.loc[rbs]
# # low_label_cols = [
# #     ("Dataset216 AMOS2022 task1", "Query Size 40"),
# #     ("Dataset135 KiTS2021", "Query Size 200"),
# #     ("Dataset027 ACDC", "Query Size 30"),
# #     ("Dataset004 Hippocampus", "Query Size 20"),
# # ]
# # keep_cols = [col for col in aubc_cols if col[:-1] not in low_label_cols]
# # aubc_gain = aubc_gain[keep_cols]
# # aubc_gain = aubc_vals.loc[best_qm] - aubc_vals.loc[rbs]
# # aubc_gain = aubc_gain[[col for col in aubc_gain if col[-1] == "AUBC"]]
# # print(aubc_gain)
# # mean_gain = aubc_gain.mean(axis=1)
# # print("Mean Gain")
# # print(mean_gain)
# # aubc_gain_pos = aubc_gain > 0
# # percentage_gain = aubc_gain_pos.mean(axis=1)
# # print("Gain Scenarios")
# # print(percentage_gain)


# import IPython

# IPython.embed()
