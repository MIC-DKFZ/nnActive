from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import (
    alexandergovern,
    f_oneway,
    kendalltau,
    spearmanr,
    ttest_ind,
    wilcoxon,
)
from setup import BASEPATH, RENAMING_DICT

from nnactive.analyze.analysis import SettingAnalysis

savepath = Path(
    "/home/c817h/Documents/projects/nnactive_project/nnactive/results/horeka_rsync_final/"
)

# 2nd value is always value which is expected to be better than the first. E.g. smaller QS is expected to be better.
# TODO: enable this for all settings!
settings = {
    "AMOS Small Label": [
        "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-40__qs-80",
        "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-40__qs-40",
        "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-40__qs-20",
    ],
    "AMOS Large Label": [
        "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-500__qs-1000",
        "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-500__qs-500",
        "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-500__qs-250",
    ],
    "KiTS Small Label": [
        "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-40__qs-80",
        "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-40__qs-40",
        "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-40__qs-20",
    ],
    "KiTS Large Label": [
        "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-500__qs-1000",
        "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-500__qs-500",
        "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-500__qs-250",
    ],
    "ACDC Small Label": [
        "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-30__qs-60",
        "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-30__qs-30",
        "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-30__qs-15",
    ],
    "ACDC Large Label": [
        "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-90__qs-180",
        "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-90__qs-90",
        "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-90__qs-45",
    ],
}

CUSTOM_ORDER = [
    "mutual_information",
    "power_bald",
    "softrank_bald",
    "pred_entropy",
    "power_pe",
    # "random",
    # "random-label",
    # "random-label2",
]


# Training Lenth Ablations
def compute_difference(two_dfs: list[pd.DataFrame], mean_key, std_key):
    df_diff = two_dfs[1][mean_key] - two_dfs[0][mean_key]
    df_std = np.sqrt((two_dfs[0][std_key] ** 2 + two_dfs[1][std_key] ** 2))
    df_diff = pd.concat(
        [df_diff, df_std],
        axis=1,
        keys=[(mean_key[0], "mean"), (std_key[0], "mean std")],
    )
    return df_diff


def compute_ttest(aucval_list: list[pd.DataFrame], metric, significance: float = 0.05):
    test_groups = []
    for aucval in aucval_list:
        g = aucval.groupby("Query Method")[metric]
        test_groups.append(g)

    results = {"Query Method": []}
    for i in range(len(test_groups)):
        for j in range(i + 1, len(test_groups)):
            results[f"t-statistic {i}-{j}"] = []
            results[f"p-value {i}-{j}"] = []
            results[f"significance {i}-{j}"] = []

    methods = test_groups[0].groups.keys()
    for method in methods:
        if all(method in g.groups for g in test_groups):
            results["Query Method"].append(method)
            for i in range(len(test_groups)):
                for j in range(i + 1, len(test_groups)):
                    t_stat, p_value = ttest_ind(
                        test_groups[j].get_group(method),
                        test_groups[i].get_group(method),
                        alternative="greater",
                    )
                    results[f"t-statistic {i}-{j}"].append(t_stat)
                    results[f"p-value {i}-{j}"].append(p_value)
                    results[f"significance {i}-{j}"].append(p_value < significance)

    results_df = pd.DataFrame(results).set_index("Query Method")
    return results_df


final_significances = pd.DataFrame()
aubc_siginificances = pd.DataFrame()
for name in settings:
    print("-" * 10)
    print(name)
    analysis_list: list[SettingAnalysis] = []
    unique_budget_list = []
    auc_list = []
    aucval_list = []

    for path in settings[name]:
        path = BASEPATH / path
        analysis = SettingAnalysis.load(path / "analysis.pkl")
        analysis_list.append(analysis)
        unique_budget_list.append(analysis.df[analysis.budget_key].unique())

    overlapping_budgets = list(set.intersection(*map(set, unique_budget_list)))

    main_metric = "Mean Dice"
    for analysis in analysis_list:
        orig_size = len(analysis.df)
        analysis.df = analysis.df[
            analysis.df[analysis.budget_key].isin(overlapping_budgets)
        ]
        enforce_full_loops = len(analysis.df) == orig_size
        auc = analysis.compute_auc_df(enforce_full=enforce_full_loops)
        auc_list.append(auc)
        aucval_list.append(pd.DataFrame(analysis._compute_auc_row_dicts([main_metric])))

    score = "AUBC"
    metric = f"{main_metric} {score}"
    mean_key = (metric, "mean")
    std_key = (metric, "std")

    auc_diff = compute_difference(auc_list, mean_key, std_key)

    for i, auc in enumerate(auc_list):
        auc_diff[(metric, f"mean{i}")] = auc[mean_key]

    auc_diff = auc_diff[~auc_diff[mean_key].isna()]
    #

    for i, auc in enumerate(auc_list):
        auc_diff[(metric, f"ranking QS {i}")] = auc.loc[auc_diff.index][mean_key].rank(
            ascending=False
        )

    results_df = compute_ttest(aucval_list, metric)

    results_df.columns = pd.MultiIndex.from_product([["t-test"], results_df.columns])
    merged_df = pd.merge(auc_diff, results_df, left_index=True, right_index=True)
    merged_df = merged_df.reindex(CUSTOM_ORDER)
    merged_df.rename(RENAMING_DICT, axis=0, inplace=True)
    print(merged_df)
    sig_cols = [col for col in merged_df.columns if col[1].startswith("significance")]

    merged_df[("t-test", "full-significance")] = 0
    for col in sig_cols:
        merged_df[("t-test", "full-significance")] += merged_df[col].astype(int)

    print(merged_df[("t-test", "full-significance")])
    aubc_siginificances[name] = merged_df[("t-test", "full-significance")]

    score = "Final"
    metric = f"{main_metric} {score}"
    mean_key = (metric, "mean")
    std_key = ("Mean Dice Final", "std")
    final_diff = compute_difference(auc_list, mean_key, std_key)

    for i, auc in enumerate(auc_list):
        final_diff[(metric, f"mean{i}")] = auc[mean_key]
    final_diff = final_diff[~final_diff[mean_key].isna()]

    final_diff = final_diff[~final_diff[mean_key].isna()]
    #

    for i, auc in enumerate(auc_list):
        final_diff[(metric, f"ranking QS {i}")] = auc.loc[auc_diff.index][
            mean_key
        ].rank(ascending=False)

    results_df = compute_ttest(aucval_list, metric)
    results_df.columns = pd.MultiIndex.from_product([["t-test"], results_df.columns])
    merged_df = pd.merge(final_diff, results_df, left_index=True, right_index=True)
    merged_df = merged_df.reindex(CUSTOM_ORDER)
    merged_df.rename(RENAMING_DICT, axis=0, inplace=True)
    print(merged_df)
    sig_cols = [col for col in merged_df.columns if col[1].startswith("significance")]

    merged_df[("t-test", "full-significance")] = 0
    for col in sig_cols:
        merged_df[("t-test", "full-significance")] += merged_df[col].astype(int)

    print(merged_df[("t-test", "full-significance")])
    final_significances[name] = merged_df[("t-test", "full-significance")]


print("Final AUBC Significances")
column_map = {}
for col in aubc_siginificances.columns:
    s_col = col.split(" ")
    column_map[col] = (s_col[0], " ".join(s_col[1:2]))
aubc_siginificances.columns = pd.MultiIndex.from_tuples(
    [column_map[col] for col in aubc_siginificances.columns]
)
aubc_siginificances["Mean"] = aubc_siginificances.mean(axis=1).round(2)
print(aubc_siginificances)


aubc_siginificances.to_latex(savepath / "ablation-query_aubc_significances.tex")


print("Final Mean DICE Significances")
column_map = {}
for col in final_significances.columns:
    s_col = col.split(" ")
    column_map[col] = (s_col[0], " ".join(s_col[1:2]))
final_significances.columns = pd.MultiIndex.from_tuples(
    [column_map[col] for col in final_significances.columns]
)
final_significances["Mean"] = final_significances.mean(axis=1).round(2)
print(final_significances)
final_significances.to_latex(savepath / "ablation-query_final_significances.tex")
