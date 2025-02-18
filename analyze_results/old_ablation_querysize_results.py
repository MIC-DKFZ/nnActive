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
from setup import RENAMING_DICT

from nnactive.analyze.analysis import SettingAnalysis

intermediate_path = Path(
    "/home/c817h/Documents/projects/nnactive_project/nnactive/results/horeka_rsync/"
)

savepath = Path(
    "/home/c817h/Documents/projects/nnactive_project/nnactive/results/horeka_rsync/"
)

# 2nd value is always value which is expected to be better than the first. E.g. smaller QS is expected to be better.
settings = {
    # "AMOS Training Length 500 epochs": [
    #     "Dataset216_AMOS2022_task1/tr-nnActiveTrainer_500epochs__patch-32_74_74__sb-random-label2-all-classes__sbs-500__qs-500__precomputed-queries",
    #     "Dataset216_AMOS2022_task1/tr-nnActiveTrainer_500epochs__patch-32_74_74__sb-random-label2-all-classes__sbs-500__qs-500",
    # ],
    # "AMOS Training Length": [
    #     "Dataset216_AMOS2022_task1/tr-nnActiveTrainer_500epochs__patch-32_74_74__sb-random-label2-all-classes__sbs-200__qs-200__precomputed-queries",
    #     "Dataset216_AMOS2022_task1/tr-nnActiveTrainer_500epochs__patch-32_74_74__sb-random-label2-all-classes__sbs-200__qs-200",
    # ],
    # "AMOS Small Label": [
    #     "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-40__qs-40",
    #     "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-40__qs-20",
    # ],
    # "AMOS Small_2 Label": [
    #     "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-40__qs-80",
    #     "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-40__qs-40",
    # ],
    # "AMOS Large Label": [
    #     "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-500__qs-500",
    #     "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-500__qs-250",
    # ],
    # "AMOS Large Label": [
    #     "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-500__qs-1000",
    #     "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-500__qs-500",
    # ],
    # # Doublecheck here the effect!
    # "KiTS Training Length": [
    #     "Dataset135_KiTS2021/tr-nnActiveTrainer_500epochs__patch-64_64_64__sb-random-label2-all-classes__sbs-200__qs-200__precomputed-queries",
    #     "Dataset135_KiTS2021/tr-nnActiveTrainer_500epochs__patch-64_64_64__sb-random-label2-all-classes__sbs-200__qs-200",
    # ],
    # "KiTS  Small Label": [
    #     "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-40__qs-40",
    #     "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-40__qs-20",
    # ],
    # "KiTS  Small_2 Label": [
    #     "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-40__qs-80",
    #     "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-40__qs-20",
    # ],
    # "KiTS  Large Label": [
    #     "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-500__qs-500",
    #     "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-500__qs-250",
    # ],
    # "KiTS  Large_2 Label": [
    #     "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-500__qs-1000",
    #     "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-500__qs-500",
    # ],
    # "ACDC Small Label": [
    #     "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-30__qs-30",
    #     "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-30__qs-15",
    # ],
    # "ACDC Small_2 Label": [
    #     "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-30__qs-60",
    #     "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-30__qs-30",
    # ],
    # "ACDC Large Label": [
    #     "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-90__qs-90",
    #     "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-90__qs-45",
    # ],
    # "ACDC Large_2 Label": [
    #     "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-90__qs-180",
    #     "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-90__qs-90",
    # ],
}

CUSTOM_ORDER = [
    "mutual_information",
    "power_bald",
    "softrank_bald",
    "pred_entropy",
    "power_pe",
    # "random", # disable for Training Length ablations
    "random-label",
    "random-label2",
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
    g1 = aucval_list[0].groupby("Query Method")[metric]
    g2 = aucval_list[1].groupby("Query Method")[metric]
    results = {"Query Method": [], "t-statistic": [], "p-value": [], "significance": []}

    for method in g1.groups.keys():
        if method in g2.groups:
            t_stat, p_value = ttest_ind(
                g2.get_group(method), g1.get_group(method), alternative="greater"
            )
            results["Query Method"].append(method)
            results["t-statistic"].append(t_stat)
            results["p-value"].append(p_value)
            results["significance"].append(p_value < significance)

    results_df = pd.DataFrame(results).set_index("Query Method")
    return results_df


final_signifiances = pd.DataFrame()
auc_significances = pd.DataFrame()
for name in settings:
    print("-" * 10)
    print(name)
    analysis_list: list[SettingAnalysis] = []
    unique_budget_list = []
    auc_list = []
    aucval_list = []

    assert len(settings[name]) == 2
    for path in settings[name]:
        path = intermediate_path / path
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

    auc_diff[(metric, "mean1")] = auc_list[0][mean_key]
    auc_diff[(metric, "mean2")] = auc_list[1][mean_key]

    auc_diff = auc_diff[~auc_diff[mean_key].isna()]
    #
    auc_diff[(metric, "ranking QS big")] = (
        auc_list[0].loc[auc_diff.index][mean_key].rank(ascending=False)
    )
    auc_diff[(metric, "ranking QS small")] = (
        auc_list[1].loc[auc_diff.index][mean_key].rank(ascending=False)
    )
    wilcoxon_stat, wilcoxon_p_value = wilcoxon(
        auc_list[0].loc[auc_diff.index][mean_key],
        auc_list[1].loc[auc_diff.index][mean_key],
    )

    results_df = compute_ttest(aucval_list, metric)

    results_df.columns = pd.MultiIndex.from_product([["t-test"], results_df.columns])
    merged_df = pd.merge(auc_diff, results_df, left_index=True, right_index=True)
    merged_df = merged_df.reindex(CUSTOM_ORDER)
    merged_df.rename(RENAMING_DICT, axis=0, inplace=True)
    print(merged_df)
    spearman, p_value_spearman = spearmanr(
        merged_df[(metric, "ranking QS big")],
        merged_df[(metric, "ranking QS small")],
        alternative="greater",
    )
    rho, p_value_kendall = kendalltau(
        merged_df[(metric, "ranking QS big")],
        merged_df[(metric, "ranking QS small")],
        alternative="greater",
    )
    print(f"Spearman's Correlation: {spearman:.2f}, p-value: {p_value_spearman:.4f}")
    print(f"Kendall's Correlation: {rho:.2f}, p-value: {p_value_kendall:.4f}")
    print(
        f"Wilcoxon Test Statistics: {wilcoxon_stat:.2f}, p-value: {wilcoxon_p_value:.4f}"
    )
    auc_significances[name] = merged_df[("t-test", "significance")]

    score = "Final"
    metric = f"{main_metric} {score}"
    mean_key = (metric, "mean")
    std_key = ("Mean Dice Final", "std")
    final_diff = compute_difference(auc_list, mean_key, std_key)
    final_diff[(metric, "mean1")] = auc_list[0][mean_key]
    final_diff[(metric, "mean2")] = auc_list[1][mean_key]
    final_diff = final_diff[~final_diff[mean_key].isna()]
    final_diff[(metric, "ranking QS big")] = (
        auc_list[0].loc[final_diff.index][mean_key].rank(ascending=False)
    )
    final_diff[(metric, "ranking QS small")] = (
        auc_list[1].loc[final_diff.index][mean_key].rank(ascending=False)
    )

    results_df = compute_ttest(aucval_list, metric)
    results_df.columns = pd.MultiIndex.from_product([["t-test"], results_df.columns])
    merged_df = pd.merge(final_diff, results_df, left_index=True, right_index=True)
    merged_df = merged_df.reindex(CUSTOM_ORDER)
    merged_df.rename(RENAMING_DICT, axis=0, inplace=True)
    print(merged_df)
    spearman, p_value_spearman = spearmanr(
        merged_df[(metric, "ranking QS big")],
        merged_df[(metric, "ranking QS small")],
        alternative="greater",
    )
    rho, p_value_kendall = kendalltau(
        merged_df[(metric, "ranking QS big")],
        merged_df[(metric, "ranking QS small")],
        alternative="greater",
    )
    print(f"Spearman's Correlation: {spearman:.2f}, p-value: {p_value_spearman:.4f}")
    print(f"Kendall's Correlation: {rho:.2f}, p-value: {p_value_kendall:.4f}")
    final_signifiances[name] = merged_df[("t-test", "significance")]
print("\n" * 2)
print("Final Significances")
print(final_signifiances)
print("\n" * 2)
print("AUBC Significances")
print(auc_significances)
