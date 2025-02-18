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
from setup import BASEPATH, RENAMING_DICT, df_to_multicol

from nnactive.analyze.analysis import SettingAnalysis

savepath = Path(
    "/home/c817h/Documents/projects/nnactive_project/nnactive/results/horeka_rsync_final/"
)

# 2nd value is always value which is expected to be better than the first. E.g. smaller QS is expected to be better.
settings = {
    "AMOS Medium Training Length": [
        "Dataset216_AMOS2022_task1/tr-nnActiveTrainer_500epochs__patch-32_74_74__sb-random-label2-all-classes__sbs-200__qs-200__precomputed-queries",
        "Dataset216_AMOS2022_task1/tr-nnActiveTrainer_500epochs__patch-32_74_74__sb-random-label2-all-classes__sbs-200__qs-200",
    ],
    "AMOS High Training Length": [
        "Dataset216_AMOS2022_task1/tr-nnActiveTrainer_500epochs__patch-32_74_74__sb-random-label2-all-classes__sbs-500__qs-500__precomputed-queries",
        "Dataset216_AMOS2022_task1/tr-nnActiveTrainer_500epochs__patch-32_74_74__sb-random-label2-all-classes__sbs-500__qs-500",
    ],
    # # Doublecheck here the effect!
    "KiTS Medium Training Length": [
        "Dataset135_KiTS2021/tr-nnActiveTrainer_500epochs__patch-64_64_64__sb-random-label2-all-classes__sbs-200__qs-200__precomputed-queries",
        "Dataset135_KiTS2021/tr-nnActiveTrainer_500epochs__patch-64_64_64__sb-random-label2-all-classes__sbs-200__qs-200",
    ],
    "KiTS High Training Length": [
        "Dataset135_KiTS2021/tr-nnActiveTrainer_500epochs__patch-64_64_64__sb-random-label2-all-classes__sbs-500__qs-500__precomputed-queries",
        "Dataset135_KiTS2021/tr-nnActiveTrainer_500epochs__patch-64_64_64__sb-random-label2-all-classes__sbs-500__qs-500",
    ],
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

QUERYMETHODS = [
    "BALD",
    "PowerBALD",
    "SoftrankBALD",
    "Predictive Entropy",
    "PowerPE",
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


final_diffs = pd.DataFrame()
aubc_diffs = pd.DataFrame()
final_significances = pd.DataFrame()
aubc_significances = pd.DataFrame()
final_corrs = pd.DataFrame()
aubc_corrs = pd.DataFrame()
final_corr_pval = pd.DataFrame()
aubc_corr_pval = pd.DataFrame()
for name in settings:
    print("-" * 10)
    print(name)
    analysis_list: list[SettingAnalysis] = []
    unique_budget_list = []
    auc_list = []
    aucval_list = []

    assert len(settings[name]) == 2
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

    auc_diff[(metric, "mean1")] = auc_list[0][mean_key]
    auc_diff[(metric, "mean2")] = auc_list[1][mean_key]

    auc_diff = auc_diff[~auc_diff[mean_key].isna()]
    #
    auc_diff[(metric, "ranking 200 epoch Queries")] = (
        auc_list[0].loc[auc_diff.index][mean_key].rank(ascending=False)
    )
    auc_diff[(metric, "ranking 500 epoch Queries")] = (
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
        merged_df[(metric, "ranking 200 epoch Queries")],
        merged_df[(metric, "ranking 500 epoch Queries")],
        alternative="greater",
    )
    rho, p_value_kendall = kendalltau(
        merged_df[(metric, "ranking 200 epoch Queries")],
        merged_df[(metric, "ranking 500 epoch Queries")],
        alternative="greater",
    )
    print(f"Spearman's Correlation: {spearman:.2f}, p-value: {p_value_spearman:.4f}")
    print(f"Kendall's Correlation: {rho:.2f}, p-value: {p_value_kendall:.4f}")
    print(
        f"Wilcoxon Test Statistics: {wilcoxon_stat:.2f}, p-value: {wilcoxon_p_value:.4f}"
    )
    merged_df = merged_df.reindex(QUERYMETHODS, axis=0)
    aubc_significances[name] = merged_df[("t-test", "significance")]
    aubc_diffs[name] = merged_df[(metric, "mean")]
    aubc_corrs[name] = np.array([rho])
    aubc_corr_pval[name] = np.array([p_value_kendall])

    score = "Final"
    metric = f"{main_metric} {score}"
    mean_key = (metric, "mean")
    std_key = ("Mean Dice Final", "std")
    final_diff = compute_difference(auc_list, mean_key, std_key)
    final_diff[(metric, "mean1")] = auc_list[0][mean_key]
    final_diff[(metric, "mean2")] = auc_list[1][mean_key]
    final_diff = final_diff[~final_diff[mean_key].isna()]
    final_diff[(metric, "ranking 200 epoch Queries")] = (
        auc_list[0].loc[final_diff.index][mean_key].rank(ascending=False)
    )
    final_diff[(metric, "ranking 500 epoch Queries")] = (
        auc_list[1].loc[final_diff.index][mean_key].rank(ascending=False)
    )

    results_df = compute_ttest(aucval_list, metric)
    results_df.columns = pd.MultiIndex.from_product([["t-test"], results_df.columns])
    merged_df = pd.merge(final_diff, results_df, left_index=True, right_index=True)
    merged_df = merged_df.reindex(CUSTOM_ORDER)
    merged_df.rename(RENAMING_DICT, axis=0, inplace=True)
    print(merged_df)
    spearman, p_value_spearman = spearmanr(
        merged_df[(metric, "ranking 200 epoch Queries")],
        merged_df[(metric, "ranking 500 epoch Queries")],
        alternative="greater",
    )
    rho, p_value_kendall = kendalltau(
        merged_df[(metric, "ranking 200 epoch Queries")],
        merged_df[(metric, "ranking 500 epoch Queries")],
        alternative="greater",
    )
    print(f"Spearman's Correlation: {spearman:.2f}, p-value: {p_value_spearman:.4f}")
    print(f"Kendall's Correlation: {rho:.2f}, p-value: {p_value_kendall:.4f}")
    merged_df = merged_df.reindex(QUERYMETHODS, axis=0)
    final_significances[name] = merged_df[("t-test", "significance")]
    final_diffs[name] = merged_df[(metric, "mean")]
    final_corrs[name] = np.array([rho])
    final_corr_pval[name] = np.array([p_value_kendall])
print("\n" * 2)
print("Final Significances")


df_to_multicol(final_significances)
df_to_multicol(final_diffs)
df_to_multicol(final_corrs)
df_to_multicol(final_corr_pval)
final_diffs = final_diffs.apply(lambda x: np.round(x * 100, 2))
styled: pd.DataFrame = final_diffs.copy(deep=True)
styled = styled.applymap(lambda x: f"{x:.2f}")
styled[final_significances == True] = styled[final_significances == True].applymap(
    lambda x: f"\\textbf{{{x}}}"
)
styled.to_latex(savepath / "ablation-training-final_diffs.tex")
print(final_significances)
final_significances.to_latex(savepath / "ablation-training-final_significances.tex")
final_corrs.to_latex(savepath / "ablation-training-final_corrs.tex")
print(final_diffs)
print(final_corrs)
print(final_corr_pval)


print("\n" * 2)
print("AUBC Significances")
df_to_multicol(aubc_significances)
df_to_multicol(aubc_diffs)
df_to_multicol(aubc_corrs)
df_to_multicol(aubc_corr_pval)
aubc_diffs = aubc_diffs.apply(lambda x: np.round(x * 100, 2))
styled: pd.DataFrame = aubc_diffs.copy(deep=True)
styled = styled.applymap(lambda x: f"{x:.2f}")
styled[aubc_significances == True] = styled[aubc_significances == True].applymap(
    lambda x: f"\\textbf{{{x}}}"
)
styled.to_latex(savepath / "ablation-training-aubc_diffs.tex")
print(aubc_significances)
print(aubc_diffs)
aubc_significances.to_latex(savepath / "ablation-training-aubc_significances.tex")

print(aubc_corrs)
aubc_corrs.to_latex(savepath / "ablation-training-aubc_corrs.tex")
print(aubc_corr_pval)
