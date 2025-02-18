from pathlib import Path

import matplotlib.colors as mcolors
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
from setup import BASEPATH, RENAMING_DICT, apply_latex_coloring, df_to_multicol

from nnactive.analyze.analysis import SettingAnalysis

savepath = Path(
    "/home/c817h/Documents/projects/nnactive_project/nnactive/results/horeka_rsync_final/"
)

# 2nd value is always value which is expected to be better than the first. E.g. smaller QS is expected to be better.
# TODO: enable this for all settings!
settings = {
    "AMOS Low Label": [
        "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-40__qs-80",
        "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-40__qs-40",
        "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-40__qs-20",
    ],
    "AMOS High Label": [
        "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-500__qs-1000",
        "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-500__qs-500",
        "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-500__qs-250",
    ],
    "KiTS Low Label": [
        "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-40__qs-80",
        "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-40__qs-40",
        "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-40__qs-20",
    ],
    "KiTS High Label": [
        "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-500__qs-1000",
        "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-500__qs-500",
        "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-500__qs-250",
    ],
    "ACDC Low Label": [
        "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-30__qs-60",
        "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-30__qs-30",
        "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-30__qs-15",
    ],
    "ACDC High Label": [
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


def compute_correlation_from_dfs(
    dfs: list[pd.DataFrame], qms, metric, significance: float = 0.10
):
    results = []
    for qm in qms:
        values = []
        compare_ranking = []
        for i, df in enumerate(dfs):
            df_sub = df[df["Query Method"] == qm]
            values.extend([v for v in df_sub[metric]])
            compare_ranking.extend([i] * len(df_sub[metric]))
        corr, pval = kendalltau(values, compare_ranking)
        results.append(
            {
                "Query Method": qm,
                "corr": corr,
                "pval": pval,
                "significance": pval < significance,
            }
        )
    return pd.DataFrame(results).set_index("Query Method")


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


VALMAP = {
    -2: "#FF0000",  # red
    -1: "#F08080",  # lightcoral
    0: "#FFFFFF",  # white
    1: "#90EE90",  # light green
    2: "#008000",  # green
}


def _get_cmap(
    values: np.ndarray, significances: np.ndarray, colormapping: dict[int, str] = VALMAP
):
    vmap = np.zeros(values.shape, dtype=np.int8)
    vmap[values > 0] = 1
    vmap[values < 0] = -1
    vmap[significances] = vmap[significances] * 2
    cmap = np.array([[colormapping[v] for v in row] for row in vmap])
    return cmap


final_significances = pd.DataFrame()
aubc_siginificances = pd.DataFrame()
final_corrs = pd.DataFrame()
aubc_corrs = pd.DataFrame()
aubc_corr_pval = pd.DataFrame()
final_corr_pval = pd.DataFrame()

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
    auc_corr = compute_correlation_from_dfs(aucval_list, CUSTOM_ORDER, metric)
    auc_corr.rename(RENAMING_DICT, axis=0, inplace=True)

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
    final_corr = compute_correlation_from_dfs(aucval_list, CUSTOM_ORDER, metric)
    final_corr.rename(RENAMING_DICT, axis=0, inplace=True)
    final_corrs[name] = final_corr["corr"]
    aubc_corrs[name] = auc_corr["corr"]
    final_corr_pval[name] = final_corr["significance"]
    aubc_corr_pval[name] = auc_corr["significance"]


print("Final AUBC Significances")
df_to_multicol(aubc_siginificances)
df_to_multicol(aubc_corrs)
df_to_multicol(aubc_corr_pval)
aubc_siginificances["Mean"] = aubc_siginificances.mean(axis=1).round(2)
print(aubc_siginificances)
styled_corrs = aubc_corrs.copy(deep=True)
styled_corrs = styled_corrs.applymap(lambda x: f"{x:.4f}")
cmap = _get_cmap(aubc_corrs.values, aubc_corr_pval.values)
styled_corrs = apply_latex_coloring(styled_corrs, cmap)
styled_corrs.to_latex(savepath / "ablation-query_aubc_corrs.tex")
print(aubc_corrs)
print(aubc_corr_pval)


aubc_siginificances.to_latex(savepath / "ablation-query_aubc_significances.tex")


print("Final Mean DICE Significances")
df_to_multicol(final_significances)
df_to_multicol(final_corrs)
df_to_multicol(final_corr_pval)
final_significances["Mean"] = final_significances.mean(axis=1).round(2)
print(final_significances)
print(final_corrs)
print(final_corr_pval)

styled_corrs = final_corrs.copy(deep=True)
styled_corrs = styled_corrs.applymap(lambda x: f"{x:.4f}")
cmap = _get_cmap(final_corrs.values, final_corr_pval.values)
styled_corrs = apply_latex_coloring(styled_corrs, cmap)

final_significances.to_latex(savepath / "ablation-query_final_significances.tex")
styled_corrs.to_latex(savepath / "ablation-query_final_corrs.tex")
