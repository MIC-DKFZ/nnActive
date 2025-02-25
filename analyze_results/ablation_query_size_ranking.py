from itertools import product
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
from setup import (
    BASEPATH,
    RENAMING_DICT,
    apply_latex_coloring,
    df_to_multicol,
    save_styled_to_latex,
)

from nnactive.analyze.analysis import SettingAnalysis

savepath = Path(
    "/home/c817h/Documents/projects/nnactive_project/nnactive/results/horeka_rsync_final/"
)

# 2nd value is always value which is expected to be better than the first. E.g. smaller QS is expected to be better.
# TODO: enable this for all settings!
settings = {
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
}

# only compute correlations in between Query Methods as we are interested is
# how the perfmance of the query methods is correlated with the Query Size
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

TESTSIDED = "greater"
SIGNIFICANCE = 0.05


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


VALMAP = {
    -2: "#FF0000",  # red
    -1: "#F08080",  # lightcoral
    0: "#FFFFFF",  # white
    1: "#90EE90",  # light green
    2: "#008000",  # green
}

BUDGET_ORDER = {"Low": 0, "Medium": 1, "High": 2}
MEANROW = "kendall-tau mean"


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


rankings = []
for name in settings:
    print("-" * 10)
    print(name)
    analysis_list: list[SettingAnalysis] = []
    unique_budget_list = []
    auc_list = []
    aucval_list = []
    exp_identifiers = []

    main_metric = "Mean Dice"

    for path in settings[name]:
        path = BASEPATH / path
        analysis = SettingAnalysis.load(path / "analysis.pkl")
        exp_identifiers.append(analysis.df["query_size"].unique()[0])

        analysis_list.append(analysis)
        unique_budget_list.append(analysis.df[analysis.budget_key].unique())

    overlapping_budgets = list(set.intersection(*map(set, unique_budget_list)))

    for analysis in analysis_list:
        orig_size = len(analysis.df)
        analysis.df = analysis.df[
            analysis.df[analysis.budget_key].isin(overlapping_budgets)
        ]
        enforce_full_loops = len(analysis.df) == orig_size
        auc = analysis.compute_auc_df(enforce_full=enforce_full_loops)
        auc_list.append(auc.reindex(CUSTOM_ORDER))
        aucval_list.append(
            pd.DataFrame(analysis._compute_auc_row_dicts([main_metric])).reindex(
                CUSTOM_ORDER
            )
        )

    scores = ["AUBC", "Final"]
    for score in scores:
        metric = f"{main_metric} {score}"
        mean_key = (metric, "mean")
        std_key = (metric, "std")
        dataset = name.split(" ")[0]
        budget = name.split(" ")[1]
        ranking = {"Metric": score, "Dataset": dataset, "Budget": budget}
        for i, auc in enumerate(auc_list):
            # ranking[f"ranking QS {i}"] = auc[mean_key].rank(ascending=False)
            pass

        exp_identifiers = ["x0.5", "x1", "x2"]

        count = 0
        ranking[MEANROW] = 0
        # ranking["kendall-tau pvals"] = []
        for i, j in product(range(len(auc_list)), repeat=2):
            if i == j or j < i:
                continue
            tau, alpha = kendalltau(auc_list[i][mean_key], auc_list[j][mean_key])
            ranking[f"kendall-tau QS{exp_identifiers[i]} vs QS{exp_identifiers[j]}"] = (
                tau
            )
            ranking[
                f"kendall-tau QS{exp_identifiers[i]} vs QS{exp_identifiers[j]} pval"
            ] = alpha
            ranking[MEANROW] += tau
            # ranking["kendall-tau pvals"] += [alpha]
            count += 1
        ranking[MEANROW] /= count if count > 0 else 1

        rankings.append(ranking)

ranking_df = pd.DataFrame(rankings)

for score in scores:
    mean_ranking_df = (
        ranking_df[ranking_df["Metric"] == score]
        .groupby(["Dataset", "Budget"])
        .mean(numeric_only=True)
    )

    mean_ranking_df_t = mean_ranking_df.T
    cols = mean_ranking_df_t.columns

    new_order = []
    for l_1 in cols.levels[0]:
        sub_cols = []
        for col in cols:
            if col[0] == l_1:
                sub_cols.append(col[1])
        sub_cols = sorted(sub_cols, key=lambda x: BUDGET_ORDER[x])
        new_order.extend([(l_1, sub_col) for sub_col in sub_cols])

    mean_ranking_df_t = mean_ranking_df_t[new_order]

    mean_row = mean_ranking_df_t.loc[MEANROW]
    mean_ranking_df_t = mean_ranking_df_t.drop(MEANROW)
    mean_ranking_df_t.loc[MEANROW] = mean_row

    print(mean_ranking_df_t)

    significance_df = []
    tex_ranking_df = []
    for index in mean_ranking_df_t.index:
        print(index)
        if index.endswith("pval"):
            significance_df.append(index)
            # significance_df.loc[index] = mean_ranking_df_t.loc[index]
        else:
            # tex_ranking_df.loc[index] = mean_ranking_df_t.loc[index]
            tex_ranking_df.append(index)

    # print(significance_df)

    # print(tex_ranking_df)
    # break

    tex_ranking_df = mean_ranking_df_t.loc[tex_ranking_df]
    significance_df = mean_ranking_df_t.loc[significance_df]

    significance_df = significance_df <= SIGNIFICANCE
    significance_df.loc[MEANROW] = False

    styled = tex_ranking_df.copy(deep=True)
    cmap = _get_cmap(tex_ranking_df.values, significance_df.values)
    styled: pd.DataFrame = styled.applymap(lambda x: f"{x:.3f}")
    styled = apply_latex_coloring(styled, cmap)
    save_styled_to_latex(styled, savepath / f"ablation-query_size_ranking_{score}.tex")

    # import IPython

    # IPython.embed()

    # break

    # import IPython

    # IPython.embed()

    # print(mean_ranking_df_f)


#     for i, auc in enumerate(auc_list):
#         auc_diff[(metric, f"mean{i}")] = auc[mean_key]

#     auc_diff = auc_diff[~auc_diff[mean_key].isna()]
#     #

#     for i, auc in enumerate(auc_list):
#         auc_diff[(metric, f"ranking QS {i}")] = auc.loc[auc_diff.index][mean_key].rank(
#             ascending=False
#         )

#     results_df = compute_ttest(aucval_list, metric)
#     auc_corr = compute_correlation_from_dfs(aucval_list, CUSTOM_ORDER, metric)
#     auc_corr.rename(RENAMING_DICT, axis=0, inplace=True)

#     results_df.columns = pd.MultiIndex.from_product([["t-test"], results_df.columns])
#     merged_df = pd.merge(auc_diff, results_df, left_index=True, right_index=True)
#     merged_df = merged_df.reindex(CUSTOM_ORDER)
#     merged_df.rename(RENAMING_DICT, axis=0, inplace=True)
#     print(merged_df)
#     sig_cols = [col for col in merged_df.columns if col[1].startswith("significance")]

#     merged_df[("t-test", "full-significance")] = 0
#     for col in sig_cols:
#         merged_df[("t-test", "full-significance")] += merged_df[col].astype(int)

#     print(merged_df[("t-test", "full-significance")])
#     aubc_siginificances[name] = merged_df[("t-test", "full-significance")]

#     score = "Final"
#     metric = f"{main_metric} {score}"
#     mean_key = (metric, "mean")
#     std_key = ("Mean Dice Final", "std")
#     final_diff = compute_difference(auc_list, mean_key, std_key)

#     for i, auc in enumerate(auc_list):
#         final_diff[(metric, f"mean{i}")] = auc[mean_key]
#     final_diff = final_diff[~final_diff[mean_key].isna()]

#     final_diff = final_diff[~final_diff[mean_key].isna()]
#     #

#     for i, auc in enumerate(auc_list):
#         final_diff[(metric, f"ranking QS {i}")] = auc.loc[auc_diff.index][
#             mean_key
#         ].rank(ascending=False)

#     results_df = compute_ttest(aucval_list, metric)
#     results_df.columns = pd.MultiIndex.from_product([["t-test"], results_df.columns])
#     merged_df = pd.merge(final_diff, results_df, left_index=True, right_index=True)
#     merged_df = merged_df.reindex(CUSTOM_ORDER)
#     merged_df.rename(RENAMING_DICT, axis=0, inplace=True)
#     print(merged_df)
#     sig_cols = [col for col in merged_df.columns if col[1].startswith("significance")]

#     merged_df[("t-test", "full-significance")] = 0
#     for col in sig_cols:
#         merged_df[("t-test", "full-significance")] += merged_df[col].astype(int)

#     print(merged_df[("t-test", "full-significance")])
#     final_significances[name] = merged_df[("t-test", "full-significance")]
#     final_corr = compute_correlation_from_dfs(aucval_list, CUSTOM_ORDER, metric)
#     final_corr.rename(RENAMING_DICT, axis=0, inplace=True)
#     final_corrs[name] = final_corr["corr"]
#     aubc_corrs[name] = auc_corr["corr"]
#     final_corr_pval[name] = final_corr["significance"]
#     aubc_corr_pval[name] = auc_corr["significance"]


# print("Final AUBC Significances")
# df_to_multicol(aubc_siginificances)
# df_to_multicol(aubc_corrs)
# df_to_multicol(aubc_corr_pval)
# aubc_siginificances["Mean"] = aubc_siginificances.mean(axis=1).round(2)
# print(aubc_siginificances)
# styled_corrs = aubc_corrs.copy(deep=True)
# styled_corrs = styled_corrs.applymap(lambda x: f"{x:.4f}")
# cmap = _get_cmap(aubc_corrs.values, aubc_corr_pval.values)
# styled_corrs = apply_latex_coloring(styled_corrs, cmap)
# styled_corrs.to_latex(savepath / "ablation-query_aubc_corrs.tex")
# print(aubc_corrs)
# print(aubc_corr_pval)


# aubc_siginificances.to_latex(savepath / "ablation-query_aubc_significances.tex")


# print("Final Mean DICE Significances")
# df_to_multicol(final_significances)
# df_to_multicol(final_corrs)
# df_to_multicol(final_corr_pval)
# final_significances["Mean"] = final_significances.mean(axis=1).round(2)
# print(final_significances)
# print(final_corrs)
# print(final_corr_pval)

# styled_corrs = final_corrs.copy(deep=True)
# styled_corrs = styled_corrs.applymap(lambda x: f"{x:.4f}")
# cmap = _get_cmap(final_corrs.values, final_corr_pval.values)
# styled_corrs = apply_latex_coloring(styled_corrs, cmap)

# final_significances.to_latex(savepath / "ablation-query_final_significances.tex")
# styled_corrs.to_latex(savepath / "ablation-query_final_corrs.tex")
