from pathlib import Path

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from evaluator import (
    get_settings_for_combination,
    load_settings,
    rename_settings_in_analysis,
)
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
    SAVEPATH,
    VALUE_TO_COLOR_MAP,
    apply_latex_coloring,
    calculate_difference_with_std,
    compute_kendalltau_correlation_from_dfs,
    df_to_multicol,
    get_ranking_cmap,
    save_styled_to_latex,
)

USE_SETTINGS = ["QSx2", "Main", "QSx1/2"]
COMPARATIVE = True
settings = get_settings_for_combination(USE_SETTINGS)
setting_analyses = load_settings(settings, comparative=COMPARATIVE)
RENAME_SETTINGS = {
    "Main": "QSx1",
}
COLLEVELNAMES = ["Dataset", "Label Regime"]

rename_settings_in_analysis(setting_analyses, RENAME_SETTINGS)

# only compute correlations in between Query Methods as we are interested is
# how the perfmance of the query methods is correlated with the Query Size
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

savepath = SAVEPATH / "tex" / "ablation_query_size"
savepath.mkdir(parents=True, exist_ok=True)
MAIN_METRIC = "Mean Dice"
SCORES = ["AUBC", "Final"]


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


def compute_multi_ttest(
    aucval_dict: dict[str, pd.DataFrame], metric, significance: float = 0.05
):
    test_groups = {}
    for aucval in aucval_dict:
        g = aucval_dict[aucval].groupby("Query Method")[metric]
        test_groups[aucval] = g

    results = {"Query Method": []}
    exp_row_names = list(aucval_dict.keys())
    for i, i_name in enumerate(exp_row_names):
        for j in range(i + 1, len(exp_row_names)):
            j_name = exp_row_names[j]
            results[f"t-statistic {i_name}-{j_name}"] = []
            results[f"p-value {i_name}-{j_name}"] = []
            results[f"significance {i_name}-{j_name}"] = []

    methods = test_groups[exp_row_names[0]].groups.keys()

    for method in methods:
        if all(method in g.groups for g in test_groups.values()):
            results["Query Method"].append(method)
            for i, i_name in enumerate(exp_row_names):
                for j in range(i + 1, len(exp_row_names)):
                    j_name = exp_row_names[j]
                    t_stat, p_value = ttest_ind(
                        test_groups[j_name].get_group(method),
                        test_groups[i_name].get_group(method),
                        alternative="greater",
                    )
                    results[f"t-statistic {i_name}-{j_name}"].append(t_stat)
                    results[f"p-value {i_name}-{j_name}"].append(p_value)
                    results[f"significance {i_name}-{j_name}"].append(
                        p_value < significance
                    )

    results_df = pd.DataFrame(results).set_index("Query Method")
    return results_df


def obtain_values(
    auc_dict: dict[str, pd.DataFrame], aucval_dict: dict[str, pd.DataFrame], score: str
):
    metric = f"{MAIN_METRIC} {score}"
    mean_key = (metric, "mean")
    std_key = (metric, "std")

    exp_row_names = list(auc_dict.keys())

    # only take into account values which we are interested in
    for exp_row in exp_row_names:
        auc_dict[exp_row] = auc_dict[exp_row].reindex(CUSTOM_ORDER)

    auc_diff = calculate_difference_with_std(
        auc_dict[exp_row_names[1]], auc_dict[exp_row_names[0]], mean_key, std_key
    )

    for exp_row in exp_row_names:
        auc_diff[(metric, exp_row)] = auc_dict[exp_row][mean_key]

    auc_diff = auc_diff[~auc_diff[mean_key].isna()]
    #

    for exp_row in exp_row_names:
        auc_diff[(metric, "ranking " + exp_row)] = (
            auc_dict[exp_row].loc[auc_diff.index][mean_key].rank(ascending=False)
        )

    results_df = compute_multi_ttest(aucval_dict, metric)
    auc_corr = compute_kendalltau_correlation_from_dfs(
        list(aucval_dict.values()), CUSTOM_ORDER, metric
    )
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
    return auc_corr, merged_df


def process_experiments(setting_analyses, main_metric, scores):
    """Processes experiments, computes AUBC and Final significance/correlations."""
    final_significances, aubc_siginificances = pd.DataFrame(), pd.DataFrame()
    final_corrs, aubc_corrs = pd.DataFrame(), pd.DataFrame()
    final_corr_pval, aubc_corr_pval = pd.DataFrame(), pd.DataFrame()

    for dataset_name in setting_analyses:
        for budget in setting_analyses[dataset_name]:
            name = f"{dataset_name} {budget}"
            print(f"{'-' * 10}\n{name}")
            auc_dict, aucval_dict = extract_auc_data(
                setting_analyses[dataset_name][budget], main_metric
            )

            for score in scores:
                auc_corr, merged_df = obtain_values(auc_dict, aucval_dict, score)
                store_results(
                    score,
                    name,
                    merged_df,
                    auc_corr,
                    final_significances,
                    aubc_siginificances,
                    final_corrs,
                    aubc_corrs,
                    final_corr_pval,
                    aubc_corr_pval,
                )

    return (
        final_significances,
        aubc_siginificances,
        final_corrs,
        aubc_corrs,
        final_corr_pval,
        aubc_corr_pval,
    )


def extract_auc_data(experiments, main_metric):
    """Extracts AUC data from experiments."""
    auc_dict, aucval_dict = {}, {}
    for exp_row, analysis in experiments.items():
        auc = analysis.compute_auc_df(enforce_full=not COMPARATIVE)
        auc_dict[exp_row] = auc
        aucval_dict[exp_row] = pd.DataFrame(
            analysis._compute_auc_row_dicts([main_metric])
        )
    return auc_dict, aucval_dict


def store_results(
    score,
    name,
    merged_df,
    auc_corr,
    final_significances,
    aubc_siginificances,
    final_corrs,
    aubc_corrs,
    final_corr_pval,
    aubc_corr_pval,
):
    """Stores significance and correlation results based on score type."""
    if score == "Final":
        final_significances[name] = merged_df[("t-test", "full-significance")]
        final_corrs[name] = auc_corr["corr"]
        final_corr_pval[name] = auc_corr["significance"]
    else:
        aubc_siginificances[name] = merged_df[("t-test", "full-significance")]
        aubc_corrs[name] = auc_corr["corr"]
        aubc_corr_pval[name] = auc_corr["significance"]


def print_and_save_results(
    significances: pd.DataFrame,
    corrs: pd.DataFrame,
    corr_pval: pd.DataFrame,
    savepath: Path,
    prefix: str,
):
    """Prints and saves significance and correlation results to LaTeX."""
    print(f"Final {prefix} Significances")
    df_to_multicol(significances)
    df_to_multicol(corrs)
    df_to_multicol(corr_pval)
    corrs.columns.names = COLLEVELNAMES
    significances.columns.names = COLLEVELNAMES
    significances["Mean"] = significances.mean(axis=1).round(2)
    print(significances)
    print(corrs)
    print(corr_pval)

    corrs.round(3)
    styled_corrs = corrs.copy().map(lambda x: f"{x:.3f}")
    cmap = get_ranking_cmap(corrs.values, corr_pval.values)
    styled_corrs = apply_latex_coloring(styled_corrs, cmap)

    save_styled_to_latex(
        significances, savepath / f"ablation-query_{prefix.lower()}_significances.tex"
    )
    save_styled_to_latex(
        styled_corrs, savepath / f"ablation-query_{prefix.lower()}_corrs.tex"
    )


if __name__ == "__main__":
    # Main Execution
    (
        final_significances,
        aubc_siginificances,
        final_corrs,
        aubc_corrs,
        final_corr_pval,
        aubc_corr_pval,
    ) = process_experiments(setting_analyses, MAIN_METRIC, SCORES)

    print_and_save_results(
        aubc_siginificances, aubc_corrs, aubc_corr_pval, savepath, "AUBC"
    )
    print_and_save_results(
        final_significances, final_corrs, final_corr_pval, savepath, "Final"
    )
