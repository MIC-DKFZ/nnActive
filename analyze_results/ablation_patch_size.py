from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr, ttest_ind, wilcoxon
from setup import BASEPATH, RENAMING_DICT, apply_latex_coloring, df_to_multicol

from nnactive.analyze.analysis import SettingAnalysis
from nnactive.utils.io import save_df_to_txt

savepath = Path(
    "/home/c817h/Documents/projects/nnactive_project/nnactive/results/horeka_rsync_final/"
)

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

QUERYMETHODS = [
    "BALD",
    "PowerBALD",
    "SoftrankBALD",
    "Predictive Entropy",
    "PowerPE",
    "Random",
    "Random 66% FG",
    "Random 33% FG",
]
SETTINGS = {
    "AMOS Low": [
        "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-40__qs-40",
        "Dataset216_AMOS2022_task1/patch-16_32_32__sb-random-label2-all-classes__sbs-40__qs-40",
    ],
    "AMOS Medium": [
        "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-200__qs-200",
        "Dataset216_AMOS2022_task1/patch-16_32_32__sb-random-label2-all-classes__sbs-200__qs-200",
    ],
    "AMOS High": [
        "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-500__qs-500",
        "Dataset216_AMOS2022_task1/patch-16_32_32__sb-random-label2-all-classes__sbs-500__qs-500",
    ],
    "KiTS Low": [
        "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-40__qs-40",
        "Dataset135_KiTS2021/patch-32_32_32__sb-random-label2-all-classes__sbs-40__qs-40",
    ],
    "KiTS Medium": [
        "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-200__qs-200",
        "Dataset135_KiTS2021/patch-32_32_32__sb-random-label2-all-classes__sbs-200__qs-200",
    ],
    "KiTS High": [
        "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-500__qs-500",
        "Dataset135_KiTS2021/patch-32_32_32__sb-random-label2-all-classes__sbs-500__qs-500",
    ],
    "ACDC Low": [
        "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-30__qs-30",
        "Dataset027_ACDC/patch-2_20_20__sb-random-label2-all-classes__sbs-30__qs-30",
    ],
    "ACDC Medium": [
        "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-60__qs-60",
        "Dataset027_ACDC/patch-2_20_20__sb-random-label2-all-classes__sbs-60__qs-60",
    ],
    "ACDC High": [
        "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-60__qs-60",
        "Dataset027_ACDC/patch-2_20_20__sb-random-label2-all-classes__sbs-60__qs-60",
    ],
}

SAVENAME = "patch_ablation"
# Two sided test
SIGNIFICANCE = 0.1
TESTING = "two-sided"


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


def compute_difference(two_dfs: list[pd.DataFrame], mean_key, std_key):
    df_diff = two_dfs[1][mean_key] - two_dfs[0][mean_key]
    df_std = np.sqrt((two_dfs[0][std_key] ** 2 + two_dfs[1][std_key] ** 2))
    df_diff = pd.concat(
        [df_diff, df_std],
        axis=1,
        keys=[(mean_key[0], "mean diff"), (std_key[0], "mean std")],
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

    return pd.DataFrame(results).set_index("Query Method")


def process_analysis_pair(setting_paths, main_metric="Mean Dice"):
    analysis_list, unique_budget_list, auc_list, aucval_list = [], [], [], []

    for path in setting_paths:
        analysis = SettingAnalysis.load((BASEPATH / path) / "analysis.pkl")
        analysis_list.append(analysis)
        unique_budget_list.append(analysis.df[analysis.budget_key].unique())

    overlapping_budgets = list(set.intersection(*map(set, unique_budget_list)))

    for analysis in analysis_list:
        analysis.df = analysis.df[
            analysis.df[analysis.budget_key].isin(overlapping_budgets)
        ]
        auc_list.append(
            analysis.compute_auc_df(enforce_full=len(analysis.df) == len(analysis.df))
        )
        aucval_list.append(pd.DataFrame(analysis._compute_auc_row_dicts([main_metric])))

    return auc_list, aucval_list


def compute_statistical_tests(merged_df, metric):
    spearman, p_value_spearman = spearmanr(
        merged_df[(metric, "ranking Small Patch")],
        merged_df[(metric, "ranking Large Patch")],
        alternative="greater",
    )
    rho, p_value_kendall = kendalltau(
        merged_df[(metric, "ranking Small Patch")],
        merged_df[(metric, "ranking Large Patch")],
        alternative="greater",
    )
    wilcoxon_stat, wilcoxon_p_value = wilcoxon(
        merged_df[(metric, "ranking Small Patch")],
        merged_df[(metric, "ranking Large Patch")],
    )
    return (
        spearman,
        p_value_spearman,
        rho,
        p_value_kendall,
        wilcoxon_stat,
        wilcoxon_p_value,
    )


def main(settings: dict[str, list[str]], save: bool = True):
    final_diffs, aubc_diffs = pd.DataFrame(), pd.DataFrame()
    final_significances, aubc_significances = pd.DataFrame(), pd.DataFrame()
    final_corrs, aubc_corrs = pd.DataFrame(), pd.DataFrame()
    final_corr_pval, aubc_corr_pval = pd.DataFrame(), pd.DataFrame()
    merged_dfs = {}

    metric_list = ["AUBC", "Final"]
    metric_format = "Mean Dice {score}"

    for name, setting_paths in settings.items():
        print(f"Processing: {name}")
        auc_list, aucval_list = process_analysis_pair(setting_paths)
        merged_dfs[name] = {}

        for score in metric_list:
            metric = metric_format.format(score=score)
            mean_key, std_key = (metric, "mean"), (metric, "std")
            diff_df = compute_difference(auc_list, mean_key, std_key)
            results_df = compute_ttest(aucval_list, metric)
            results_df.columns = pd.MultiIndex.from_product(
                [["t-test"], results_df.columns]
            )

            merged_df = pd.merge(diff_df, results_df, left_index=True, right_index=True)
            merged_df = merged_df.reindex(CUSTOM_ORDER)
            merged_df[(metric, "mean Small Patch")] = auc_list[0][mean_key]
            merged_df[(metric, "mean Large Patch")] = auc_list[1][mean_key]
            merged_df[(metric, "ranking Small Patch")] = auc_list[0][mean_key].rank(
                ascending=False
            )
            merged_df[(metric, "ranking Large Patch")] = auc_list[1][mean_key].rank(
                ascending=False
            )

            merged_df.rename(RENAMING_DICT, axis=0, inplace=True)

            (
                spearman,
                p_value_spearman,
                rho,
                p_value_kendall,
                wilcoxon_stat,
                wilcoxon_p_value,
            ) = compute_statistical_tests(merged_df, metric)

            print(
                f"Spearman's Correlation: {spearman:.2f}, p-value: {p_value_spearman:.4f}"
            )
            print(f"Kendall's Correlation: {rho:.2f}, p-value: {p_value_kendall:.4f}")
            print(
                f"Wilcoxon Test Statistics: {wilcoxon_stat:.2f}, p-value: {wilcoxon_p_value:.4f}"
            )

            merged_df = merged_df.reindex(QUERYMETHODS, axis=0)
            merged_dfs[name][score] = merged_df

            if score == "AUBC":
                aubc_significances[name] = merged_df[("t-test", "significance")]
                aubc_diffs[name] = merged_df[(metric, "mean diff")]
                aubc_corrs[name] = np.array([rho])
                aubc_corr_pval[name] = np.array([p_value_kendall])
            else:
                final_significances[name] = merged_df[("t-test", "significance")]
                final_diffs[name] = merged_df[(metric, "mean diff")]
                final_corrs[name] = np.array([rho])
                final_corr_pval[name] = np.array([p_value_kendall])

    print("Final Analysis Completed")
    if save:
        final_diffs.to_latex(savepath / f"{SAVENAME}_final_diffs.tex")
        final_significances.to_latex(savepath / f"{SAVENAME}_final_significances.tex")
        final_corrs.to_latex(savepath / f"{SAVENAME}_final_corrs.tex")
        aubc_diffs.to_latex(savepath / f"{SAVENAME}_aubc_diffs.tex")
        aubc_significances.to_latex(savepath / f"{SAVENAME}_aubc_significances.tex")
        aubc_corrs.to_latex(savepath / f"{SAVENAME}_aubc_corrs.tex")

    print(
        final_diffs,
        final_corrs,
        final_corr_pval,
        aubc_diffs,
        aubc_corrs,
        aubc_corr_pval,
    )

    for score in metric_list:
        score_name = metric_format.format(score=score)
        rankings: dict[str, pd.DataFrame] = {
            "ranking Small Patch": pd.DataFrame(),
            "ranking Large Patch": pd.DataFrame(),
        }
        for setting in merged_dfs:
            for r_name in rankings:
                rankings[r_name][setting] = merged_dfs[setting][score][
                    (score_name, r_name)
                ]
        for r_name in rankings:
            df_to_multicol(rankings[r_name])

        mean_rank_key = "Mean"
        mean_rank_keys = [mean_rank_key]
        d_sets = rankings["ranking Small Patch"].columns.levels[0]
        for r_name in rankings:
            rankings[r_name][mean_rank_key] = rankings[r_name].mean(axis=1)
        for d_set in d_sets:
            for r_name in rankings:
                rankings[r_name][(d_set, mean_rank_key)] = rankings[r_name][d_set].mean(
                    axis=1
                )
                mean_rank_keys.append((d_set, mean_rank_key))

        for r_name in rankings:
            print("-" * 20)
            print(score)
            print(r_name)
            print(rankings[r_name])

        correlations = pd.DataFrame()
        for mean_rank_key in mean_rank_keys:
            spearman, p_value_spearman = spearmanr(
                rankings["ranking Small Patch"][mean_rank_key],
                rankings["ranking Large Patch"][mean_rank_key],
                alternative=TESTING,
            )
            rho, p_value_kendall = kendalltau(
                rankings["ranking Small Patch"][mean_rank_key],
                rankings["ranking Large Patch"][mean_rank_key],
                alternative=TESTING,
            )
            wilcoxon_stat, wilcoxon_p_value = wilcoxon(
                rankings["ranking Small Patch"][mean_rank_key],
                rankings["ranking Large Patch"][mean_rank_key],
            )
            final_key = (
                mean_rank_key[0] if isinstance(mean_rank_key, tuple) else mean_rank_key
            )
            correlations[final_key] = {
                "spearman": spearman,
                "p_value_spearman": p_value_spearman,
                "rho": rho,
                "p_value_kendall": p_value_kendall,
                "wilcoxon_stat": wilcoxon_stat,
                "wilcoxon_p_value": wilcoxon_p_value,
            }

        save_df_to_txt(
            correlations,
            savepath / f"{SAVENAME}_{score}_rank_correlations.txt",
        )

        cmap = _get_cmap(
            correlations.loc["rho":"rho"].values,
            correlations.loc["p_value_kendall":"p_value_kendall"].values < SIGNIFICANCE,
        )
        styled = correlations.loc["rho":"rho"].round(3)
        styled = apply_latex_coloring(styled, cmap)
        styled.to_latex(savepath / f"{SAVENAME}_{score}_rank_correlations.tex")


if __name__ == "__main__":
    main(SETTINGS, save=False)
