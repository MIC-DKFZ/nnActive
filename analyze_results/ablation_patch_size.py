from pathlib import Path

import numpy as np
import pandas as pd
from evaluator import (
    get_settings_for_combination,
    load_settings,
    rename_settings_in_analysis,
)
from scipy.stats import kendalltau, spearmanr, ttest_ind, wilcoxon
from setup import (
    BASEPATH,
    CUSTOM_ORDER,
    RENAMING_DICT,
    SAVEPATH,
    VALUE_TO_COLOR_MAP,
    apply_latex_coloring,
    calculate_difference_with_std,
    df_to_multicol,
    get_ranking_cmap,
)

from nnactive.analyze.analysis import SettingAnalysis
from nnactive.utils.io import save_df_to_txt

savepath = SAVEPATH / "tex" / "ablation_patch_size"
savepath.mkdir(parents=True, exist_ok=True)

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


COMPARATIVE = False
USE_SETTINGS = ["Main", "Patchx1/2"]

RENAME_SETTINGS = {
    "Main": "Patchx1",
}

settings = get_settings_for_combination(USE_SETTINGS)
settings_analyses = load_settings(settings, comparative=COMPARATIVE)
rename_settings_in_analysis(settings_analyses, RENAME_SETTINGS)

MAIN_METRIC = "Mean Dice"


SAVENAME = "patch_ablation"
# Two sided test
SIGNIFICANCE = 0.1
TESTING = "two-sided"


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


def main(
    setting_analyses: dict[str, dict[str, dict[str, SettingAnalysis]]],
    save: bool = True,
):
    merged_dfs = {}

    SCORES = ["AUBC", "Final"]

    for dataset_name in setting_analyses:
        for budget in setting_analyses[dataset_name]:
            name = f"{dataset_name} {budget}"
            merged_dfs[name] = {}

            print(f"Processing: {name}")
            auc_dicts = {}
            for exp_row in setting_analyses[dataset_name][budget]:
                analysis = setting_analyses[dataset_name][budget][exp_row]
                auc = analysis.compute_auc_df(enforce_full=not COMPARATIVE).reindex(
                    CUSTOM_ORDER
                )
                auc.rename(RENAMING_DICT, axis=0, inplace=True)
                auc_dicts[exp_row] = auc

            for score in SCORES:
                metric = f"{MAIN_METRIC} {score}"
                mean_key, std_key = (metric, "mean"), (metric, "std")
                exp_row_names = list(auc_dicts.keys())
                print(exp_row_names)
                diff_df = calculate_difference_with_std(
                    auc_dicts[exp_row_names[1]],
                    auc_dicts[exp_row_names[0]],
                    mean_key,
                    std_key,
                )

                ## just try this out
                merged_df = diff_df.copy()
                for exp_row in exp_row_names:
                    merged_df[(metric, "mean " + exp_row)] = auc_dicts[exp_row][
                        mean_key
                    ]
                    # doublecheck whether we already threw out unnecessary values
                    # for ranking this however does not matter super much
                    merged_df[(metric, "ranking " + exp_row)] = auc_dicts[exp_row][
                        mean_key
                    ].rank(ascending=False)

                merged_dfs[name][score] = merged_df

    for score in SCORES:
        metric = f"{MAIN_METRIC} {score}"

        rankings: dict[str, pd.DataFrame] = {
            "ranking " + RENAME_SETTINGS.get(x, x): pd.DataFrame() for x in USE_SETTINGS
        }
        for setting in merged_dfs:
            for r_name in rankings:
                rankings[r_name][setting] = merged_dfs[setting][score][(metric, r_name)]
        for r_name in rankings:
            df_to_multicol(rankings[r_name])

        mean_rank_key = "Mean"
        mean_rank_keys = [mean_rank_key]
        d_sets = list(rankings.values())[0].columns.levels[0]
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

        compare_rank_keys = list(rankings.keys())
        assert (
            len(compare_rank_keys) == 2
        )  # these tests currently only allow to compare two rankings
        comp_i = compare_rank_keys[0]
        comp_j = compare_rank_keys[1]

        correlations = pd.DataFrame()
        for mean_rank_key in mean_rank_keys:
            spearman, p_value_spearman = spearmanr(
                rankings[comp_i][mean_rank_key],
                rankings[comp_j][mean_rank_key],
                alternative=TESTING,
            )
            rho, p_value_kendall = kendalltau(
                rankings[comp_i][mean_rank_key],
                rankings[comp_j][mean_rank_key],
                alternative=TESTING,
            )
            wilcoxon_stat, wilcoxon_p_value = wilcoxon(
                rankings[comp_i][mean_rank_key],
                rankings[comp_j][mean_rank_key],
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

        cmap = get_ranking_cmap(
            correlations.loc["rho":"rho"].values,
            correlations.loc["p_value_kendall":"p_value_kendall"].values < SIGNIFICANCE,
        )
        styled = correlations.loc["rho":"rho"].round(3)
        styled = apply_latex_coloring(styled, cmap)
        styled.to_latex(savepath / f"{SAVENAME}_{score}_rank_correlations.tex")


if __name__ == "__main__":
    main(settings_analyses, save=False)
