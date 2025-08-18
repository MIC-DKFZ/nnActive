from pathlib import Path
from typing import Iterable

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
    apply_latex_coloring,
    calculate_difference_with_std,
    df_to_multicol,
    get_ranking_cmap,
    save_styled_to_latex,
)

from nnactive.analyze.analysis import SettingAnalysis

savepath = SAVEPATH / "tex" / "ablation_training"
savepath.mkdir(parents=True, exist_ok=True)

COMPARATIVE = False
# second values is always expected to have better performance then the first
USE_SETTINGS_LIST = [
    ["Precomputed", "500 Epochs"],
    ["Main", "500 Epochs"],
    ["Main", "Precomputed"],
]
RENAME_SETTINGS = {
    "Main": "200 Epochs",
}
COPY_VALUES_LIST = [["random", "random-label", "random-label2"], None, None]
SIGNIFICANCE = 0.1
ALTERNATIVE = "two-sided"


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

COPY_VALUES = ["random", "random-label", "random-label2"]
SCORES = ["Final", "AUBC"]
MAIN_METRIC = "Mean Dice"
COLLEVELNAMES = ["Dataset", "Label Regime"]

QUERYMETHODS = [
    "BALD",
    "PowerBALD",
    "SoftrankBALD",
    "Predictive Entropy",
    "PowerPE",
]


def compute_ttest(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    metric: str,
    significance: float = 0.05,
    alternative="greater",
):
    g1 = df1.groupby("Query Method")[metric]
    g2 = df2.groupby("Query Method")[metric]
    results = {"Query Method": [], "t-statistic": [], "p-value": [], "significance": []}

    for method in g1.groups.keys():
        if method in g2.groups:
            t_stat, p_value = ttest_ind(
                g1.get_group(method), g2.get_group(method), alternative=alternative
            )
            results["Query Method"].append(method)
            results["t-statistic"].append(t_stat)
            results["p-value"].append(p_value)
            results["significance"].append(p_value < significance)

    results_df = pd.DataFrame(results).set_index("Query Method")
    return results_df


def analyze_settings(
    setting_analyses: dict[str, dict[str, dict[str, SettingAnalysis]]],
    savename: str | None = None,
    copy_values: Iterable[str] | None = None,
):
    diffs_dict: dict[str, pd.DataFrame] = {score: pd.DataFrame() for score in SCORES}
    significances_dict: dict[str, pd.DataFrame] = {
        score: pd.DataFrame() for score in SCORES
    }
    corrs_dict: dict[str, pd.DataFrame] = {score: pd.DataFrame() for score in SCORES}
    corr_pval_dict: dict[str, pd.DataFrame] = {
        score: pd.DataFrame() for score in SCORES
    }
    for dataset_name in setting_analyses:
        for budget in setting_analyses[dataset_name]:
            name = f"{dataset_name} {budget}"
            print(name)

            assert len(setting_analyses[dataset_name][budget]) == 2
            auc_dicts = {}
            aucval_dicts = {}
            for exp_row in setting_analyses[dataset_name][budget]:
                analysis = setting_analyses[dataset_name][budget][exp_row]
                auc = analysis.compute_auc_df(enforce_full=not COMPARATIVE)
                auc_dicts[exp_row] = auc
                aucval_dicts[exp_row] = pd.DataFrame(
                    analysis._compute_auc_row_dicts([MAIN_METRIC])
                )

            for score in SCORES:
                metric, merged_df, rho, p_value_kendall = calculate_auc_statistics(
                    auc_dicts, aucval_dicts, MAIN_METRIC, score, copy_values
                )

                significances_dict[score][name] = merged_df[("t-test", "significance")]
                diffs_dict[score][name] = merged_df[(metric, "mean")]
                corrs_dict[score][name] = np.array([rho])
                corr_pval_dict[score][name] = np.array([p_value_kendall])

    for score in SCORES:
        print("\n" * 2)
        print("Results for score: ", score)
        df_to_multicol(significances_dict[score])
        df_to_multicol(diffs_dict[score])
        df_to_multicol(corrs_dict[score])
        df_to_multicol(corr_pval_dict[score])
        diffs_dict[score] = diffs_dict[score].map(lambda x: np.round(x * 100, 2))
        print(diffs_dict[score])
        if savename is not None:
            basename = f"ablation-training-{savename}"
            print("Saving to: ", savepath / basename)
            significances_dict[score].columns.names = COLLEVELNAMES
            corrs_dict[score].columns.names = COLLEVELNAMES
            diffs_dict[score].columns.names = COLLEVELNAMES
            styled: pd.DataFrame = diffs_dict[score].copy(deep=True)
            styled = styled.map(lambda x: f"{x:.2f}")
            ##### Significant values are bold
            # styled[significances_dict[score] == True] = styled[
            #     significances_dict[score] == True
            # ].map(lambda x: f"\\textbf{{{x}}}")
            ##### Colorbased significance
            cmap = get_ranking_cmap(
                diffs_dict[score].values, significances_dict[score].values
            )
            styled = apply_latex_coloring(styled, cmap)
            save_styled_to_latex(
                styled, savepath / f"{basename}-{score.lower()}_diffs.tex"
            )
            # significances_dict[score].columns.names = COLLEVELNAMES
            # corrs_dict[score].columns.names = COLLEVELNAMES
            # save_styled_to_latex(
            #     significances_dict[score],
            #     savepath / f"{basename}-{score.lower()}_significances.tex",
            # )
            # save_styled_to_latex(
            #     corrs_dict[score], savepath / f"{basename}-{score.lower()}_corrs.tex"
            # )
    return diffs_dict, significances_dict, corrs_dict, corr_pval_dict


def calculate_auc_statistics(
    auc_dict: dict[str, pd.DataFrame],
    aucval_dict: dict[str, pd.DataFrame],
    main_metric: str,
    score: str,
    copy_values: Iterable[str] | None = None,
):
    metric = f"{main_metric} {score}"
    mean_key = (metric, "mean")
    std_key = (metric, "std")
    exp_row_names = list(auc_dict.keys())

    # This is only done for the ranking analysis
    # This is a hack to copy the values of the random methods to the other methods
    # Only works if all values are identical across these methods
    # So e.g. for different training length this is not a valid approach
    if copy_values is not None:
        for method in copy_values:
            auc_dict[exp_row_names[1]].loc[method] = auc_dict[exp_row_names[0]].loc[
                method
            ]

    for exp_row in exp_row_names:
        auc_dict[exp_row] = auc_dict[exp_row].reindex(CUSTOM_ORDER)

    auc_diff = calculate_difference_with_std(
        auc_dict[exp_row_names[1]], auc_dict[exp_row_names[0]], mean_key, std_key
    )

    for exp_row in exp_row_names:
        auc_diff[(metric, exp_row)] = auc_dict[exp_row][mean_key]

    for exp_row in exp_row_names:
        auc_diff[(metric, "ranking " + exp_row)] = (
            auc_dict[exp_row].loc[auc_diff.index][mean_key].rank(ascending=False)
        )

    results_df = compute_ttest(
        aucval_dict[exp_row_names[1]],
        aucval_dict[exp_row_names[0]],
        metric,
        SIGNIFICANCE,
        ALTERNATIVE,
    )

    results_df.columns = pd.MultiIndex.from_product([["t-test"], results_df.columns])
    merged_df = pd.merge(auc_diff, results_df, left_index=True, right_index=True)
    merged_df.rename(RENAMING_DICT, axis=0, inplace=True)
    print(merged_df)
    rho, p_value_kendall = kendalltau(
        merged_df[(metric, "ranking " + exp_row_names[1])],
        merged_df[(metric, "ranking " + exp_row_names[0])],
        alternative=ALTERNATIVE,
    )
    print(f"Kendall's Correlation: {rho:.2f}, p-value: {p_value_kendall:.4f}")
    merged_df = merged_df.reindex(QUERYMETHODS, axis=0)
    return metric, merged_df, rho, p_value_kendall


def concatenate_results(corrs_settings_dict):
    df_list = []
    for outer_key, inner_dict in corrs_settings_dict.items():
        for inner_key, df in inner_dict.items():
            df = df.copy()
            df.index = pd.MultiIndex.from_product(
                [[outer_key], [inner_key]], names=["Setting", "Metric"]
            )
            df_list.append(df)

    # Concatenate everything
    final_df = pd.concat(df_list)
    return final_df


if __name__ == "__main__":
    corrs_settings_dict = {}
    corrs_pval_settings_dict = {}
    for i, (use_settings, copy_values) in enumerate(
        zip(USE_SETTINGS_LIST, COPY_VALUES_LIST)
    ):
        print("Compare: ", use_settings)
        dictname = " \\& ".join([RENAME_SETTINGS.get(s, s) for s in use_settings])
        savename = "-".join(use_settings)
        savename = savename.replace(" ", "")
        savename = savename.lower()
        settings = get_settings_for_combination(use_settings)
        setting_analyses = load_settings(settings, comparative=COMPARATIVE)
        rename_settings_in_analysis(setting_analyses, RENAME_SETTINGS)
        diffs_dict, significances_dict, corrs_dict, corr_pval_dict = analyze_settings(
            setting_analyses, savename=savename, copy_values=copy_values
        )
        corrs_settings_dict[dictname] = corrs_dict
        corrs_pval_settings_dict[dictname] = corr_pval_dict

    corrs_pval_settings_df = concatenate_results(corrs_pval_settings_dict)
    corrs_sig_settings_df = corrs_pval_settings_df < SIGNIFICANCE
    corrs_settings_df = concatenate_results(corrs_settings_dict)
    cmap = get_ranking_cmap(corrs_settings_df.values, corrs_sig_settings_df.values)

    corrs_settings_df.columns.names = COLLEVELNAMES
    corrs_settings_df = corrs_settings_df.round(3).map(lambda x: f"{x:.3f}")
    corrs_settings_df = apply_latex_coloring(corrs_settings_df, cmap)
    save_styled_to_latex(corrs_settings_df, savepath / "ablation-training-corrs.tex")

    for score in SCORES:
        filter_corrs = corrs_settings_df.xs(score, level="Metric")
        save_styled_to_latex(
            filter_corrs, savepath / f"ablation-training-corrs-{score}.tex"
        )
