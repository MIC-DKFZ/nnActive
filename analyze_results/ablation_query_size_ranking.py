from itertools import product
from pathlib import Path

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
    SAVEPATH,
    VALUE_TO_COLOR_MAP,
    apply_latex_coloring,
    calculate_difference_with_std,
    df_to_multicol,
    get_ranking_cmap,
    save_styled_to_latex,
)

from nnactive.analyze.analysis import SettingAnalysis

savepath = SAVEPATH / "tex" / "ablation_query_size"
savepath.mkdir(parents=True, exist_ok=True)

# 2nd value is always value which is expected to be better than the first. E.g. smaller QS is expected to be better.
# TODO: enable this for all settings!
USE_SETTINGS = ["QSx2", "Main", "QSx1/2"]
COMPARATIVE = True
settings = get_settings_for_combination(USE_SETTINGS)
setting_analyses = load_settings(settings, comparative=COMPARATIVE)
RENAME_SETTINGS = {
    "Main": "QSx1",
}
rename_settings_in_analysis(setting_analyses, RENAME_SETTINGS)

SCORES = ["AUBC", "Final"]

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


BUDGET_ORDER = {"Low": 0, "Medium": 1, "High": 2}
MEANROW = "Mean"


rankings = []
for dataset_name in setting_analyses:
    for budget in setting_analyses[dataset_name]:
        name = f"{dataset_name} {budget}"
        print("-" * 10)
        print(name)
        auc_dict = {}

        main_metric = "Mean Dice"

        for exp_row in setting_analyses[dataset_name][budget]:
            analysis = setting_analyses[dataset_name][budget][exp_row]
            auc = analysis.compute_auc_df(enforce_full=not COMPARATIVE)
            auc_dict[exp_row] = auc.reindex(CUSTOM_ORDER)
        exp_row_names = list(auc_dict.keys())

        scores = ["AUBC", "Final"]
        for score in scores:
            metric = f"{main_metric} {score}"
            mean_key = (metric, "mean")
            std_key = (metric, "std")
            dataset = name.split(" ")[0]
            budget = name.split(" ")[1]
            ranking = {"Metric": score, "Dataset": dataset, "Label Regime": budget}

            for exp_row in exp_row_names:
                ranking["ranking " + exp_row] = auc_dict[exp_row][mean_key].rank(
                    ascending=False
                )

            count = 0
            ranking[MEANROW] = 0
            ranking["kendall-tau pvals"] = []
            for i, j in product(range(len(exp_row_names)), repeat=2):
                if i == j or j < i:
                    continue
                i_name, j_name = exp_row_names[i], exp_row_names[j]
                tau, alpha = kendalltau(
                    auc_dict[i_name][mean_key], auc_dict[j_name][mean_key]
                )
                ranking[f"{i_name} vs {j_name}"] = tau
                ranking[f"{i_name} vs {j_name} pval"] = alpha
                ranking[MEANROW] += tau
                ranking["kendall-tau pvals"] += [alpha]
                count += 1
            ranking[MEANROW] /= count if count > 0 else 1

            rankings.append(ranking)

ranking_df = pd.DataFrame(rankings)

for score in scores:
    mean_ranking_df = (
        ranking_df[ranking_df["Metric"] == score]
        .groupby(["Dataset", "Label Regime"])
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
        if index.endswith("pval"):
            significance_df.append(index)
        else:
            tex_ranking_df.append(index)

    tex_ranking_df = mean_ranking_df_t.loc[tex_ranking_df]
    significance_df = mean_ranking_df_t.loc[significance_df]

    significance_df = significance_df <= SIGNIFICANCE
    significance_df.loc[MEANROW] = False

    styled = tex_ranking_df.copy(deep=True)
    cmap = get_ranking_cmap(tex_ranking_df.values, significance_df.values)
    styled.index.name = "Setting"
    styled.round(3)
    styled: pd.DataFrame = styled.map(lambda x: f"{x:.3f}")
    styled = apply_latex_coloring(styled, cmap)
    save_styled_to_latex(styled, savepath / f"ablation-query_size_ranking_{score}.tex")
