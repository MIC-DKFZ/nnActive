from pathlib import Path

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
import seaborn as sns
from evaluator import (
    get_settings_for_combination,
    load_settings,
    rename_settings_in_analysis,
)
from matplotlib.axes import Axes
from scipy.stats import friedmanchisquare
from setup import (
    BASEPATH,
    CUSTOM_ORDER,
    FINAL_COLUMNS,
    MAIN_ORDER,
    QM_TO_COLOR,
    ROLL_OUT_ORDER,
    SAVEPATH,
    SAVETYPE,
    compute_column_normalized_gmap,
    get_ranking_cmap,
    load_setting_data_to_df,
)

USETEX = False
if USETEX:
    matplotlib.rcParams["text.usetex"] = True
    matplotlib.rcParams["font.family"] = "Computer Modern"

savepath = SAVEPATH / "new_tables"
savepath.mkdir(exist_ok=True, parents=True)

NAME = "main_method_means"
COLLEVELANMES = ["Dataset", "Label Regime", "Metric"]
SCORES = ["AUBC", "Final Dice"]


def generate_colored_latex_report_table(
    CMAP,
    whole_data: pd.DataFrame,
    savepath: str | Path,
    subset: list[str],
    colorization: str | None = "linear",
    error_val="StE",
):

    print_data = whole_data.copy(deep=True)

    for n in print_data.index:
        print_data.rename(index={n: n.replace("%", "\%")}, inplace=True)
    if colorization == "linear":
        gmap = compute_column_normalized_gmap(print_data[subset], invert=True)
    elif colorization == "rank":
        gmap = print_data[subset].rank(method="min", ascending=False)
        gmap = gmap / print_data.shape[0]
    elif colorization is None:
        gmap = None
    else:
        raise ValueError(f"Colorization {colorization} not supported")
    for col in subset:
        print_data[col] = print_data[col].apply(lambda x: f"{x:.2f}")
        std_col = col + " " + error_val
        if std_col in print_data.columns:
            print_data[col] = (
                print_data[col]
                + " ± "
                + print_data[std_col].apply(lambda x: f"{x:.2f}")
            )
            del print_data[std_col]

        for del_col in list(print_data.columns):
            if del_col.startswith(col + " "):
                del print_data[del_col]

    columns = ""
    cur_col = None
    split_level = 2
    for col in print_data.columns:
        if cur_col == col[:split_level]:
            columns += "c"

        else:
            cur_col = col[:split_level]
            columns += "|c"

    if gmap is not None:
        styled = print_data.style.background_gradient(
            CMAP, axis=None, subset=subset, gmap=gmap
        )
        styled.to_latex(
            savepath,
            convert_css=True,
            hrules=True,
            multicol_align="c|",
            column_format="l" + columns + "|",
        )
    else:
        try:
            styled = print_data.style
            styled.to_latex(
                savepath,
                hrules=True,
                convert_css=True,
                multicol_align="c|",
                column_format="l" + columns + "|",
            )
        except Exception as e:
            print("Error in generating table:", e)
            print("Dataframe shape:", print_data.shape)
            print("Dataframe columns:", print_data.columns)
            import IPython

            IPython.embed()
            raise e


COMPARATIVE = False

ADD_LEGEND = False

PLOT_CONFIGS = [
    {
        "settings": ["Main", "Patchx1/2"],
        "comparative": False,
        "rename": None,
        "copy": None,
        "order": MAIN_ORDER,
        "name": NAME + "_joined",
    }
]


if __name__ == "__main__":
    for config in PLOT_CONFIGS:
        setting = config["settings"]
        rename_setting = config["rename"]
        custom_order = config["order"]
        plot_prefix = config["name"]
        print(setting)
        print_setting = "_".join(setting).replace(" ", "").replace("/", "-")
        setting_paths = get_settings_for_combination(setting)
        setting_analyses = load_settings(setting_paths, comparative=COMPARATIVE)
        if rename_setting is not None:
            rename_settings_in_analysis(setting_analyses, rename_setting)
            rename_settings_in_analysis(setting_paths, rename_setting)

        data_dict = load_setting_data_to_df(
            custom_order, FINAL_COLUMNS, setting_paths, setting_analyses
        )
        plot_metrics = SCORES

        name = "--".join(
            [s.lower().replace(" ", "").replace("/", "-") for s in setting]
        )
        out_data = {}
        whole_data: dict[str, dict[str, pd.DataFrame]] = {}
        for dataset in data_dict:
            whole_data[dataset] = {}
            for budget in data_dict[dataset]:
                whole_data[dataset][budget] = {
                    s: data_dict[dataset][budget][s] for s in setting
                }

        for s in setting:
            out_data[s] = {}
            for dataset in whole_data:
                out_data[s][dataset] = pd.concat(
                    [whole_data[dataset][b][s] for b in whole_data[dataset]],
                    axis=1,
                    keys=whole_data[dataset].keys(),
                    names=COLLEVELANMES[1:],
                )
            out_data[s] = pd.concat(
                out_data[s].values(),
                axis=1,
                keys=out_data[s].keys(),
                names=COLLEVELANMES,
            )

        metric_table = {}
        for metric in plot_metrics:
            metric_means = []
            metric_stds = []
            for s in out_data:
                df = out_data[s]
                metric_mean = df.loc[:, df.columns.get_level_values("Metric") == metric]
                metric_std = (
                    df.loc[:, df.columns.get_level_values("Metric") == f"{metric} std"]
                    ** 2
                )
                metric_means.append(metric_mean)
                metric_stds.append(metric_std)
            metric_means = (sum(metric_means) / len(metric_means)).mean(axis=1)
            # metric_stds = (
            #     sum(metric_stds).sum(axis=1)
            #     / ((len(metric_stds) * len(metric_stds[0].columns)) ** 2 * 4)
            # ) ** 0.5
            metric_stds = (
                sum(metric_stds).sum(axis=1)
                / ((len(metric_stds) * len(metric_stds[0].columns)) ** 2 * 4)
            ) ** 0.5

            print(metric)
            metric_data = pd.DataFrame(
                {"Mean": metric_means, "StE": metric_stds, "CI": 1.96 * metric_stds}
            )
            print(metric_data)
            metric_table[metric] = metric_data
        metric_table_df = pd.concat(
            metric_table.values(), axis=1, keys=metric_table.keys()
        ).round(2)
        print(metric_table_df)
        metric_table_df.to_csv(savepath / f"{plot_prefix}_mean_performance_table.csv")

        # two column level to single column df
        metric_table_df.columns = [
            f"{metric}" if stat == "Mean" else f"{metric} {stat}"
            for metric, stat in metric_table_df.columns
        ]

        cols = ["AUBC", "Final Dice"]
        generate_colored_latex_report_table(
            "Greens_r",
            metric_table_df,
            savepath / f"{plot_prefix}_mean_performance_table.tex",
            cols,
            error_val="StE",
        )
        generate_colored_latex_report_table(
            "Greens_r",
            metric_table_df,
            savepath / f"{plot_prefix}_mean_performance_table_CI.tex",
            cols,
            error_val="CI",
        )
