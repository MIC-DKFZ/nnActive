from pathlib import Path

import numpy as np
import pandas as pd
from evaluator import (
    get_settings_for_combination,
    load_settings,
    rename_settings_in_analysis,
)
from setup import (
    BASEPATH,
    CUSTOM_ORDER,
    FINAL_COLUMNS,
    MAIN_ORDER,
    SAVEPATH,
    compute_column_normalized_gmap,
    load_setting_data_to_df,
)

from nnactive.analyze.aggregate_results import pretty_auc
from nnactive.analyze.analysis import SettingAnalysis
from nnactive.utils.io import save_df_to_txt

COMPARATIVE = False
CMAP = "Oranges"
COLLEVELNAMES = ["Dataset", "Label Regime", "Setting", "Metric"]

savepath = SAVEPATH

TABLE_CONFIGS = [
    {
        "settings": ["Main"],
        "comparative": False,
        "rename": None,
        "copy": None,
        "order": CUSTOM_ORDER,
    },
    {
        "settings": ["500 Epochs"],
        "comparative": False,
        "rename": None,
        "copy": None,
        "order": CUSTOM_ORDER,
    },
    # {
    #     "settings": ["Precomputed"],
    #     "comparative": False,
    #     "rename": None,
    #     "copy": None,
    # },
    # {
    #     "settings": ["QSx2"],
    #     "comparative": False,
    #     "rename": None,
    #     "copy": None,
    # },
    # {
    #     "settings": ["QSx1/2"],
    #     "comparative": False,
    #     "rename": None,
    #     "copy": None,
    # },
    {
        "settings": ["Patchx1/2"],
        "comparative": False,
        "rename": None,
        "copy": None,
        "order": MAIN_ORDER,
    },
    # {
    #     "settings": ["Main", "Precomputed", "500 Epochs"],
    #     "comparative": True,
    #     "rename": {"Main": "200 Epochs"},
    #     "copy": {
    #         "Source": "Precomputed",
    #         "Target": "500 Epochs",
    #         "Copy": ["Random", "Random 33% FG", "Random 66% FG"],
    #         "Transfer_fct": lambda x: (slice(None), x),
    #     },
    # },
    # {
    #     "settings": ["QSx1/2", "Main", "QSx2"],
    #     "comparative": True,
    #     "rename": {"Main": "QSx1"},
    #     "copy": None,
    # },
    # {
    #     "settings": ["Patchx1/2", "Main"],
    #     "comparative": True,
    #     "rename": {"Main": "Patchx1"},
    #     "copy": None,
    # },
]

# ensure that comparative is set to True for all settings with multiple settings
for config in TABLE_CONFIGS:
    if len(config["settings"]) > 1:
        config["comparative"] = True
    if config.get("order") is None:
        config["order"] = CUSTOM_ORDER
    if config.get("merge_datasets") is None:
        config["merge_datasets"] = False


def generate_colored_latex_report_table(
    CMAP,
    FINAL_COLUMNS: list[dict],
    whole_data: pd.DataFrame,
    savepath: str | Path,
    colorization: str | None = "linear",
    copy_setting: dict[str, str | list[str]] | None = None,
):
    is_better = [c["PrintCol"] for c in FINAL_COLUMNS if c["better"] == "higher"]
    subset = [col for col in whole_data.columns if col[-1] in is_better]

    print_data = whole_data.copy(deep=True)
    if copy_setting is not None:
        # using .values is necessary here to avoid NaNs due to multicolumn indexing
        print_data.loc[
            copy_setting["Copy"], copy_setting["Transfer_fct"](copy_setting["Target"])
        ] = print_data.loc[
            copy_setting["Copy"], copy_setting["Transfer_fct"](copy_setting["Source"])
        ].values

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
        std_col = tuple(list(col[:-1]) + [col[-1] + " std"])
        if std_col in print_data.columns:
            print_data[col] = (
                print_data[col]
                + " ± "
                + print_data[std_col].apply(lambda x: f"{x:.2f}")
            )
            del print_data[std_col]

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


if __name__ == "__main__":
    for config in TABLE_CONFIGS:
        setting = config["settings"]
        rename_setting = config["rename"]
        cp_setting = config["copy"]
        comparative = config["comparative"]
        custom_order = config["order"]
        print(setting)
        print_setting = "_".join(setting).replace(" ", "").replace("/", "-")
        setting_paths = get_settings_for_combination(setting)
        setting_analyses = load_settings(setting_paths, comparative=comparative)
        if rename_setting is not None:
            rename_settings_in_analysis(setting_analyses, rename_setting)
            rename_settings_in_analysis(setting_paths, rename_setting)

        colorization = "linear" if len(setting) == 1 else "rank"
        data_dict = load_setting_data_to_df(
            config["order"],
            FINAL_COLUMNS,
            setting_paths,
            setting_analyses,
            comparative=comparative,
        )

        # 1 table per dataset and budget with multiple settings
        print_tables: dict[str, pd.DataFrame] = {}
        if isinstance(setting, (list, tuple)) and len(setting) == 1:
            setting = setting[0]
        if isinstance(setting, str):
            collevelnames = COLLEVELNAMES.copy()
            collevelnames.remove("Setting")
            for dataset in data_dict:
                whole_data = []
                for budget in data_dict[dataset]:
                    val = data_dict[dataset][budget].get(setting)
                    if val is None:
                        print("No data for", dataset, budget, setting)
                        continue
                    whole_data.append(data_dict[dataset][budget][setting])
                if len(whole_data) == 0:
                    print(f"No data for {dataset} in {setting}")
                    continue
                    # raise ValueError(f"No data for {dataset} in {setting}")
                whole_data = pd.concat(
                    whole_data,
                    axis=1,
                    keys=data_dict[dataset].keys(),
                    names=collevelnames[1:],
                )
                whole_data = pd.concat(
                    [whole_data],
                    axis=1,
                    keys=[dataset],
                    names=collevelnames,
                )
                print_tables[dataset] = whole_data
            print_tables[print_setting] = pd.concat(
                print_tables.values(),
                axis=1,
            )
            # print_tables[print_setting + "_nostd"] = print_tables[print_setting][
            #     [col for col in print_tables[print_setting] if "std" not in col[2]]
            # ]

        else:
            collevelnames = COLLEVELNAMES.copy()
            collevelnames.remove("Label Regime")
            for dataset in data_dict:
                for budget in data_dict[dataset]:
                    whole_data = []
                    for sett in data_dict[dataset][budget]:
                        whole_data.append(data_dict[dataset][budget][sett])
                    if len(whole_data) == 0:
                        print(f"No data for {dataset} in {budget}")
                        continue
                    whole_data = pd.concat(
                        whole_data,
                        axis=1,
                        keys=data_dict[dataset][budget].keys(),
                        names=collevelnames[1:],
                    )
                    whole_data = pd.concat(
                        [whole_data],
                        axis=1,
                        keys=[dataset],
                        names=collevelnames,
                    )
                    print_tables[f"{dataset}_{budget}"] = whole_data

        for print_table in print_tables:
            txt_folder = savepath / f"tables-{print_setting}".lower()
            tex_folder = savepath / "tex" / f"tables-{print_setting}".lower()
            txt_folder.mkdir(exist_ok=True, parents=True)
            tex_folder.mkdir(exist_ok=True, parents=True)
            whole_data = print_tables[print_table]
            fn = f"{print_setting}--{print_table}".lower()
            tex_fn = tex_folder / f"{fn}.tex"
            generate_colored_latex_report_table(
                CMAP,
                FINAL_COLUMNS,
                whole_data,
                tex_fn,
                colorization=colorization,
                copy_setting=cp_setting,
            )
            nocolor_tex_fn = tex_folder / f"{fn}-nocolor.tex"
            generate_colored_latex_report_table(
                CMAP,
                FINAL_COLUMNS,
                whole_data,
                nocolor_tex_fn,
                colorization=None,
                copy_setting=cp_setting,
            )
            save_df_to_txt(whole_data, txt_folder / f"{fn}.txt")

            # summer_tex_fn = tex_folder / f"{fn}-summer.tex"
            # generate_colored_latex_report_table(
            #     "summer",
            #     FINAL_COLUMNS,
            #     whole_data,
            #     summer_tex_fn,
            #     colorization=colorization,
            #     copy_setting=cp_setting,
            # )

            ylgn_tex_fn = tex_folder / f"{fn}-greens.tex"
            generate_colored_latex_report_table(
                "Greens_r",
                FINAL_COLUMNS,
                whole_data,
                ylgn_tex_fn,
                colorization=colorization,
                copy_setting=cp_setting,
            )
