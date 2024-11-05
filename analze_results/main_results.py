from pathlib import Path

import numpy as np
import pandas as pd

from nnactive.analyze.aggregate_results import pretty_auc
from nnactive.analyze.analysis import SettingAnalysis
from nnactive.utils.io import save_df_to_txt

base_path = Path(
    "/home/c817h/Documents/projects/nnactive_project/nnactive/results/horeka-main_symlink/"
)
paths = [
    base_path
    / "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-40__qs-40",
    base_path
    / "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-200__qs-200",
    base_path
    / "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-500__qs-500",
    base_path
    / "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-30__qs-30",
    base_path
    / "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-90__qs-90",
    # base_path / "Dataset004_Hippocampus/patch-20_20_20__sb-random-label2-all-classes__sbs-20__qs-20",
    # base_path / "Dataset004_Hippocampus/patch-20_20_20__sb-random-label2-all-classes__sbs-40__qs-40",
    base_path
    / "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-40__qs-40",
    base_path
    / "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-200__qs-200",
    base_path
    / "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-500__qs-500",
]
paths = [Path(p) for p in paths]


def _compute_gmap(data: pd.DataFrame, invert: bool):
    import matplotlib

    # NOTE: Manually compute gradient map because Normalize returns 0 if vmax - vmin == 0, but we
    # NOTE:   want it to be 1 in that case

    gmap = data.to_numpy(float)
    gmap_min = np.nanmin(gmap, axis=0)
    gmap_max = np.nanmax(gmap, axis=0)

    for col in range(gmap.shape[1]):
        vmin = gmap_min[col] - (0.0001 if invert else 0)
        vmax = gmap_max[col] + (0 if invert else 0.0001)
        gmap_use = gmap
        if invert:
            vmin_0 = vmin
            vmin = -vmax
            vmax = -vmin_0
            gmap_use = -gmap

        gmap[:, col] = matplotlib.colors.Normalize(vmin, vmax)(gmap_use[:, col])

    return gmap


data_dicts = []
for path in paths:
    data_dict = {}
    data_dict["df_auc"] = (
        pd.read_json(path / "auc.json")[
            [
                "('Mean Dice AUBC', 'mean')",
                "('Mean Dice AUBC', 'std')",
                "('Mean Dice Final', 'mean')",
                "('Mean Dice Final', 'std')",
            ]
        ]
        .rename(
            columns={
                "('Mean Dice AUBC', 'mean')": "AUBC",
                "('Mean Dice AUBC', 'std')": "AUBC std",
                "('Mean Dice Final', 'mean')": "Final",
                "('Mean Dice Final', 'std')": "Final std",
            }
        )
        .apply(lambda x: np.round(x * 100, 2))
    )
    data_dict["Dataset"] = path.parent.name.replace("_", " ")
    data_dict["Setting"] = "Query Size " + path.name.split("qs-")[1].split("__")[0]
    data_dict["df_beta"] = (
        pd.read_json(path / "beta.json")
        .set_index("Query Method")
        .apply(lambda x: np.round(x, 2))
    ).rename(columns={"beta_std": "beta std"})
    data_dict["df"] = pd.concat([data_dict["df_auc"], data_dict["df_beta"]], axis=1)[
        ["AUBC", "AUBC std", "beta", "beta std", "Final", "Final std"]
    ]
    data_dict["df"].reset_index(inplace=True)
    print(data_dict["df"].columns)
    data_dict["df"]["index"] = data_dict["df"]["index"].map(
        lambda x: x.replace("_", " ")
    )
    data_dict["df"] = data_dict["df"].set_index("index")
    data_dicts.append(data_dict)

order = ["Dataset", "Setting", "df"]

datasets = set([data["Dataset"] for data in data_dicts])
path = Path(
    "/home/c817h/Documents/projects/nnactive_project/nnactive/results/horeka-main/"
)

whole_data = {}
for dataset in datasets:
    whole_data[dataset] = {}
    for data in data_dicts:
        if data["Dataset"] == dataset:
            whole_data[dataset][data["Setting"]] = data["df"]
    whole_data[dataset] = pd.concat(
        whole_data[dataset],
        axis=1,
        keys=whole_data[dataset].keys(),
        names=["Setting"],
    )
    d_folder = dataset.replace(" ", "_")
    save_df_to_txt(whole_data[dataset], path / d_folder / "main_table.txt")
whole_data = pd.concat(whole_data, axis=1, keys=whole_data.keys(), names=["Dataset"])
save_df_to_txt(whole_data, path / "main_table.txt")

with open(path / "main_table.md", "w") as f:
    f.write(whole_data.to_markdown())

cmap = "Oranges"
higher_is_better = ["AUBC", "Final", "beta"]
subset = [col for col in whole_data.columns if col[-1] in higher_is_better]
columns = [
    "c" * len(whole_data["Dataset135 KiTS2021"].columns)
    for _ in range(len(whole_data.columns.levels[0]))
]

print_data = whole_data.copy(deep=True)
gmap = _compute_gmap(print_data[subset], invert=True)
for col in subset:
    std_col = tuple(list(col[:-1]) + [col[-1] + " std"])
    print_data[col] = (
        print_data[col].apply(lambda x: f"{x:.2f}")
        + " ± "
        + print_data[std_col].apply(lambda x: f"{x:.2f}")
    )
    del print_data[std_col]

columns = [
    "c" * len(whole_data["Dataset135 KiTS2021"].columns)
    for _ in range(len(whole_data.columns.levels[0]))
]
columns = ""
levels = [whole_data.columns.levels]
cur_col = None
split_level = 2
for col in print_data.columns:
    if cur_col == col[:split_level]:
        columns += "c"

    else:
        cur_col = col[:split_level]
        columns += "|c"


styled = print_data.style.background_gradient(
    "Oranges", axis=None, subset=subset, gmap=gmap
)
styled.to_latex(
    path / "out2.tex",
    convert_css=True,
    hrules=True,
    multicol_align="c|",
    column_format="l" + columns + "|",
)
import IPython

IPython.embed()
