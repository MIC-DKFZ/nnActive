from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from setup import BASEPATH, RENAMING_DICT, SAVEPATH, SAVETYPE

from nnactive.analyze.aggregate_results import pretty_auc
from nnactive.analyze.analysis import SettingAnalysis
from nnactive.utils.io import save_df_to_txt

# Creation of these plots is so brutally hacky...
# I'm sorry for the mess, but it works and I don't want to touch it anymore
# This is due to mutiple issues in the code it is built upon

savepath = SAVEPATH / "tex" / "ablation_powerbald"
savepath.mkdir(parents=True, exist_ok=True)
figpath = SAVEPATH / "figures"
figpath.mkdir(parents=True, exist_ok=True)
print("Savepath:", savepath)

basepath = BASEPATH.parent / (BASEPATH.name + "_pbald_ablation")
COLLEVELNAMES = ["Dataset", "Label Regime", "Metric"]
SETTINGS = {
    "ACDC": {
        "Low": [
            basepath
            / "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-30__qs-30_revision",
            BASEPATH
            / "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-30__qs-30",
        ],
        "Medium": [
            basepath
            / "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-60__qs-60_revision",
            BASEPATH
            / "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-60__qs-60",
        ],
        "High": [
            basepath
            / "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-90__qs-90_revision",
            BASEPATH
            / "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-90__qs-90",
        ],
    },
    "AMOS": {
        "Low": [
            basepath
            / "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-40__qs-40",
            basepath
            / "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-40__qs-40_v0",
            BASEPATH
            / "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-40__qs-40",
        ],
        "Medium": [
            basepath
            / "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-200__qs-200",
            basepath
            / "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-200__qs-200_v0",
            BASEPATH
            / "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-200__qs-200",
        ],
        "High": [
            basepath
            / "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-500__qs-500",
            basepath
            / "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-500__qs-500_v0",
            BASEPATH
            / "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-500__qs-500",
        ],
    },
    "KiTS": {
        "Low": [
            basepath
            / "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-40__qs-40",
            basepath
            / "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-40__qs-40_v0",
            BASEPATH
            / "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-40__qs-40",
        ],
        "Medium": [
            basepath
            / "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-200__qs-200",
            basepath
            / "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-200__qs-200_v0",
            BASEPATH
            / "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-200__qs-200",
        ],
        "High": [
            basepath
            / "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-500__qs-500",
            basepath
            / "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-500__qs-500_v0",
            BASEPATH
            / "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-500__qs-500",
        ],
    },
}


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


CUSTOM_ORDER = [
    "power bald",
    "power bald b5",
    "power bald b10",
    "power bald b20",
    "power bald b40",
    "mutual information",
]

RENAMING_DICT = {
    "power bald": "PowerBALD (b=1)",
    "power bald b5": "PowerBALD (b=5)",
    "power bald b10": "PowerBALD (b=10)",
    "power bald b20": "PowerBALD (b=20)",
    "power bald b40": "PowerBALD (b=40)",
    "mutual information": "PowerBALD (b=$\\infty$)",
}

entire_data = []


def _load_and_format_auc_df(path: Path) -> pd.DataFrame:
    df_auc = (
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
                "('Mean Dice Final', 'mean')": "Final Dice",
                "('Mean Dice Final', 'std')": "Final Dice std",
            }
        )
        .apply(lambda x: np.round(x * 100, 2))
    )

    return df_auc


def _load_and_format_beta_df(path: Path) -> pd.DataFrame:
    return (
        pd.read_json(path / "beta.json")
        .set_index("Query Method")
        .apply(lambda x: np.round(x, 2))
    ).rename(columns={"beta": "FG-Eff", "beta_std": "FG-Eff std"})


for dataset_name in SETTINGS:
    data_dicts = []
    for col_name in SETTINGS[dataset_name]:
        data_dict = {}
        df_auc = []
        df_beta = []
        for path in SETTINGS[dataset_name][col_name]:
            if not path.is_dir():
                print(f"Skipping {path}")
                continue
            df_auc.append(_load_and_format_auc_df(path))
            df_beta.append(_load_and_format_beta_df(path))
        if len(df_auc) == 0:
            print(f"Skipping {dataset_name} {col_name}")
            continue
        df_auc = pd.concat(df_auc, axis=0)
        df_beta = pd.concat(df_beta, axis=0)
        data_dict["df_auc"] = df_auc
        data_dict["df_beta"] = df_beta
        data_dict["Setting"] = col_name
        data_dict["Dataset"] = dataset_name.split("_")[0]
        data_dict["df"] = pd.concat(
            [data_dict["df_auc"], data_dict["df_beta"]], axis=1
        )[
            [
                "AUBC",
                "AUBC std",
                "Final Dice",
                "Final Dice std",
                "FG-Eff",
                "FG-Eff std",
            ]
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
            names=COLLEVELNAMES[1:],
        )
        d_folder = dataset.replace(" ", "_")
    if len(whole_data) == 0:
        print(f"Skipping {dataset_name}")
        continue
    whole_data = pd.concat(
        whole_data, axis=1, keys=whole_data.keys(), names=COLLEVELNAMES
    )
    whole_data = whole_data.reindex(CUSTOM_ORDER, level=0)
    whole_data = whole_data.rename(RENAMING_DICT, axis=0)
    whole_data.index.name = "Query Method"

    cmap = "Oranges"
    higher_is_better = ["AUBC", "Final Dice", "FG-Eff"]
    subset = [col for col in whole_data.columns if col[-1] in higher_is_better]

    print_data = whole_data.copy(deep=True)
    for n in print_data.index:
        print_data.rename(index={n: n.replace("%", "\%")}, inplace=True)
    gmap = _compute_gmap(print_data[subset], invert=True)
    for col in subset:
        std_col = tuple(list(col[:-1]) + [col[-1] + " std"])
        print_data[col] = (
            print_data[col].apply(lambda x: f"{x:.2f}")
            + " ± "
            + print_data[std_col].apply(lambda x: f"{x:.2f}")
        )
        del print_data[std_col]

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
    tex_fn = savepath / f"pbald_{dataset_name}.tex"
    styled.to_latex(
        tex_fn,
        convert_css=True,
        hrules=True,
        multicol_align="c|",
        column_format="l" + columns + "|",
    )

    entire_data.append(whole_data)


from itertools import product

import matplotlib.pyplot as plt
import seaborn as sns

entire_data = pd.concat(entire_data, axis=1)

df = entire_data.unstack().unstack(level=2)


datasets = df.index.levels[0]
metrics = ["AUBC", "Final Dice", "FG-Eff"]
fig, axs = plt.subplots(
    nrows=len(datasets), ncols=len(metrics), squeeze=False, sharex=True
)
for (j, dataset), (i, metric) in product(enumerate(datasets), enumerate(metrics)):
    sns.lineplot(
        ax=axs[i, j],
        data=df.loc[dataset],
        x="Query Method",
        y=metric,
        hue="Label Regime",
        marker="o",
    )

    axs[i, j].legend().remove()
    axs[i, j].set_ylabel(None)
    if i == 0:
        axs[i, j].set_title(dataset)
        ax = axs[len(datasets) - 1, j]
        ax.set_xlabel("$\\beta$-Parameter")
        ax.set_xticklabels(["1", "5", "10", "20", "40", "$\\infty$"])

    axs[i, 0].set_ylabel(metric)

fig.tight_layout()
axs[axs.shape[-1] - 1, 0].legend(loc=(0.5, -0.7), handlelength=4, ncols=4)
plt.savefig(figpath / f"ablation-powerbald.{SAVETYPE}", bbox_inches="tight")

for dataset in datasets:
    fig, axs = plt.subplots(
        nrows=1, ncols=len(metrics), squeeze=False, sharex=False, figsize=(6, 2)
    )
    i = 0
    for j, metric in enumerate(metrics):
        ax: Axes = axs[i, j]
        sns.lineplot(
            ax=ax,
            data=df.loc[dataset],
            x="Query Method",
            y=metric,
            hue="Label Regime",
            marker="o",
        )

        ax.legend().remove()
        # axs[i, j].set_ylabel(None)
        if i == 0:
            # axs[i, j].set_title(dataset)
            # ax = axs[0, j]
            ax.set_xlabel("$\\beta$-Parameter")
            ax.set_xticklabels(["1", "5", "10", "20", "40", "$\\infty$"])

        if metric == "FG-Eff":
            ax.text(
                0.95,
                0.95,
                r"Noise strength",
                horizontalalignment="right",
                verticalalignment="top",
                transform=ax.transAxes,
                fontsize=8,
            )
            ax.text(
                0.9,
                0.88,
                r"$\longleftarrow$",
                horizontalalignment="right",
                verticalalignment="top",
                transform=ax.transAxes,
                fontsize=18,
            )

        # axs[i, 0].set_ylabel(metric)

    fig.tight_layout()
    axs[0, 0].legend(loc=(0.5, -0.6), handlelength=4, ncols=4)
    plt.savefig(
        figpath / f"ablation-powerbald_{dataset}.{SAVETYPE}", bbox_inches="tight"
    )
