from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nnactive.utils.io import save_df_to_txt

qm_to_color = {
    "mutual information": "#bcbd22",  # Yellow-green
    "power bald": "#ff7f0e",  # Orange
    "power pe": "#2ca02c",  # Green
    "pred entropy": "#1f77b4",  # Blue
    "softrank bald": "#7f7f7f",  # Gray
    "random": "#9467bd",  # Purple
    "random-label": "#e377c2",  # Light Red
    "random-label2": "#d62728",  # Red
}

base_path = Path("/home/j211b/experiments/nnactive/analysis/visualization-main")
out_path = Path("/home/j211b/experiments/nnactive/analysis/output")

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
    / "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-60__qs-60",
    base_path
    / "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-90__qs-90",
    base_path
    / "Dataset004_Hippocampus/patch-20_20_20__sb-random-label2-all-classes__sbs-20__qs-20",
    base_path
    / "Dataset004_Hippocampus/patch-20_20_20__sb-random-label2-all-classes__sbs-40__qs-40",
    base_path
    / "Dataset004_Hippocampus/patch-20_20_20__sb-random-label2-all-classes__sbs-60__qs-60",
    base_path
    / "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-40__qs-40",
    base_path
    / "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-200__qs-200",
    base_path
    / "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-500__qs-500",
]
paths = [Path(p) for p in paths]


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
    data_dict["Dataset"] = (
        path.parent.name.replace("Dataset004_Hippocampus", "Hippocampus")
        .replace("Dataset216_AMOS2022_task1", "AMOS")
        .replace("Dataset027_ACDC", "ACDC")
        .replace("Dataset135_KiTS2021", "KiTS")
    )
    data_dict["Setting"] = "QS " + path.name.split("qs-")[1].split("__")[0]
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

whole_data = pd.concat(
    whole_data,
    axis=1,
    keys=whole_data.keys(),
    names=["Dataset"],
)  # .sort_index(axis=1, level=0)

# Remove power bald ablations and kmeans bald
whole_data = whole_data.drop(["power bald b10", "power bald b5", "kmeans bald"])


# Sort by first and second levels, using QS numeric values for the second level
def sort_key(col):
    # Extract the numeric part of the second-level column (e.g., 'QS 20' -> 20)
    first_level, second_level = col
    second_level_numeric = int(second_level.split(" ")[-1])
    return (first_level, second_level_numeric)


for metric in ["AUBC", "beta", "Final"]:
    # Compute method rankings
    ranks = (
        whole_data.loc[:, (slice(None), slice(None), metric)]
        .droplevel(level=2, axis=1)
        .rank(ascending=False, method="min")
    )

    # whole_data = whole_data[sorted(whole_data.columns, key=sort_key)]
    ranks = ranks[sorted(ranks.columns, key=sort_key)]

    # Save ranking table to txt file
    save_df_to_txt(ranks, out_path / f"method_ranking_{metric}.txt")

    # Create ranking line plot
    plt.figure(figsize=(12, 6))
    ax = plt.gcf().gca()

    # for i in range(ranks.shape[0]):
    for i, method_name in enumerate(qm_to_color):
        ax.plot(
            ranks.iloc[i, :].values,
            marker="o",
            label=method_name,
            color=qm_to_color[method_name],
            ls="--" if "random" in method_name else "-",
        )

    ax.vlines(
        [2.5, 5.5, 8.5],
        1,
        ranks.shape[0],
        colors="k",
        linestyles="--",
        lw=0.5,
        zorder=-1,
    )

    plt.xticks(
        ticks=np.arange(ranks.shape[-1]),
        labels=[
            (f"{c[1]}\n\n{c[0]}" if i % 3 == 1 else c[1])
            for i, c in enumerate(ranks.columns)
        ],
    )
    plt.yticks(ticks=np.arange(ranks.shape[0]) + 1)
    plt.ylabel(f"Method Rank ({metric})")
    plt.legend(loc=(1.02, 0.3))
    plt.tight_layout()
    plt.savefig(out_path / f"method_ranking_{metric}.png")

if __name__ == "__main__":
    pass
