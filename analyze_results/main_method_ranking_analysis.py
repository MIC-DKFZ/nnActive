from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from setup import BASEPATH, QM_TO_COLOR, RENAMING_DICT

from nnactive.utils.io import save_df_to_txt

out_path = Path("/home/j211b/experiments/nnactive/analysis/output")
out_path = Path(
    "/home/c817h/Documents/projects/nnactive_project/nnactive/results/horeka_rsync_final"
)

paths = [
    BASEPATH
    / "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-40__qs-40",
    BASEPATH
    / "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-200__qs-200",
    BASEPATH
    / "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-500__qs-500",
    BASEPATH
    / "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-30__qs-30",
    BASEPATH
    / "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-60__qs-60",
    BASEPATH
    / "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-90__qs-90",
    BASEPATH
    / "Dataset004_Hippocampus/patch-20_20_20__sb-random-label2-all-classes__sbs-20__qs-20__5loops",
    BASEPATH
    / "Dataset004_Hippocampus/patch-20_20_20__sb-random-label2-all-classes__sbs-40__qs-40",
    BASEPATH
    / "Dataset004_Hippocampus/patch-20_20_20__sb-random-label2-all-classes__sbs-60__qs-60",
    BASEPATH
    / "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-40__qs-40",
    BASEPATH
    / "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-200__qs-200",
    BASEPATH
    / "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-500__qs-500",
]
paths = [Path(p) for p in paths]


data_dicts = []
for path in paths:
    data_dict = {}
    data_dict["df_auc"] = pd.read_json(path / "auc.json")[
        [
            "('Mean Dice AUBC', 'mean')",
            "('Mean Dice AUBC', 'std')",
            "('Mean Dice Final', 'mean')",
            "('Mean Dice Final', 'std')",
        ]
    ].rename(
        columns={
            "('Mean Dice AUBC', 'mean')": "AUBC",
            "('Mean Dice AUBC', 'std')": "AUBC std",
            "('Mean Dice Final', 'mean')": "Final Dice",
            "('Mean Dice Final', 'std')": "Final Dice std",
        }
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
        ["AUBC", "AUBC std", "beta", "beta std", "Final Dice", "Final Dice std"]
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
whole_data = whole_data.drop(
    [
        "power bald b10",
        "power bald b5",
        "power bald b20",
        "power bald b40",
        "kmeans bald",
    ],
    errors="ignore",
)
whole_data = whole_data.rename(mapper=RENAMING_DICT)


# Sort by first and second levels, using QS numeric values for the second level
def sort_key(col):
    # Extract the numeric part of the second-level column (e.g., 'QS 20' -> 20)
    first_level, second_level = col
    second_level_numeric = int(second_level.split(" ")[-1])
    return (first_level, second_level_numeric)


for metric in ["AUBC", "beta", "Final Dice"]:
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

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        # sharey=True,
        figsize=(14, 6),
        width_ratios=(12, 1),
        gridspec_kw={
            "wspace": 0.05,  # 0.03
        },
    )
    # plt.figure(figsize=(12, 6))
    # ax = plt.gcf().gca()

    # Create ranking line plot
    for method_name in QM_TO_COLOR:
        ax1.plot(
            ranks.loc[method_name, :].values,
            marker="o",
            label=method_name,
            color=QM_TO_COLOR[method_name],
            ls="--" if "random" in method_name else "-",
            lw=2,
            markerfacecolor=(
                "white" if "random" in method_name else QM_TO_COLOR[method_name]
            ),
        )

    ax1.add_patch(
        patches.Rectangle((-0.5, 0), 3, 8.5, linewidth=1, facecolor="k", alpha=0.1)
    )
    ax1.add_patch(
        patches.Rectangle((2.5, 0), 3, 8.5, linewidth=1, facecolor="k", alpha=0.03)
    )
    ax1.add_patch(
        patches.Rectangle((5.5, 0), 3, 8.5, linewidth=1, facecolor="k", alpha=0.1)
    )
    ax1.add_patch(
        patches.Rectangle((8.5, 0), 3, 8.5, linewidth=1, facecolor="k", alpha=0.03)
    )

    # ax.vlines(
    #     [2.5, 5.5, 8.5],
    #     0.5,
    #     ranks.shape[0] + 0.5,
    #     colors="k",
    #     linestyles="-",
    #     lw=2,
    #     alpha=0.2,
    #     zorder=-1,
    # )
    ax1.set_xlim(-0.5, 11.5)
    ax1.set_ylim(0.5, ranks.shape[0] + 0.5)
    ax1.grid(axis="y")

    ax1.set_xticks(
        ticks=np.arange(ranks.shape[-1]),
        labels=[
            (f"{c[1]}\n\n{c[0]}" if i % 3 == 1 else c[1])
            for i, c in enumerate(ranks.columns)
        ],
    )
    ax1.set_yticks(ticks=np.arange(ranks.shape[0]) + 1)
    ax1.set_ylabel(f"Method Rank ({metric})")
    ax1.legend(loc=(0.1, -0.25), handlelength=4, ncols=4)

    # Create mean rank plot
    avg_ranks = ranks.mean(axis=1)
    std_ranks = ranks.std(axis=1)
    ranks_sorted = avg_ranks.rank(ascending=True, method="first")
    # print(avg_ranks)
    # print(ranks_sorted)
    for i, method_name in enumerate(QM_TO_COLOR):
        # ax2.add_patch(
        #     patches.Rectangle(
        #         (-0.4, avg_ranks[method_name]),
        #         0.8,
        #         0.1,
        #         # linewidth=1,
        #         facecolor=qm_to_color[method_name],
        #         alpha=0.3,
        #     )
        # )
        ax2.errorbar(
            -0.5 + ranks_sorted[method_name] / (len(ranks_sorted) + 1),
            [avg_ranks[method_name]],
            yerr=[std_ranks[method_name]],
            fmt=".",
            markersize=8,
            color=QM_TO_COLOR[method_name],
            markerfacecolor=(
                "white" if "random" in method_name else QM_TO_COLOR[method_name]
            ),
        )

    ax2.set_xlim(-0.5, 0.5)
    ax2.set_ylim(0.5, ranks.shape[0] + 0.5)
    ax2.set_xticks(ticks=[0], labels=["Mean Rank"])
    ax2.set_yticks(ticks=np.arange(ranks.shape[0]) + 1)
    ax2.tick_params("x", length=0, pad=7)
    ax2.grid(axis="y")

    # Save figure
    # plt.tight_layout()
    plt.savefig(out_path / f"method_ranking_{metric}.png", bbox_inches="tight")

if __name__ == "__main__":
    pass
