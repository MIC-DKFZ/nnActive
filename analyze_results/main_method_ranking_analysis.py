from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
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
    QM_TO_COLOR,
    RENAMING_DICT,
    SAVEPATH,
    load_setting_data_to_df,
)

from nnactive.utils.io import save_df_to_txt

savepath = SAVEPATH / "figures"
savepath.mkdir(exist_ok=True, parents=True)

NAME = "main_method_ranking"
COLLEVELANMES = ["Dataset", "Budget", "Metric"]

USE_SETTINGS_LIST = [
    ["Main"],
]
SCORES = ["AUBC", "Final Dice", "beta"]

RENAME_SETTINGS_LIST = [
    None,
]

COMPARATIVE = False
FINAL_COLUMNS = [
    {"ReadCol": "('Mean Dice AUBC', 'mean')", "PrintCol": "AUBC", "better": "higher"},
    {"ReadCol": "('Mean Dice AUBC', 'std')", "PrintCol": "AUBC std", "better": None},
    {"ReadCol": "('Mean Dice Final', 'mean')", "PrintCol": "Final", "better": "higher"},
    {"ReadCol": "('Mean Dice Final', 'std')", "PrintCol": "Final std", "better": None},
    {"ReadCol": "beta", "PrintCol": "beta", "better": "higher"},
    {"ReadCol": "beta_std", "PrintCol": "beta std", "better": None},
]


# Sort by first and second levels, using QS numeric values for the second level
def sort_key(col):
    # Extract the numeric part of the second-level column (e.g., 'QS 20' -> 20)
    first_level, second_level = col
    second_level_numeric = int(second_level.split(" ")[-1])
    return (first_level, second_level_numeric)


# SETTINGS = {
#     "KiTS": [
#         BASEPATH
#         / "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-40__qs-40",
#         BASEPATH
#         / "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-200__qs-200",
#         BASEPATH
#         / "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-500__qs-500",
#     ],
#     "ACDC": [
#         BASEPATH
#         / "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-30__qs-30",
#         BASEPATH
#         / "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-60__qs-60",
#         BASEPATH
#         / "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-90__qs-90",
#     ],
#     "Hippocampus": [
#         BASEPATH
#         / "Dataset004_Hippocampus/patch-20_20_20__sb-random-label2-all-classes__sbs-20__qs-20__5loops",
#         BASEPATH
#         / "Dataset004_Hippocampus/patch-20_20_20__sb-random-label2-all-classes__sbs-40__qs-40",
#         BASEPATH
#         / "Dataset004_Hippocampus/patch-20_20_20__sb-random-label2-all-classes__sbs-60__qs-60",
#     ],
#     "AMOS": [
#         BASEPATH
#         / "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-40__qs-40",
#         BASEPATH
#         / "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-200__qs-200",
#         BASEPATH
#         / "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-500__qs-500",
#     ],
# }

# for name in SETTINGS:
#     SETTINGS[name] = [Path(p) for p in SETTINGS[name]]


# data_dicts = []
# for name in SETTINGS:
#     paths = SETTINGS[name]
#     for i, path in enumerate(paths):
#         data_dict = {}
#         data_dict["df_auc"] = pd.read_json(path / "auc.json")[
#             [
#                 "('Mean Dice AUBC', 'mean')",
#                 "('Mean Dice AUBC', 'std')",
#                 "('Mean Dice Final', 'mean')",
#                 "('Mean Dice Final', 'std')",
#             ]
#         ].rename(
#             columns={
#                 "('Mean Dice AUBC', 'mean')": "AUBC",
#                 "('Mean Dice AUBC', 'std')": "AUBC std",
#                 "('Mean Dice Final', 'mean')": "Final Dice",
#                 "('Mean Dice Final', 'std')": "Final Dice std",
#             }
#         )
#         data_dict["Dataset"] = name
#         # data_dict["Setting"] = "QS " + path.name.split("qs-")[1].split("__")[0]
#         data_dict["Setting"] = COLNAMES[i]

#         data_dict["df_beta"] = (
#             pd.read_json(path / "beta.json")
#             .set_index("Query Method")
#             .apply(lambda x: np.round(x, 2))
#         ).rename(columns={"beta_std": "beta std"})
#         data_dict["df"] = pd.concat(
#             [data_dict["df_auc"], data_dict["df_beta"]], axis=1
#         )[["AUBC", "AUBC std", "beta", "beta std", "Final Dice", "Final Dice std"]]
#         data_dict["df"].reset_index(inplace=True)
#         print(data_dict["df"].columns)
#         data_dict["df"]["index"] = data_dict["df"]["index"].map(
#             lambda x: x.replace("_", " ")
#         )
#         data_dict["df"] = data_dict["df"].set_index("index")
#         data_dicts.append(data_dict)


# datasets = set([data["Dataset"] for data in data_dicts])

# whole_data = {}
# for dataset in datasets:
#     whole_data[dataset] = {}
#     for data in data_dicts:
#         if data["Dataset"] == dataset:
#             whole_data[dataset][data["Setting"]] = data["df"]
#     whole_data[dataset] = pd.concat(
#         whole_data[dataset],
#         axis=1,
#         keys=whole_data[dataset].keys(),
#         names=["Setting"],
#     )

# whole_data = pd.concat(
#     whole_data,
#     axis=1,
#     keys=whole_data.keys(),
#     names=["Dataset"],
# )  # .sort_index(axis=1, level=0)

# # Remove power bald ablations and kmeans bald
# whole_data = whole_data.drop(
#     [
#         "power bald b10",
#         "power bald b5",
#         "power bald b20",
#         "power bald b40",
#         "kmeans bald",
#     ],
#     errors="ignore",
# )
# whole_data = whole_data.rename(mapper=RENAMING_DICT)


# # Sort by first and second levels, using QS numeric values for the second level
# def sort_key(col):
#     # Extract the numeric part of the second-level column (e.g., 'QS 20' -> 20)
#     first_level, second_level = col
#     second_level_numeric = int(second_level.split(" ")[-1])
#     return (first_level, second_level_numeric)


# for metric in ["AUBC", "beta", "Final Dice"]:
#     # Compute method rankings
#     ranks = (
#         whole_data.loc[:, (slice(None), slice(None), metric)]
#         .droplevel(level=2, axis=1)
#         .rank(ascending=False, method="min")
#     )

#     # whole_data = whole_data[sorted(whole_data.columns, key=sort_key)]
#     # ranks = ranks[sorted(ranks.columns, key=sort_key)]

#     # Save ranking table to txt file
#     # save_df_to_txt(ranks, out_path / f"method_ranking_{metric}.txt")

#     fig, (ax1, ax2) = plt.subplots(
#         1,
#         2,
#         # sharey=True,
#         figsize=(14, 6),
#         width_ratios=(12, 1),
#         gridspec_kw={
#             "wspace": 0.05,  # 0.03
#         },
#     )
#     # plt.figure(figsize=(12, 6))
#     # ax = plt.gcf().gca()

#     # Create ranking line plot
#     for method_name in QM_TO_COLOR:
#         ax1.plot(
#             ranks.loc[method_name, :].values,
#             marker="o",
#             label=method_name,
#             color=QM_TO_COLOR[method_name],
#             ls="--" if "random" in method_name else "-",
#             lw=2,
#             markerfacecolor=(
#                 "white" if "random" in method_name else QM_TO_COLOR[method_name]
#             ),
#         )

#     ax1.add_patch(
#         patches.Rectangle((-0.5, 0), 3, 8.5, linewidth=1, facecolor="k", alpha=0.1)
#     )
#     ax1.add_patch(
#         patches.Rectangle((2.5, 0), 3, 8.5, linewidth=1, facecolor="k", alpha=0.03)
#     )
#     ax1.add_patch(
#         patches.Rectangle((5.5, 0), 3, 8.5, linewidth=1, facecolor="k", alpha=0.1)
#     )
#     ax1.add_patch(
#         patches.Rectangle((8.5, 0), 3, 8.5, linewidth=1, facecolor="k", alpha=0.03)
#     )

#     # ax.vlines(
#     #     [2.5, 5.5, 8.5],
#     #     0.5,
#     #     ranks.shape[0] + 0.5,
#     #     colors="k",
#     #     linestyles="-",
#     #     lw=2,
#     #     alpha=0.2,
#     #     zorder=-1,
#     # )
#     ax1.set_xlim(-0.5, 11.5)
#     ax1.set_ylim(0.5, ranks.shape[0] + 0.5)
#     ax1.grid(axis="y")

#     ax1.set_xticks(
#         ticks=np.arange(ranks.shape[-1]),
#         labels=[
#             (f"{c[1]}\n\n{c[0]}" if i % 3 == 1 else c[1])
#             for i, c in enumerate(ranks.columns)
#         ],
#     )
#     ax1.set_yticks(ticks=np.arange(ranks.shape[0]) + 1)
#     ax1.set_ylabel(f"Method Rank ({metric})")
#     ax1.legend(loc=(0.1, -0.25), handlelength=4, ncols=4)

#     # Create mean rank plot
#     avg_ranks = ranks.mean(axis=1)
#     std_ranks = ranks.std(axis=1)
#     ranks_sorted = avg_ranks.rank(ascending=True, method="first")
#     for i, method_name in enumerate(QM_TO_COLOR):
#         ax2.errorbar(
#             -0.5 + ranks_sorted[method_name] / (len(ranks_sorted) + 1),
#             [avg_ranks[method_name]],
#             yerr=[std_ranks[method_name]],
#             fmt=".",
#             markersize=8,
#             color=QM_TO_COLOR[method_name],
#             markerfacecolor=(
#                 "white" if "random" in method_name else QM_TO_COLOR[method_name]
#             ),
#         )

#     ax2.set_xlim(-0.5, 0.5)
#     ax2.set_ylim(0.5, ranks.shape[0] + 0.5)
#     ax2.set_xticks(ticks=[0], labels=["Mean Rank"])
#     ax2.set_yticks(ticks=np.arange(ranks.shape[0]) + 1)
#     ax2.tick_params("x", length=0, pad=7)
#     ax2.grid(axis="y")

#     # Save figure
#     # plt.tight_layout()
#     plt.savefig(savepath / f"{name}_{metric}.png", bbox_inches="tight")

if __name__ == "__main__":
    for setting, rename_setting in zip(USE_SETTINGS_LIST, RENAME_SETTINGS_LIST):
        print(setting)
        print_setting = "_".join(setting).replace(" ", "").replace("/", "-")
        setting_paths = get_settings_for_combination(setting)
        setting_analyses = load_settings(setting_paths, comparative=COMPARATIVE)
        if rename_setting is not None:
            rename_settings_in_analysis(setting_analyses, rename_setting)
            rename_settings_in_analysis(setting_paths, rename_setting)

        data_dict = load_setting_data_to_df(
            CUSTOM_ORDER, FINAL_COLUMNS, setting_paths, setting_analyses
        )

        if len(setting) == 1:
            name = setting[0].lower().replace(" ", "")
            whole_data: dict[str, dict[str, pd.DataFrame]] = {}
            for dataset in data_dict:
                whole_data[dataset] = {}
                for budget in data_dict[dataset]:
                    whole_data[dataset][budget] = data_dict[dataset][budget][setting[0]]
                whole_data[dataset] = pd.concat(
                    whole_data[dataset].values(),
                    axis=1,
                    keys=whole_data[dataset].keys(),
                    names=COLLEVELANMES[1:],
                )
            whole_data = pd.concat(
                whole_data,
                axis=1,
                keys=whole_data.keys(),
                names=COLLEVELANMES,
            )

            for metric in [
                c["PrintCol"] for c in FINAL_COLUMNS if c["better"] == "higher"
            ]:
                # Compute method rankings
                ranks = (
                    whole_data.loc[:, (slice(None), slice(None), metric)]
                    .droplevel(level=2, axis=1)
                    .rank(ascending=False, method="min")
                )

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

                # Create ranking line plot
                for method_name in QM_TO_COLOR:
                    ax1.plot(
                        ranks.loc[method_name, :].values,
                        marker="o",
                        label=method_name,
                        color=QM_TO_COLOR[method_name],
                        ls="--" if "Random" in method_name else "-",
                        lw=2,
                        markerfacecolor=(
                            "white"
                            if "Random" in method_name
                            else QM_TO_COLOR[method_name]
                        ),
                    )

                ax1.add_patch(
                    patches.Rectangle(
                        (-0.5, 0), 3, 8.5, linewidth=1, facecolor="k", alpha=0.1
                    )
                )
                ax1.add_patch(
                    patches.Rectangle(
                        (2.5, 0), 3, 8.5, linewidth=1, facecolor="k", alpha=0.03
                    )
                )
                ax1.add_patch(
                    patches.Rectangle(
                        (5.5, 0), 3, 8.5, linewidth=1, facecolor="k", alpha=0.1
                    )
                )
                ax1.add_patch(
                    patches.Rectangle(
                        (8.5, 0), 3, 8.5, linewidth=1, facecolor="k", alpha=0.03
                    )
                )

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
                for i, method_name in enumerate(QM_TO_COLOR):
                    ax2.errorbar(
                        -0.5 + ranks_sorted[method_name] / (len(ranks_sorted) + 1),
                        [avg_ranks[method_name]],
                        yerr=[std_ranks[method_name]],
                        fmt=".",
                        markersize=8,
                        color=QM_TO_COLOR[method_name],
                        markerfacecolor=(
                            "white"
                            if "Random" in method_name
                            else QM_TO_COLOR[method_name]
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
                metric_name = metric.lower().replace(" ", "")
                plt.savefig(
                    savepath / f"{name}--{metric_name}.png", bbox_inches="tight"
                )

        else:
            raise NotImplementedError
