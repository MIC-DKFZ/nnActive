from pathlib import Path

import numpy as np
import pandas as pd
from setup import BASEPATH, RENAMING_DICT

from nnactive.analyze.aggregate_results import pretty_auc
from nnactive.analyze.analysis import SettingAnalysis
from nnactive.utils.io import save_df_to_txt

STANDARD_COLNAMES = ["Low", "Medium", "High"]
COMPUTE_COLNAMES = ["200 Epochs", "Precomputed 500 Epochs", "500 Epochs"]
SETTINGS = {
    "AMOS": [
        "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-40__qs-40",
        "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-200__qs-200",
        "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-500__qs-500",
    ],
    "KiTS": [
        "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-40__qs-40",
        "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-200__qs-200",
        "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-500__qs-500",
    ],
    "ACDC": [
        "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-30__qs-30",
        "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-60__qs-60",
        "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-90__qs-90",
    ],
    "Hippocampus": [
        "Dataset004_Hippocampus/patch-20_20_20__sb-random-label2-all-classes__sbs-20__qs-20__5loops",
        "Dataset004_Hippocampus/patch-20_20_20__sb-random-label2-all-classes__sbs-40__qs-40",
        "Dataset004_Hippocampus/patch-20_20_20__sb-random-label2-all-classes__sbs-60__qs-60",
    ],
    "AMOS_500epochs": [
        "Dataset216_AMOS2022_task1/tr-nnActiveTrainer_500epochs__patch-32_74_74__sb-random-label2-all-classes__sbs-200__qs-200",
        "Dataset216_AMOS2022_task1/tr-nnActiveTrainer_500epochs__patch-32_74_74__sb-random-label2-all-classes__sbs-500__qs-500",
    ],
    "AMOS_500epochs_precomp": [
        "Dataset216_AMOS2022_task1/tr-nnActiveTrainer_500epochs__patch-32_74_74__sb-random-label2-all-classes__sbs-200__qs-200__precomputed-queries",
        "Dataset216_AMOS2022_task1/tr-nnActiveTrainer_500epochs__patch-32_74_74__sb-random-label2-all-classes__sbs-500__qs-500__precomputed-queries",
    ],
    "AMOS_Medium-training": [
        "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-200__qs-200",
        "Dataset216_AMOS2022_task1/tr-nnActiveTrainer_500epochs__patch-32_74_74__sb-random-label2-all-classes__sbs-200__qs-200__precomputed-queries",
        "Dataset216_AMOS2022_task1/tr-nnActiveTrainer_500epochs__patch-32_74_74__sb-random-label2-all-classes__sbs-200__qs-200",
    ],
    "AMOS_High-training": [
        "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-500__qs-500",
        "Dataset216_AMOS2022_task1/tr-nnActiveTrainer_500epochs__patch-32_74_74__sb-random-label2-all-classes__sbs-500__qs-500__precomputed-queries",
        "Dataset216_AMOS2022_task1/tr-nnActiveTrainer_500epochs__patch-32_74_74__sb-random-label2-all-classes__sbs-500__qs-500",
    ],
    "AMOS_Low-QS": [
        "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-40__qs-20",
        "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-40__qs-40",
        "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-40__qs-80",
    ],
    "AMOS_High-QS": [
        "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-500__qs-250",
        "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-500__qs-500",
        "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-500__qs-1000",
    ],
    "KiTS_500epochs": [
        "Dataset135_KiTS2021/tr-nnActiveTrainer_500epochs__patch-64_64_64__sb-random-label2-all-classes__sbs-200__qs-200",
        "Dataset135_KiTS2021/tr-nnActiveTrainer_500epochs__patch-64_64_64__sb-random-label2-all-classes__sbs-500__qs-500",
    ],
    "KiTS_500epochs__precomp": [
        "Dataset135_KiTS2021/tr-nnActiveTrainer_500epochs__patch-64_64_64__sb-random-label2-all-classes__sbs-200__qs-200__precomputed-queries",
        "Dataset135_KiTS2021/tr-nnActiveTrainer_500epochs__patch-64_64_64__sb-random-label2-all-classes__sbs-500__qs-500__precomputed-queries",
    ],
    "KiTS_Medium-training": [
        "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-200__qs-200",
        "Dataset135_KiTS2021/tr-nnActiveTrainer_500epochs__patch-64_64_64__sb-random-label2-all-classes__sbs-200__qs-200__precomputed-queries",
        "Dataset135_KiTS2021/tr-nnActiveTrainer_500epochs__patch-64_64_64__sb-random-label2-all-classes__sbs-200__qs-200",
    ],
    "KiTS_High-training": [
        "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-500__qs-500",
        "Dataset135_KiTS2021/tr-nnActiveTrainer_500epochs__patch-64_64_64__sb-random-label2-all-classes__sbs-500__qs-500__precomputed-queries",
        "Dataset135_KiTS2021/tr-nnActiveTrainer_500epochs__patch-64_64_64__sb-random-label2-all-classes__sbs-500__qs-500",
    ],
    "KiTS_Low-QS": [
        "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-40__qs-20",
        "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-40__qs-40",
        "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-40__qs-80",
    ],
    "KiTS_High-QS": [
        "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-500__qs-250",
        "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-500__qs-500",
        "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-500__qs-1000",
    ],
    "ACDC_Low-QS": [
        "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-30__qs-15_revision",
        "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-30__qs-30",
        "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-30__qs-60_revision",
    ],
    "ACDC_High-QS": [
        "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-90__qs-45_revision",
        "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-90__qs-90",
        "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-90__qs-180_revision",
    ],
    "AMOS_patchablation": [
        "Dataset216_AMOS2022_task1/patch-16_32_32__sb-random-label2-all-classes__sbs-40__qs-40",
        "Dataset216_AMOS2022_task1/patch-16_32_32__sb-random-label2-all-classes__sbs-200__qs-200",
        "Dataset216_AMOS2022_task1/patch-16_32_32__sb-random-label2-all-classes__sbs-500__qs-500",
    ],
    "KiTS_patchablation": [
        "Dataset135_KiTS2021/patch-32_32_32__sb-random-label2-all-classes__sbs-40__qs-40",
        "Dataset135_KiTS2021/patch-32_32_32__sb-random-label2-all-classes__sbs-200__qs-200",
        "Dataset135_KiTS2021/patch-32_32_32__sb-random-label2-all-classes__sbs-500__qs-500",
    ],
    "ACDC_patchablation": [
        "Dataset027_ACDC/patch-2_20_20__sb-random-label2-all-classes__sbs-30__qs-30",
        "Dataset027_ACDC/patch-2_20_20__sb-random-label2-all-classes__sbs-60__qs-60",
        "Dataset027_ACDC/patch-2_20_20__sb-random-label2-all-classes__sbs-90__qs-90",
    ],
}

for name in SETTINGS:
    vals = []
    for path in SETTINGS[name]:
        fn = path.split("/")[-1]
        qs = fn.split("qs-")[1].split("_")[0]
        sbs = fn.split("sbs-")[1].split("_")[0]
        vals.append({"fn": fn, "qs": qs, "sbs": sbs})

    if "low" in name and "QS" in name:
        colnames = [f"Low ({v['qs']})" for v in vals]
    elif "high" in name and "QS" in name:
        colnames = [f"High ({v['qs']})" for v in vals]
    elif "training" in name:
        budget = name.split("-")[0].split("_")[-1]
        colnames = [f"{budget} ({v})" for v in COMPUTE_COLNAMES]
    elif "500epochs" in name:
        # small query size is not ablated atm
        colnames = STANDARD_COLNAMES[1:]
    elif "patchablation" in name:
        colnames = STANDARD_COLNAMES
    else:
        colnames = STANDARD_COLNAMES

    SETTINGS[name] = {
        "paths": [BASEPATH / p for p in SETTINGS[name]],
        "colnames": colnames,
    }

savepath = Path(
    "/home/c817h/Documents/projects/nnactive_project/nnactive/results/horeka_rsync_final/"
)

for name in SETTINGS:
    SETTINGS[name]["paths"] = [BASEPATH / p for p in SETTINGS[name]["paths"]]

CUSTOM_ORDER = [
    "mutual information",
    "power bald",
    "softrank bald",
    "pred entropy",
    "power pe",
    "random",
    "random-label",
    "random-label2",
]

entire_data = []

for name in SETTINGS:
    paths = SETTINGS[name]["paths"]
    colnames = SETTINGS[name]["colnames"]

    fn = name

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
    for path, colname in zip(paths, colnames):
        if path.is_dir():
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
            analysis = SettingAnalysis.load(path / "analysis.pkl")
            query_size = path.name.split("qs-")[1].split("__")[0]
            starting_budget = path.name.split("sbs-")[1].split("__")[0]

            # data_dict["Dataset"] = path.parent.name.replace("_", " ")
            data_dict["Dataset"] = name.split("_")[0]

            data_dict["Setting"] = colname
            data_dict["df_beta"] = (
                pd.read_json(path / "beta.json")
                .set_index("Query Method")
                .apply(lambda x: np.round(x, 2))
            ).rename(columns={"beta_std": "beta std"})
            data_dict["df"] = pd.concat(
                [data_dict["df_auc"], data_dict["df_beta"]], axis=1
            )[["AUBC", "AUBC std", "beta", "beta std", "Final", "Final std"]]
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
        d_folder = dataset.replace(" ", "_")
        # save_df_to_txt(whole_data[dataset], savepath / d_folder / f"{fn}.txt")
    if len(whole_data) == 0:
        print(f"Skipping {fn}")
        continue
    whole_data = pd.concat(
        whole_data, axis=1, keys=whole_data.keys(), names=["Dataset"]
    )
    whole_data = whole_data.reindex(CUSTOM_ORDER, level=0)
    whole_data = whole_data.rename(RENAMING_DICT, axis=0)
    save_df_to_txt(whole_data, savepath / f"{fn}.txt")

    with open(savepath / f"{fn}.md", "w") as f:
        f.write(whole_data.to_markdown())

    cmap = "Oranges"
    higher_is_better = ["AUBC", "Final", "beta"]
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
    tex_fn = savepath / f"{fn}.tex"
    styled.to_latex(
        tex_fn,
        convert_css=True,
        hrules=True,
        multicol_align="c|",
        column_format="l" + columns + "|",
    )

    entire_data.append(whole_data)
    # aubc_cols = [col for col in whole_data.columns if col[-1] == "AUBC"]
    # aubc_std_cols = [col for col in whole_data.columns if col[-1] == "AUBC std"]
    # aubc_vals = whole_data[aubc_cols]
    # aubc_std_vals = whole_data[aubc_std_cols]
    # for col in aubc_vals:
    #     aubc_vals[(*col[:-1], "AUBC rank")] = aubc_vals[col].rank(ascending=False)

    # entire_data.append(aubc_vals)

# whole_data = pd.concat(entire_data, axis=1)
# aubc_cols = [col for col in whole_data.columns if col[-1] == "AUBC"]
# aubc_std_cols = [col for col in whole_data.columns if col[-1] == "AUBC std"]
# aubc_vals = whole_data[aubc_cols]
# aubc_std_vals = whole_data[aubc_std_cols]
# rank_cols = [(*col[:-1], "AUBC rank") for col in aubc_vals]
# for col in aubc_vals:
#     aubc_vals[(*col[:-1], "AUBC rank")] = aubc_vals[col].rank(ascending=False)


# mean_rank = aubc_vals[rank_cols].mean(axis=1).sort_values()
# meadian_rank = aubc_vals[rank_cols].median(axis=1).sort_values()
# mean_aubc = whole_data[aubc_cols].mean(axis=1).sort_values()
# mean_aubc_rank = mean_aubc.rank(ascending=False)

# aggregated_rankings_aubc = pd.concat(
#     [mean_rank, meadian_rank, mean_aubc_rank, mean_aubc],
#     axis=1,
#     keys=["Mean Rank (AUBC)", "Median Rank (AUBC)", "Rank (Mean AUBC)", "Mean AUBC"],
# )
# aggregated_rankings_aubc = aggregated_rankings_aubc.reindex(CUSTOM_ORDER)

# print(aggregated_rankings_aubc)


# print("Analysing for best QM")
# # best_qm = "power bald"
# best_qm = "softrank bald"
# rbs = ["random-label", "random-label2", "random"]
# print(best_qm)
# print("\n")

# aubc_gain = aubc_vals.loc[best_qm] - aubc_vals.loc[rbs]
# aubc_gain = aubc_gain[aubc_cols]

# mean_gain = aubc_gain.mean(axis=1)
# print("Mean Gain")
# print(mean_gain)
# aubc_gain_pos = aubc_gain > 0
# percentage_gain = aubc_gain_pos.mean(axis=1)
# print("Gain Scenarios")
# print(percentage_gain)


# # print("Only Medium and Large Label Regimes")
# # aubc_gain = aubc_vals.loc[best_qm] - aubc_vals.loc[rbs]
# # low_label_cols = [
# #     ("Dataset216 AMOS2022 task1", "Query Size 40"),
# #     ("Dataset135 KiTS2021", "Query Size 200"),
# #     ("Dataset027 ACDC", "Query Size 30"),
# #     ("Dataset004 Hippocampus", "Query Size 20"),
# # ]
# # keep_cols = [col for col in aubc_cols if col[:-1] not in low_label_cols]
# # aubc_gain = aubc_gain[keep_cols]
# # aubc_gain = aubc_vals.loc[best_qm] - aubc_vals.loc[rbs]
# # aubc_gain = aubc_gain[[col for col in aubc_gain if col[-1] == "AUBC"]]
# # print(aubc_gain)
# # mean_gain = aubc_gain.mean(axis=1)
# # print("Mean Gain")
# # print(mean_gain)
# # aubc_gain_pos = aubc_gain > 0
# # percentage_gain = aubc_gain_pos.mean(axis=1)
# # print("Gain Scenarios")
# # print(percentage_gain)


# import IPython

# IPython.embed()
