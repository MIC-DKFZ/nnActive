from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from setup import BASEPATH, RENAMING_DICT

from nnactive.analyze.aggregate_results import pretty_auc
from nnactive.analyze.analysis import SettingAnalysis
from nnactive.analyze.metrics import PairwisePenaltyMatrix
from nnactive.utils.io import save_df_to_txt

# For final version in paper
# When using increase dpi and save to .pdf as vector graphic!
# Increase font size or reduce figure size to make text readable in paper!
# matplotlib.rcParams["text.usetex"] = True
# matplotlib.rcParams["font.family"] = "Computer Modern"

MAIN_ORDER = [
    "mutual information",
    "power bald",
    "softrank bald",
    "pred entropy",
    "power pe",
    "random",
    "random-label",
    "random-label2",
]

NORANDOM_ORDER = [
    "mutual information",
    "power bald",
    "softrank bald",
    "pred entropy",
    "power pe",
    "random-label",
    "random-label2",
]

SETTING = {
    "Main": {
        "comparisons": {
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
        },
        "custom_order": MAIN_ORDER,
    },
    "Long-Training": {
        "comparisons": {
            "AMOS": [
                "Dataset216_AMOS2022_task1/tr-nnActiveTrainer_500epochs__patch-32_74_74__sb-random-label2-all-classes__sbs-200__qs-200",
                "Dataset216_AMOS2022_task1/tr-nnActiveTrainer_500epochs__patch-32_74_74__sb-random-label2-all-classes__sbs-500__qs-500",
            ],
            "KiTS": [
                "Dataset135_KiTS2021/tr-nnActiveTrainer_500epochs__patch-64_64_64__sb-random-label2-all-classes__sbs-200__qs-200",
                "Dataset135_KiTS2021/tr-nnActiveTrainer_500epochs__patch-64_64_64__sb-random-label2-all-classes__sbs-500__qs-500",
            ],
        },
        "custom_order": NORANDOM_ORDER,
    },
}


savepath = Path(
    "/home/c817h/Documents/projects/nnactive_project/nnactive/results/horeka_rsync_final/"
)

for comparison_name in SETTING:
    comparison_setting = SETTING[comparison_name]
    comparison = comparison_setting["comparisons"]
    for name in comparison:
        comparison[name] = [BASEPATH / p for p in comparison[name]]

    custom_order = comparison_setting["custom_order"]

    for i in range(len(custom_order)):
        custom_order[i] = custom_order[i].replace(" ", "_")

    all_matrices = []
    for name, paths in comparison.items():

        matrices: list[PairwisePenaltyMatrix] = []
        for path in paths:
            matrix = PairwisePenaltyMatrix.load(BASEPATH / path / "ppm.json")

            del_algs = [a for a in matrix.algs if a not in custom_order]
            for a in del_algs:
                matrix.delete_alg(a)
            ex_name = path.name
            save_name = f"{name}_{ex_name}_ppm.txt"
            matrices.append(matrix)
            all_matrices.append(matrix)

        full_matrix = PairwisePenaltyMatrix.create_merged_matrix(matrices)
        full_matrix = full_matrix.custom_order_matrix(custom_order)
        full_matrix.rename_algs(RENAMING_DICT)
        full_matrix.plot_pairwise_matrix(
            full_matrix,
            savepath=savepath / f"{comparison_name.lower()}_{name}_ppm.png",
            max_poss_ent=None,
            title_tag=f"{comparison_name} " + name + " [%]",
            norm_val=full_matrix.max_pos_ent,
        )
    full_matrix = PairwisePenaltyMatrix.create_merged_matrix(all_matrices)
    full_matrix = full_matrix.custom_order_matrix(custom_order)
    full_matrix.rename_algs(RENAMING_DICT)
    full_matrix.plot_pairwise_matrix(
        full_matrix,
        savepath=savepath / f"{comparison_name.lower()}_ppm.png",
        max_poss_ent=None,
        title_tag=f"{comparison_name} Experiments [%]",
        norm_val=full_matrix.max_pos_ent,
    )
