from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from nnactive.analyze.aggregate_results import pretty_auc
from nnactive.analyze.analysis import SettingAnalysis
from nnactive.analyze.metrics import PairwisePenaltyMatrix
from nnactive.utils.io import save_df_to_txt

# For final version in paper
# When using increase dpi and save to .pdf as vector graphic!
# Increase font size or reduce figure size to make text readable in paper!
# matplotlib.rcParams["text.usetex"] = True
# matplotlib.rcParams["font.family"] = "Computer Modern"


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
        "Dataset004_Hippocampus/patch-20_20_20__sb-random-label2-all-classes__sbs-20__qs-20",
        "Dataset004_Hippocampus/patch-20_20_20__sb-random-label2-all-classes__sbs-40__qs-40",
        "Dataset004_Hippocampus/patch-20_20_20__sb-random-label2-all-classes__sbs-60__qs-60",
    ],
}

intermediate_path = Path(
    "/home/c817h/Documents/projects/nnactive_project/nnactive/results/horeka_rsync/"
)

savepath = Path(
    "/home/c817h/Documents/projects/nnactive_project/nnactive/results/horeka_rsync/"
)

for name in SETTINGS:
    SETTINGS[name] = [intermediate_path / p for p in SETTINGS[name]]

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

for i in range(len(CUSTOM_ORDER)):
    CUSTOM_ORDER[i] = CUSTOM_ORDER[i].replace(" ", "_")

all_matrices = []
for name, paths in SETTINGS.items():

    matrices: list[PairwisePenaltyMatrix] = []
    for path in paths:
        matrix = PairwisePenaltyMatrix.load(intermediate_path / path / "ppm.json")

        del_algs = [a for a in matrix.algs if a not in CUSTOM_ORDER]
        for a in del_algs:
            matrix.delete_alg(a)
        ex_name = path.name
        save_name = f"{name}_{ex_name}_ppm.txt"
        matrices.append(matrix)
        all_matrices.append(matrix)

    full_matrix = PairwisePenaltyMatrix.create_merged_matrix(matrices * 2)
    full_matrix = full_matrix.custom_order_matrix(CUSTOM_ORDER)
    full_matrix.plot_pairwise_matrix(
        full_matrix,
        savepath=savepath / f"main_{name}_ppm.png",
        max_poss_ent=None,
        title_tag=f"Main " + name,
        norm_val=len(paths),
    )
full_matrix = PairwisePenaltyMatrix.create_merged_matrix(all_matrices)
full_matrix = full_matrix.custom_order_matrix(CUSTOM_ORDER)
full_matrix.plot_pairwise_matrix(
    full_matrix,
    savepath=savepath / f"main_ppm.png",
    max_poss_ent=None,
    title_tag="Main Experiments",
    norm_val=sum([len(paths) for paths in SETTINGS.values()]),
)
