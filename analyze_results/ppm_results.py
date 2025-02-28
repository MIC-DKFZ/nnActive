from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from evaluator import (
    get_settings_for_combination,
    load_settings,
    rename_settings_in_analysis,
)
from setup import BASEPATH, CUSTOM_ORDER, RENAMING_DICT, SAVEPATH

from nnactive.analyze.aggregate_results import pretty_auc
from nnactive.analyze.analysis import SettingAnalysis
from nnactive.analyze.metrics import PairwisePenaltyMatrix
from nnactive.utils.io import save_df_to_txt

# For final version in paper
# When using increase dpi and save to .pdf as vector graphic!
# Increase font size or reduce figure size to make text readable in paper!
# matplotlib.rcParams["text.usetex"] = True
# matplotlib.rcParams["font.family"] = "Computer Modern"

MAIN_ORDER = CUSTOM_ORDER

NORANDOM_ORDER = MAIN_ORDER.copy()
NORANDOM_ORDER.remove("random")
savepath = SAVEPATH / "figures"

COMPARATIVE = False

USE_SETTINGS_LIST = [
    {"setting_names": ["Main"], "custom_order": MAIN_ORDER},
    {"setting_names": ["500 Epochs"], "custom_order": NORANDOM_ORDER},
    {"setting_names": ["Precomputed"], "custom_order": NORANDOM_ORDER},
    {"setting_names": ["Precomputed"], "custom_order": MAIN_ORDER},
    {"setting_names": ["Patchx1/2"], "custom_order": MAIN_ORDER},
]

RENAME_SETTINGS = None


for setting in USE_SETTINGS_LIST:
    setting_names = setting["setting_names"]
    custom_order = setting["custom_order"]
    setting_paths = get_settings_for_combination(setting_names)
    setting_data = load_settings(setting_paths)
    if RENAME_SETTINGS is not None:
        rename_settings_in_analysis(setting_data)
    save_setting = "_".join(setting_names).replace(" ", "").replace("/", "-").lower()
    print_setting = " & ".join(setting_names)

    all_matrices = {}
    for dataset in setting_data:
        all_matrices[dataset] = {}
        for budget in setting_data[dataset]:
            all_matrices[dataset][budget] = {}
            for subsetting in setting_data[dataset][budget]:
                analysis = setting_data[dataset][budget][subsetting]
                matrix = analysis.compute_pairwise_penalty()
                del_algs = [a for a in matrix.algs if a not in custom_order]
                for a in del_algs:
                    matrix.delete_alg(a)
                matrix = matrix.custom_order_matrix(custom_order)
                matrix.rename_algs(RENAMING_DICT)
                all_matrices[dataset][budget][subsetting] = matrix

    for dataset in all_matrices:
        merged_matrix = PairwisePenaltyMatrix.create_merged_matrix(
            [
                mat[subsetting]
                for subsetting in setting_names
                for mat in all_matrices[dataset].values()
            ]
        )
        PairwisePenaltyMatrix.plot_pairwise_matrix(
            merged_matrix,
            savepath=savepath / f"{save_setting}_{dataset}_ppm.png",
            max_poss_ent=None,
            title_tag=f"{print_setting} {dataset} [%]",
            norm_val=merged_matrix.max_pos_ent,
        )
    merged_matrix = PairwisePenaltyMatrix.create_merged_matrix(
        [
            all_matrices[d][b][s]
            for d in all_matrices
            for b in all_matrices[d]
            for s in all_matrices[d][b]
        ]
    )
    PairwisePenaltyMatrix.plot_pairwise_matrix(
        merged_matrix,
        savepath=savepath / f"{save_setting}_ppm.png",
        max_poss_ent=None,
        title_tag=f"{print_setting} [%]",
        norm_val=merged_matrix.max_pos_ent,
    )
