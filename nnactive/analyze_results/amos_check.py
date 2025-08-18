import os
import re
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from setup import BASEPATH, RENAMING_DICT, RESULTSPATH, SAVETYPE

from nnactive.analyze.aggregate_results import pretty_auc
from nnactive.analyze.analysis import SettingAnalysis
from nnactive.utils.io import save_df_to_txt

SETTINGS = {
    "AMOS": [
        "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-40__qs-40",
        "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-200__qs-200",
        "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-500__qs-500",
    ],
    "AMOS_Precomputed": [
        "Dataset216_AMOS2022_task1/tr-nnActiveTrainer_500epochs__patch-32_74_74__sb-random-label2-all-classes__sbs-200__qs-200__precomputed-queries",
        "Dataset216_AMOS2022_task1/tr-nnActiveTrainer_500epochs__patch-32_74_74__sb-random-label2-all-classes__sbs-500__qs-500__precomputed-queries",
    ],
    "AMOS_500epochs": [
        "Dataset216_AMOS2022_task1/tr-nnActiveTrainer_500epochs__patch-32_74_74__sb-random-label2-all-classes__sbs-200__qs-200",
        "Dataset216_AMOS2022_task1/tr-nnActiveTrainer_500epochs__patch-32_74_74__sb-random-label2-all-classes__sbs-500__qs-500",
    ],
    "AMOS_minipatch": [
        "Dataset216_AMOS2022_task1/patch-16_32_32__sb-random-label2-all-classes__sbs-40__qs-40",
        "Dataset216_AMOS2022_task1/patch-16_32_32__sb-random-label2-all-classes__sbs-200__qs-200",
        "Dataset216_AMOS2022_task1/patch-16_32_32__sb-random-label2-all-classes__sbs-500__qs-500",
    ],
}

COMPARISONS = [
    "random-label",
    "random-label2",
    "mutual_information",
    "pred_entropy",
    "power_pe",
    "random",
]

BASELINES = ["random-label", "random-label2", "random"]

IDENTIFIER_DICT = {
    40: "Low-Label Regime",
    200: "Medium-Label Regime",
    500: "High-Label Regime",
}

savepath = RESULTSPATH / "amos_check"
if not savepath.exists():
    os.makedirs(savepath)


for name in SETTINGS:
    SETTINGS[name] = [BASEPATH / p for p in SETTINGS[name]]


for name, paths in SETTINGS.items():
    fn = name.replace("_", " ")
    savepath_fn = savepath / f"{name}"
    os.makedirs(savepath_fn, exist_ok=True)
    for path in paths:
        setting = SettingAnalysis.load(path / "analysis.pkl")
        values_dict = {}
        perf_values_dict = {}
        for unc in COMPARISONS:
            identifiers = setting.df["query_size"].unique()
            assert len(identifiers) == 1
            identifier = identifiers[0]
            df_filter = setting.df[setting.df[setting.query_key] == unc]
            df_filter = df_filter[df_filter["Loop"] == 4]
            perc_classes = [
                col
                for col in df_filter.columns
                if "percentage_of_voxels_per_cls_" in col
            ]
            perf_classes = [
                col for col in df_filter.columns if bool(re.search(r"Class \d+", col))
            ]
            values = []
            perf_values = []
            for col in perc_classes:
                values.append(df_filter[col].mean())
            for col in perf_classes:
                perf_values.append(df_filter[col].mean())
            perf_values.append(0)
            perf_values = np.array(perf_values)
            values_dict[unc] = np.array(values)

            perc_classes = [int(col.split("_")[-1]) for col in perc_classes]
            perf_classes = [col.split(" ")[-2] for col in perf_classes]
            perf_classes.append(0)
            perf_classes = np.array(perf_classes).astype(int)
            sorting = np.argsort(perf_classes)
            perf_classes = np.take_along_axis(perf_classes, sorting, axis=0)
            perf_values = np.take_along_axis(perf_values, sorting, axis=0)
            perf_values_dict[unc] = perf_values

            fig, axs = plt.subplots(2, 1, figsize=(10, 10))
            ax = axs[0]
            sns.barplot(x=perc_classes, y=values, ax=ax)
            ax.set_xlabel("Class")
            ax.set_ylabel("Mean percentage of voxels for class")
            ax.set_title(f"{RENAMING_DICT[unc]} on {fn} {IDENTIFIER_DICT[identifier]}")
            ax = axs[1]
            sns.barplot(x=perf_classes, y=perf_values, ax=ax)
            ax.set_xlabel("Class")
            ax.set_ylabel("Final Dice")
            fig.tight_layout()

            plt.savefig(
                savepath_fn / f"{name}_{identifier}_{unc}_perc_classes.{SAVETYPE}"
            )
            plt.close()

        compared = "random-label2"
        for unc, compared in product(COMPARISONS, BASELINES):
            if unc == compared:
                continue
            values = values_dict[unc] - values_dict[compared]
            perf_values = perf_values_dict[unc] - perf_values_dict[compared]
            fig, axs = plt.subplots(2, 1, figsize=(10, 10))
            ax = axs[0]
            sns.barplot(x=perc_classes, y=values, ax=ax)
            ax.set_xlabel("Class")
            ax.set_ylabel(r"$\Delta$ Mean percentage of voxels for class")
            ax.set_title(
                f"Difference from {RENAMING_DICT[unc]} to {RENAMING_DICT[compared]} on {fn} {IDENTIFIER_DICT[identifier]}"
            )
            ax = axs[1]
            sns.barplot(x=perf_classes, y=perf_values, ax=ax)
            ax.set_xlabel("Class")
            ax.set_ylabel(r"$\Delta$ Final DICE")
            fig.tight_layout()

            plt.savefig(
                savepath_fn
                / f"{name}_{identifier}_diff_{compared}_{unc}_perc_classes.{SAVETYPE}"
            )
            plt.close()
