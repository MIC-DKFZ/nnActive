import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from setup import BASEPATH, QM_TO_COLOR, RENAMING_DICT, RESULTSPATH, SAVETYPE

from nnactive.analyze.analysis import SettingAnalysis
from nnactive.utils.io import load_pickle

qm_list = [
    "power_pe",
    "pred_entropy",
    "random-label",
]


SAVEPATH = RESULTSPATH / "fg_effplot"

if SAVEPATH.exists() is False:
    os.makedirs(SAVEPATH)

PATHS = [
    BASEPATH
    / "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-200__qs-200",
]

if __name__ == "__main__":
    for i in range(len(PATHS)):
        PATHS[i] = Path(PATHS[i])

    x_axs = "percentage_of_voxels_foreground"
    y_axs = "Mean Dice"

    for path in PATHS:
        setting: SettingAnalysis = load_pickle(path / "analysis.pkl")

        trainer = str(setting.df["trainer"].unique()[0])
        trainer_use = "nnUNetTrainer"
        if len(trainer.split("_")) > 1:
            epochs = trainer.split("_")
            trainer_use = f"{trainer_use}_{epochs[-1]}"
        betas = setting.compute_beta_curve(
            trainer_use,
            x_axs,
        )
        datasetbeta = setting.compute_beta_curve(trainer_use, x_axs, y_axs)
        from pprint import pprint

        x_min = setting.df[x_axs].min()
        x_max = setting.df[x_axs].max()
        x_range = np.linspace(x_min, x_max, 100)
        factor = 4 / 5
        fig, ax = plt.subplots(figsize=(8 * factor, 6 * factor))
        for qm in qm_list:
            qm_name = RENAMING_DICT[qm]
            y_vals = datasetbeta.compute(x_range, qm)
            qm_df = setting.df[setting.df[setting.query_key] == qm]
            ax.plot(
                x_range * 100,
                y_vals,
                label=qm_name
                + r" ($\gamma=$"
                + str(round(datasetbeta.beta_dict[qm], 2))
                + ")",
                color=QM_TO_COLOR[qm_name],
            )
            ax.scatter(
                qm_df[x_axs] * 100,
                qm_df[y_axs],
                color=QM_TO_COLOR[qm_name],
            )
        ax.set_xlabel("Percentage of Voxels Foreground [%]")
        ax.set_ylabel(y_axs.replace("_", " "))
        plt.hlines(
            datasetbeta.c,
            x_min * 100,
            x_max * 100,
            color="black",
            linestyle="--",
            label=r"$\hat{y}_{\text{full}}$",
        )
        ax.legend()
        plt.savefig(SAVEPATH / f"fg-eff_plot.{SAVETYPE}", bbox_inches="tight")
        pprint(datasetbeta)
