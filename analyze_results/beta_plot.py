import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit

from nnactive.analyze.analysis import GridPlotter, SettingAnalysis
from nnactive.analyze.analyze_results import PALETTE, MultiExperimentAnalysis
from nnactive.analyze.metrics import DatasetBeta
from nnactive.utils.io import load_pickle, save_df_to_txt

if __name__ == "__main__":

    paths = [
        "/home/c817h/Documents/projects/nnactive_project/nnactive/results/horeka-main/Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-200__qs-200",
    ]
    output_dir = Path(".")

    # paths = [
    # "/home/c817h/Documents/projects/nnactive_project/nnactive/results/horeka-main/Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-200__qs-200"
    # ]

    # path = Path(
    #     "/home/c817h/Documents/projects/nnactive_project/nnactive/results/horeka-main"
    # )
    # paths = list(path.rglob("analysis.pkl"))
    # paths = [p.parent for p in paths if "Dataset004" not in str(p)]

    for i in range(len(paths)):
        paths[i] = Path(paths[i])

    x_axs = "percentage_of_voxels_foreground"
    y_axs = "Mean Dice"

    for path in paths:
        setting: SettingAnalysis = load_pickle(path / "analysis.pkl")

        trainer = str(setting.df["trainer"].unique()[0])
        trainer_use = "nnUNetTrainer"
        if len(trainer.split("_")) > 1:
            epochs = trainer.split("_")
            trainer_use = f"{trainer_use}_{epochs[-1]}"
        betas = setting.compute_beta_curve(
            trainer_use,
            "percentage_of_voxels_foreground",
        )
        datasetbeta = setting.compute_beta_curve(trainer_use, x_axs, y_axs)
        from pprint import pprint

        beta_df = datasetbeta.to_beta_df()
        # beta_df.to_json(path / "beta.json")
        # save_df_to_txt(beta_df, path / "beta.txt")

    # data = pd.concat(data_dicts, axis=0)
    # setting.df = data
    # x_axs = "percentage_of_voxels_foreground"
    # y_axs = "Mean Dice"

    # fig, axs = plt.subplots()

    # for key, df in setting.df.groupby(setting.query_key):
    #     sns.regplot(
    #         data=df,
    #         x=x_axs,
    #         y=y_axs,
    #         logx=True,
    #     )

    # plt.savefig("fit.png", bbox_inches="tight")

    # c_ = np.mean(
    #     setting.df[setting.df["#Patches"] == setting.df["#Patches"].min()][y_axs]
    # )
    # # amos full data 200 epochs
    a = 0.8600327277955928
    # c = a - c_
    # curve = lambda x, b: a - np.exp(-x * b) * c
    # # Center around 0!
    # setting.df["x_fit"] = setting.df[x_axs] - setting.df[x_axs].min()

    # fig, axs = plt.subplots()
    # print(setting.df["x_fit"])
    # print(a, c)
    # fit_dirs = []
    # for key, df in setting.df.groupby(setting.query_key):
    #     popt, pcov = curve_fit(curve, df["x_fit"], df[y_axs])
    #     print(f"{key=}, {popt=}")
    #     df["y_fit"] = curve(df["x_fit"], *popt)
    #     sns.lineplot(
    #         data=df,
    #         x=x_axs,
    #         y="y_fit",
    #         ax=axs,
    #         hue=setting.query_key,
    #         palette=PALETTE,
    #     )
    #     sns.scatterplot(
    #         data=df,
    #         x=x_axs,
    #         y=y_axs,
    #         ax=axs,
    #         hue=setting.query_key,
    #         palette=PALETTE,
    #         legend=False,
    #     )
    #     fit_dirs.append({"Query Method": key, "b": popt[0], "b_std": pcov[0, 0]})
    # plt.savefig(output_dir / "fit2.png", bbox_inches="tight")
    # axs.set_xscale("log")

    # plt.savefig(output_dir / "fit2_log.png", bbox_inches="tight")
    # from pprint import pprint

    # pprint(pd.DataFrame(fit_dirs))

    # # curve = lambda x, b: a - np.exp(-x * b) + c_
    # # setting.df["x_fit"] = (
    # #     setting.df["percentage_of_voxels_foreground"]
    # #     - setting.df["percentage_of_voxels_foreground"].min()
    # # )

    # # fig, axs = plt.subplots()
    # # print(setting.df["x_fit"])
    # # print(a, c)
    # # for key, df in setting.df.groupby(setting.query_key):
    # #     popt, pcov = curve_fit(curve, df["x_fit"], df[y_axs])
    # #     print(f"{key=}, {popt=}")
    # #     df["y_fit"] = curve(df["x_fit"], *popt)
    # #     sns.lineplot(
    # #         data=df,
    # #         x="percentage_of_voxels_foreground",
    # #         y="y_fit",
    # #         ax=axs,
    # #         hue=setting.query_key,
    # #         palette=PALETTE,
    # #     )
    # #     sns.scatterplot(
    # #         data=df,
    # #         x="percentage_of_voxels_foreground",
    # #         y=y_axs,
    # #         ax=axs,
    # #         hue=setting.query_key,
    # #         palette=PALETTE,
    # #         legend=False,
    # #     )
    # # axs.set_xscale("log")

    # # plt.savefig(output_dir / "fit3.png", bbox_inches="tight")

    c_ = np.mean(setting.df[setting.df[x_axs] == setting.df[x_axs].min()][y_axs])
    # amos full data 200 epochs
    a = 0.8600327277955928
    c = a - c_
    curve = lambda x, b: a - np.exp(-x * b) * c
    # Center around 0!
    setting.df["x_fit"] = (
        setting.df["percentage_of_voxels_foreground"]
        - setting.df["percentage_of_voxels_foreground"].min()
    )

    fig, axs = plt.subplots()
    fit_dirs = []
    for key, df in setting.df.groupby(setting.query_key):
        popt, pcov = curve_fit(curve, df["x_fit"], df[y_axs])
        df["y_fit"] = curve(df["x_fit"], *popt)
        sns.lineplot(
            data=df,
            x="percentage_of_voxels_foreground",
            y="y_fit",
            ax=axs,
            hue=setting.query_key,
            palette=PALETTE,
        )
        sns.scatterplot(
            data=df,
            x="percentage_of_voxels_foreground",
            y=y_axs,
            ax=axs,
            hue=setting.query_key,
            palette=PALETTE,
            legend=False,
        )
        fit_dirs.append({"Query Method": key, "b": popt[0], "b_std": pcov[0, 0]})
    plt.savefig(output_dir / "fit4.png", bbox_inches="tight")
    axs.set_xscale("log")

    plt.savefig(output_dir / "fit4_log.png", bbox_inches="tight")
    # from pprint import pprint

    # pprint(pd.DataFrame(fit_dirs))

    # fitted = DatasetBeta.from_df(
    #     setting.df, x=x_axs, y=y_axs, y_max=a, qm_key=setting.query_key
    # )
    # fitted.compute(fitted.x_offset, "random")
    # df_fitted = fitted.to_beta_df()

    # save_df_to_txt(df_fitted, "fitted.txt")
