from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import gumbel_r

from nnactive.strategies.utils import power_noising
from nnactive.utils.io import load_json

fns = ["mutual_information", "power_bald", "power_pe", "pred_entropy"]


def power_noising_rate(beta, top_k=200, samples=10000):
    x = np.ones(samples, dtype=np.float32)
    x = power_noising(x, beta)
    x = np.sort(x)[-top_k:]
    return x


def categorical_boltz(x, beta=1):
    out = np.exp(x * beta)
    out = out / np.sum(out)
    return out


path = Path(
    # "/home/c817h/network/horeka-main/data_folder/nnActive_data/Dataset027_ACDC/nnUNet_raw/Dataset000_ACDC__patch-4_40_40__sb-random-label2-all-classes__sbs-30__qs-30__unc-mutual_information__seed-12345"
    # "/home/c817h/network/horeka-main/data_folder/nnActive_data/Dataset135_KiTS2021/nnUNet_raw/Dataset101_KiTS2021__patch-64_64_64__sb-random-label2-all-classes__sbs-500__qs-500__unc-mutual_information__seed-12345"
    # "/home/c817h/network/horeka-main/data_folder/nnActive_data/Dataset135_KiTS2021/nnUNet_raw/Dataset105_KiTS2021__patch-64_64_64__sb-random-label2-all-classes__sbs-500__qs-500__unc-power_bald__seed-12345"
    "/home/c817h/network/horeka-main/data_folder/nnActive_data/Dataset216_AMOS2022_task1/nnUNet_raw/Dataset132_AMOS2022_task1__patch-32_74_74__sb-random-label2-all-classes__sbs-500__qs-500__unc-mutual_information__seed-12345"
    # "/home/c817h/network/horeka-main/data_folder/nnActive_data/Dataset216_AMOS2022_task1/nnUNet_raw/Dataset136_AMOS2022_task1__patch-32_74_74__sb-random-label2-all-classes__sbs-500__qs-500__unc-power_bald__seed-12345"
)
output_path = Path(".")
if __name__ == "__main__":
    files = [
        f
        for f in path.iterdir()
        if f.is_file() and any([f.name.startswith(fn) for fn in fns])
    ]
    files.sort()
    fn = "_".join(files[0].name.split("_")[:-1])
    scores = []
    for f in files:
        scores.append(load_json(f))

    dfs = []
    for score in scores:
        df = pd.DataFrame(score).sort_values("score")
        dfs.append(df)
    qs = int(path.name.split("qs-")[1].split("__")[0])

    output_path = output_path / path.name
    output_path.mkdir(exist_ok=True)

    fig, axs = plt.subplots(1, len(dfs) + 1, figsize=(20, 5))
    full_plot_ind = len(dfs)
    for i, df in enumerate(dfs):
        sns.histplot(data=df, ax=axs[i], x="score", kde=True)
        axs[i].axvline(df["score"].iloc[-qs], 0, 1)
        axs[i].set_title(f"Loop {i}")

        sns.histplot(data=df, ax=axs[full_plot_ind], x="score", kde=True)
        axs[full_plot_ind].axvline(df["score"].iloc[-qs], 0, 1)
        # print(df["score"].mean())

        # print(df["score"].std())
    fig.tight_layout()

    plt.savefig(output_path / "score.png")
    plt.clf()

    if fn in ["mutual_information", "pred_entropy"]:
        fig, axs = plt.subplots(1, len(dfs) + 1, figsize=(20, 5))
        plot_val = "log(score)"
        for i, df in enumerate(dfs):
            df[plot_val] = np.log(df["score"])
            sns.histplot(data=df, ax=axs[i], x=plot_val, kde=True)
            axs[i].axvline(df[plot_val].iloc[-qs], 0, 1)
            axs[i].set_title(f"Loop {i}")

            sns.histplot(data=df, ax=axs[full_plot_ind], x=plot_val, kde=True)
            axs[full_plot_ind].axvline(df[plot_val].iloc[-qs], 0, 1)
        fig.tight_layout()
        plt.savefig(output_path / f"{plot_val}.png")
        plt.clf()

        fig, axs = plt.subplots(1, len(dfs) + 1, figsize=(20, 5))
        plot_val = "log(score)"
        plot_y = "categorical_boltz"
        for i, df in enumerate(dfs):
            df[plot_val] = np.log(df["score"])
            df[plot_y] = categorical_boltz(df["score"], beta=50)
            counts, bin_edges = np.histogram(df[plot_y], bins="auto")
            df["bin"] = bin_edges[np.digitize(df[plot_y], bin_edges) - 1]

            sns.lineplot(data=df.groupby("bin").sum(), ax=axs[i], x="bin", y=plot_y)
            # sns.histplot(data=df, ax=axs[i], x=plot_val, y=plot_y, kde=True)
            # axs[i].axvline(df[plot_val].iloc[-qs], 0, 1)
            # axs[i].set_title(f"Loop {i}")

            # sns.histplot(data=df, ax=axs[full_plot_ind], x=plot_val, kde=True)
            # axs[full_plot_ind].axvline(df[plot_val].iloc[-qs], 0, 1)
        fig.tight_layout()
        plt.savefig(output_path / f"{plot_y}-{plot_val}.png")
        plt.clf()

    beta = 1

    df["score"] = 1
    df["score"] = power_noising(df["score"].to_numpy(), beta=1)
    fig, ax = plt.subplots()
    sns.histplot(data=df, ax=ax, x="score", kde=True)
    ax.set_title(f"Gumbel Beta {beta}")
    plt.savefig(output_path / f"gumbel(b={beta}).png")
    plt.clf()

    n_samples = 58 * 57 * 57
    x = power_noising_rate(1, top_k=qs, samples=n_samples)
    fig, ax = plt.subplots()
    sns.histplot(data=x, ax=ax, kde=True)
    ax.set_title(f"Gumbel Beta {beta} ({qs} out of {n_samples})")
    # import IPython

    # IPython.embed()
    plt.savefig(output_path / "gumbel(b={beta})_acc.png")

    print(gumbel_r.isf(qs / n_samples, beta))
