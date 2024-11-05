from typing import Iterable

import matplotlib.pyplot as plt
import seaborn as sns


def plot_dataframe(
    axs,
    df,
    x_name: str,
    y_name: str,
    hue_key: str,
    plot_title: str | None = None,
    palette: dict | None = None,
    x_ticks: Iterable | None = None,
    legend: str | None = "best",
    style: str | None = None,
):
    axs = sns.lineplot(
        data=df,
        x=x_name,
        y=y_name,
        hue=hue_key,
        errorbar="sd",
        style=style,
        ax=axs,
        markers=True,
        palette=palette,
    )
    axs.set_ylabel(y_name)
    axs.set_xlabel(x_name)
    if legend is not None:
        axs.legend(loc=legend)
    if plot_title is not None:
        axs.set_title(plot_title)
    if x_ticks is not None:
        axs.set_xticks(x_ticks)
    return axs
