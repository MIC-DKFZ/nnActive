from typing import Iterable

import matplotlib.pyplot as plt
import seaborn as sns

REMOVE_LIST = [" ", "_", "-"]


def create_unique_name(
    x_name: str, y_name: str, identifier: tuple | None, ignore_ident: list[int]
) -> str:
    ident_name = tuple([k for i, k in enumerate(identifier) if i not in ignore_ident])
    ident_name = f"{ident_name}"

    for rm_char in REMOVE_LIST:
        x_name = x_name.replace(rm_char, "")
        y_name = y_name.replace(rm_char, "")
        ident_name = ident_name.replace(rm_char, "")

    full_name = f"{y_name}-{x_name}__{ident_name}"[:250]
    return full_name


def plot_dataframe(
    axs,
    df,
    x_name: str,
    y_name: str,
    hue_key: str,
    plot_title: str,
    palette: dict = None,
    x_ticks: Iterable | None = None,
):
    axs = sns.lineplot(
        data=df,
        x=x_name,
        y=y_name,
        hue=hue_key,
        errorbar="sd",
        ax=axs,
        markers=True,
        palette=palette,
    )
    axs.set_ylabel(y_name)
    axs.set_xlabel(x_name)
    axs.legend(loc="best")
    axs.set_title(plot_title)
    if x_ticks is not None:
        axs.set_xticks(x_ticks)
    return axs
