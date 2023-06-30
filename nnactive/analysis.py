import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from nnactive.config.struct import ActiveConfig

sns.set_style("whitegrid")


def load_results(filenames: list[Path]) -> list[dict]:
    out_list = []
    for filename in filenames:
        out_dict = dict()
        with open(filename, "r") as file:
            file_dict = json.load(file)
        out_dict["Mean Dice"] = file_dict["foreground_mean"]["Dice"]
        out_dict["Loop"] = int(filename.parent.name.split("_")[1])
        out_list.append(out_dict)
    return out_list


def get_experiment_results(experiment_path: Path):
    filenames = [fn for fn in experiment_path.rglob("summary.json")]
    # make use of loop_XXX folder structure
    filenames.sort()
    dict_list = load_results(filenames)

    config = ActiveConfig.from_json(experiment_path / ActiveConfig.filename()).__dict__
    for dictval in dict_list:
        dictval["Experiment Name"] = experiment_path.name
        dictval.update(config)

    return dict_list


def compare_multi_experiment_results(base_path: Path):
    """WIP version to plot and combine results of multiple experiments.
    Plots results of the current experiments in current folder.

    Args:
        base_path (Path): $nnActive_results
    """
    experiment_vals = []
    for exp_path in base_path.iterdir():
        if exp_path.name.startswith("Dataset"):
            experiment_vals.extend(get_experiment_results(exp_path))
    df = pd.DataFrame(experiment_vals)

    # TODO: multiple different datasets
    # Currently this is not supported at all!

    skip_keys = [
        "Experiment Name",
        "seed",
        "num_processes",
        "Loop",
        "Mean Dice",
        "uncertainty",
    ]
    vals = [seperator for seperator in df.columns if seperator not in skip_keys]
    max_loop_ind = vals.index("query_steps")
    for key, df_g in df.groupby(vals):
        fig, axs = plt.subplots()
        sns.lineplot(
            data=df_g,
            x="Loop",
            y="Mean Dice",
            hue="uncertainty",
            errorbar="sd",
            ax=axs,
            markers="O",
        )
        axs.set_xticks(np.arange(0, key[max_loop_ind]))
        plt.savefig(f"{key}.png")
