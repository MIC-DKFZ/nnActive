import json
import os
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from nnactive.nnunet.utils import convert_id_to_dataset_name
from nnactive.paths import get_nnActive_results


def load_results(filenames: list[Path]):
    out_list = []
    for filename in filenames:
        out_dict = dict()
        with open(filename, "r") as file:
            file_dict = json.load(file)
        out_dict["Mean Dice"] = file_dict["foreground_mean"]["Dice"]
        out_dict["Loop"] = int(filename.parent.name.split("_")[1])
        out_list.append(out_dict)
    return out_list


nnActive_results = get_nnActive_results()
if __name__ == "__main__":
    parser = ArgumentParser()
    # TODO: help
    parser.add_argument("-d", "--dataset_id", type=int)
    args = parser.parse_args()
    dataset_id = args.dataset_id
    experiment_results_path: Path = nnActive_results / convert_id_to_dataset_name(
        dataset_id
    )
    filenames = [fn for fn in experiment_results_path.rglob("summary.json")]
    filenames.sort()
    out_list = load_results(filenames)
    df = pd.DataFrame(out_list)
    fig, ax = plt.subplots()

    x = "Loop"
    y = "Mean Dice"
    sns.lineplot(data=df, x=x, y=y, ax=ax)
    plt.savefig(f"Dataset{dataset_id:03d}_DICE.pdf")
