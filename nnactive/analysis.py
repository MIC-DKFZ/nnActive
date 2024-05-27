import json
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from nnactive.config.struct import ActiveConfig, Final
from nnactive.utils.io import load_json

sns.set_style("whitegrid")

PALETTE = {
    "random": "tab:blue",
    "pred_entropy": "tab:green",
    "mutual_information": "tab:orange",
    # "other_1": "tab:purple",
    "random-label": "tab:red",
    # "other_3": "tab:cyan",
}

DATASET_PERFORMANCES = []
for result in (Path(__file__).parent.parent / "full_data_results").iterdir():
    if result.suffix == ".json":
        with open(result, "r") as file:
            summary = load_json(result)
            summary["Dataset"] = result.name.split("__")[0]
            summary["Trainer"] = result.name.split("__")[1].split(".")[0]
            DATASET_PERFORMANCES.append(summary)


FULL_DATASET_PERFORMANCE = {
    "Dataset004_Hippocampus": 0.895,
}

for dataset_summary in DATASET_PERFORMANCES:
    FULL_DATASET_PERFORMANCE[dataset_summary["Dataset"]] = dataset_summary[
        "foreground_mean"
    ]["Dice"]


def load_results(filenames: list[Path]) -> list[dict]:
    out_list = []
    for filename in filenames:
        out_dict = dict()
        with open(filename, "r") as file:
            file_dict = json.load(file)
        out_dict["Mean Dice"] = file_dict["foreground_mean"]["Dice"]
        # for cls in file_dict["mean"]:
        #     out_dict[f"Class Dice {cls}"] = file_dict["mean"][cls]["Dice"]

        out_dict["Loop"] = int(filename.parent.name.split("_")[1])
        out_list.append(out_dict)
    return out_list


def get_experiment_results(experiment_path: Path, filter=True) -> list[dict]:
    # check that summary.jsons are read from loop_XXX in results format
    filenames = [
        fn for fn in experiment_path.rglob("summary.json") if "loop_" in fn.__str__()
    ]
    # make use of loop_XXX folder structure
    filenames.sort()
    dict_list = load_results(filenames)

    config_item = ActiveConfig.from_json(experiment_path / ActiveConfig.filename())
    config_dict = config_item.to_dict()

    final_item = Final.from_json(experiment_path / Final.filename())
    # final_dict = final_item.to_dict()
    if filter:
        if final_item.final is False:
            print(f"Skipping Experiment: {experiment_path.name}")
            return []

    for dictval in dict_list:
        dictval["Experiment Name"] = experiment_path.name
        dictval.update(config_dict)
        # dictval.update(final_dict)

    return dict_list


def compare_multi_experiment_results(
    base_path: Path, base_dataset_id: Union[int, None] = None
):
    """WIP version to plot and combine results of multiple experiments.
    Plots results of the current experiments in current folder.

    Args:
        base_path (Path): $nnActive_results
    """
    experiment_vals = []

    experiment_paths = [fn.parent for fn in base_path.rglob("config.json")]

    # for exp_path in base_path.iterdir():
    #     if exp_path.name.startswith("Dataset"):
    for exp_path in experiment_paths:
        experiment_vals.extend(get_experiment_results(exp_path))
    df = pd.DataFrame(experiment_vals)
    if base_dataset_id:
        df = (
            df[df["dataset"].str.startswith(f"Dataset{base_dataset_id:03d}")]
            .reset_index()
            .drop("index", axis=1)
        )

    df["#Patches"] = (df["Loop"]) * df["query_size"] + df["starting_budget_size"]

    skip_keys = [
        "Experiment Name",
        "seed",
        "num_processes",
        "Loop",
        "Mean Dice",
        "uncertainty",
        "#Patches",
    ]
    query_key = "uncertainty"
    vals = [seperator for seperator in df.columns if seperator not in skip_keys]
    max_loop_ind = vals.index("query_steps")
    dataset_ind = vals.index("dataset")
    sb_ind = vals.index("starting_budget_size")
    qs_ind = vals.index("query_size")
    for key, df_g in df.groupby(vals):
        dataset = key[dataset_ind]
        fig, axs = plt.subplots()
        sns.lineplot(
            data=df_g,
            x="Loop",
            y="Mean Dice",
            hue=query_key,
            errorbar="sd",
            ax=axs,
            markers=True,
            palette=PALETTE,
        )

        if dataset in FULL_DATASET_PERFORMANCE:
            axs.axhline(
                y=FULL_DATASET_PERFORMANCE[dataset],
                label="Full Data Performance",
                linestyle="-",
                color="black",
            )
        axs.set_xticks(np.arange(0, key[max_loop_ind]))
        axs.legend(loc="best")
        axs.set_title(dataset)
        key_plot = tuple([k for i, k in enumerate(key) if i != dataset_ind])
        plt.savefig(f"Performance-{dataset}__{key_plot}.png")

        fig, axs = plt.subplots()
        sns.lineplot(
            data=df_g,
            x="#Patches",
            y="Mean Dice",
            hue=query_key,
            errorbar="sd",
            ax=axs,
            markers="O",
            palette=PALETTE,
        )

        # # Value for Hippocampus Dataset
        if dataset in FULL_DATASET_PERFORMANCE:
            axs.axhline(
                y=FULL_DATASET_PERFORMANCE[dataset],
                label="Full Data Performance",
                linestyle="-",
                color="black",
            )

        axs.set_xticks(
            np.arange(
                key[sb_ind],
                key[sb_ind] + (key[qs_ind] * key[max_loop_ind]),
                key[qs_ind],
            )
        )
        # axs.axhline(y=0.895, label="Ful Data Performance", linestyle="-", color="black")
        # axs.set_ylim(0.84, 0.90)
        # axs.set_xlim(10, 200)
        # # axs.set_xticks(np.arange(0, key[max_loop_ind]))
        # axs.legend(loc="best")
        plt.savefig(f"PerformancePatch-{dataset}__{key_plot}.png")

        ### Label Efficency Plot starts here

        label_eff_plot = []

        # try:
        if False:
            #######################
            # version for each random
            #######################
            # df_g_random = df_g[df_g[query_key] == "random"]
            # for val, df_g_unc in df_g.groupby(query_key):
            #     if val == "random":
            #         continue

            #     # version for each random select best query
            #     for index, row in df_g_random.iterrows():
            #         label_efficency = (row["Loop"] + 1) / (
            #             min(df_g_unc[df_g_unc["Mean Dice"] >= row["Mean Dice"]]["Loop"])
            #             + 1
            #         )
            #         label_eff_plot.append(
            #             {
            #                 "Label Efficiency": label_efficency,
            #                 "Mean Dice": row["Mean Dice"],
            #                 query_key: val,
            #             }
            #         )

            # version for each random select one for each seed of query
            # for seed, df_g_unc_seed in df_g_unc.groupby("seed"):
            #     for index, row in df_g_random.iterrows():
            #         try:
            #             label_efficency = (row["Loop"] + 1) / (
            #                 min(
            #                     df_g_unc_seed[
            #                         df_g_unc_seed["Mean Dice"] >= row["Mean Dice"]
            #                     ]["Loop"]
            #                 )
            #                 + 1
            #             )
            #             label_eff_plot.append(
            #                 {
            #                     "Label Efficiency": label_efficency,
            #                     "Mean Dice": row["Mean Dice"],
            #                     query_key: val,
            #                     "seed": seed,
            #                 }
            #             )
            #         except:
            #             pass

            # version for means
            out_dfs = dict()
            for query, df_g_query in df_g.groupby(query_key):
                print(query)
                count = 0
                for seed, df_seed in df_g_query.groupby("seed"):
                    print(f"Seed {seed}: len DataFrame {len(df_seed)}")
                    if len(df_seed) < df_seed["query_steps"].unique()[0]:
                        continue
                    if count == 0:
                        df_g_mean = df_seed.sort_values(by=["Loop"]).reset_index()
                    else:
                        df_g_mean["Mean Dice"] = (
                            df_g_mean["Mean Dice"]
                            + df_seed.sort_values(by=["Loop"]).reset_index()[
                                "Mean Dice"
                            ]
                        )
                    count += 1
                df_g_mean["Mean Dice"] = df_g_mean["Mean Dice"] / count
                out_dfs[query] = df_g_mean

            for val, df_g_unc in out_dfs.items():
                if val == "random":
                    continue

                # version for each random select best query
                for index, row in out_dfs["random"].iterrows():
                    try:
                        label_efficency = (row["Loop"] + 1) / (
                            min(
                                df_g_unc[
                                    df_g_unc["Mean Dice"] >= (row["Mean Dice"] - 0.0002)
                                ]["Loop"]
                            )
                            + 1
                        )
                        label_eff_plot.append(
                            {
                                "Label Efficiency": label_efficency,
                                "Mean Dice": row["Mean Dice"],
                                query_key: val,
                            }
                        )
                    except:
                        pass
            fig, axs = plt.subplots()

            sns.lineplot(
                data=pd.DataFrame(label_eff_plot),
                x="Mean Dice",
                y="Label Efficiency",
                hue="uncertainty",
                errorbar="sd",
                ax=axs,
                markers="O",
                palette=PALETTE,
            )
            plt.savefig(f"Efficiency__{key}.png")
        # except Exception as exception:
        #     print(f"No Label Efficency Plot for Setting {key}")
        #     print("Error Message:")
        #     print(exception)
