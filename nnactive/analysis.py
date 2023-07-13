import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from nnactive.config.struct import ActiveConfig

sns.set_style("whitegrid")

PALETTE = {
    "random": "tab:blue",
    "pred_entropy": "tab:green",
    "mutual_information": "tab:orange",
    # "other_1": "tab:purple",
    # "other_2": "tab:red",
    # "other_3": "tab:cyan",
}


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
    query_key = "uncertainty"
    vals = [seperator for seperator in df.columns if seperator not in skip_keys]
    max_loop_ind = vals.index("query_steps")
    for key, df_g in df.groupby(vals):
        fig, axs = plt.subplots()
        sns.lineplot(
            data=df_g,
            x="Loop",
            y="Mean Dice",
            hue=query_key,
            errorbar="sd",
            ax=axs,
            markers="O",
        )

        # Value for Hippocampus Dataset
        axs.axhline(y=0.895, label="Ful Data Performance", linestyle="-", color="black")
        axs.set_xticks(np.arange(0, key[max_loop_ind]))
        axs.legend(loc="best")
        plt.savefig(f"Performance__{key}.png")

        ### Label Efficency Plot starts here

        label_eff_plot = []

        # try:
        if True:
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
                                    df_g_unc["Mean Dice"] >= (row["Mean Dice"] - 0.002)
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
