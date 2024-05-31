import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Union

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
    "expected_dice": "tab:purple",
    "random-label": "tab:red",
    # "other_3": "tab:cyan",
}
FULL_LINESTYLE = ["-", "- "]

DATASET_PERFORMANCES = []
for result in (Path(__file__).parent.parent / "full_data_results").iterdir():
    if result.suffix == ".json":
        with open(result, "r") as file:
            summary = load_json(result)
            summary["Dataset"] = result.name.split("__")[0]
            summary["Trainer"] = result.name.split("__")[1].split(".")[0]
            DATASET_PERFORMANCES.append(summary)


class MeanValue:
    def __init__(self, value: str):
        self.value = value

    def get_from_dict(self, file_dict: dict):
        return file_dict["foreground_mean"]["Dice"]

    @property
    def name(self):
        return "Mean Dice"


class ClassValue:
    def __init__(self, value: str, cls: Any):
        self.value = value
        self.cls = cls

    def get_from_dict(self, file_dict: dict):
        return file_dict["mean"][self.cls][self.value]

    @property
    def name(self):
        return f"Class {self.value} {self.cls}"


class SingleExperimentResults:
    def __init__(self, experiment_path):
        self.experiment_path = experiment_path

    @property
    def summary_files(self) -> list[Path]:
        filenames = [
            fn
            for fn in self.experiment_path.rglob("summary.json")
            if "loop_" in fn.__str__()
        ]
        filenames.sort()
        return filenames

    @property
    def config(self) -> ActiveConfig:
        return ActiveConfig.from_json(self.experiment_path / ActiveConfig.filename())

    @property
    def results(self):
        out_results = []
        for summary_file in self.summary_files:
            temp_dict = {}
            temp_dict["summary"] = load_json(summary_file)
            temp_dict["Loop"] = int(summary_file.parent.name.split("_")[1])
            temp_dict["#Patches"] = (
                temp_dict["Loop"] * self.config.query_size
                + self.config.starting_budget_size
            )
            temp_dict["Experiment"] = self.experiment_path.name
            out_results.append(temp_dict)
        return out_results

    # TODO: retrieve class names from dataset.json
    @property
    def label_names(self) -> dict[str, int]:
        return {}

    @property
    def plot_name(self) -> str:
        return "plot_val"

    @property
    def plot_skip_keys(self):
        skip_keys = [
            self.plot_name,
            "Experiment",  # Possibly Experiment Name
            "seed",
            "Loop",
            "uncertainty",
            "#Patches",
        ]
        return skip_keys

    def value_dict(self, plot_val: str = "Dice"):
        # better to do this with classes for names of plots
        plot_dict = {f"Mean {plot_val}": MeanValue(plot_val)}
        for cls in self.results[0]["summary"]["mean"]:
            # use deepcopy here as otherwise cls is changed in lambda function
            plot_dict[f"Class {cls} {plot_val}"] = ClassValue(plot_val, cls)
        return plot_dict

    def to_df_row_dicts(self, plot_fct=lambda x: x["foreground_mean"]["Dice"]):
        out = []
        for result in self.results:
            append_dict = {}
            for k in result:
                if k != "summary":
                    append_dict[k] = result[k]
            append_dict[self.plot_name] = plot_fct(result["summary"])
            append_dict.update(self.config.to_dict())
            out.append(append_dict)
        return out

    def full_dataset_performance(self, plot_fct=lambda x: x["foreground_mean"]["Dice"]):
        y_fulls = []
        for dataset_performance in DATASET_PERFORMANCES:
            if dataset_performance["Dataset"] == self.config.dataset:
                y_fulls.append(
                    {
                        "y": plot_fct(dataset_performance),
                        "label": "{} full dataset performance".format(
                            dataset_performance["Trainer"]
                        ),
                        "linestyle": FULL_LINESTYLE[len(y_fulls)],
                        "color": "black",
                    }
                )
        return y_fulls


def compare_multi_experiment_results(
    base_path: Path,
    base_dataset_id: Union[int, None] = None,
    filter_final: bool = True,
    all_plots: bool = True,
):
    """WIP version to plot and combine results of multiple experiments.
    Plots results of the current experiments in current folder.

    Args:
        base_path (Path): $nnActive_results
    """
    experiment_paths = [fn.parent for fn in base_path.rglob("config.json")]
    if filter_final:
        experiment_paths = [
            exp_path
            for exp_path in experiment_paths
            if Final.from_json(exp_path / Final.filename()).final
        ]

    unique_datasets = set()
    experiments: list[SingleExperimentResults] = []
    for exp_path in experiment_paths:
        single_exp = SingleExperimentResults(exp_path)
        experiments.append(single_exp)
        unique_datasets.add(single_exp.config.base_id)

    for unique_id in unique_datasets:
        dataset_experiments = [
            exp
            for exp in experiments
            if exp.config.base_id == unique_id and len(exp.results) > 0
        ]
        value = "Dice"
        plot_name = "Mean Dice"
        if all_plots:
            plot_names = dataset_experiments[0].value_dict(plot_val=value).keys()
        else:
            plot_names = ["Mean Dice"]

        for plot_name in plot_names:
            plot_fct = dataset_experiments[0].value_dict(plot_val=value)[plot_name]
            y_fulls = dataset_experiments[0].full_dataset_performance(
                plot_fct.get_from_dict
            )
            # print(unique_id)
            # print(plot_name)
            # print(y_fulls)
            # print("--")
            # for plot_val in plot_vals...
            df_row_dicts = []
            for exp in dataset_experiments:
                df_row_dicts.extend(
                    exp.to_df_row_dicts(plot_fct=plot_fct.get_from_dict)
                )

            df = pd.DataFrame(df_row_dicts)

            query_key = "uncertainty"
            vals = [
                seperator
                for seperator in df.columns
                if seperator not in dataset_experiments[0].plot_skip_keys
            ]
            max_loop_ind = vals.index("query_steps")
            dataset_ind = vals.index("dataset")
            sb_ind = vals.index("starting_budget_size")
            qs_ind = vals.index("query_size")

            # create plots for each unique setting for the respective dataset now
            for key, df_g in df.groupby(vals):
                dataset = key[dataset_ind]
                x = "Loop"
                x_name = "Loop"

                fig, axs = plt.subplots()
                sns.lineplot(
                    data=df_g,
                    x=x,
                    y=dataset_experiments[0].plot_name,
                    hue=query_key,
                    errorbar="sd",
                    ax=axs,
                    markers=True,
                    palette=PALETTE,
                )
                axs.set_ylabel(plot_name)
                for y_full in y_fulls:
                    axs.axhline(**y_full)
                axs.set_xticks(np.arange(0, key[max_loop_ind]))
                axs.legend(loc="best")
                axs.set_title(dataset)
                key_plot = tuple([k for i, k in enumerate(key) if i != dataset_ind])
                key_plot_file = f"{key_plot}".replace(" ", "")
                plot_name_file = plot_name.replace(" ", "")

                plt.savefig(f"{dataset}-{plot_name_file}-{x_name}__{key_plot_file}.png")

                x = "#Patches"
                x_name = "Patches"
                fig, axs = plt.subplots()
                sns.lineplot(
                    data=df_g,
                    x=x,
                    y=dataset_experiments[0].plot_name,
                    hue=query_key,
                    errorbar="sd",
                    ax=axs,
                    markers=True,
                    palette=PALETTE,
                )
                axs.set_ylabel(plot_name)

                for y_full in y_fulls:
                    axs.axhline(**y_full)

                axs.set_xticks(
                    np.arange(
                        key[sb_ind],
                        key[sb_ind] + (key[qs_ind] * key[max_loop_ind]),
                        key[qs_ind],
                    )
                )
                axs.set_title(dataset)
                axs.legend(loc="best")
                plt.savefig(f"{dataset}-{plot_name_file}-{x_name}__{key_plot_file}.png")
                plt.close("all")

        # #     ### Label Efficency Plot starts here

        # #     label_eff_plot = []

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
