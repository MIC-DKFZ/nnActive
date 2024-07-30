from __future__ import annotations

import os
from functools import cached_property
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

from nnactive.analyze.experiment_results import SingleExperimentResults
from nnactive.analyze.experiment_statistics import SingleExperimentStastistics
from nnactive.config.struct import Final
from nnactive.utils.io import load_json
from nnactive.utils.plot import create_unique_name, plot_dataframe
from nnactive.utils.pyutils import merge_dict_lists_on_indices

sns.set_style("whitegrid")

PALETTE = {
    "random": "tab:blue",
    "pred_entropy": "tab:green",
    "mutual_information": "tab:orange",
    "expected_dice": "tab:purple",
    "random-label": "tab:red",
    "random-label2": "tab:cyan",
    "power_bald": "tab:brown",
    "power_pe": "tab:gray",
    "softrank_bald": "tab:pink",
    "kmeans_bald": "tab:olive",
}


class MultiExperimentAnalysis:
    def __init__(
        self,
        base_results_path: Path,
        base_raw_path: Path | None = None,
        filter_final: bool = True,
    ):
        """Allows analysis of multiple experiments from a base_folder.
        Finding all subsequent folders containing results and aggregates and plots them.

        For in-depth analysis with statistics it requires info from $nnActive_raw/nnUNet_raw/DatasetXXX

        We do all work on a dataset level as performance metrics and statistics do change across datasets.
        e.g. amount of classes etc.
        Therefore to avoid dataframes with missing values etc. the dataframes are separately created for each dataset.
        Also experiment comparisons only make sense for the same dataset.

        Args:
            base_results_path (Path): Base_folder for analysis
            base_raw_path (Path | None, optional): Base_folder for Raw Data. Defaults to None.
            filter_final (bool, optional): Filter out based on final.json. Defaults to True.
        """
        self.base_results_path = base_results_path
        self.base_raw_path = base_raw_path
        self.filter_final = filter_final

    @cached_property
    def exp_results_paths(self):
        experiment_paths = [
            fn.parent for fn in self.base_results_path.rglob("config.json")
        ]
        if self.filter_final:
            experiment_paths = [
                exp_path
                for exp_path in experiment_paths
                if Final.from_json(exp_path / Final.filename()).final
            ]
        return experiment_paths

    @cached_property
    def exp_results(self) -> list[SingleExperimentResults]:
        exp_results = []
        for exp_path in self.exp_results_paths:
            single_exp = SingleExperimentResults(exp_path)
            if len(single_exp.results) == 0:
                print(f"Skippig Experiment in {exp_path} due to no results.")
                continue
            exp_results.append(single_exp)
        return exp_results

    @cached_property
    def exp_raw_paths(self):
        raw_paths = []
        for experiment in self.exp_results:
            rel_raw_path = str(experiment.experiment_path)[
                len(str(self.base_results_path)) + 1 :
            ]
            rel_raw_path = rel_raw_path.replace("nnActive_results/", "nnUNet_raw/")
            raw_paths.append(self.base_raw_path / rel_raw_path)
        return raw_paths

    @cached_property
    def exp_statistics(self) -> list[SingleExperimentStastistics]:
        exp_statistics = []
        for i, exp_path in enumerate(self.exp_raw_paths):
            single_stat = SingleExperimentStastistics(
                exp_path, self.exp_results[i].experiment_path
            )
            exp_statistics.append(single_stat)
        return exp_statistics

    @property
    def unique_datasets(self) -> set[int]:
        unique_dset = set([dset.config.base_id for dset in self.exp_results])
        return unique_dset

    @property
    def query_key(self) -> str:
        return "uncertainty"

    def plot_single_experiment(
        self,
        df_g: pd.DataFrame,
        y_name: str,
        x_name: str,
        dataset: str | None = None,
        x_ticks: Iterable | None = None,
        hline_printers: list[dict, Any] | None = None,
    ):
        fig, axs = plt.subplots()
        axs = plot_dataframe(
            axs,
            df_g,
            x_name,
            y_name,
            hue_key=self.query_key,
            plot_title=dataset,
            palette=PALETTE,
            x_ticks=x_ticks,
        )

        # add vertical line
        if hline_printers is not None:
            for y_full in hline_printers:
                axs.axhline(**y_full)
        return fig, axs

    def plot_experiment_overview(
        self,
        df: pd.DataFrame,
        selected_classes: list[int] | list[tuple[int]] | None = None,
        horizontal_lines: dict[str, Any] | None = None,
        x_axis_dict: dict[str, Any] | None = None,
    ):
        n_rows, n_cols = 3, 9
        n_performance_cols = 3
        plot_size = 4
        if selected_classes is None:
            selected_classes = [
                int(i.split(" ")[1]) for i in df.columns if i.startswith("Class")
            ][:3]
            while len(selected_classes) < n_performance_cols:
                selected_classes.append(None)

        cols = [[] for _ in range(n_cols)]
        col_num = 0
        # fill first 4 columns
        col0_x = [
            "#Patches",
            "percentage_of_voxels_foreground",
            "avg_percentage_of_voxels_fg_cls",
        ]
        for x_n in col0_x:
            cols[col_num].append({"x_name": x_n, "y_name": "Mean Dice"})

        for i, cls_index in enumerate(selected_classes):
            col_num += 1
            if cls_index is None:
                x_names = [None] * n_rows
                y_names = [None] * n_rows
            else:
                y_names = [f"Class {cls_index} Dice"] * n_rows
                x_names = [
                    "#Patches",
                    f"percentage_of_voxels_per_cls_{cls_index[0]}",
                    "avg_percentage_of_voxels_fg_cls",
                ]

            cols[col_num].extend(
                [{"x_name": x_n, "y_name": y_n} for x_n, y_n in zip(x_names, y_names)]
            )

        # fill last 4 columns
        x_names = [
            "percentage_of_num_voxels",
            "percentage_of_num_voxels",
            "#Patches",
        ]
        y_names = [
            "percentage_of_voxels_foreground",
            "avg_percentage_of_voxels_fg_cls",
            "patches_foreground",
        ]
        col_num += 1
        for x_n, y_n in zip(x_names, y_names):
            cols[col_num].append({"x_name": x_n, "y_name": y_n})
        for i, cls_index in enumerate(selected_classes):
            col_num += 1
            if cls_index is None:
                x_names = [None] * n_rows
                y_names = [None] * n_rows
            else:
                x_names = [
                    "percentage_of_num_voxels",
                    None,
                    "#Patches",
                ]
                y_names = [
                    f"percentage_of_voxels_per_cls_{cls_index[0]}",
                    None,
                    f"patches_per_cls_{cls_index[0]}",
                ]
            cols[col_num].extend(
                [{"x_name": x_n, "y_name": y_n} for x_n, y_n in zip(x_names, y_names)]
            )

        x_names = [
            "Loop",
            "#Patches",
            "#Patches",
        ]
        y_names = [
            "percentage_of_voxel_percentage_foreground",
            "percentage_of_num_unique_files",
            "num_unique_files",
        ]
        col_num += 1
        for x_n, y_n in zip(x_names, y_names):
            cols[col_num].append({"x_name": x_n, "y_name": y_n})

        fig, axs = plt.subplots(
            nrows=n_rows, ncols=n_cols, figsize=(n_cols * plot_size, n_rows * plot_size)
        )

        for i, j in product(range(n_rows), range(n_cols)):
            x_name, y_name = cols[j][i]["x_name"], cols[j][i]["y_name"]
            if x_name is None or y_name is None:
                axs[i, j].set_axis_off()
                continue
            if x_name in x_axis_dict:
                x_kwargs = x_axis_dict[x_name]
            else:
                x_kwargs = {}

            axs[i, j] = plot_dataframe(
                axs[i, j],
                df,
                x_name,
                y_name,
                hue_key=self.query_key,
                palette=PALETTE,
                legend=None,
                **x_kwargs,
            )
            if y_name in horizontal_lines:
                hline_printers = horizontal_lines[y_name]
                for y_full in hline_printers:
                    axs[i, j].axhline(**y_full)
        handles, labels = axs[0][0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            # loc="lower center",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=8,
        )
        fig.tight_layout()
        for i, j in product(range(n_rows), range(n_cols)):
            legend = axs[i, j].get_legend()
            if legend is not None:
                legend.remove()

        return fig, axs

    def dataset_analyze_performance(
        self, unique_id: int, all_plots: bool = True, output_dir: Path = Path(".")
    ):
        dataset_results = [
            exp for exp in self.exp_results if exp.config.base_id == unique_id
        ]
        value = "Dice"

        df, vals = self.create_df(dataset_results, value)

        y_full_dict = dataset_results[0].to_full_dataset_performance_dict(value)

        output_dir = output_dir / dataset_results[0].config.dataset
        if not output_dir.is_dir():
            os.makedirs(output_dir)

        if all_plots:
            y_names = dataset_results[0].get_value_dict(plot_val=value).keys()
        else:
            y_names = ["Mean Dice"]

        max_loop_ind = vals.index("query_steps")
        dataset_ind = vals.index("dataset")
        sb_ind = vals.index("starting_budget_size")
        qs_ind = vals.index("query_size")
        pre_suffix_ind = vals.index("pre_suffix")

        # create plots for each unique setting of the respective dataset
        for key, df_g in df.groupby(vals):
            save_dir: Path = output_dir / key[pre_suffix_ind][2:] / "performance"
            if not save_dir.is_dir():
                os.makedirs(save_dir)
            dataset = key[dataset_ind]
            x_name_dict = {
                "Loop": {"x_ticks": np.arange(0, key[max_loop_ind] + 1)},
                "#Patches": {
                    "x_ticks": np.arange(
                        key[sb_ind],
                        key[sb_ind] + key[qs_ind] * (key[max_loop_ind] + 1),
                        key[qs_ind],
                    )
                },
            }
            for y_name, x_name in product(y_names, x_name_dict):
                fig, axs = self.plot_single_experiment(
                    df_g,
                    y_name,
                    x_name,
                    dataset,
                    hline_printers=y_full_dict[y_name],
                    **x_name_dict[x_name],
                )
                file_name = create_unique_name(
                    x_name, y_name, key, [dataset_ind, pre_suffix_ind]
                )

                plt.savefig(save_dir / f"{file_name}.png")
                plt.close("all")

    def create_df(self, dataset_results, value):
        df_results_dicts: list[dict] = []
        for exp in dataset_results:
            df_exp_dict, exp_skip_keys = exp.to_df_row_dicts(value)
            df_results_dicts.extend(df_exp_dict)

        df = pd.DataFrame(df_results_dicts)
        vals = [seperator for seperator in df.columns if seperator not in exp_skip_keys]
        return df, vals

    def dataset_analyze_statistics(
        self, unique_id: int, all_plots: bool = True, output_dir: Path = Path(".")
    ):
        if not output_dir.is_dir():
            os.makedirs(output_dir)
        dataset_statistics = [
            exp for exp in self.exp_statistics if exp.base_id == unique_id
        ]

        # how to get pre_suffix?
        output_dir = output_dir / dataset_statistics[0].source_dataset_path.name

        if all_plots:
            y_names = dataset_statistics[0].plot_vals
        else:
            y_names = ["percentage_of_voxels_foreground"]

        df_row_dicts = []
        for exp in dataset_statistics:
            df_row_dict, skip_keys = exp.to_df_row_dicts()
            df_row_dicts.extend(df_row_dict)

        df = pd.DataFrame(df_row_dicts)

        vals = [seperator for seperator in df.columns if seperator not in skip_keys]
        max_loop_ind = vals.index("query_steps")
        dataset_ind = vals.index("dataset")
        pre_suffix_ind = vals.index("pre_suffix")

        # create plots for each unique setting of the respective dataset
        for key, df_g in df.groupby(vals):
            save_dir: Path = output_dir / key[pre_suffix_ind][2:] / "statistics"
            if not save_dir.is_dir():
                os.makedirs(save_dir)
            dataset = key[dataset_ind]
            x_name_dict = {"Loop": {"x_ticks": np.arange(0, key[max_loop_ind] + 1)}}
            for y_name, x_name in product(y_names, x_name_dict):

                fig, axs = self.plot_single_experiment(
                    df_g,
                    y_name,
                    x_name,
                    dataset,
                    **x_name_dict[x_name],
                )
                file_name = create_unique_name(
                    x_name, y_name, key, [dataset_ind, pre_suffix_ind]
                )

                plt.savefig(save_dir / f"{file_name}.png")
                plt.close("all")

    def dataset_analyze_statistics_results(
        self,
        unique_id: int,
        all_plots: bool = True,
        output_dir: Path = Path("."),
        value: str = "Dice",
    ):
        dataset_statistics = [
            exp for exp in self.exp_statistics if exp.base_id == unique_id
        ]
        dataset_results = [
            exp for exp in self.exp_results if exp.config.base_id == unique_id
        ]

        output_dir = output_dir / dataset_results[0].config.dataset
        if not output_dir.is_dir():
            os.makedirs(output_dir)

        df_stat_dicts: list[dict] = []
        for exp in dataset_statistics:
            df_stat_dict, stat_skip_keys = exp.to_df_row_dicts()
            df_stat_dicts.extend(df_stat_dict)

        df_results_dicts: list[dict] = []
        for exp in dataset_results:
            df_exp_dict, exp_skip_keys = exp.to_df_row_dicts(value)
            df_results_dicts.extend(df_exp_dict)

        indices = ["Experiment", "Loop"]
        merged_dicts = merge_dict_lists_on_indices(
            df_results_dicts, df_stat_dicts, indices
        )

        df = pd.DataFrame(merged_dicts)

        vals = [
            seperator
            for seperator in df.columns
            if seperator not in (exp_skip_keys + stat_skip_keys)
        ]

        dataset_ind = vals.index("dataset")

        if all_plots:
            x_names = dataset_statistics[0].plot_vals
            y_names = dataset_results[0].get_value_dict(plot_val=value).keys()
        else:
            x_names = ["percentage_of_voxels_foreground"]
            y_names = ["Mean Dice"]

        y_full_dict = dataset_results[0].to_full_dataset_performance_dict(value)

        pre_suffix_ind = vals.index("pre_suffix")
        max_loop_ind = vals.index("query_steps")
        dataset_ind = vals.index("dataset")
        qs_ind = vals.index("query_size")
        sb_ind = vals.index("starting_budget_size")
        pre_suffix_ind = vals.index("pre_suffix")
        for key, df_g in df.groupby(vals):
            # create plots for each unique setting of the respective dataset
            save_dir: Path = output_dir / key[pre_suffix_ind][2:] / "result_statistics"
            if not save_dir.is_dir():
                os.makedirs(save_dir)
            dataset = key[dataset_ind]

            selected_classes = None
            if dataset == "Dataset216_AMOS2022_task1":
                selected_classes = [1, 13, 15]
            if dataset == "Dataset137_BraTS2021":
                selected_classes = [(1, 2, 3), (2, 3), (3,)]

            x_name_dict = {
                "Loop": {"x_ticks": np.arange(0, key[max_loop_ind] + 1)},
                "#Patches": {
                    "x_ticks": np.arange(
                        key[sb_ind],
                        key[sb_ind] + key[qs_ind] * (key[max_loop_ind] + 1),
                        key[qs_ind],
                    )
                },
            }

            fig, axs = self.plot_experiment_overview(
                df_g,
                selected_classes=selected_classes,
                horizontal_lines=y_full_dict,
                x_axis_dict=x_name_dict,
            )
            fig.suptitle(
                dataset,
                y=1.05,
            )
            filename = "overview.png"
            plt.savefig(save_dir.parent / filename, bbox_inches="tight")
            plt.close("all")

            x_name_dict = {x_n: {} for x_n in x_names}
            for x_name, y_name in product(x_name_dict, y_names):
                # create plots for each value to be plotted

                fig, axs = self.plot_single_experiment(
                    df_g,
                    y_name,
                    x_name,
                    dataset,
                    hline_printers=y_full_dict[y_name],
                    **x_name_dict[x_name],
                )

                file_name = create_unique_name(
                    x_name, y_name, key, [dataset_ind, pre_suffix_ind]
                )

                plot_name_file = file_name.split("-")[0]  # y_name
                save_dir_final = save_dir / plot_name_file
                if not save_dir_final.is_dir():
                    os.makedirs(save_dir_final)

                plt.savefig(save_dir_final / f"{file_name}.png", bbox_inches="tight")
                plt.close("all")

    def analyze_multi_datasets(
        self,
        output_dir: Path = Path("."),
        all_results_plots: bool = True,
        all_raw_plots: bool = True,
        all_combi_plots: bool = True,
    ):
        for dataset_id in self.unique_datasets:
            logger.info(
                f"Analyzing results for experiments derived from dataset id {dataset_id}"
            )
            self.dataset_analyze_performance(
                unique_id=dataset_id, all_plots=all_results_plots, output_dir=output_dir
            )
            self.dataset_analyze_statistics(
                unique_id=dataset_id, all_plots=all_raw_plots, output_dir=output_dir
            )
            self.dataset_analyze_statistics_results(
                unique_id=dataset_id, all_plots=all_combi_plots, output_dir=output_dir
            )


def analyze_multi_experiment_results(
    base_path: Path,
    base_raw_path: Path | None,
    filter_final: bool = True,
    all_plots: bool = True,
    output_dir: bool = Path("."),
):
    """Analyze Experiments return a multi folder structure contatining plots for performance,
    query statistics and performance vs. query statistics.

    Args:
        base_path (Path): path containing nnActive_results
        base_raw_path (Path | None): path containing nnActive_data
        filter_final (bool, optional): filtering. Defaults to True.
        all_plots (bool, optional): create all plots or only subset. Defaults to True.
        output_dir (bool, optional): where to save output images. Defaults to Path(".").
    """
    analysis = MultiExperimentAnalysis(
        base_results_path=base_path,
        base_raw_path=base_raw_path,
        filter_final=filter_final,
    )
    analysis.analyze_multi_datasets(
        output_dir=output_dir,
        all_results_plots=all_plots,
        all_raw_plots=all_plots,
        all_combi_plots=all_plots,
    )

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
                        + df_seed.sort_values(by=["Loop"]).reset_index()["Mean Dice"]
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


if __name__ == "__main__":
    # exp_path = "/home/c817h/network/cluster-data/Dataset004_Hippocampus/nnUNet_raw/Dataset005_Hippocampus__patch-20_20_20__qs-20__unc-mutual_information__seed-12345"

    # exp_path = "/home/c817h/network/cluster-data/Dataset135_KiTS2021/nnUNet_raw/Dataset001_KiTS2021__patch-64_64_64__qs-20__unc-random-label__seed-12345"
    # exp_path = Path(exp_path)
    # statistics = SingleExperimentStastistics(exp_path)
    # output_path = Path(__file__).parent.parent / "results" / "raw_plots"

    # statistics.plot_experiment(output_path=output_path)

    # full_data_stat = statistics.full_data_statistic

    base_path = "/home/c817h/network/cluster-data"
    unique_id = 137

    base_path = Path(base_path)
    analysis = MultiExperimentAnalysis(base_path, base_path)

    output_path = Path(__file__).parent.parent / "results" / "allplot"
    analysis.analyze_multi_datasets(output_dir=output_path)

    # output_path = (
    #     Path(__file__).parent.parent / "results" / "raw_plots" / str(unique_id)
    # )
    # analysis.dataset_analyze_statistics(unique_id, output_dir=output_path)

    # output_path = (
    #     Path(__file__).parent.parent / "results" / "results_plots" / str(unique_id)
    # )
    # analysis.dataset_analyze_performance(unique_id=unique_id, output_dir=output_path)

    # output_path = (
    #     Path(__file__).parent.parent / "results" / "resraw_plots" / str(unique_id)
    # )
    # analysis.dataset_analyze_statistics_results(
    #     unique_id=unique_id, output_dir=output_path
    # )
