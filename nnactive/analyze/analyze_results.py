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

from nnactive.analyze.analysis import SettingAnalysis
from nnactive.analyze.experiment_results import SingleExperimentResults
from nnactive.analyze.experiment_statistics import SingleExperimentStastistics
from nnactive.config.struct import Final
from nnactive.utils.io import load_json, save_df_to_txt
from nnactive.utils.pyutils import create_string_identifier, merge_dict_lists_on_indices

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

SELECTED_CLASSES_OVERVIEW = {
    "Dataset216_AMOS2022_task1": [1, 13, 15],
    "Dataset137_BraTS2021": [(1, 2, 3), (2, 3), (3,)],
}

REMOVE_IND_COLS_SETTING_KEY = [
    "pre_suffix",
    "base_id",
    "dataset",
    "model_plans",
    "model_config",
    "train_folds",
    "starting_budget",
]


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
            fn.parent for fn in self.base_results_path.rglob("*/config.json")
        ]
        logger.debug(f"Found {len(experiment_paths)} experiments.")
        if self.filter_final:
            experiment_paths = [
                exp_path
                for exp_path in experiment_paths
                if Final.from_json(exp_path / Final.filename()).final
            ]

        # Filter out debug and prototype runs
        experiment_paths = [
            e
            for e in experiment_paths
            if not "DEBUG" in str(e)
            and not "PROTOTYPE" in str(e)
            and not "small" in str(e)
        ]

        return experiment_paths

    @cached_property
    def exp_results(self) -> list[SingleExperimentResults]:
        """Returns list of SingleExperimentResults for all experiments in base_results_path.
        Skips experiments with no results.
        """
        exp_results = []
        for exp_path in self.exp_results_paths:
            single_exp = SingleExperimentResults(exp_path)
            if len(single_exp.results) == 0:
                print(f"Skippig Experiment in {exp_path} due to no results.")
                continue
            exp_results.append(single_exp)
        return exp_results

    @cached_property
    def exp_raw_paths(self) -> list[Path]:
        """Returns list of paths to raw data for all experiments in base_results_path."""
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

    @property
    def merge_keys(self) -> list[str]:
        return ["Experiment", "Loop"]

    def create_results_df(
        self, dataset_results: list[SingleExperimentResults], value: str | None
    ) -> tuple[pd.DataFrame, list[str]]:
        df_results_dicts: list[dict] = []
        for exp in dataset_results:
            df_exp_dict, exp_skip_keys = exp.to_df_row_dicts(value)
            df_results_dicts.extend(df_exp_dict)

        df = pd.DataFrame(df_results_dicts)
        df = self.ensure_df_elt_hashable(df)

        return df, exp_skip_keys

    def create_statistics_df(
        self, dataset_statistics: list[SingleExperimentStastistics]
    ) -> tuple[pd.DataFrame, list[str]]:
        df_row_dicts: list[dict] = []
        for exp in dataset_statistics:
            df_row_dict, skip_keys = exp.to_df_row_dicts()
            df_row_dicts.extend(df_row_dict)

        df = pd.DataFrame(df_row_dicts)
        df = self.ensure_df_elt_hashable(df)

        return df, skip_keys

    def create_merged_df(
        self,
        dataset_statistics: list[SingleExperimentStastistics],
        dataset_results: list[SingleExperimentResults],
        value: str = "Dice",
    ):
        df_stat_dicts: list[dict] = []
        for exp in dataset_statistics:
            df_stat_dict, stat_skip_keys = exp.to_df_row_dicts()
            df_stat_dicts.extend(df_stat_dict)

        df_results_dicts: list[dict] = []
        for exp in dataset_results:
            df_exp_dict, exp_skip_keys = exp.to_df_row_dicts(value)
            df_results_dicts.extend(df_exp_dict)

        merged_dicts = merge_dict_lists_on_indices(
            df_results_dicts, df_stat_dicts, self.merge_keys
        )

        merged_skip_keys = list(set(exp_skip_keys + stat_skip_keys))

        df = pd.DataFrame(merged_dicts)
        df = self.ensure_df_elt_hashable(df)
        return df, merged_skip_keys

    @staticmethod
    def ensure_df_elt_hashable(df: pd.DataFrame):
        for col in df.columns:
            if df[col].dtype == object:
                if len(df[col]) > 0 and isinstance(df[col][0], list):
                    df[col] = df[col].apply(lambda x: tuple(x))
        return df

    def dataset_analyze_statistics_results(
        self,
        unique_id: int,
        output_dir: Path = Path("."),
        value: str = "Dice",
        save_df: bool = False,
    ):
        dataset_statistics = [
            exp for exp in self.exp_statistics if exp.base_id == unique_id
        ]
        dataset_results = [
            exp for exp in self.exp_results if exp.config.base_id == unique_id
        ]

        dataset_name = dataset_results[0].config.dataset
        output_dir = output_dir / dataset_name
        if not output_dir.is_dir():
            os.makedirs(output_dir)

        df, skip_keys = self.create_merged_df(
            dataset_statistics, dataset_results, value
        )

        vals = [seperator for seperator in df.columns if seperator not in skip_keys]

        y_full_dict = dataset_results[0].to_full_dataset_performance_dict(value)

        remove_ind = [vals.index(col_name) for col_name in REMOVE_IND_COLS_SETTING_KEY]

        if save_df:
            temp_file = (
                Path(__file__).parent.parent.parent / "temp" / (dataset_name + ".json")
            )
            if not temp_file.parent.is_dir():
                os.makedirs(temp_file.parent)
            logger.debug(f"Saving temporary json to {temp_file}")
            df.to_json(temp_file)

        for key, df_g in df.groupby(vals, as_index=False):
            pre_suffix = df_g["pre_suffix"].iloc[0]
            identifier = create_string_identifier(key, ignore_ident=remove_ind)
            # create plots for each unique setting of the respective dataset
            setting_dir: Path = output_dir / (pre_suffix[2:])
            if not setting_dir.is_dir():
                os.makedirs(setting_dir)

            # most default values from SettingAnalysis are already set for this analysis
            analysis = SettingAnalysis(
                df_g,
                dataset=dataset_name,
                query_key=self.query_key,
                main_performance_key="Mean Dice",
                budget_key="#Patches",
                statistic_keys=dataset_statistics[0].plot_vals,
                performance_keys=dataset_results[0]
                .get_value_dict(plot_val=value)
                .keys(),
                full_performance_dict=y_full_dict,
                palette=PALETTE,
                string_id=identifier,
            )

            analysis.save(save_path=setting_dir / "analysis.pkl")

            # overview metrics
            auc_df = analysis.compute_auc_df()
            # pprint(auc_df)
            auc_df.to_json(setting_dir / "auc.json")
            save_df_to_txt(auc_df, setting_dir / "auc.txt")

            ppm = analysis.compute_pairwise_penalty("Mean Dice")
            ppm.plot_pairwise_matrix(ppm.matrix, savepath=setting_dir / "ppm.png")
            ppm.save(setting_dir / "ppm.json")

            # overview plots
            selected_classes = SELECTED_CLASSES_OVERVIEW.get(dataset_name, None)
            x_names = ["Loop", "#Patches"]
            analysis.save_overview_plots(
                save_dir=setting_dir,
                selected_classes=selected_classes,
                horizontal_lines=y_full_dict,
                x_names=x_names,
            )

            # performance plots
            x_names = ["Loop", "#Patches"]
            y_names = analysis.performance_keys
            analysis.save_setting_plots(
                setting_dir / "results",
                y_names,
                x_names,
                x_ticks=True,
                y_full_dict=y_full_dict,
            )

            # statistic plots
            x_names = ["Loop"]
            y_names = analysis.statistic_keys
            analysis.save_setting_plots(
                setting_dir / "statistics", y_names, x_names, x_ticks=True
            )

            # statistic results plots
            x_names = analysis.statistic_keys
            y_names = analysis.performance_keys
            for y_name in y_names:
                y_names_ = [y_name]
                analysis.save_setting_plots(
                    setting_dir / "results_statistics" / y_name,
                    y_names_,
                    x_names,
                    y_full_dict=y_full_dict,
                    x_ticks=False,
                )

    def analyze_multi_datasets(
        self,
        output_dir: Path = Path("."),
    ):
        for dataset_id in self.unique_datasets:
            logger.info(
                f"Analyzing results for experiments derived from dataset id {dataset_id}"
            )
            self.dataset_analyze_statistics_results(
                unique_id=dataset_id, output_dir=output_dir
            )


def analyze_multi_experiment_results(
    base_path: Path,
    base_raw_path: Path | None,
    filter_final: bool = True,
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
    )


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
