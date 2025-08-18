from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from functools import cached_property
from pathlib import Path

import pandas as pd
import seaborn as sns
from loguru import logger

from nnactive.analyze.aggregate_results import pretty_auc
from nnactive.analyze.analysis import GridPlotter, SettingAnalysis
from nnactive.analyze.experiment_results import SingleExperimentResults
from nnactive.analyze.experiment_statistics import (
    SingleExperimentStastistics,
    efficient_multistatitistics_nested_labels,
)
from nnactive.config.struct import Final
from nnactive.utils.io import save_df_to_txt
from nnactive.utils.pyutils import (
    create_string_identifier,
    merge_dict_lists_on_indices,
    rglob_follow_symlinks,
)

sns.set_style("whitegrid")


# TODO: Final version
# Load all strategies from nnactive/strategies/__init__.py
PALETTE = {
    "random": None,
    "pred_entropy": None,
    "mutual_information": None,
    "expected_dice": None,
    "random-label": None,
    "random-label2": None,
    "power_bald": None,
    "power_bald_b5": None,
    "power_bald_b10": None,
    "power_bald_b20": None,
    "power_bald_b40": None,
    "power_bald_b100": None,
    "power_pe": None,
    "softrank_bald": None,
    "kmeans_bald": None,
}

unique_colors = sns.color_palette("husl", len(PALETTE))
for i, key in enumerate(PALETTE):
    PALETTE[key] = unique_colors[i]

# Defuault is Class 1, 2, 3, ....
# For some datasets we only want to plot a subset of classes
# If Less than 3 classes are available, None is used to fill the list
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
        base_raw_path: Path,
        filter_final: bool = True,
        trainer_use: str = "nnUNetTrainer",
        strict: bool = False,
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
        self.trainer_use = trainer_use
        self.strict = strict

    @cached_property
    def exp_results_paths(self):
        search_paths = [self.base_results_path]
        experiment_paths = []
        while len(search_paths) > 0:
            search_path = search_paths.pop(0)
            if search_path.name == "nnActive_data":
                results_paths = [
                    s_p / "nnActive_results"
                    for s_p in search_path.iterdir()
                    if s_p.name.startswith("Dataset")
                ]
                search_paths.extend(results_paths)
            else:
                experiment_paths.extend(
                    [
                        fn.parent
                        for fn in rglob_follow_symlinks(search_path, "*/config.json")
                    ]
                    # [fn.parent for fn in search_path.rglob("*/config.json")]
                )
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
                # print(f"Skippig Experiment in {exp_path} due to no results.")
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
        num_processes: int = 4,
    ):
        """Creates a merged dataframe for one unique base experiment."""
        df_stat_dicts: list[dict] = []

        efficient_multistatitistics_nested_labels(dataset_statistics)
        if num_processes == 0:
            for exp in dataset_statistics:
                df_stat_dict, stat_skip_keys = exp.to_df_row_dicts()
                df_stat_dicts.extend(df_stat_dict)
        else:
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                futures = [
                    executor.submit(exp.to_df_row_dicts) for exp in dataset_statistics
                ]
                for future in futures:
                    df_stat_dict, stat_skip_keys = future.result()
                    df_stat_dicts.extend(df_stat_dict)

        df_results_dicts: list[dict] = []
        if num_processes == 0:
            for exp in dataset_results:
                df_stat_dict, stat_skip_keys = exp.to_df_row_dicts()
                df_stat_dicts.extend(df_stat_dict)
        else:
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                futures = [
                    executor.submit(exp.to_df_row_dicts, value)
                    for exp in dataset_results
                ]
                for future in futures:
                    df_exp_dict, exp_skip_keys = future.result()
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
                    df[col] = df[col].apply(
                        lambda x: tuple(x) if isinstance(x, list) else x
                    )
        return df

    def dataset_analyze_statistics_results(
        self,
        unique_id: int,
        output_dir: Path = Path("."),
        value: str = "Dice",
        save_df: bool = False,
        create_detailed_plots: bool = False,
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

        logger.info(f"Building dataframe for {dataset_name}")

        df, skip_keys = self.create_merged_df(
            dataset_statistics, dataset_results, value
        )
        logger.info(f"Finished buiding dataframe for {dataset_name}")

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

        # TEMPORARY FIX FOR agg_stride to allow for grouping
        # TODO: clean this up!
        if any([isinstance(val, tuple) for val in df["agg_stride"].unique()]):
            df["agg_stride"] = df["agg_stride"].apply(
                lambda x: tuple([x] * 3) if isinstance(x, int) else x
            )
        output_dirs = []
        for key, df_g in df.groupby(vals, as_index=False):
            identifier = create_string_identifier(key, ignore_ident=remove_ind)
            logger.info(f"Creating plots for Identifier: {identifier}")
            # create plots for each unique setting of the respective dataset
            pre_suffix = df_g["pre_suffix"].iloc[0]
            setting_dir: Path = output_dir / (pre_suffix[2:])
            if self.strict:
                i = 0
                while setting_dir in output_dirs:
                    setting_dir = setting_dir.parent / (setting_dir.name + f"_v{i}")
                    i += 1
                output_dirs.append(setting_dir)
                if not setting_dir.is_dir():
                    os.makedirs(setting_dir)

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

            else:
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
                if setting_dir in output_dirs:
                    logger.info(
                        f"Found existing analysis in {setting_dir}. Merging experiments."
                    )
                    prev_ana = SettingAnalysis.load(setting_dir / "analysis.pkl")
                    analysis.df = pd.concat(
                        [prev_ana.df, analysis.df], ignore_index=True
                    )
                output_dirs.append(setting_dir)
            logger.info(f"Saving results to {setting_dir.name}")

            analysis.save(save_path=setting_dir / "analysis.pkl")

            # overview metrics
            auc_df = analysis.compute_auc_df()
            # pprint(auc_df)
            auc_df.to_json(setting_dir / "auc.json")
            save_df_to_txt(auc_df, setting_dir / "auc.txt")
            save_df_to_txt(
                pretty_auc(pd.read_json(setting_dir / "auc.json"), seeds=True),
                setting_dir / "auc_pretty.txt",
            )

            ppm = analysis.compute_pairwise_penalty("Mean Dice")
            ppm.plot_pairwise_matrix(ppm.matrix, savepath=setting_dir / "ppm.png")
            ppm.save(setting_dir / "ppm.json")

            trainer = str(analysis.df["trainer"].unique()[0])
            if len(trainer.split("_")) > 1:
                epochs = trainer.split("_")
                trainer_use = f"{self.trainer_use}_{epochs[-1]}"
                logger.info(f"Using Full Performance Trainer: {trainer_use}")
            trainers = [
                f.label
                for f in analysis.full_performance_dict[analysis.main_performance_key]
            ]
            compute_beta = True
            if trainer_use not in trainers:
                if len(trainers) > 0:
                    trainer_use = trainers[0]
                    logger.info(
                        f"Using substitute Full Performance Trainer {trainer_use} from {trainers}"
                    )
                else:
                    compute_beta = False
            if compute_beta:
                betas = analysis.compute_beta_curve(
                    trainer_use,
                    "percentage_of_voxels_foreground",
                )
                betas_df = betas.to_beta_df()
                betas_df.to_json(setting_dir / "beta.json")
                save_df_to_txt(betas_df, setting_dir / "beta.txt")

            # overview plots
            selected_classes = SELECTED_CLASSES_OVERVIEW.get(dataset_name, None)
            n_performance_cols = 3
            if selected_classes is None:
                selected_classes = [
                    int(i.split(" ")[1]) for i in df_g.columns if i.startswith("Class")
                ][:3]
                while len(selected_classes) < n_performance_cols:
                    selected_classes.append(None)
            grid = self.generate_grid(selected_classes)
            x_names = ["Loop", "#Patches"]
            analysis.save_overview_plots(
                save_dir=setting_dir,
                grid=grid,
                horizontal_lines=y_full_dict,
                x_names=x_names,
            )

            if create_detailed_plots:
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

    @staticmethod
    def generate_grid(
        selected_classes: list[int] | list[tuple[int,]],
    ) -> GridPlotter:
        n_rows = 3
        n_cols = 9
        grid = GridPlotter(n_rows, n_cols)

        # fill column 1 with main performance metric
        col_num = 0
        x_names = [
            "#Patches",
            "percentage_of_voxels_foreground",
            "avg_percentage_of_voxels_fg_cls",
        ]
        y_names = ["Mean Dice"] * n_rows
        grid.set_col_from_values(col_num, x_names, y_names)

        # fill columns 2-4 with class performance metrics
        for i, cls_index in enumerate(selected_classes):
            col_num += 1
            if cls_index is None:
                x_names = [None] * n_rows
                y_names = [None] * n_rows
            else:
                y_names = [f"Class {cls_index} Dice"] * n_rows
                if isinstance(cls_index, (tuple, list)):
                    perctentage_index = cls_index[0]
                else:
                    perctentage_index = cls_index
                x_names = [
                    "#Patches",
                    f"percentage_of_voxels_per_cls_{perctentage_index}",
                    "avg_percentage_of_voxels_fg_cls",
                ]
            grid.set_col_from_values(col_num, x_names, y_names)

        # fill column 5 with statistic metrics
        col_num += 1
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
        grid.set_col_from_values(col_num, x_names, y_names)

        # fill column 6-8 with statistic metrics per class
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
                if isinstance(cls_index, (tuple, list)):
                    perctentage_index = cls_index[0]
                else:
                    perctentage_index = cls_index
                y_names = [
                    f"percentage_of_voxels_per_cls_{perctentage_index}",
                    None,
                    f"patches_per_cls_{perctentage_index}",
                ]
            col = [{"x_name": x_n, "y_name": y_n} for x_n, y_n in zip(x_names, y_names)]
            grid.set_col_from_dicts(col_num, col)

        # fill column 9 with file statistics
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
        grid.set_col_from_values(col_num, x_names, y_names)
        return grid


def analyze_multi_experiment_results(
    base_path: Path,
    base_raw_path: Path,
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
