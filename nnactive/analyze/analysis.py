import os
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib import pyplot as plt

from nnactive.analyze.metrics import PairwisePenaltyMatrix, compute_auc
from nnactive.utils.io import save_df_to_txt
from nnactive.utils.plot import create_unique_name, plot_dataframe

SELECTED_CLASSES = {
    "Dataset216_AMOS2022_task1": [1, 13, 15],
    "Dataset137_BraTS2021": [(1, 2, 3), (2, 3), (3,)],
}


class SettingAnalysis:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        dataset: str | None = None,
        query_key: str = "uncertainty",
        performance_val: str = "Dice",
        performance_key: str = "Mean Dice",
        budget_key: str = "#Patches",
        palette: dict[str, str] | None = None,
        results_skip_keys: list[str] | None = None,
        statistics_skip_keys: list[str] | None = None,
        key: Iterable | None = None,
    ):
        """Analyse the dataset statistics and performance of the experiments corresponding to one Dataset.

        Notes:
        key: already cleaned of vales that should not be used for naming the files.
        """
        self.df = dataframe
        self.palette = palette
        self.query_key = query_key
        self.performance_val = performance_val
        self.performance_key = performance_key
        self.results_skip_keys = [] if results_skip_keys is None else results_skip_keys
        self.statistics_skip_keys = (
            [] if statistics_skip_keys is None else statistics_skip_keys
        )
        self.budget_key = budget_key
        self.key = key
        self.dataset = dataset

    @property
    def results_group_keys(self) -> list[str]:
        return [col for col in self.df.columns if col not in self.results_skip_keys]

    @property
    def statistics_group_keys(self) -> list[str]:
        return [col for col in self.df.columns if col not in self.statistics_skip_keys]

    @property
    def all_group_keys(self) -> list[str]:
        return [
            col
            for col in self.df.columns
            if col not in self.statistics_skip_keys + self.results_skip_keys
        ]

    def create_filename(self, x_name: str, y_name: str) -> str:
        fn = create_unique_name(
            x_name,
            y_name,
            self.key,
        )
        return fn

    def compute_auc_row_dicts(self) -> list[dict]:
        # TODO: replace this placeholder
        performance_cols = [self.performance_key]
        # group each experiment by query_key and seed
        df_grouped = self.df.groupby([self.query_key, "seed"])

        df_row_dicts = []
        for name, group_df in df_grouped:
            row_dict = {"Query Method": name[0], "seed": name[1]}
            for performance_col in performance_cols:
                # compute AUC for each group
                values = group_df[performance_col]
                n_loops = len(values)
                auc = compute_auc(values)
                final_performance = values.iloc[-1]
                row_dict[performance_col + " AUBC"] = auc
                row_dict[performance_col + " Final"] = final_performance
                row_dict["#Loops"] = (
                    min(row_dict["#Loops"], n_loops)
                    if "#Loops" in row_dict
                    else n_loops
                )
            df_row_dicts.append(row_dict)

        return df_row_dicts

    def compute_auc_df(self) -> pd.DataFrame:
        df_row_dicts = self.compute_auc_row_dicts()
        df = pd.DataFrame(df_row_dicts)
        num_loops = self.df["query_steps"].max()
        # TODO: This check could perhaps be done in a post init method
        assert all(num_loops == self.df["query_steps"].unique())
        df = df[df["#Loops"] == num_loops]
        df: pd.DataFrame = df[
            [
                "Query Method",
                self.performance_key + " AUBC",
                self.performance_key + " Final",
            ]
        ]
        df = df.groupby("Query Method").aggregate(["mean", "std", "count"])
        return df

    def compute_pairwise_penalty(self, alpha: float = 0.05) -> PairwisePenaltyMatrix:
        return PairwisePenaltyMatrix(
            self.df,
            alpha=alpha,
            value_key=self.performance_key,
            qm_key=self.query_key,
            budget_key=self.budget_key,
        )

    def plot_single_experiment(
        self,
        df_g: pd.DataFrame,
        y_name: str,
        x_name: str,
        dataset: str | None = None,
        x_ticks: Iterable | None = None,
        hline_printers: list[dict, Any] | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        fig, axs = plt.subplots()
        axs = plot_dataframe(
            axs,
            df_g,
            x_name,
            y_name,
            hue_key=self.query_key,
            plot_title=dataset,
            palette=self.palette,
            x_ticks=x_ticks,
        )

        # add vertical line
        if hline_printers is not None:
            for y_full in hline_printers:
                axs.axhline(**y_full)
        return fig, axs

    def plot_experiment_overview(
        self,
        selected_classes: list[int] | list[tuple[int]] | None = None,
        horizontal_lines: dict[str, Any] | None = None,
        x_axis_dict: dict[str, Any] | None = None,
    ) -> tuple[plt.Figure, list[list[plt.Axes]]]:
        n_rows, n_cols = 3, 9
        n_performance_cols = 3
        plot_size = 4
        if selected_classes is None:
            selected_classes = [
                int(i.split(" ")[1]) for i in self.df.columns if i.startswith("Class")
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
                if isinstance(cls_index, (tuple, list)):
                    perctentage_index = cls_index[0]
                else:
                    perctentage_index = cls_index
                x_names = [
                    "#Patches",
                    f"percentage_of_voxels_per_cls_{perctentage_index}",
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
                if isinstance(cls_index, (tuple, list)):
                    perctentage_index = cls_index[0]
                else:
                    perctentage_index = cls_index
                y_names = [
                    f"percentage_of_voxels_per_cls_{perctentage_index}",
                    None,
                    f"patches_per_cls_{perctentage_index}",
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
                self.df,
                x_name,
                y_name,
                hue_key=self.query_key,
                palette=self.palette,
                legend=None,
                **x_kwargs,
            )
            if horizontal_lines is not None and y_name in horizontal_lines:
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

    def save_setting_plots(
        self,
        save_dir: Path,
        y_names: list[str],
        x_names: list[str],
        y_full_dict: dict | None = None,
        x_ticks: bool = True,
        short_name: bool = False,
    ):
        if not save_dir.is_dir():
            os.makedirs(save_dir)
        if x_ticks:
            x_name_dict = dict()
            for x_name in x_names:
                x_name_dict[x_name] = {"x_ticks": self.df[x_name].unique()}
        else:
            x_name_dict = {x_name: {} for x_name in x_names}

        for y_name, x_name in product(y_names, x_name_dict):
            fig, axs = self.plot_single_experiment(
                self.df,
                y_name,
                x_name,
                self.dataset,
                hline_printers=y_full_dict[y_name] if y_full_dict is not None else None,
                **x_name_dict[x_name],
            )
            file_name = self.create_filename(x_name, y_name)

            plt.savefig(save_dir / f"{file_name}.png", bbox_inches="tight")
            plt.close("all")

    def save_overview_plots(
        self,
        save_dir: Path,
        selected_classes: list[int] | list[tuple[int]] | None = None,
        horizontal_lines: dict[str, Any] | None = None,
        x_names: list[str] | tuple[str,] = tuple(),
    ):
        x_axis_dict = dict()
        for x_name in x_names:
            x_axis_dict[x_name] = {"x_ticks": self.df[x_name].unique()}
        fig, axs = self.plot_experiment_overview(
            selected_classes=selected_classes,
            horizontal_lines=horizontal_lines,
            x_axis_dict=x_axis_dict,
        )
        plt.savefig(save_dir / "overview.png", bbox_inches="tight")
        plt.close("all")

    @staticmethod
    def ensure_df_elt_hashable(df: pd.DataFrame):
        for col in df.columns:
            if df[col].dtype == object:
                if len(df[col]) > 0 and isinstance(df[col][0], list):
                    df[col] = df[col].apply(lambda x: tuple(x))
        return df

    # TODO: move this method to the class which orchestrates everything
    def analyze_all(self, save_dir: Path, y_full_dict: dict[str, Any] | None = None):
        if not save_dir.is_dir():
            os.makedirs(save_dir)
        selected_classes = SELECTED_CLASSES.get(self.dataset, None)

        # overview plot
        x_names = ["Loop", "#Patches"]
        self.save_overview_plots(
            save_dir=save_dir,
            selected_classes=selected_classes,
            horizontal_lines=y_full_dict,
            x_names=x_names,
        )

        # performance plots
        x_names = ["Loop", "#Patches"]
        y_names = [col for col in self.df.columns if col.endswith(self.performance_val)]
        self.save_setting_plots(
            save_dir / "results",
            y_names,
            x_names,
            x_ticks=True,
            y_full_dict=y_full_dict,
        )

        # statistic plots
        x_names = ["Loop"]
        # TODO: clear reading out what statistics are
        y_names = ["percentage_of_patches_percentage_foreground"]
        self.save_setting_plots(save_dir / "statistics", y_names, x_names, x_ticks=True)

        # statistic results plots
        # TODO: clear reading out what statistics are
        x_names = [
            "percentage_of_patches_percentage_foreground",
            "avg_percentage_of_voxels_fg_cls",
        ]
        y_names = [col for col in self.df.columns if col.endswith(self.performance_val)]
        for y_name in y_names:
            y_names_ = [y_name]
            self.save_setting_plots(
                save_dir / "results_statistics" / y_name,
                y_names_,
                x_names,
                y_full_dict=y_full_dict,
                x_ticks=False,
            )

        auc_df = self.compute_auc_df()
        # pprint(auc_df)
        auc_df.to_json(save_dir / "auc.json")
        save_df_to_txt(auc_df, save_dir / "auc.txt")

        ppm = self.compute_pairwise_penalty()
        ppm.plot_pairwise_matrix(ppm.matrix, savepath=save_dir / "ppm.png")
        ppm.save(save_dir / "ppm.json")


if __name__ == "__main__":
    from pprint import pprint

    d_set = "Dataset216_AMOS2022_task1"
    temp_file = Path(__file__).parent.parent.parent / "temp" / f"{d_set}.json"
    df = pd.read_json(temp_file)
    groups = df.groupby(["pre_suffix"], as_index=False)
    l_groups = list(groups)
    analysis = SettingAnalysis(l_groups[1][1])
    df_auc = analysis.compute_auc_df()

    pprint(df_auc)

    save_dir = temp_file.parent / "test_analysis" / d_set
    if not save_dir.is_dir():
        os.makedirs(save_dir)
    analysis.analyze_all(save_dir)
