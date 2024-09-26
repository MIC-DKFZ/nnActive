import os
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib import pyplot as plt
from pydantic.dataclasses import dataclass
from typing_extensions import Self

from nnactive.analyze.metrics import PairwisePenaltyMatrix, compute_auc
from nnactive.utils.io import load_pickle, save_df_to_txt, save_pickle
from nnactive.utils.plot import plot_dataframe


@dataclass
class HorizontalLine:
    y: float
    label: str
    color: str = "black"

    def to_dict(self) -> dict[str, Any]:
        return {"y": self.y, "label": self.label, "color": self.color}


@dataclass(config={"arbitrary_types_allowed": True})
class SettingAnalysis:
    df: pd.DataFrame
    dataset: str | None = None
    seed_key: str = "seed"
    query_key: str = "uncertainty"
    budget_key: str = "#Patches"
    max_loops_key: str = "query_steps"
    main_performance_key: str = "Mean Dice"
    main_statistic_key: str = "avg_percentage_of_voxels_fg_cls"
    full_performance_dict: dict[str, list[HorizontalLine]] | None = (
        None  # possibly for each performance key multiple horizontal line
    )
    performance_keys: list[str] | None = None
    statistic_keys: list[str] | None = None
    palette: dict[str, str] | None = None
    string_id: str | None = None

    def __post_init__(self):
        if self.statistic_keys is None:
            self.statistic_keys = []
        if self.performance_keys is None:
            self.performance_keys = []

    def create_filename(self, x_name: str, y_name: str) -> str:
        fn = f"{y_name}-{x_name}__{self.string_id}"[:250]
        return fn

    def _compute_auc_row_dicts(self, performance_keys: list[str]) -> list[dict]:
        # group each experiment by query_key and seed
        df_grouped = self.df.groupby([self.query_key, self.seed_key])

        df_row_dicts = []
        for name, group_df in df_grouped:
            row_dict = {"Query Method": name[0], "seed": name[1]}
            for performance_col in performance_keys:
                # compute AUC for each group
                group_df = group_df.sort_values(self.budget_key)
                values = group_df[performance_col]
                n_loops = len(values)
                if n_loops > 1:
                    auc = compute_auc(values)
                else:
                    auc = np.nan
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

    def compute_auc_df(
        self, performance_vals: str | Iterable[str] | None = None
    ) -> pd.DataFrame:
        performance_vals = self.get_performance_vals(performance_vals)
        df_row_dicts = self._compute_auc_row_dicts(performance_vals)
        df = pd.DataFrame(df_row_dicts)
        num_loops = self.df[self.max_loops_key].max()

        # Ensure that all experiments have the same number of loops (maximum amount)
        assert all(num_loops == self.df[self.max_loops_key].unique())

        df = df[df["#Loops"] == num_loops]

        df_cols = [
            [performance_val + " AUBC", performance_val + " Final"]
            for performance_val in performance_vals
        ]
        qm_key = "Query Method"
        df_cols = [qm_key] + [item for sublist in df_cols for item in sublist]
        df: pd.DataFrame = df[df_cols]
        df = df.groupby(qm_key).aggregate(["mean", "std", "count"])
        return df

    def get_performance_vals(
        self, performance_keys: str | Iterable[str] | None
    ) -> list[str]:
        if performance_keys is None and len(self.performance_keys) > 0:
            performance_keys = self.performance_keys
        elif performance_keys is None and self.main_performance_key is not None:
            performance_keys = [self.main_performance_key]
        elif isinstance(performance_keys, str):
            performance_keys = [performance_keys]
        else:
            performance_keys = list(performance_keys)
        assert (
            len(performance_keys) > 0
        )  # performance_keys has to be longer than 0. See this function to know your options!
        return performance_keys

    def compute_pairwise_penalty(
        self, performance_key: str | None = None, alpha: float = 0.05
    ) -> PairwisePenaltyMatrix:
        if performance_key is None:
            performance_key = self.main_performance_key

        return PairwisePenaltyMatrix(
            self.df,
            alpha=alpha,
            value_key=performance_key,
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
        hline_printers: list[dict, Any] | list[HorizontalLine] | None = None,
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
                if isinstance(y_full, HorizontalLine):
                    axs.axhline(**(y_full.to_dict()))
                else:
                    axs.axhline(**y_full)
        return fig, axs

    def plot_experiment_overview(
        self,
        selected_classes: list[int] | list[tuple[int]] | None = None,
        horizontal_lines: (
            dict[str, list[dict]] | dict[str, list[HorizontalLine]] | None
        ) = None,
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
                    if isinstance(y_full, HorizontalLine):
                        axs[i, j].axhline(**(y_full.to_dict()))
                    else:
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

    def save(self, save_path: Path, save_df: bool = True):
        """Saves the SettingAnalysis object as a binary pickle file and the dataframe for easy access."""
        save_pickle(self, save_path)
        if save_df:
            fn = save_path.name.split(".")[0]
            fn += "_df.pkl"
            self.df.to_pickle(save_path.parent / fn)

    @classmethod
    def load(cls, load_path: Path) -> Self:
        """Initializes the Setting Analysis object from a pickle file."""
        return load_pickle(load_path)


if __name__ == "__main__":
    from pprint import pprint

    d_set = "Dataset216_AMOS2022_task1"
    temp_file = Path(__file__).parent.parent.parent / "temp" / f"{d_set}.json"
    df = pd.read_json(temp_file)
    groups = df.groupby(["pre_suffix"], as_index=False)
    l_groups = list(groups)
    analysis = SettingAnalysis(l_groups[1][1])
    performance_val = "Dice"
    performance_cols = [
        col for col in analysis.df.columns if col.endswith(performance_val)
    ]
    df_auc = analysis.compute_auc_df(performance_cols)

    pprint(df_auc)

    save_dir = temp_file.parent / "test_analysis" / d_set
    if not save_dir.is_dir():
        os.makedirs(save_dir)

    SELECTED_CLASSES = {
        "Dataset216_AMOS2022_task1": [1, 13, 15],
        "Dataset137_BraTS2021": [(1, 2, 3), (2, 3), (3,)],
    }
    selected_classes = SELECTED_CLASSES.get(d_set, None)
    y_full_dict = None

    if not save_dir.is_dir():
        os.makedirs(save_dir)

    # overview plot
    x_names = ["Loop", "#Patches"]
    analysis.save_overview_plots(
        save_dir=save_dir,
        selected_classes=selected_classes,
        horizontal_lines=y_full_dict,
        x_names=x_names,
    )

    # performance plots
    x_names = ["Loop", "#Patches"]
    y_names = [col for col in analysis.df.columns if col.endswith(performance_val)]
    analysis.save_setting_plots(
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
    analysis.save_setting_plots(save_dir / "statistics", y_names, x_names, x_ticks=True)

    # statistic results plots
    x_names = [
        "percentage_of_patches_percentage_foreground",
        "avg_percentage_of_voxels_fg_cls",
    ]
    y_names = [col for col in analysis.df.columns if col.endswith(performance_val)]
    for y_name in y_names:
        y_names_ = [y_name]
        analysis.save_setting_plots(
            save_dir / "results_statistics" / y_name,
            y_names_,
            x_names,
            y_full_dict=y_full_dict,
            x_ticks=False,
        )

    df_auc.to_json(save_dir / "auc.json")
    save_df_to_txt(df_auc, save_dir / "auc.txt")

    ppm = analysis.compute_pairwise_penalty(performance_key="Mean Dice")
    ppm.plot_pairwise_matrix(ppm.matrix, savepath=save_dir / "ppm.png")
    ppm.save(save_dir / "ppm.json")
