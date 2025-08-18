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

from nnactive.analyze.metrics import DatasetBeta, PairwisePenaltyMatrix, compute_auc
from nnactive.utils.io import load_pickle, save_df_to_txt, save_pickle
from nnactive.utils.plot import plot_dataframe


@dataclass
class HorizontalLine:
    y: float
    label: str
    linestyle: str = "-"
    color: str = "black"

    def to_dict(self) -> dict[str, Any]:
        return {
            "y": self.y,
            "label": self.label,
            "color": self.color,
            "linestyle": self.linestyle,
        }


class GridPlotter:
    def __init__(self, n_rows: int, n_cols: int):
        """A class to help with plotting multiple plots in a grid.
        For an example usecase see the SettingAnalysis class."""
        self.grid: list[list[dict[str, str]]] = [[None] * n_cols for _ in range(n_rows)]
        self.n_rows = n_rows
        self.n_cols = n_cols

    def set_spot(self, row: int, col: int, x_name: str, y_name: str, **kwargs):
        self.grid[row][col] = {"x_name": x_name, "y_name": y_name, **kwargs}

    def set_col_from_dicts(self, col: int, column_dicts: list[dict[str, str]]):
        assert len(column_dicts) == self.n_rows
        for row, col_dict in enumerate(column_dicts):
            self.grid[row][col] = col_dict

    def set_row_from_dicts(self, row: int, row_dicts: list[dict[str, str]]):
        assert len(row_dicts) == self.n_cols
        self.grid[row] = row_dicts

    def set_col_from_values(
        self,
        col: int,
        x_names: list[str],
        y_names: list[str],
        additional: dict[str, list[str]] = {},
    ):
        assert len(x_names) == self.n_rows
        assert len(y_names) == self.n_rows

        for row, (x_name, y_name) in enumerate(zip(x_names, y_names)):
            add = {key: value[row] for key, value in additional.items()}
            self.grid[row][col] = {"x_name": x_name, "y_name": y_name, **add}

    def set_row_from_values(
        self,
        row: int,
        x_names: list[str],
        y_names: list[str],
        additional: dict[str, list[str]],
    ):
        assert len(x_names) == self.n_cols
        assert len(y_names) == self.n_cols
        for col, (x_name, y_name) in enumerate(zip(x_names, y_names)):
            add = {key: value[col] for key, value in additional.items()}
            self.grid[row][col] = {"x_name": x_name, "y_name": y_name, **add}

    def __repr__(self) -> str:
        return "\n".join(
            ["[{}]".format(",\t".join([str(elt) for elt in row])) for row in self.grid]
        )

    def shape(self) -> tuple[int, int]:
        return self.n_rows, self.n_cols

    def __getitem__(self, key: tuple[int, int]) -> dict[str, str]:
        return self.grid[key[0]][key[1]]

    def __setitem__(self, key: tuple[int, int], value: dict[str, str]):
        self.grid[key[0]][key[1]] = value

    def from_dict(
        self,
        grid_dict: list[list[dict[str, str]]],
        row_offset: int = 0,
        col_offset: int = 0,
    ):
        for row, col_dicts in enumerate(grid_dict):
            for col, plot_dict in enumerate(col_dicts):
                self.grid[row + row_offset][col + col_offset] = plot_dict
        for i, j in product(range(self.n_rows), range(self.n_cols)):
            if self.grid[i][j] is not None:
                keys = list(self.grid[i][j].keys())
                assert "x_name" in keys
                assert "y_name" in keys


@dataclass(config={"arbitrary_types_allowed": True})
class SettingAnalysis:
    """
    A class to analyze settings and performance metrics from a DataFrame.

    Attributes:
        df (pd.DataFrame): The DataFrame containing the data to be analyzed.
        dataset (str | None): The name of the dataset. Defaults to None.
        seed_key (str): The column name for the seed values. Defaults to "seed".
        query_key (str): The column name for the query method. Defaults to "uncertainty".
        budget_key (str): The column name for the budget values. Defaults to "#Patches".
        max_loops_key (str): The column name for the maximum loops. Defaults to "query_steps".
        main_performance_key (str): The column name for the main performance metric. Defaults to "Mean Dice".
        main_statistic_key (str): The column name for the main statistic. Defaults to "avg_percentage_of_voxels_fg_cls".
        full_performance_dict (dict[str, list[HorizontalLine]] | None): A dictionary containing performance metrics for each key. Defaults to None.
        performance_keys (list[str] | None): A list of performance keys. Defaults to None.
        statistic_keys (list[str] | None): A list of statistic keys. Defaults to None.
        palette (dict[str, str] | None): A dictionary mapping keys to colors for plotting. Defaults to None.
        string_id (str | None): A string identifier for the analysis. Defaults to None.
    """

    df: pd.DataFrame
    dataset: str | None = None
    seed_key: str = "seed"
    query_key: str = "uncertainty"
    budget_key: str = "#Patches"
    max_loops_key: str = "query_steps"
    main_performance_key: str = "Mean Dice"
    main_statistic_key: str = "avg_percentage_of_voxels_fg_cls"
    full_performance_dict: dict[
        str, list[HorizontalLine]
    ] | None = None  # possibly for each performance key multiple horizontal line
    performance_keys: list[str] | None = None
    statistic_keys: list[str] | None = None
    palette: dict[str, Any] | None = None
    string_id: str | None = None

    def __post_init__(self):
        if self.statistic_keys is None:
            self.statistic_keys = []
        if self.performance_keys is None:
            self.performance_keys = []

    def create_filename(self, x_name: str, y_name: str) -> str:
        """Creates a filename based on x_name, y_name, and string_id."""
        fn = f"{y_name}-{x_name}__{self.string_id}"[:250]
        return fn

    @property
    def auc_qm_key(self) -> str:
        """Returns the key for the query method used for AUC computation."""
        return "Query Method"

    @property
    def auc_loops_key(self) -> str:
        """Returns the key for the number of loops used for AUC computation."""
        return "#Loops"

    def _compute_auc_row_dicts(self, performance_keys: list[str]) -> list[dict]:
        """
        Computes the Area Under the Curve (AUC) for each group of experiments and returns a list of dictionaries containing the results.

        Args:
            performance_keys (list[str]): A list of column names representing the performance metrics to compute AUC for.

        Returns:
            list[dict]: A list of dictionaries where each dictionary contains the AUC and final performance values for each performance metric,
                        along with the query key, seed key, and the number of loops used in the AUC computation.

        Notes:
            - The DataFrame `self.df` is expected to have columns corresponding to `self.query_key`, `self.seed_key`, and `self.budget_key`.
            - The AUC is computed for each group of experiments defined by unique combinations of `self.query_key` and `self.seed_key`.
            - If a group has only one value, the AUC is set to NaN.
            - The final performance value is taken as the last value in the sorted group.
        """
        # group each experiment by query_key and seed
        df_grouped = self.df.groupby([self.query_key, self.seed_key])

        df_row_dicts = []
        for name, group_df in df_grouped:
            row_dict = {self.auc_qm_key: name[0], self.seed_key: name[1]}
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
                row_dict[self.auc_loops_key] = (
                    min(row_dict[self.auc_loops_key], n_loops)
                    if self.auc_loops_key in row_dict
                    else n_loops
                )
            df_row_dicts.append(row_dict)

        return df_row_dicts

    def compute_auc_df(
        self,
        performance_vals: str | Iterable[str] | None = None,
        enforce_full: bool = True,
    ) -> pd.DataFrame:
        """
        Computes the Area Under the Curve (AUC) DataFrame for the given performance values.

        This method processes performance values to compute AUC-related metrics and returns
        a DataFrame containing the aggregated results.

        Args:
            performance_vals (str | Iterable[str] | None): A string or an iterable of strings
            representing the performance values to be considered. If None, default
            performance values will be used.

        Returns:
            pd.DataFrame: A DataFrame containing the aggregated AUC metrics, including mean,
            standard deviation, and count for each performance value.

        Raises:
            AssertionError: If the number of loops is not consistent across all experiments.
        """
        performance_vals = self.get_performance_vals(performance_vals)
        df_row_dicts = self._compute_auc_row_dicts(performance_vals)
        df = pd.DataFrame(df_row_dicts)
        num_loops = self.df[self.max_loops_key].max()

        # Ensure that all experiments have the same number of loops (maximum amount)
        assert all(num_loops == self.df[self.max_loops_key].unique())

        if enforce_full:
            df = df[df[self.auc_loops_key] == num_loops]
        else:
            df = df[df[self.auc_loops_key] == df[self.auc_loops_key].max()]

        df_cols = [
            [performance_val + " AUBC", performance_val + " Final"]
            for performance_val in performance_vals
        ]
        qm_key = self.auc_qm_key
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

    def compute_beta_curve(
        self,
        trainer: str,
        budget_key: str | None = None,
        performance_key: str | None = None,
    ):
        if performance_key is None:
            performance_key = self.main_performance_key
        if budget_key is None:
            budget_key = self.budget_key
        full_performance = [
            f.y
            for f in self.full_performance_dict[performance_key]
            if f.label == trainer
        ]
        if len(full_performance) == 0:
            trainers = [f.label for f in self.full_performance_dict[performance_key]]
            raise ValueError(
                f"""
                Could not find full performance for trainer {trainer} and performance key {performance_key}.
                Please try one following {trainers}
                """
            )
        full_performance = full_performance[0]

        return DatasetBeta.from_df(
            self.df, budget_key, performance_key, full_performance, self.query_key
        )

    def compute_pairwise_penalty(
        self, performance_key: str | None = None, alpha: float = 0.05
    ) -> PairwisePenaltyMatrix:
        if performance_key is None:
            performance_key = self.main_performance_key

        return PairwisePenaltyMatrix.from_df(
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
        """
        Plots a single experiment from the given DataFrame.

        Args:
            df_g (pd.DataFrame): The DataFrame containing the data to plot.
            y_name (str): The name of the column to use for the y-axis.
            x_name (str): The name of the column to use for the x-axis.
            dataset (str | None, optional): The title of the plot. Defaults to None.
            x_ticks (Iterable | None, optional): Custom x-ticks for the plot. Defaults to None.
            hline_printers (list[dict, Any] | list[HorizontalLine] | None, optional):
            List of dictionaries or HorizontalLine objects to add horizontal lines to the plot. Defaults to None.

        Returns:
            tuple[plt.Figure, plt.Axes]: The figure and axes of the plot.
        """
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

    def plot_setting_overview(
        self,
        grid: list[list[dict[str, str]]] | GridPlotter,
        horizontal_lines: (
            dict[str, list[dict]] | dict[str, list[HorizontalLine]] | None
        ) = None,
        x_axis_dict: dict[str, Any] | None = None,
        plot_size: float = 4,
        style: str | None = None,
    ) -> tuple[plt.Figure, list[list[plt.Axes]]]:
        if isinstance(grid, list):
            grid = GridPlotter(len(grid), len(grid[0]))
            grid.from_dict(grid)

        n_rows, n_cols = grid.shape()
        fig, axs = plt.subplots(
            nrows=n_rows, ncols=n_cols, figsize=(n_cols * plot_size, n_rows * plot_size)
        )

        for i, j in product(range(n_rows), range(n_cols)):
            plot_dict = grid[i, j]
            if plot_dict is None:
                axs[i, j].set_axis_off()
                continue
            x_name, y_name = plot_dict["x_name"], plot_dict["y_name"]
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
                style=style,
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
        grid: list[list[dict[str, str]]] | GridPlotter,
        horizontal_lines: dict[str, Any] | None = None,
        x_names: list[str] | tuple[str,] = tuple(),
        style: str | None = None,
    ):
        x_axis_dict = dict()
        for x_name in x_names:
            x_axis_dict[x_name] = {"x_ticks": self.df[x_name].unique()}
        fig, axs = self.plot_setting_overview(
            grid=grid,
            horizontal_lines=horizontal_lines,
            x_axis_dict=x_axis_dict,
            style=style,
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
