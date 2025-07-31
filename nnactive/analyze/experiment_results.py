from __future__ import annotations

from abc import abstractmethod
from functools import cached_property
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from loguru import logger
from pydantic.dataclasses import dataclass

from nnactive.analyze.analysis import HorizontalLine
from nnactive.analyze.metrics import compute_auc
from nnactive.config.struct import ActiveConfig
from nnactive.utils.io import load_json

DATASET_PERFORMANCES = []
full_data_results_folder = Path(__file__).parent.parent.parent / "full_data_results"
if full_data_results_folder.is_dir():
    try:
        for result in full_data_results_folder.iterdir():
            if result.suffix == ".json":
                with open(result, "r") as file:
                    summary = load_json(result)
                    summary["Dataset"] = result.name.split("__")[0]
                    summary["Trainer"] = result.name.split("__")[1].split(".")[0]
                    DATASET_PERFORMANCES.append(summary)
    except Exception as err:
        logger.info("Error on retreiving full data results! Ignoring this...")
else:
    logger.info(
        f"Folder for full_data_results does not exist.\n{full_data_results_folder}"
    )

FULL_LINESTYLE = ["solid", "dashed", "dashdot", "dotted"]


class AbstractValue:
    @abstractmethod
    def get_from_dict(self, file_dict: dict) -> float:
        return 0

    @property
    def name(self) -> str:
        return ""


class MeanValue(AbstractValue):
    def __init__(self, value: str):
        self.value = value

    def get_from_dict(self, file_dict: dict) -> float:
        return file_dict["foreground_mean"][self.value]

    @property
    def name(self) -> str:
        return f"Mean {self.value}"


class ClassValue(AbstractValue):
    def __init__(self, value: str, cls: Any):
        self.value = value
        self.cls = cls

    def get_from_dict(self, file_dict: dict) -> float:
        return file_dict["mean"][self.cls][self.value]

    @property
    def name(self) -> str:
        return f"Class {self.value} {self.cls}"


class SingleExperimentResults:
    def __init__(self, experiment_path: Path):
        """Class to handle results of a single experiment.
        Where one experiment consists of multiple loops."""
        self.experiment_path = experiment_path
        self.performance_metric = "Dice"

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

    @cached_property
    def results(self) -> list[dict]:
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

    @cached_property
    def overview(self) -> pd.DataFrame:
        """Computes the AUBC and final value for each metric in the summary files.

        Returns:
            float: _description_
        """
        df, _ = self.to_df_row_dicts()
        df = pd.DataFrame(df)

        out_list = []
        for col in df.columns:
            if self.performance_metric in col:
                out_dict = {}
                out_dict["Metric"] = col
                out_dict["AUBC"] = compute_auc(df[col])
                out_dict["Final"] = df[col].iloc[-1]
                out_list.append(out_dict)
        out_df = pd.DataFrame(out_list)

        return out_df

    # TODO: retrieve class names from dataset.json
    @property
    def label_names(self) -> dict[str, int]:
        return {}

    @property
    def plot_name(self) -> str:
        return "plot_val"

    @property
    def plot_skip_keys(self) -> list[str]:
        skip_keys = [
            self.plot_name,
            "Experiment",
            "seed",
            "Loop",
            "uncertainty",
            "queries_from_experiment",
            "#Patches",
        ]
        return skip_keys

    def get_value_dict(self, plot_val: str | None = None) -> dict[str, AbstractValue]:
        if plot_val is None:
            plot_val = self.performance_metric
        # better to do this with classes for names of plots
        plot_dict = {f"Mean {plot_val}": MeanValue(plot_val)}
        for cls in self.results[0]["summary"]["mean"]:
            # use deepcopy here as otherwise cls is changed in lambda function
            plot_dict[f"Class {cls} {plot_val}"] = ClassValue(plot_val, cls)
        return plot_dict

    def get_plot_vals(self, plot_val: str | None = "Dice") -> list[str]:
        if plot_val is None:
            plot_val = self.performance_metric
        return list(self.get_value_dict(plot_val).keys())

    def to_df_row_dicts(
        self, plot_val: str | None = "Dice"
    ) -> tuple[list[dict], list[str]]:
        if plot_val is None:
            plot_val = self.performance_metric

        value_dict = self.get_value_dict(plot_val)
        out = []
        skip_keys = [] + self.plot_skip_keys
        for result in self.results:
            append_dict = {}
            for k in result:
                if k != "summary":
                    append_dict[k] = result[k]
            for key in value_dict:
                append_dict[key] = value_dict[key].get_from_dict(result["summary"])
                skip_keys.append(key)
            append_dict.update(self.config.to_dict())
            out.append(append_dict)
        skip_keys = list(set(skip_keys))
        return out, skip_keys

    def to_full_dataset_performance_dict(
        self, value: str | None = "Dice"
    ) -> dict[str, list[HorizontalLine]]:
        out = {}
        if value is None:
            raise NotImplementedError
        else:
            value_dicts = self.get_value_dict(value)
            for key in value_dicts:
                out[key] = self.full_dataset_performance(value_dicts[key].get_from_dict)
        return out

    def to_plot_name_df_row_dicts(
        self,
        plot_fct: Callable[[dict], float] = lambda x: x["foreground_mean"]["Dice"],
    ) -> list[dict]:
        """Returns list of dict with experiment results where self.plot_name is key for the value obtained with plot_fct.

        Args:
            plot_fct (_type_, optional): _description_. Defaults to lambda:x["foreground_mean"]["Dice"].

        Returns:
        """
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

    def full_dataset_performance(
        self, plot_fct=lambda x: x["foreground_mean"]["Dice"]
    ) -> list[HorizontalLine]:
        y_fulls = []
        for dataset_performance in DATASET_PERFORMANCES:
            if dataset_performance["Dataset"] == self.config.dataset:
                y_fulls.append(
                    HorizontalLine(
                        plot_fct(dataset_performance),
                        label=dataset_performance["Trainer"],
                        linestyle=FULL_LINESTYLE[len(y_fulls)],
                        color="black",
                    )
                )
        return y_fulls


if __name__ == "__main__":
    exp_path = Path(
        "/home/c817h/Documents/projects/nnactive_project/nnActive_data/Dataset004_Hippocampus/nnActive_results/Dataset040_Hippocampus__DEBUG__patch-20_20_20__sb-random-label2-all-classes__sbs-10__qs-10__unc-random-label__seed-12345"
    )
    exp = SingleExperimentResults(exp_path)
    from pprint import pprint

    pprint(exp.overview)
