from __future__ import annotations

import os
from copy import deepcopy
from functools import cached_property
from itertools import product
from pathlib import Path
from typing import Any, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from pydantic.dataclasses import dataclass

from nnactive.config.struct import ActiveConfig, Final
from nnactive.data import Patch
from nnactive.loops.loading import get_nested_patches_from_loop_files
from nnactive.nnunet.utils import get_raw_path
from nnactive.paths import set_raw_paths
from nnactive.utils.io import load_json, load_label_map, save_json
from nnactive.utils.patches import get_slices_for_file_from_patch
from nnactive.utils.pyutils import get_clean_dataclass_dict

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

CONFIGSKIPKEYS = ["seed", "uncertainty", "#Patches"]


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


@dataclass
class Statistics:
    files: list[str]
    classes: list[int]
    background_label: None | int = 0

    @staticmethod
    def internal_values():
        return ["patches_per_cls", "voxels_per_cls", "patches_background"]

    def __post_init__(self):
        self.patches_per_cls = {c: 0 for c in self.classes}
        self.voxels_per_cls = {c: 0 for c in self.classes}
        self.patches_background = 0

    def update_patch(self, patch_labels: dict[int, int]):
        background_patch = True
        for patch_class, class_count in patch_labels.items():
            if class_count > 0:
                self.patches_per_cls[patch_class] += 1
                if (
                    self.background_label is not None
                    and patch_class != self.background_label
                ):
                    background_patch = False
            self.voxels_per_cls[patch_class] += class_count
        if background_patch:
            self.patches_background += 1

    def update_statistics(self, statistics: Statistics):
        assert self.classes == statistics.classes
        assert self.background_label == statistics.background_label
        self.files += statistics.files
        self.patches_background += statistics.patches_background
        for c in self.classes:
            self.voxels_per_cls[c] += statistics.voxels_per_cls[c]
            self.patches_per_cls[c] += statistics.patches_per_cls[c]

    @property
    def voxels_foreground(self) -> int:
        foreground_voxels = 0
        for c in self.classes:
            if self.background_label is not None and c != self.background_label:
                foreground_voxels += self.voxels_per_cls[c]
        return foreground_voxels

    @property
    def num_voxels(self) -> int:
        num_voxels = 0
        for c in self.classes:
            num_voxels += self.voxels_per_cls[c]
        return num_voxels

    @property
    def patches_foreground(self) -> int:
        return self.num_patches - self.patches_background

    @property
    def num_unique_files(self) -> int:
        return len(set(self.files))

    @property
    def num_patches(self) -> int:
        # num files gives the amount of patches as each patch has one file
        return len(self.files)

    def to_dict(self) -> dict[str, Any]:
        out_dict = {
            "voxels_foreground": self.voxels_foreground,
            "num_voxels": self.num_voxels,
            "patches_foreground": self.patches_foreground,
            "num_patches": self.num_patches,
            "num_unique_files": self.num_unique_files,
            "voxel_percentage_foreground": self.voxels_foreground / self.num_voxels,
            "patches_percentage_foreground": self.patches_foreground / self.num_patches,
        }
        for c in self.classes:
            out_dict[f"voxels_per_cls_{c}"] = self.voxels_per_cls[c]
            out_dict[f"patches_per_cls_{c}"] = self.patches_per_cls[c]
        return out_dict

    @staticmethod
    def from_json(filepath: Path | str) -> Statistics:
        file_dict = load_json(filepath)
        out = Statistics(
            [],
            [],
        )
        for key in file_dict:
            out.__setattr__(key, file_dict[key])

        for c in out.classes:
            str_c = str(c)
            if str_c in out.voxels_per_cls.keys():
                out.voxels_per_cls[c] = out.voxels_per_cls.pop(str(c))
            if str_c in out.patches_per_cls.keys():
                out.patches_per_cls[c] = out.patches_per_cls.pop(str(c))

        return out

    def to_json_dict(self) -> dict[str, dict[str, Any]]:
        return get_clean_dataclass_dict(self)


# TODO: Possibly delete results depending on ease of aggregation!
class SingleExperimentStastistics:
    def __init__(self, raw_path: Path, results_path: Path | None = None):
        self.raw_path = raw_path
        self.results_path = results_path

    @property
    def dataset_json(self) -> dict:
        return load_json(self.raw_path / "dataset.json")

    @property
    def config(self) -> ActiveConfig | None:
        if self.results_path is not None:
            return ActiveConfig.from_json(self.results_path / ActiveConfig.filename())
        else:
            return None

    @cached_property
    def full_data_statistic(self):
        savefile = self.source_dataset_path / "labelsTr_statistics.json"
        if savefile.is_file():
            return Statistics.from_json(savefile)
        else:
            labels_path = self.source_dataset_path / "labelsTr"
            files = [
                f.name
                for f in (labels_path).iterdir()
                if (f.name).endswith(self.dataset_json["file_ending"])
            ]
            full_data_stat = Statistics(files, self.unique_dataset_classes())
            for f in files:
                patch_labels = load_label_map(f, labels_path, "")
                unique_cls, counts = np.unique(patch_labels, return_counts=True)

                patch_stastics = {
                    int(unique_cl): int(count)
                    for unique_cl, count in zip(unique_cls, counts)
                }
                full_data_stat.update_patch(patch_stastics)
            save_json(full_data_stat.to_json_dict(), savefile)
            return full_data_stat

    @property
    def base_id(self) -> int:
        base_id = self.dataset_json["annotated_id"]
        return base_id

    @property
    def source_dataset_path(self) -> Path:
        with set_raw_paths():
            source_path = get_raw_path(self.base_id)
        return source_path

    @property
    def dataset_labels(self) -> dict[str, int | list[int]]:
        return self.dataset_json["labels"]

    def unique_dataset_classes(
        self,
        no_ignore: bool = True,
        no_background: bool = False,
    ) -> list[int]:
        out = []
        ignore_list = []
        if no_ignore:
            ignore_list.append("ignore")
        if no_background:
            ignore_list.append("background")
        for dataset_label in self.dataset_labels:
            if dataset_label not in ignore_list:
                classes = self.dataset_labels[dataset_label]
                if isinstance(classes, int):
                    out.append(classes)
                elif isinstance(classes, (list, tuple)):
                    out.extend(list(classes))
                else:
                    raise NotImplementedError
        out = list(set(out))
        out.sort()
        return out

    @property
    def nested_patches(self) -> list[list[Patch]]:
        return get_nested_patches_from_loop_files(self.raw_path)

    @cached_property
    def nested_patch_labels(self) -> list[list[dict[int, int]]]:
        nested_labels = []
        for loop_patches in self.nested_patches:
            loop_labels = []
            for patch in loop_patches:
                label_image = load_label_map(
                    patch.file, self.source_dataset_path / "labelsTr", ""
                )
                patch_access = get_slices_for_file_from_patch([patch], patch.file)[0]
                patch_labels = label_image[patch_access]
                # fill statistics
                unique_cls, counts = np.unique(patch_labels, return_counts=True)

                patch_stastics = {
                    int(unique_cl): int(count)
                    for unique_cl, count in zip(unique_cls, counts)
                }
                loop_labels.append(patch_stastics)
            nested_labels.append(loop_labels)
        return nested_labels

    @property
    def nested_statstics(self) -> list[Statistics]:
        nested_statistics = []
        for loop_labels, loop_patches in zip(
            self.nested_patch_labels, self.nested_patches
        ):
            loop_statistics = Statistics(
                [patch.file for patch in loop_patches],
                self.unique_dataset_classes(no_ignore=True),
            )

            for patch_labels in loop_labels:
                loop_statistics.update_patch(patch_labels)
            nested_statistics.append(loop_statistics)
        return nested_statistics

    @property
    def statistics(self) -> list[Statistics]:
        statistics = self.nested_statstics
        for i in range(1, len(statistics)):
            statistics[i].update_statistics(statistics[i - 1])
        return statistics

    @property
    def plot_vals(self) -> list[str]:
        plot_vals = []
        for key in self.full_data_statistic.to_dict():
            plot_vals.append(key)
            plot_vals.append(f"percentage_of_{key}")
        return plot_vals

    def skip_keys(self) -> list[str]:
        skip_keys = ["Loop", "Experiment"]
        return skip_keys

    def to_df_row_dicts(self) -> tuple[list[dict], list[str]]:
        out_results = []
        full_dict = self.full_data_statistic.to_dict()
        percentage_dict_keys = full_dict.keys()
        for i, statistic in enumerate(self.statistics):
            temp_dict = statistic.to_dict()
            for key in percentage_dict_keys:
                temp_dict[f"percentage_of_{key}"] = temp_dict[key] / full_dict[key]
            temp_dict["Loop"] = i
            temp_dict["Experiment"] = self.raw_path.name

            skip_keys = list(temp_dict.keys()) + CONFIGSKIPKEYS
            if self.config is not None:
                temp_dict.update(self.config.to_dict())
            out_results.append(temp_dict)
        return out_results, skip_keys

    def plot_experiment(self, output_path: Path | str | None = None):
        df = pd.DataFrame(self.statistics.to_df_row_dicts())
        for key in df.columns:
            if key in ["Loop", "Experiment"]:
                continue
            if self.config is not None and key in self.config.to_dict().keys():
                continue

            fig, axs = plt.subplots()
            sns.lineplot(df, x="Loop", y=key, ax=axs)
            if output_path is None:
                plt.show()
            else:
                output_path = Path(output_path)
                os.makedirs(output_path, exist_ok=True)

                plt.savefig(output_path / f"{self.raw_path.name}-{key}.png")
            plt.close("all")


class SingleExperimentResults:
    def __init__(self, experiment_path: Path):
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

    def to_df_row_dicts(
        self, plot_fct=lambda x: x["foreground_mean"]["Dice"]
    ) -> list[dict]:
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
    ) -> list[dict]:
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


class MultiExperimentAnalysis:
    def __init__(
        self,
        base_results_path: Path,
        base_raw_path: Path | None = None,
        filter_final: bool = True,
    ):
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
            rel_raw_path: str = experiment.experiment_path.__str__()[
                len(self.base_raw_path.__str__()) + 1 :
            ]
            rel_raw_path = rel_raw_path.replace("/nnActive_results/", "/nnUNet_raw/")
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

    def dataset_analyze_performance(
        self, unique_id: int, all_plots: bool = True, output_dir: Path = Path(".")
    ):
        dataset_results = [
            exp for exp in self.exp_results if exp.config.base_id == unique_id
        ]

        output_dir = output_dir / dataset_results[0].config.dataset / "performance"
        if not output_dir.is_dir():
            os.makedirs(output_dir)

        value = "Dice"
        if all_plots:
            plot_names = dataset_results[0].value_dict(plot_val=value).keys()
        else:
            plot_names = ["Mean Dice"]

        for plot_name in plot_names:
            plot_fct = dataset_results[0].value_dict(plot_val=value)[plot_name]
            y_fulls = dataset_results[0].full_dataset_performance(
                plot_fct.get_from_dict
            )

            df_row_dicts = []
            for exp in dataset_results:
                df_row_dicts.extend(
                    exp.to_df_row_dicts(plot_fct=plot_fct.get_from_dict)
                )

            df = pd.DataFrame(df_row_dicts)

            vals = [
                seperator
                for seperator in df.columns
                if seperator not in dataset_results[0].plot_skip_keys
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
                    y=dataset_results[0].plot_name,
                    hue=self.query_key,
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

                plt.savefig(
                    output_dir
                    / f"{dataset}-{plot_name_file}-{x_name}__{key_plot_file}.png"
                )

                x = "#Patches"
                x_name = "Patches"
                fig, axs = plt.subplots()
                sns.lineplot(
                    data=df_g,
                    x=x,
                    y=dataset_results[0].plot_name,
                    hue=self.query_key,
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
                plt.savefig(
                    output_dir
                    / f"{dataset}-{plot_name_file}-{x_name}__{key_plot_file}.png"
                )
                plt.close("all")

    def dataset_analyze_statistics(
        self, unique_id: int, all_plots: bool = True, output_dir: Path = Path(".")
    ):
        if not output_dir.is_dir():
            os.makedirs(output_dir)
        dataset_statistics = [
            exp for exp in self.exp_statistics if exp.base_id == unique_id
        ]

        output_dir = (
            output_dir / dataset_statistics[0].source_dataset_path.name / "statistics"
        )
        if not output_dir.is_dir():
            os.makedirs(output_dir)

        if all_plots:
            plot_names = dataset_statistics[0].plot_vals
        else:
            plot_names = ["percentage_of_voxels_foreground"]

        for plot_name in plot_names:
            df_row_dicts = []
            for exp in dataset_statistics:
                df_row_dict, skip_keys = exp.to_df_row_dicts()
                df_row_dicts.extend(df_row_dict)

            df = pd.DataFrame(df_row_dicts)

            vals = [seperator for seperator in df.columns if seperator not in skip_keys]
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
                    y=plot_name,
                    hue=self.query_key,
                    errorbar="sd",
                    ax=axs,
                    markers=True,
                    palette=PALETTE,
                )
                axs.set_ylabel(plot_name)
                axs.set_xticks(np.arange(0, key[max_loop_ind]))
                axs.legend(loc="best")
                axs.set_title(dataset)
                key_plot = tuple([k for i, k in enumerate(key) if i != dataset_ind])
                key_plot_file = f"{key_plot}".replace(" ", "")
                plot_name_file = plot_name.replace(" ", "")
                plot_x_name = x_name.replace("_", "")
                file_name = f"{dataset}-{plot_name_file}-{plot_x_name}__{key_plot_file}"
                file_name = file_name[:250]  # can exceed max filename!
                save_dir = output_dir
                if not save_dir.is_dir():
                    os.makedirs(save_dir)

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

        output_dir = (
            output_dir / dataset_results[0].config.dataset / "performance_statistics"
        )
        if not output_dir.is_dir():
            os.makedirs(output_dir)

        df_stat_dicts: list[dict] = []
        for exp in dataset_statistics:
            df_stat_dict, stat_skip_keys = exp.to_df_row_dicts()
            df_stat_dicts.extend(df_stat_dict)

        if all_plots:
            x_names = dataset_statistics[0].plot_vals
            y_names = dataset_results[0].value_dict(plot_val=value).keys()
        else:
            x_names = ["percentage_of_voxels_foreground"]
            y_names = ["Mean Dice"]

        for x_name, y_name in product(x_names, y_names):
            plot_fct = dataset_results[0].value_dict(plot_val=value)[y_name]
            y_fulls = dataset_results[0].full_dataset_performance(
                plot_fct.get_from_dict
            )

            df_results_dicts: list[dict] = []
            for exp in dataset_results:
                df_results_dicts.extend(
                    exp.to_df_row_dicts(plot_fct=plot_fct.get_from_dict)
                )

            results_skip_keys = dataset_results[0].plot_skip_keys

            indices = ["Experiment", "Loop"]
            merged_dicts = []
            for i in range(len(df_results_dicts)):
                merged_dict = df_results_dicts[i].copy()
                extended = False
                for j in range(len(df_stat_dicts)):
                    accept = True
                    for index in indices:
                        if merged_dict[index] != df_stat_dicts[j][index]:
                            accept = False
                    if accept:
                        merged_dict.update(df_stat_dicts[j])
                        extended = True
                        break
                if not extended:
                    raise ValueError(
                        "One dictionary in the list does not have a partner."
                    )
                else:
                    merged_dicts.append(merged_dict)

            df = pd.DataFrame(merged_dicts)

            vals = [
                seperator
                for seperator in df.columns
                if seperator not in (results_skip_keys + stat_skip_keys)
            ]

            dataset_ind = vals.index("dataset")

            # create plots for each unique setting for the respective dataset now
            for key, df_g in df.groupby(vals):
                dataset = key[dataset_ind]

                fig, axs = plt.subplots()
                sns.lineplot(
                    data=df_g,
                    x=x_name,
                    y=dataset_results[0].plot_name,
                    hue=self.query_key,
                    errorbar="sd",
                    ax=axs,
                    markers=True,
                    palette=PALETTE,
                )
                axs.set_ylabel(y_name)
                axs.legend(loc="best")
                axs.set_title(dataset)

                key_plot = tuple([k for i, k in enumerate(key) if i != dataset_ind])
                key_plot_file = f"{key_plot}".replace(" ", "")
                plot_name_file = y_name.replace(" ", "")
                plot_x_name = x_name.replace("_", "")
                file_name = f"{dataset}-{plot_name_file}-{plot_x_name}__{key_plot_file}"
                file_name = file_name[:250]  # can exceed max filename!
                save_dir = output_dir / plot_name_file
                if not save_dir.is_dir():
                    os.makedirs(save_dir)

                plt.savefig(save_dir / f"{file_name}.png")
                plt.close("all")

    def analyze_multi_datasets(
        self,
        output_dir: Path = Path("."),
        all_results_plots: bool = True,
        all_raw_plots: bool = True,
        all_combi_plots: bool = True,
    ):
        for dataset_id in self.unique_datasets:
            logger.log(
                f"Analaying results for experiments derived from dataset id {dataset_id}"
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

    # TODO: Doublecheck plot_experiment after this again!

    # exp_path = "/home/c817h/network/cluster-data/Dataset135_KiTS2021/nnUNet_raw/Dataset001_KiTS2021__patch-64_64_64__qs-20__unc-random-label__seed-12345"
    # exp_path = Path(exp_path)
    # statistics = SingleExperimentStastistics(exp_path)
    output_path = Path(__file__).parent.parent / "results" / "raw_plots"

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
