from __future__ import annotations

import os
from functools import cached_property
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pydantic.dataclasses import dataclass

from nnactive.config.struct import ActiveConfig
from nnactive.data import Patch
from nnactive.loops.loading import get_nested_patches_from_loop_files
from nnactive.nnunet.utils import get_raw_path
from nnactive.paths import set_raw_paths
from nnactive.utils.io import load_json, load_label_map, save_json
from nnactive.utils.patches import get_slices_for_file_from_patch
from nnactive.utils.pyutils import get_clean_dataclass_dict

CONFIGSKIPKEYS = ["seed", "uncertainty", "#Patches", "queries_from_experiment"]


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

    @cached_property
    def efficient_nested_patch_labels(self) -> list[list[dict[int, int]]]:
        files = [
            patch.file for loop_patches in self.nested_patches for patch in loop_patches
        ]
        nested_labels = [
            [None] * len(loop_patches) for loop_patches in self.nested_patches
        ]
        for file in files:
            label_image = load_label_map(
                file, self.source_dataset_path / "labelsTr", ""
            )
            for i, loop_patches in enumerate(self.nested_patches):
                for j, patch in enumerate(loop_patches):
                    if patch.file == file:
                        patch_access = get_slices_for_file_from_patch(
                            [patch], patch.file
                        )[0]
                        patch_labels = label_image[patch_access]
                        # fill statistics
                        unique_cls, counts = np.unique(patch_labels, return_counts=True)

                        nested_labels[i][j] = {
                            int(unique_cl): int(count)
                            for unique_cl, count in zip(unique_cls, counts)
                        }
        return nested_labels

    @property
    def nested_statstics(self) -> list[Statistics]:
        nested_statistics = []
        for loop_labels, loop_patches in zip(
            self.efficient_nested_patch_labels, self.nested_patches
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
        plot_vals.append("avg_percentage_of_voxels_fg_cls")
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

            avg_fg_classes = []
            for key in temp_dict:
                if key.startswith("percentage_of_voxels_per_cls") and int(
                    key.split("_")[-1]
                ) not in [self.full_data_statistic.background_label]:
                    avg_fg_classes.append(temp_dict[key])
            avg_fg_classes = float(np.array(avg_fg_classes).mean())
            temp_dict["avg_percentage_of_voxels_fg_cls"] = avg_fg_classes
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
