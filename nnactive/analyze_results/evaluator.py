from pathlib import Path

import pandas as pd
from setup import BASEPATH

from nnactive.analyze.analyze_results import SettingAnalysis

FULL_SETTINGS = {
    "ACDC": {
        "Low": {
            "Main": "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-30__qs-30",
            "QSx1/2": "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-30__qs-15_revision",
            "QSx2": "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-30__qs-60_revision",
            "Patchx1/2": "Dataset027_ACDC/patch-2_20_20__sb-random-label2-all-classes__sbs-30__qs-30",
            "PowerBALD_beta": "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-30__qs-30",  # from specific Path
            "500 Epochs": None,
            "Precomputed": None,
        },
        "Medium": {
            "Main": "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-60__qs-60",
            "QSx1/2": None,
            "QSx2": None,
            "Patchx1/2": "Dataset027_ACDC/patch-2_20_20__sb-random-label2-all-classes__sbs-60__qs-60",
            "PowerBALD_beta": "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-60__qs-60",  # from specific Path
            "500 Epochs": None,
            "Precomputed": None,
        },
        "High": {
            "Main": "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-90__qs-90",
            "QSx1/2": "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-90__qs-45_revision",
            "QSx2": "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-90__qs-180_revision",
            "Patchx1/2": "Dataset027_ACDC/patch-2_20_20__sb-random-label2-all-classes__sbs-90__qs-90",
            "PowerBALD_beta": "Dataset027_ACDC/patch-4_40_40__sb-random-label2-all-classes__sbs-90__qs-90",  # from specific Path
            "500 Epochs": None,
            "Precomputed": None,
        },
    },
    "AMOS": {
        "Low": {
            "Main": "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-40__qs-40",
            "QSx1/2": "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-40__qs-20",
            "QSx2": "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-40__qs-80",
            "Patchx1/2": "Dataset216_AMOS2022_task1/patch-16_32_32__sb-random-label2-all-classes__sbs-40__qs-40",
            "PowerBALD_beta": "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-40__qs-40",  # from specific Path
            # "500 Epochs": None,
            "500 Epochs": "Dataset216_AMOS2022_task1/tr-nnActiveTrainer_500epochs__patch-32_74_74__sb-random-label2-all-classes__sbs-40__qs-40",
            "Precomputed": None,
        },
        "Medium": {
            "Main": "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-200__qs-200",
            "QSx1/2": None,
            "QSx2": None,
            "Patchx1/2": "Dataset216_AMOS2022_task1/patch-16_32_32__sb-random-label2-all-classes__sbs-200__qs-200",
            "PowerBALD_beta": "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-200__qs-200",  # from specific Path
            "500 Epochs": "Dataset216_AMOS2022_task1/tr-nnActiveTrainer_500epochs__patch-32_74_74__sb-random-label2-all-classes__sbs-200__qs-200",
            "Precomputed": "Dataset216_AMOS2022_task1/tr-nnActiveTrainer_500epochs__patch-32_74_74__sb-random-label2-all-classes__sbs-200__qs-200__precomputed-queries",
        },
        "High": {
            "Main": "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-500__qs-500",
            "QSx1/2": "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-500__qs-250",
            "QSx2": "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-500__qs-1000",
            "Patchx1/2": "Dataset216_AMOS2022_task1/patch-16_32_32__sb-random-label2-all-classes__sbs-500__qs-500",
            "PowerBALD_beta": "Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-500__qs-500",  # from specific Path
            "500 Epochs": "Dataset216_AMOS2022_task1/tr-nnActiveTrainer_500epochs__patch-32_74_74__sb-random-label2-all-classes__sbs-500__qs-500",
            "Precomputed": "Dataset216_AMOS2022_task1/tr-nnActiveTrainer_500epochs__patch-32_74_74__sb-random-label2-all-classes__sbs-500__qs-500__precomputed-queries",
        },
    },
    "Hippocampus": {
        "Low": {
            "Main": "Dataset004_Hippocampus/patch-20_20_20__sb-random-label2-all-classes__sbs-20__qs-20__5loops",
            "QSx1/2": None,
            "QSx2": None,
            "Patchx1/2": "Dataset004_Hippocampus/patch-10_10_10__sb-random-label2-all-classes__sbs-20__qs-20",
            "PowerBALD_beta": None,
            "500 Epochs": None,
            "Precomputed": None,
        },
        "Medium": {
            "Main": "Dataset004_Hippocampus/patch-20_20_20__sb-random-label2-all-classes__sbs-40__qs-40",
            "QSx1/2": None,
            "QSx2": None,
            "Patchx1/2": "Dataset004_Hippocampus/patch-10_10_10__sb-random-label2-all-classes__sbs-40__qs-40",
            "PowerBALD_beta": None,
            "500 Epochs": None,
            "Precomputed": None,
        },
        "High": {
            "Main": "Dataset004_Hippocampus/patch-20_20_20__sb-random-label2-all-classes__sbs-60__qs-60",
            "QSx1/2": None,
            "QSx2": None,
            "Patchx1/2": "Dataset004_Hippocampus/patch-10_10_10__sb-random-label2-all-classes__sbs-60__qs-60",
            "PowerBALD_beta": None,
            "500 Epochs": None,
            "Precomputed": None,
        },
    },
    "KiTS": {
        "Low": {
            "Main": "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-40__qs-40",
            "QSx1/2": "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-40__qs-20",
            "QSx2": "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-40__qs-80",
            "Patchx1/2": "Dataset135_KiTS2021/patch-32_32_32__sb-random-label2-all-classes__sbs-40__qs-40",
            "PowerBALD_beta": "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-40__qs-40",  # from specific Path
            "500 Epochs": None,
            "Precomputed": None,
        },
        "Medium": {
            "Main": "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-200__qs-200",
            "QSx1/2": None,
            "QSx2": None,
            "Patchx1/2": "Dataset135_KiTS2021/patch-32_32_32__sb-random-label2-all-classes__sbs-200__qs-200",
            "PowerBALD_beta": "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-200__qs-200",  # from specific Path
            "500 Epochs": "Dataset135_KiTS2021/tr-nnActiveTrainer_500epochs__patch-64_64_64__sb-random-label2-all-classes__sbs-200__qs-200",
            "Precomputed": "Dataset135_KiTS2021/tr-nnActiveTrainer_500epochs__patch-64_64_64__sb-random-label2-all-classes__sbs-200__qs-200__precomputed-queries",
        },
        "High": {
            "Main": "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-500__qs-500",
            "QSx1/2": "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-500__qs-250",
            "QSx2": "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-500__qs-1000",
            "Patchx1/2": "Dataset135_KiTS2021/patch-32_32_32__sb-random-label2-all-classes__sbs-500__qs-500",
            "PowerBALD_beta": "Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-500__qs-500",  # from specific Path
            "500 Epochs": "Dataset135_KiTS2021/tr-nnActiveTrainer_500epochs__patch-64_64_64__sb-random-label2-all-classes__sbs-500__qs-500",
            "Precomputed": "Dataset135_KiTS2021/tr-nnActiveTrainer_500epochs__patch-64_64_64__sb-random-label2-all-classes__sbs-500__qs-500__precomputed-queries",
        },
    },
}

for S in FULL_SETTINGS:
    for D in FULL_SETTINGS[S]:
        for M in FULL_SETTINGS[S][D]:
            if FULL_SETTINGS[S][D][M] is not None:
                FULL_SETTINGS[S][D][M] = BASEPATH / FULL_SETTINGS[S][D][M]
            else:
                FULL_SETTINGS[S][D][M] = None


def load_settings(
    settings: dict[str, dict[str, dict[str, Path | None]]], comparative: bool = False
) -> dict[str, dict[str, dict[str, SettingAnalysis]]]:
    out_settings: dict[str, dict[str, dict[str, SettingAnalysis]]] = {}
    remove_dsets = []
    for D in settings:
        out_settings[D] = {}
        for S in settings[D]:
            remove_budgets = set()
            out_settings[D][S] = {}
            for M in settings[D][S]:
                if settings[D][S][M] is not None:
                    print(f"Loading {D} {S} {M}")
                    analysis_file = settings[D][S][M] / "analysis.pkl"
                    if analysis_file.exists():
                        out_settings[D][S][M] = SettingAnalysis.load(
                            settings[D][S][M] / "analysis.pkl"
                        )
                    else:
                        print(
                            f"Analysis file not found for {D} {S} {M}: {analysis_file}"
                        )
                        remove_budgets.add(S)
                else:
                    print(f"Skipping {D} {S} {M}: No path provided")
                    remove_budgets.add(S)

            for M in remove_budgets:
                del out_settings[D][S]
        if len(out_settings[D]) == 0:
            remove_dsets.append(D)

    for D in remove_dsets:
        del out_settings[D]

    if comparative:
        ensure_overlap(out_settings)
    return out_settings


def ensure_overlap(out_settings: dict[str, dict[str, dict[str, SettingAnalysis]]]):
    """Ensure that all settings compute values over overlapping budget values.
    This is done for each label-regime.

    Args:
        out_settings (dict[str, dict[str, dict[str, SettingAnalysis]]]): The settings to ensure overlap for.
    """
    for D in out_settings:
        for S in out_settings[D]:
            overlapping_budgets = []
            for M in out_settings[D][S]:
                overlapping_budgets.append(
                    out_settings[D][S][M].df[out_settings[D][S][M].budget_key].unique()
                )
            if len(overlapping_budgets) == 0:
                continue
            overlapping_budgets = list(set.intersection(*map(set, overlapping_budgets)))
            for M in out_settings[D][S]:
                out_settings[D][S][M].df = out_settings[D][S][M].df[
                    out_settings[D][S][M]
                    .df[out_settings[D][S][M].budget_key]
                    .isin(overlapping_budgets)
                ]


def get_settings_for_combination(key_combination: list[str] | str) -> dict:
    """Returns the settings for the given key combination as pre-defined.
    Only returns dataset and budgets that have all keys in the key combination.
    """
    result = {}
    if isinstance(key_combination, str):
        key_combination = [key_combination]
    for dataset, levels in FULL_SETTINGS.items():
        for level, entries in levels.items():
            values = [entries.get(key) for key in key_combination if key in entries]

            if all(values):  # Ensure all keys have non-None value
                result.setdefault(dataset, {}).setdefault(
                    level, dict(zip(key_combination, [None] * len(key_combination)))
                )
                for key, value in zip(key_combination, values):
                    result[dataset][level][key] = value

    return result


def rename_settings_in_analysis(
    setting_analyses: dict[str, dict[str, dict[str, SettingAnalysis]]],
    rename_settings: dict[str, str],
):
    """
    Rename 3rd layer (setting) in the dictionary and keeps order.
    """
    for d in setting_analyses:
        for b in setting_analyses[d]:
            ordered_renames_dict = {
                key: rename_settings.get(key, key)
                for key in setting_analyses[d][b].keys()
            }

            for old_key, new_key in ordered_renames_dict.items():
                setting_analyses[d][b][new_key] = setting_analyses[d][b].pop(old_key)
