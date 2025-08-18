import os
from pathlib import Path

import pandas as pd

from nnactive.analyze.analysis import GridPlotter, SettingAnalysis
from nnactive.analyze.analyze_results import MultiExperimentAnalysis
from nnactive.utils.io import load_pickle

if __name__ == "__main__":
    paths = [
        "/home/c817h/Documents/projects/nnactive_project/nnactive/results/horeka-main/Dataset216_AMOS2022_task1/patch-32_74_74__sb-random-label2-all-classes__sbs-200__qs-200",
        "/home/c817h/Documents/projects/nnactive_project/nnactive/results/horeka-main/Dataset216_AMOS2022_task1/patch-1000_1000_1000__qs-15__tr-nnUNetTrainer_200epochs",
    ]

    output_dir = "/home/c817h/Documents/projects/nnactive_project/nnactive/results/horeka-main/Dataset216_AMOS2022_task1/compare"

    output_dir = Path(output_dir)

    # paths = [
    #     "/home/c817h/Documents/projects/nnactive_project/nnactive/results/horeka-main/Dataset135_KiTS2021/patch-64_64_64__sb-random-label2-all-classes__sbs-200__qs-200",
    #     "/home/c817h/Documents/projects/nnactive_project/nnactive/results/horeka-main/Dataset135_KiTS2021/patch-1000_1000_1000__qs-22__tr-nnUNetTrainer_200epochs",
    # ]

    # output_dir = "/home/c817h/Documents/projects/nnactive_project/nnactive/results/horeka-main/Dataset135_KiTS2021/compare"

    output_dir = Path(output_dir)

    os.makedirs(output_dir, exist_ok=True)
    output_dir = output_dir

    for i in range(len(paths)):
        paths[i] = Path(paths[i])

    data_dicts = []
    for path in paths:
        data_dicts.append(pd.read_pickle(path / "analysis_df.pkl"))

    setting: SettingAnalysis = load_pickle(paths[0] / "analysis.pkl")

    data = pd.concat(data_dicts, axis=0)
    setting.df = data
    grid = MultiExperimentAnalysis.generate_grid([i + 1 for i in range(3)])

    setting.save_overview_plots(output_dir, grid, style="pre_suffix")
