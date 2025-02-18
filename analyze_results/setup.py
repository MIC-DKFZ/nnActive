import os
from pathlib import Path

import pandas as pd
from loguru import logger

from nnactive.analyze.aggregate_results import pretty_auc
from nnactive.analyze.analysis import SettingAnalysis
from nnactive.utils.io import save_df_to_txt

small_dict = {
    "mutual information": "BALD",
    "power bald": "PowerBALD",
    "softrank bald": "SoftrankBALD",
    "pred entropy": "Predictive Entropy",
    "power pe": "PowerPE",
    "random": "Random",
    "random-label": "Random 66% FG",
    "random-label2": "Random 33% FG",
}

RENAMING_DICT = {}
keys = list(small_dict.keys())
for key in keys:
    RENAMING_DICT[key] = small_dict[key]
    RENAMING_DICT[key.replace(" ", "_")] = small_dict[key]

QM_TO_COLOR = {
    "BALD": "#bcbd22",  # Yellow-green
    "PowerBALD": "#ff7f0e",  # Orange
    "SoftrankBALD": "#7f7f7f",  # Gray
    "PowerPE": "#2ca02c",  # Green
    "Predictive Entropy": "#1f77b4",  # Blue
    "Random": "#9467bd",  # Purple
    "Random 66% FG": "#e377c2",  # Light Red
    "Random 33% FG": "#d62728",  # Red
}

BASEPATH = Path(
    "/home/c817h/Documents/projects/nnactive_project/nnactive/results/horeka_rsync_final/"
)


#################### Fixing Hippocampus Experiments automatically ####################
hippcompaus_path = BASEPATH / "Dataset004_Hippocampus"
datapath = (
    hippcompaus_path / "patch-20_20_20__sb-random-label2-all-classes__sbs-20__qs-20"
)
save_path = hippcompaus_path / f"{datapath.name}__5loops"
os.makedirs(save_path, exist_ok=True)

analysis = SettingAnalysis.load(datapath / "analysis.pkl")

df_filter = analysis.df[analysis.df["Loop"] <= 4]
df_filter[analysis.max_loops_key] = 5
df_filter = df_filter.reset_index()

analysis.df = df_filter

analysis.save(save_path / "analysis.pkl")

# overview metrics
auc_df = analysis.compute_auc_df()


analysis.save(save_path=save_path / "analysis.pkl")

# overview metrics
auc_df = analysis.compute_auc_df()
# pprint(auc_df)
auc_df.to_json(save_path / "auc.json")
save_df_to_txt(auc_df, save_path / "auc.txt")
save_df_to_txt(
    pretty_auc(pd.read_json(save_path / "auc.json"), seeds=True),
    save_path / "auc_pretty.txt",
)

ppm = analysis.compute_pairwise_penalty("Mean Dice")
ppm.plot_pairwise_matrix(ppm.matrix, savepath=save_path / "ppm.png")
ppm.save(save_path / "ppm.json")

trainer = str(analysis.df["trainer"].unique()[0])
trainer_use = "nnUNetTrainer"
if len(trainer.split("_")) > 1:
    epochs = trainer.split("_")
    trainer_use = f"{trainer_use}_{epochs[-1]}"
    logger.info(f"Using Full Performance Trainer: {trainer_use}")
trainers = [
    f.label for f in analysis.full_performance_dict[analysis.main_performance_key]
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
    betas_df.to_json(save_path / "beta.json")
    save_df_to_txt(betas_df, save_path / "beta.txt")

# overview plots
selected_classes = None
n_performance_cols = 3
if selected_classes is None:
    selected_classes = [
        int(i.split(" ")[1]) for i in analysis.df.columns if i.startswith("Class")
    ][:3]
    while len(selected_classes) < n_performance_cols:
        selected_classes.append(None)
