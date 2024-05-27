import json
from datetime import datetime

import numpy as np
import pandas as pd
import wandb

api = wandb.Api()
all_runs = api.runs("ctlueth/nnactive")
finished_runs = filter(lambda run: run.state == "finished", all_runs)
finished_runs = list(
    filter(
        lambda run: datetime.fromisoformat(run.metadata["startedAt"])
        > datetime(2024, 2, 20),
        finished_runs,
    )
)


def get_epoch_time(run):
    epoch_start_timestamps = [
        row["epoch_start_timestamps"]
        for row in run.scan_history(keys=["epoch_start_timestamps"])
    ]
    epoch_end_timestamps = [
        row["epoch_end_timestamps"]
        for row in run.scan_history(keys=["epoch_end_timestamps"])
    ]

    epoch_start_timestamps = np.array(epoch_start_timestamps)
    epoch_end_timestamps = np.array(epoch_end_timestamps)

    epoch_time = np.mean(epoch_end_timestamps - epoch_start_timestamps)
    return epoch_time


kits_small_a100_name = "silver-vortex-109"
brats_small_a100_name = "still-breeze-111"
run_names = [kits_small_a100_name, brats_small_a100_name]
finished_runs = list(filter(lambda run: run.name in run_names, finished_runs))
# import IPython

# IPython.embed()
info = pd.DataFrame.from_records(
    map(
        lambda run: {
            "dataset": json.loads(run.json_config)["dataset"]["value"],
            "runtime": run.summary["_runtime"],
            "query_from_probs_time": run.summary["query_from_probs_time"]["mean"],
            "predict_from_data_iterator_time": run.summary[
                "predict_from_data_iterator_time"
            ]["mean"],
            "compose_query_of_patches_time": run.summary[
                "compose_query_of_patches_time"
            ]["mean"],
            "epoch_time": get_epoch_time(run),
        },
        finished_runs,
    )
)
print(info.to_csv())
