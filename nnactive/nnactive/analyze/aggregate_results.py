import numpy as np
import pandas as pd

from nnactive.utils.io import save_df_to_txt


def pretty_auc(data: pd.DataFrame, seeds=False) -> pd.DataFrame:
    """Return a table ready AUC from the AUC.json DataFrame."""
    out = data.copy(deep=True)
    final_cols = ["Mean DICE Final", "Mean DICE AUBC"]

    out["Mean DICE AUBC"] = data.apply(
        lambda row: f"""{row["('Mean Dice AUBC', 'mean')"]*100:0.2f} (+/- {row["('Mean Dice AUBC', 'std')"]*100:0.2f})""",
        axis=1,
    )
    out["Mean DICE Final"] = data.apply(
        lambda row: f"""{row["('Mean Dice Final', 'mean')"]*100:0.2f} (+/- {row["('Mean Dice Final', 'std')"]*100:0.2f})""",
        axis=1,
    )
    if seeds:
        out["#Seeds"] = data.apply(
            lambda row: row["('Mean Dice AUBC', 'count')"], axis=1
        )
        final_cols.append("#Seeds")

    out = out[final_cols]
    return out


if __name__ == "__main__":
    from pathlib import Path

    path = Path(
        "/home/c817h/Documents/projects/nnactive_project/nnactive/results/horeka-main/"
    )
    files = path.rglob("auc.json")
    data_dicts = []
    for file in files:
        data_dict = {}
        data_dict["df"] = pretty_auc(pd.read_json(file))
        data_dict["Dataset"] = file.parent.parent.name
        data_dict["Setting"] = (
            "Query Size " + file.parent.name.split("qs-")[1].split("__")[0]
        )
        data_dicts.append(data_dict)

    order = ["Dataset", "Setting", "df"]

    datasets = set([data["Dataset"] for data in data_dicts])

    whole_data = {}
    for dataset in datasets:
        whole_data[dataset] = {}
        for data in data_dicts:
            if data["Dataset"] == dataset:
                whole_data[dataset][data["Setting"]] = data["df"]
        whole_data[dataset] = pd.concat(
            whole_data[dataset],
            axis=1,
            keys=whole_data[dataset].keys(),
            names=["Setting"],
        )
        save_df_to_txt(whole_data[dataset], path / dataset / "entire_auc.txt")
    whole_data = pd.concat(
        whole_data, axis=1, keys=whole_data.keys(), names=["Dataset"]
    )
    save_df_to_txt(whole_data, path / "entire_auc.txt")

    with open(path / "entire_auc.md", "w") as f:
        f.write(whole_data.to_markdown())
