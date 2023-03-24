import os
from pathlib import Path

from nnactive.paths import get_nnActive_results


def convert_id_to_dataset_name(id: int) -> str:
    """Returns the name for the folder corresponding to id in $nnActive_results

    Args:
        id (int): Dataset/Experiment Identifier
    """
    results_path = get_nnActive_results()
    prefix = f"Dataset{id:03d}"
    matching = []
    for file in os.listdir(results_path):
        if file.startswith(prefix):
            matching.append(file)

    if len(matching) == 1:
        return matching[0]
    elif len(matching) == 0:
        raise FileNotFoundError("No folder fitting ID")
    raise NotImplementedError(f"Too many potential folders for ID {id}")


def get_results_folder(id: int) -> Path:
    return get_nnActive_results() / convert_id_to_dataset_name(id)
