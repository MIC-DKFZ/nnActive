import json
from pathlib import Path

from pydantic import dataclasses


def get_clean_dataclass_dict(data: dataclasses):
    datadict = data.__dict__
    popkeys = []
    for key in datadict:
        if isinstance(key, str):
            if key.startswith("__") and key.endswith("__"):
                popkeys.append(key)
    for key in popkeys:
        datadict.pop(key)
    return datadict


def save_dataclass_to_json(data: dataclasses, filepath: Path):
    """Saves the dataclass as json file without __keys__

    Args:
        data (dataclasses): _description_
        filepath (Path): _description_
    """
    with open(filepath, "w") as file:
        datadict = get_clean_dataclass_dict(data)
        json.dump(datadict, file, indent=4)
