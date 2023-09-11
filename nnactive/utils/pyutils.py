from pydantic import dataclasses


def get_clean_dataclass_dict(data: dataclasses) -> dict:
    datadict = data.__dict__
    popkeys = []
    for key in datadict:
        if isinstance(key, str):
            if key.startswith("__") and key.endswith("__"):
                popkeys.append(key)
    for key in popkeys:
        datadict.pop(key)
    return datadict
