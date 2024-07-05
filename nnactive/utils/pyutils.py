from collections import defaultdict
from copy import deepcopy
from typing import Any, Hashable

from pydantic import dataclasses


def invert_dict(d: dict[list[Any]]) -> dict[list[Any]]:
    """Inverts a dictionary where values are lists, mapping elements of those lists to their corresponding keys.

    Args:
        d (dict[list[Any]]): The input dictionary to be inverted. Keys are strings, and values are lists of elements.

    Returns:
        dict[list[Any]]: A dictionary where keys are elements from the input lists, and values are lists of keys from the input dictionary
        that correspond to those elements.

    Example:
        >>> original_dict = {'a': [1, 2], 'b': [2, 3], 'c': [1, 3]}
        >>> inverted_dict = invert_dict(original_dict)
        >>> print(inverted_dict)
        {1: ['a', 'c'], 2: ['a', 'b'], 3: ['b', 'c']}
    """
    inverted_dict = defaultdict(list)
    for key, values in d.items():
        for value in values:
            inverted_dict[value].append(key)
    return inverted_dict


def get_clean_dataclass_dict(data: dataclasses) -> dict:
    datadict = deepcopy(data.__dict__)
    popkeys = []
    for key in datadict:
        if isinstance(key, str):
            if key.startswith("__") and key.endswith("__"):
                popkeys.append(key)
    for key in popkeys:
        datadict.pop(key)
    return datadict


def merge_dict_lists_on_indices(
    init_dict: list[dict], update_dict: list[dict], indices: list[Hashable]
) -> list[dict]:
    merged_dicts = []
    for i in range(len(init_dict)):
        merged_dict = init_dict[i].copy()
        extended = False
        for j in range(len(update_dict)):
            accept = True
            for index in indices:
                if merged_dict[index] != update_dict[j][index]:
                    accept = False
            if accept:
                merged_dict.update(update_dict[j])
                extended = True
                break
        if not extended:
            raise ValueError("One dictionary in the list does not have a partner.")
        else:
            merged_dicts.append(merged_dict)
    return merged_dicts
