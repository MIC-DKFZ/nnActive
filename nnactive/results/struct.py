from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from pydantic.dataclasses import dataclass


@dataclass
class ActiveConfig:
    starting_budget: str
    trainer: str  # e.g. nnUNetDebugTrainer
    query_size: int
    patch_size: Union[tuple[int, int, int], str]

    @classmethod
    def from_json(cls, path: Path) -> ActiveConfig:
        with open(path, "r") as file:
            parsed = json.load(file)
        return parsed
