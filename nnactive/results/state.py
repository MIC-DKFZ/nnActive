from __future__ import annotations

import json
from pathlib import Path

from pydantic.dataclasses import dataclass

from nnactive.loops.loading import get_sorted_loop_files
from nnactive.nnunet.utils import get_raw_path
from nnactive.results.utils import get_results_folder

# from typing_extensions import Self


FILENAME = "state.json"


# TODO: This should/ could be redone to a more exhaustive list with all steps!
# e.g. loop, update_data, trainings, uncertainty, query...
@dataclass
class State:
    dataset_id: int
    loop: int = 0
    preprocess: bool = False
    training: bool = False
    get_performance: bool = False
    query: bool = False
    update_data: bool = False

    def new_loop(self):
        self.loop += 1
        self.preprocess = False
        self.training = False
        self.get_performance = False
        self.query = False
        self.update_data = False

    def save_state(self):
        fn = get_results_folder(self.dataset_id) / FILENAME
        with open(fn, "w") as file:
            # TODO: SAVING DATACLASS Delete pydantic key value pairs here
            json.dump(self.__dict__, file)

    def verify(self):
        # if we are in loop X, we want to have loop_XXX.json
        loop_val = len(get_sorted_loop_files(get_raw_path(self.dataset_id))) - 1
        if self.query:
            assert loop_val == self.loop + 1
        else:
            assert loop_val == self.loop

        if self.training:
            assert self.preprocess  # preprocessing before training is required

        # further we may want validation results
        if self.get_performance:
            assert (
                self.training
            )  # performance for loop requires trained models for this loop
        if self.query:
            assert self.training  # query for loop requires trained models for this loop
        if self.update_data:
            assert self.query  # updating data requires loop_XXX.json file

    @classmethod
    def from_json(cls, path: Path) -> State:
        with open(path, "r") as file:
            parsed = json.load(file)
        state = State(**parsed)
        return state

    @classmethod
    def get_id_state(cls, id: int) -> State:
        fn = get_results_folder(id) / "state.json"
        state = State.from_json(fn)
        state.verify()
        return state

    @staticmethod
    def filename():
        return FILENAME
