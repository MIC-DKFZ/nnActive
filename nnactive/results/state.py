from __future__ import annotations

import json
from pathlib import Path

from loguru import logger
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name
from pydantic.dataclasses import dataclass

from nnactive.config.struct import ActiveConfig
from nnactive.loops.loading import get_sorted_loop_files
from nnactive.nnunet.utils import get_raw_path
from nnactive.paths import get_nnActive_results
from nnactive.results.utils import get_results_folder
from nnactive.utils.io import save_dataclass_to_json


@dataclass
class State:
    dataset_id: int
    name: str
    loop: int = 0
    preprocess: bool = False
    training: bool = False
    get_performance: bool = False
    pred_tr: bool = False
    query: bool = False
    update_data: bool = False
    in_progress: bool = False

    def new_loop(self):
        self.loop += 1
        self.preprocess = False
        self.training = False
        self.get_performance = False
        self.pred_tr = False
        self.query = False
        self.update_data = False
        self.in_progress = False

    def reset(self):
        self.loop = 0
        self.preprocess = False
        self.training = False
        self.get_performance = False
        self.pred_tr = False
        self.query = False
        self.update_data = False
        self.in_progress = False

    def save_state(self):
        try:
            fn = get_results_folder(self.dataset_id) / State.filename()
        except FileNotFoundError:
            save_path: Path = (
                get_nnActive_results() / f"Dataset{self.dataset_id:03d}_{self.name}"
            )
            logger.info(f"Creating Path: {save_path}")
            save_path.mkdir(parents=True)
            fn = save_path / State.filename()
        save_dataclass_to_json(self, fn)

    def verify(self, check_loop_files: bool = True):
        if check_loop_files:
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
        if self.pred_tr:
            assert self.training
        if self.query:
            assert self.training  # query for loop requires trained models for this loop
            # TODO: better version
            # assert self.pred_tr
        if self.update_data:
            assert self.query  # updating data requires loop_XXX.json file

    @classmethod
    def from_json(cls, path: Path) -> State:
        with open(path, "r") as file:
            parsed = json.load(file)
        state = State(**parsed)
        return state

    @classmethod
    def get_id_state(cls, id: int, verify: bool = True) -> State:
        fn = get_results_folder(id) / State.filename()
        state = State.from_json(fn)
        if verify:
            state.verify()
        return state

    @classmethod
    def latest(cls, config: ActiveConfig) -> State:
        state_files = sorted(
            list(
                (config.group_dir() / "nnActive_results").glob(
                    f"Dataset*{config.name()}/state.json"
                )
            )
        )
        assert state_files, f"No state files found for {config.name()}"
        return State.from_json(state_files[-1])

    @classmethod
    def next_free_state(cls, config: ActiveConfig) -> State:
        state_files = list(
            map(
                lambda path: int(path.name[7:10]),
                list((config.group_dir() / "nnActive_results").glob("Dataset*")),
            )
        )
        if not state_files:
            return State(name=config.name(), dataset_id=0)

        return State(name=config.name(), dataset_id=max(state_files) + 1)

    @staticmethod
    def experiment_finished(path: Path) -> bool:
        """Returns True if the experiment is finished.

        Args:
            path (Path): Path to nnActive_results/DatasetXXX_name
        """
        state = State.from_json(path / State.filename())
        config = ActiveConfig.from_json(path / ActiveConfig.filename())
        if state.loop >= config.query_steps - 1:
            verified = False
            try:
                state.verify(check_loop_files=False)
                verified = True
            except AssertionError as e:
                pass
            if verified:
                return True
        return False

    @staticmethod
    def filename():
        return "state.json"
