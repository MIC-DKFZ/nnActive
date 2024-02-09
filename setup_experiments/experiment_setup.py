import os
import shutil
import subprocess
from argparse import ArgumentParser
from itertools import product
from pathlib import Path
from typing import Union

from nnactive.config import ActiveConfig
from nnactive.nnunet.utils import get_patch_size
from nnactive.results.state import State

DEFAULT_QUERIES = (
    "pred_entropy",
    "mutual_information",
    "random",
    "random-label",
)


class DatasetSetup:
    def __init__(
        self,
        base_id: int,
        query_steps: int,
        query_size: int,
        trainer: str = "nnActiveTrainer_5epochs",
        starting_budget: str = "random",
        patch_size: Union[None, list[int]] = None,
        pre_suffix="",
        add_validation="",
        add_uncertainty="",
        uncertainties: Union[list, tuple] = DEFAULT_QUERIES,
        seeds: Union[list, tuple] = (12345, 12346, 12347),
        num_processes: int = 4,
        full_folds: int = 5,
        train_folds: int = 5,
        force_override: bool = False,
        agg_stride: Union[int, list[int]] = 1,
        patch_overlap: float = 0,
        additional_overlap: float = 0.2,
    ):
        self.base_id = base_id
        # standard values
        self.force_override = force_override
        self.train_folds = train_folds
        self.full_folds = full_folds
        self.num_processes = num_processes
        self.add_uncertainty = add_uncertainty
        self.add_validation = add_validation
        self.agg_stride = agg_stride
        self.patch_overlap = patch_overlap
        self.additional_overlap = additional_overlap

        # standard values iterated over
        self.seeds = seeds
        self.uncertainties = uncertainties

        self.base_id = base_id
        self.query_steps = query_steps
        self.query_size = query_size
        self.trainer = trainer
        self.starting_budget = starting_budget
        self.patch_size = (
            get_patch_size(self.base_id) if patch_size is None else patch_size
        )
        self.pre_suffix = pre_suffix

    @property
    def base_dataset_name(self):
        for folder in self.data_path.iterdir():
            if folder.name.startswith(f"Dataset{self.base_id:03d}"):
                return folder.name
        raise RuntimeError(f"No Dataset with corresponding base id: {self.base_id}")

    @property
    def data_path(self):
        data_path = os.getenv("nnUNet_raw")
        if data_path is None:
            raise ValueError("OS variable nnUNet_raw is not set.")
        return Path(data_path)

    @property
    def vals(self):
        return [val for val in product(self.uncertainties, self.seeds)]

    @property
    def existing_dsets(self):
        existing_dsets = [
            folder.name
            for folder in self.data_path.iterdir()
            if folder.is_dir() and folder.name.startswith("Dataset")
        ]
        return existing_dsets

    def check_dataset_id(self, output_id: int):
        dset_name = f"Dataset{output_id:03d}"
        if any([dset.startswith(dset_name) for dset in self.existing_dsets]):
            print(
                f"Dataset beginning with '{dset_name}' already exists in {self.data_path}."
            )
            if not self.force_override:
                return False
            else:
                os_variables = [
                    "nnUNet_results",
                    "nnUNet_raw",
                    "nnUNet_preprocessed",
                    "nnActive_results",
                ]
                for os_variable in os_variables:
                    base_path = Path(os.getenv(os_variable))
                    rm_dirs = [
                        folder
                        for folder in base_path.iterdir()
                        if folder.name.startswith(f"Dataset{output_id:03d}")
                    ]
                    for rm_dir in rm_dirs:
                        print(f"Deleting folder: {rm_dir}")
                        shutil.rmtree(rm_dir)
        return True

    def convert_dset(self, dataset_id: int, seed: int, uncertainty: str):
        print("Converting Dataset")
        ex_command = "nnactive convert"
        past_suffix = f"__unc-{uncertainty}__seed-{seed}"
        name_suffix = self.pre_suffix + past_suffix
        # DO all of this based on dictionaries!
        patch_call = str(self.patch_size)
        for rm in ["(", ")", "[", "]", ","]:
            patch_call = patch_call.replace(rm, "")
        ex_call = f"{ex_command} -d {self.base_id} -o {dataset_id} --strategy {self.starting_budget} --seed {seed} --num-patches {self.query_size} --name-suffix {name_suffix} --patch-size {patch_call}"
        print(ex_call)
        subprocess.run(ex_call, shell=True, check=True)

    def prepare_dset(self, datset_id: int):
        subprocess.run(
            f"nnUNetv2_extract_fingerprint -d {datset_id}  -np {self.num_processes}",
            shell=True,
            check=True,
        )
        subprocess.run(
            f"nnUNetv2_plan_experiment -d {datset_id} -np {self.num_processes}",
            shell=True,
            check=True,
        )

    def setup_al(
        self,
        dataset_id: int,
        seed: int,
        uncertainty: str,
    ):
        config = ActiveConfig(
            trainer=self.trainer,
            patch_size=self.patch_size,
            uncertainty=uncertainty,
            query_size=self.query_size,
            query_steps=self.query_steps,
            starting_budget=self.starting_budget,
            seed=seed,
            num_processes=self.num_processes,
            dataset=self.base_dataset_name,
            train_folds=self.train_folds,
            full_folds=self.full_folds,
            add_uncertainty=self.add_uncertainty,
            add_validation=self.add_validation,
            agg_stride=self.agg_stride,
            patch_overlap=self.patch_overlap,
            additional_overlap=self.additional_overlap,
        )
        config.save_id(dataset_id)
        state = State(dataset_id=dataset_id)
        state.save_state()

    def rollout(
        self, start_id: int, num_experiments: int | None = None, debug: bool = False
    ):
        for i, (unc, seed) in enumerate(self.vals):
            if num_experiments:
                if i >= num_experiments:
                    break
            output_id = start_id + i
            if self.check_dataset_id(output_id):
                if debug:
                    print(f"Creating Dataset{output_id:3d}....")
                    continue
                self.convert_dset(output_id, seed, unc)
                self.prepare_dset(output_id)
                self.setup_al(output_id, seed, unc)

    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument(
            "--num_experiments",
            type=int,
            default=None,
            help="How many experiments to create counting from first",
        )
        parser.add_argument(
            "--debug",
            "-d",
            action="store_true",
            help="Activate Debug Modus. No Dataset is created.",
        )
        parser.add_argument(
            "--force_override",
            action="store_true",
            help="Overrides all Experiments defined in this batch. USE WITH CAUTION!",
        )
        return parser
