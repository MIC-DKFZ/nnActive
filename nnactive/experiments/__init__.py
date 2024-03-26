from itertools import product
from typing import Callable

from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name

from nnactive.config.struct import ActiveConfig
from nnactive.nnunet.utils import get_patch_size

__experiments = {}


def register(make_config: Callable, seeds, uncertainties):
    for seed, uncertainty in product(seeds, uncertainties):
        name, config = make_config(seed=seed, uncertainty=uncertainty)
        __experiments[name] = config


def get_experiment(name):
    return __experiments[name]


def list_experiments():
    return __experiments.keys()


def make_kits_small_config(seed: int, uncertainty: str):
    base_id = 982
    dataset_name = convert_id_to_dataset_name(base_id)
    return "kits_small", ActiveConfig(
        trainer="nnActiveTrainer_5epochs",
        base_id=base_id,
        patch_size=get_patch_size(base_id),
        uncertainty=uncertainty,
        query_size=10,
        query_steps=3,
        starting_budget="random-label-all-classes",
        seed=seed,
        num_processes=4,
        dataset=dataset_name,
        pre_suffix="__patch-full_patch",
        train_folds=2,
        full_folds=5,
        add_validation="--disable_tta",
        add_uncertainty="--diable_tta",
        agg_stride=8,
        patch_overlap=0,
        additional_overlap=0.2,
    )


register(
    make_kits_small_config,
    seeds=[12345, 12346, 12347],
    uncertainties=["pred_entropy", "mutual_information", "random"],
)
