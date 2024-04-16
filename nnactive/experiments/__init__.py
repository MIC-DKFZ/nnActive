from itertools import product
from typing import Callable

import nnunetv2.paths as paths
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name

from nnactive.config.struct import ActiveConfig
from nnactive.nnunet.utils import get_patch_size
from nnactive.paths import set_raw_paths

__experiments = {}


def register(
    make_config: Callable[[int, str], ActiveConfig],
    seeds: list[int],
    uncertainties: list[str],
    **kwargs,
):
    for seed, uncertainty in product(seeds, uncertainties):
        try:
            config = make_config(seed=seed, uncertainty=uncertainty, **kwargs)
            __experiments[config.name()] = config
        except RuntimeError:
            continue


def get_experiment(name):
    return __experiments[name]


def list_experiments():
    return __experiments.keys()


def make_kits_small_config(seed: int, uncertainty: str):
    with set_raw_paths():
        base_id = 982
        dataset_name = convert_id_to_dataset_name(base_id)
        patch_size = get_patch_size(base_id)
    return ActiveConfig(
        trainer="nnActiveTrainer_5epochs",
        base_id=base_id,
        patch_size=patch_size,
        uncertainty=uncertainty,
        query_size=10,
        query_steps=3,
        starting_budget="random-label-all-classes",
        seed=seed,
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


def make_kits_config(seed: int, uncertainty: str):
    with set_raw_paths():
        base_id = 135
        dataset_name = convert_id_to_dataset_name(base_id)
    return ActiveConfig(
        trainer="nnActiveTrainer_200epochs",
        base_id=base_id,
        patch_size=[64, 64, 64],
        uncertainty=uncertainty,
        query_size=10,
        query_steps=10,
        starting_budget="random-label-all-classes",
        seed=seed,
        dataset=dataset_name,
        pre_suffix="__patch-full_patch",
        train_folds=5,
        full_folds=5,
        add_validation="",
        add_uncertainty="",
        agg_stride=8,
        patch_overlap=0,
        additional_overlap=0.2,
    )


def make_brats_small_config(seed: int, uncertainty: str):
    with set_raw_paths():
        base_id = 981
        dataset_name = convert_id_to_dataset_name(base_id)
        patch_size = get_patch_size(base_id)
    return ActiveConfig(
        trainer="nnActiveTrainer_5epochs",
        base_id=base_id,
        patch_size=patch_size,
        uncertainty=uncertainty,
        query_size=10,
        query_steps=3,
        starting_budget="random-label-all-classes",
        seed=seed,
        dataset=dataset_name,
        pre_suffix="__patch-full_patch",
        train_folds=5,
        full_folds=5,
        add_validation="--disable_tta",
        add_uncertainty="--diable_tta",
        agg_stride=8,
        patch_overlap=0,
        additional_overlap=0.2,
    )


def make_brats_config(seed: int, uncertainty: str):
    with set_raw_paths():
        base_id = 137
        dataset_name = convert_id_to_dataset_name(base_id)
    return ActiveConfig(
        trainer="nnActiveTrainer_200epochs",
        base_id=base_id,
        patch_size=[20, 20, 20],
        uncertainty=uncertainty,
        query_size=20,
        query_steps=10,
        starting_budget="random-label-all-classes",
        seed=seed,
        dataset=dataset_name,
        pre_suffix="__patch-20",
        train_folds=5,
        full_folds=5,
        add_validation="--disable_tta",
        add_uncertainty="--diable_tta",
        agg_stride=8,
        patch_overlap=0,
        additional_overlap=0.2,
    )


def make_hippocampus_config(
    seed: int, uncertainty: str, query_size: int = 20, query_steps: int = 10
):
    with set_raw_paths():
        base_id = 4
        dataset_name = convert_id_to_dataset_name(base_id)
    return ActiveConfig(
        trainer="nnActiveTrainer_5epochs",
        base_id=base_id,
        patch_size=[20, 20, 20],
        uncertainty=uncertainty,
        query_size=query_size,
        query_steps=query_steps,
        starting_budget="random-label-all-classes",
        seed=seed,
        dataset=dataset_name,
        pre_suffix=f"__patch-20__qs{query_size}",
        train_folds=5,
        full_folds=5,
        add_validation="--disable_tta",
        add_uncertainty="--diable_tta",
        agg_stride=8,
        patch_overlap=0,
        additional_overlap=0.2,
    )


def make_acdc_config(seed: int, uncertainty: str):
    with set_raw_paths():
        base_id = 27
        dataset_name = convert_id_to_dataset_name(base_id)
    return ActiveConfig(
        trainer="nnActiveTrainer_200epochs",
        base_id=base_id,
        patch_size=[4, 40, 40],
        uncertainty=uncertainty,
        query_size=15,
        query_steps=10,
        starting_budget="random-label-all-classes",
        seed=seed,
        dataset=dataset_name,
        pre_suffix="__patch-full",
        train_folds=5,
        full_folds=5,
        add_validation="",
        add_uncertainty="",
        agg_stride=8,
        patch_overlap=0,
        additional_overlap=0.2,
    )


def make_acdc_small_config(seed: int, uncertainty: str):
    with set_raw_paths():
        base_id = 985
        dataset_name = convert_id_to_dataset_name(base_id)
        patch_size = get_patch_size(base_id)
    return ActiveConfig(
        trainer="nnActiveTrainer_5epochs",
        base_id=base_id,
        patch_size=patch_size,
        uncertainty=uncertainty,
        query_size=25,
        query_steps=5,
        starting_budget="random-label-all-classes",
        seed=seed,
        dataset=dataset_name,
        pre_suffix="__patch-full",
        train_folds=5,
        full_folds=5,
        add_validation="",
        add_uncertainty="",
        agg_stride=8,
        patch_overlap=0,
        additional_overlap=0.2,
    )


def make_amos_config(seed: int, uncertainty: str):
    with set_raw_paths():
        base_id = 216
        dataset_name = convert_id_to_dataset_name(base_id)
        patch_size = get_patch_size(base_id)
    return ActiveConfig(
        trainer="nnActiveTrainer_200epochs",
        base_id=base_id,
        patch_size=patch_size,
        uncertainty=uncertainty,
        query_size=32,
        query_steps=10,
        starting_budget="random-label-all-classes",
        seed=seed,
        dataset=dataset_name,
        pre_suffix="__patch-full_patch",
        train_folds=5,
        full_folds=5,
        add_validation="",
        add_uncertainty="",
        agg_stride=8,
        patch_overlap=0,
        additional_overlap=0.2,
    )


def make_amos_small_config(seed: int, uncertainty: str):
    with set_raw_paths():
        base_id = 984
        dataset_name = convert_id_to_dataset_name(base_id)
        patch_size = get_patch_size(base_id)
    return ActiveConfig(
        trainer="nnActiveTrainer_5epochs",
        base_id=base_id,
        patch_size=patch_size,
        uncertainty=uncertainty,
        query_size=60,
        query_steps=3,
        starting_budget="random-label-all-classes",
        seed=seed,
        dataset=dataset_name,
        pre_suffix="__patch-full_patch",
        train_folds=5,
        full_folds=5,
        add_validation="",
        add_uncertainty="",
        agg_stride=8,
        patch_overlap=0,
        additional_overlap=0.2,
    )


def make_airway_config(seed: int, uncertainty: str):
    with set_raw_paths():
        base_id = 980
        dataset_name = convert_id_to_dataset_name(base_id)
        patch_size = get_patch_size(base_id)
    return ActiveConfig(
        trainer="nnActiveTrainer_airway_200epochs",
        base_id=base_id,
        patch_size=patch_size,
        uncertainty=uncertainty,
        query_size=10,
        query_steps=10,
        starting_budget="random-label-all-classes",
        seed=seed,
        dataset=dataset_name,
        pre_suffix="__patch-full_patch",
        train_folds=5,
        full_folds=5,
        add_validation="",
        add_uncertainty="",
        agg_stride=8,
        patch_overlap=0,
        additional_overlap=0.2,
    )


def make_airway_small_config(seed: int, uncertainty: str):
    with set_raw_paths():
        base_id = 983
        dataset_name = convert_id_to_dataset_name(base_id)
        patch_size = get_patch_size(base_id)
    return ActiveConfig(
        trainer="nnActiveTrainer_airway_5epochs",
        base_id=base_id,
        patch_size=patch_size,
        uncertainty=uncertainty,
        query_size=10,
        query_steps=3,
        starting_budget="random-label-all-classes",
        seed=seed,
        dataset=dataset_name,
        pre_suffix="__patch-std_patch__sb-random-label-all-classes",
        train_folds=1,
        full_folds=5,
        add_validation="",
        add_uncertainty="",
        agg_stride=8,
        patch_overlap=0,
        additional_overlap=0.2,
    )


register(
    make_kits_small_config,
    seeds=[12345, 12346, 12347],
    uncertainties=["pred_entropy", "mutual_information", "random"],
)

register(
    make_kits_config,
    seeds=[12345, 12346, 12347],
    uncertainties=["pred_entropy", "mutual_information", "random-label", "random"],
)

register(
    make_brats_small_config,
    seeds=[12345, 12346, 12347],
    uncertainties=["pred_entropy", "mutual_information", "random"],
)

register(
    make_brats_config,
    seeds=[12345, 12346, 12347],
    uncertainties=["pred_entropy", "mutual_information", "random-label", "random"],
)

register(
    make_hippocampus_config,
    seeds=[12345, 12346, 12347],
    uncertainties=["pred_entropy", "mutual_information", "random-label", "random"],
    query_size=20,
    query_steps=10,
)

register(
    make_hippocampus_config,
    seeds=[12345, 12346, 12347],
    uncertainties=["pred_entropy", "mutual_information", "random-label", "random"],
    query_size=40,
    query_steps=5,
)

register(
    make_acdc_config,
    seeds=[12345, 12346, 12347],
    uncertainties=["pred_entropy", "mutual_information", "random-label", "random"],
)

register(
    make_acdc_small_config,
    seeds=[12345, 12346, 12347],
    uncertainties=["pred_entropy", "mutual_information", "random"],
)

register(
    make_amos_config,
    seeds=[12345, 12346, 12347],
    uncertainties=["pred_entropy", "mutual_information", "random-label", "random"],
)

register(
    make_amos_small_config,
    seeds=[12345, 12346, 12347],
    uncertainties=["pred_entropy", "mutual_information", "random"],
)

register(
    make_airway_config,
    seeds=[12345, 12346, 12347],
    uncertainties=["pred_entropy", "mutual_information", "random-label", "random"],
)

register(
    make_airway_small_config,
    seeds=[12345, 12346, 12347],
    uncertainties=["pred_entropy", "mutual_information", "random"],
)
