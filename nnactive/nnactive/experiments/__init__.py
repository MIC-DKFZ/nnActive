from itertools import product
from pathlib import Path
from typing import Callable

import nnunetv2.paths as paths
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name

from nnactive.config.struct import ActiveConfig
from nnactive.nnunet.utils import get_patch_size
from nnactive.paths import get_nnActive_data, set_raw_paths
from nnactive.results.state import State

__experiments = {}

__seeds = list(i + 12345 for i in range(4))

__strategies = [
    "random-label",
    "random-label2",
    "random",
    "pred_entropy",
    "mutual_information",
    "power_bald",
    "power_pe",
    "softrank_bald",
]

__standard_trainer = "nnActiveTrainer_200epochs"
__standard_starting_budget = "random-label2-all-classes"

__standard_pre_suffix_format = "__patch-{patch_size}__sb-{starting_budget}__sbs-{starting_budget_size}__qs-{query_size}"

# TODO Separate naming for experiments that need to be re-done. Remove this once that all revision are complete.
__revision_pre_suffix_format = "__patch-{patch_size}__sb-{starting_budget}__sbs-{starting_budget_size}__qs-{query_size}_revision"


def register(
    make_config: Callable[[int, str], ActiveConfig],
    seeds: list[int],
    uncertainties: list[str],
    **kwargs,
):
    for seed, uncertainty in product(seeds, uncertainties):
        try:
            config: ActiveConfig = make_config(
                seed=seed, uncertainty=uncertainty, **kwargs
            )
            __experiments[config.name()] = config
        except RuntimeError:
            continue


def get_experiment(name):
    return __experiments[name]


def list_experiments():
    return __experiments.keys()


def list_prepared_experiments(base_id: int):
    with set_raw_paths():
        dataset_name = convert_id_to_dataset_name(base_id)
    # use preprocessed path to ensure that entire setup pipeline finished.
    results_path: Path = get_nnActive_data() / dataset_name / "nnUNet_preprocessed"
    if not results_path.exists():
        return []
    out_list = [
        file.name for file in results_path.iterdir() if file.name.startswith("Dataset")
    ]
    out_list.sort()
    return out_list


def list_finished_experiments(base_id: int):
    with set_raw_paths():
        dataset_name = convert_id_to_dataset_name(base_id)
    results_path: Path = get_nnActive_data() / dataset_name / "nnActive_results"
    if not results_path.exists():
        return []
    out_list = [
        file.name
        for file in results_path.iterdir()
        if file.name.startswith("Dataset")
        and file.is_dir()
        and State.experiment_finished(file)
    ]
    return out_list


def make_config(
    seed: int,
    uncertainty: str,
    base_id: int,
    query_size: int,
    query_steps: int,
    patch_size: list[int] | None = None,
    agg_stride: int = 8,
    trainer: str = __standard_trainer,
    starting_budget: str = __standard_starting_budget,
    pre_suffix_format: str = __standard_pre_suffix_format,
    **config_kwargs,
) -> ActiveConfig:
    with set_raw_paths():
        dataset_name = convert_id_to_dataset_name(base_id)
        if patch_size is None:
            patch_size = get_patch_size(base_id)

    config = ActiveConfig(
        trainer=trainer,
        base_id=base_id,
        patch_size=patch_size,
        uncertainty=uncertainty,
        query_size=query_size,
        query_steps=query_steps,
        starting_budget=starting_budget,
        seed=seed,
        dataset=dataset_name,
        train_folds=5,
        full_folds=5,
        agg_stride=agg_stride,
        patch_overlap=0,
        **config_kwargs,
    )
    config.set_pre_suffix(pre_suffix_format)
    return config


def make_kits_small_config(
    seed: int,
    uncertainty: str,
    query_size: int = 10,
    query_steps: int = 3,
    patch_size: list[int] = [64, 64, 64],
    starting_budget: str = __standard_starting_budget,
    pre_suffix_format: str = __standard_pre_suffix_format,
):
    with set_raw_paths():
        base_id = 982
        dataset_name = convert_id_to_dataset_name(base_id)
        if patch_size is None:
            patch_size = get_patch_size(base_id)

    config = ActiveConfig(
        trainer="nnActiveTrainer_5epochs",
        base_id=base_id,
        patch_size=patch_size,
        uncertainty=uncertainty,
        query_size=query_size,
        query_steps=query_steps,
        starting_budget=starting_budget,
        seed=seed,
        dataset=dataset_name,
        train_folds=5,
        full_folds=5,
        agg_stride=8,
        patch_overlap=0,
    )
    config.set_pre_suffix(pre_suffix_format)
    return config


def make_kits_config(
    seed: int,
    uncertainty: str,
    query_size: int = 20,
    query_steps: int = 10,
    patch_size: list[int] = [64, 64, 64],
    trainer: str = __standard_trainer,
    starting_budget: str = __standard_starting_budget,
    pre_suffix_format: str = __standard_pre_suffix_format,
):
    with set_raw_paths():
        base_id = 135
        dataset_name = convert_id_to_dataset_name(base_id)
        if patch_size is None:
            patch_size = get_patch_size(base_id)

    config = ActiveConfig(
        trainer=trainer,
        base_id=base_id,
        patch_size=patch_size,
        uncertainty=uncertainty,
        query_size=query_size,
        query_steps=query_steps,
        starting_budget=starting_budget,
        seed=seed,
        n_patch_per_image=20,
        dataset=dataset_name,
        train_folds=5,
        full_folds=5,
        agg_stride=8,
        patch_overlap=0,
    )
    config.set_pre_suffix(pre_suffix_format)
    return config


def make_kits_prototyping_config(
    seed: int,
    uncertainty: str,
    query_size: int = 20,
    query_steps: int = 3,
    patch_size: list[int] = [64, 64, 64],
    starting_budget: str = __standard_starting_budget,
    pre_suffix_format: str = "__PROTOTYPE" + __standard_pre_suffix_format,
):
    with set_raw_paths():
        base_id = 982
        dataset_name = convert_id_to_dataset_name(base_id)
        if patch_size is None:
            patch_size = get_patch_size(base_id)

    config = ActiveConfig(
        trainer=__standard_trainer,
        base_id=base_id,
        patch_size=patch_size,
        uncertainty=uncertainty,
        query_size=query_size,
        query_steps=query_steps,
        starting_budget=starting_budget,
        seed=seed,
        dataset=dataset_name,
        train_folds=5,
        full_folds=5,
        agg_stride=8,
        patch_overlap=0,
    )
    config.set_pre_suffix(pre_suffix_format)
    return config


def make_kits_debug_config(
    seed: int,
    uncertainty: str,
    query_size: int = 10,
    query_steps: int = 3,
    patch_size: list[int] = [64, 64, 64],
    starting_budget: str = __standard_starting_budget,
    pre_suffix_format: str = "__DEBUG" + __standard_pre_suffix_format,
):
    with set_raw_paths():
        base_id = 982
        dataset_name = convert_id_to_dataset_name(base_id)
        if patch_size is None:
            patch_size = get_patch_size(base_id)
    config = ActiveConfig(
        trainer="nnActiveTrainer_5epochs",
        base_id=base_id,
        patch_size=patch_size,
        uncertainty=uncertainty,
        query_size=query_size,
        query_steps=query_steps,
        starting_budget=starting_budget,
        seed=seed,
        dataset=dataset_name,
        train_folds=2,
        full_folds=5,
        agg_stride=8,
        patch_overlap=0,
    )
    config.set_pre_suffix(pre_suffix_format)
    return config


def make_hippocampus_config(
    seed: int,
    uncertainty: str,
    query_size: int = 20,
    query_steps: int = 10,
    patch_size: list[int] = [20, 20, 20],
    starting_budget: str = __standard_starting_budget,
    pre_suffix_format: str = __standard_pre_suffix_format,
):
    with set_raw_paths():
        base_id = 4
        dataset_name = convert_id_to_dataset_name(base_id)
        if patch_size is None:
            patch_size = get_patch_size(base_id)

    config = ActiveConfig(
        trainer=__standard_trainer,
        base_id=base_id,
        patch_size=patch_size,
        uncertainty=uncertainty,
        query_size=query_size,
        query_steps=query_steps,
        starting_budget=starting_budget,
        seed=seed,
        n_patch_per_image=20,
        dataset=dataset_name,
        train_folds=5,
        full_folds=5,
        agg_stride=1,
        patch_overlap=0,
    )
    config.set_pre_suffix(pre_suffix_format)
    return config


def make_hippocampus_debug_config(
    seed: int,
    uncertainty: str,
    query_size: int = 10,
    query_steps: int = 3,
    patch_size: list[int] = [20, 20, 20],
    starting_budget: str = __standard_starting_budget,
    pre_suffix_format: str = "__DEBUG" + __standard_pre_suffix_format,
):
    with set_raw_paths():
        base_id = 4
        dataset_name = convert_id_to_dataset_name(base_id)
        if patch_size is None:
            patch_size = get_patch_size(base_id)
    config = ActiveConfig(
        trainer="nnActiveTrainer_5epochs",
        base_id=base_id,
        patch_size=patch_size,
        uncertainty=uncertainty,
        query_size=query_size,
        query_steps=query_steps,
        starting_budget=starting_budget,
        seed=seed,
        dataset=dataset_name,
        train_folds=2,
        full_folds=5,
        agg_stride=1,
        patch_overlap=0,
    )
    config.set_pre_suffix(pre_suffix_format)
    return config


def make_acdc_config(
    seed: int,
    uncertainty: str,
    query_size: int = 15,
    query_steps: int = 10,
    patch_size: list[int] = [4, 40, 40],
    starting_budget: str = __standard_starting_budget,
    pre_suffix_format: str = __standard_pre_suffix_format,
):
    with set_raw_paths():
        base_id = 27
        dataset_name = convert_id_to_dataset_name(base_id)
        if patch_size is None:
            patch_size = get_patch_size(base_id)

    config = ActiveConfig(
        trainer=__standard_trainer,
        base_id=base_id,
        patch_size=patch_size,
        uncertainty=uncertainty,
        query_size=query_size,
        query_steps=query_steps,
        starting_budget=starting_budget,
        seed=seed,
        n_patch_per_image=20,
        dataset=dataset_name,
        train_folds=5,
        full_folds=5,
        agg_stride=[1, 8, 8],
        patch_overlap=0,
    )

    config.set_pre_suffix(pre_suffix_format)
    return config


def make_acdc_small_config(
    seed: int,
    uncertainty: str,
    query_size: int = 20,
    query_steps: int = 3,
    patch_size: list[int] = [4, 40, 40],
    starting_budget: str = __standard_starting_budget,
    pre_suffix_format: str = __standard_pre_suffix_format,
):
    with set_raw_paths():
        base_id = 985
        dataset_name = convert_id_to_dataset_name(base_id)
        if patch_size is None:
            patch_size = get_patch_size(base_id)

    config = ActiveConfig(
        trainer="nnActiveTrainer_5epochs",
        base_id=base_id,
        patch_size=patch_size,
        uncertainty=uncertainty,
        query_size=query_size,
        query_steps=query_steps,
        starting_budget=starting_budget,
        seed=seed,
        dataset=dataset_name,
        train_folds=5,
        full_folds=5,
        agg_stride=[1, 8, 8],
        patch_overlap=0,
    )
    config.set_pre_suffix(pre_suffix_format)
    return config


def make_amos_config(
    seed: int,
    uncertainty: str,
    query_size: int = 20,
    query_steps: int = 10,
    patch_size: list[int] | None = [32, 74, 74],
    trainer: str = __standard_trainer,
    starting_budget: str = __standard_starting_budget,
    pre_suffix_format: str = __standard_pre_suffix_format,
):
    with set_raw_paths():
        base_id = 216
        dataset_name = convert_id_to_dataset_name(base_id)
        if patch_size is None:
            patch_size = get_patch_size(base_id)

    config = ActiveConfig(
        trainer=trainer,
        base_id=base_id,
        patch_size=patch_size,
        uncertainty=uncertainty,
        query_size=query_size,
        query_steps=query_steps,
        starting_budget=starting_budget,
        seed=seed,
        n_patch_per_image=20,
        dataset=dataset_name,
        train_folds=5,
        full_folds=5,
        agg_stride=8,
        patch_overlap=0,
    )
    config.set_pre_suffix(pre_suffix_format)
    return config


def make_amos_prototyping_config(
    seed: int,
    uncertainty: str,
    query_size: int = 20,
    query_steps: int = 3,
    patch_size: list[int] = [32, 74, 74],
    starting_budget: str = __standard_starting_budget,
    pre_suffix_format: str = "__PROTOTYPE" + __standard_pre_suffix_format,
):
    with set_raw_paths():
        base_id = 984
        dataset_name = convert_id_to_dataset_name(base_id)
        if patch_size is None:
            patch_size = get_patch_size(base_id)

    config = ActiveConfig(
        trainer=__standard_trainer,
        base_id=base_id,
        patch_size=patch_size,
        uncertainty=uncertainty,
        query_size=query_size,
        query_steps=query_steps,
        starting_budget=starting_budget,
        seed=seed,
        n_patch_per_image=20,
        dataset=dataset_name,
        train_folds=5,
        full_folds=5,
        agg_stride=8,
        patch_overlap=0,
    )
    config.set_pre_suffix(pre_suffix_format)
    return config


############### Base Experiments (3 label settings) ###############

register(
    make_kits_small_config,
    seeds=__seeds,
    uncertainties=__strategies,
)

_query_sizes = [40, 200, 500]
for qs in _query_sizes:
    register(
        make_kits_config,
        seeds=__seeds,
        uncertainties=__strategies,
        query_size=qs,
        query_steps=5,
    )

register(
    make_hippocampus_config,
    seeds=__seeds,
    uncertainties=__strategies,
    query_size=20,
    query_steps=5,
)

register(
    make_hippocampus_config,
    seeds=__seeds,
    uncertainties=__strategies,
    query_size=40,
    query_steps=5,
)

register(
    make_hippocampus_config,
    seeds=__seeds,
    uncertainties=__strategies,
    query_size=60,
    query_steps=5,
)

_query_sizes = [30, 60, 90]
for qs in _query_sizes:
    register(
        make_acdc_config,
        seeds=__seeds,
        uncertainties=__strategies,
        query_size=qs,
        query_steps=5,
    )

register(
    make_acdc_small_config,
    seeds=__seeds,
    uncertainties=__strategies,
)

_query_sizes = [40, 200, 500]
for qs in _query_sizes:
    register(
        make_amos_config,
        seeds=__seeds,
        uncertainties=__strategies,
        query_size=qs,
        query_steps=5,
    )


################## Smaller Query Size Ablation Experiments ################

dataset_configs: list[dict] = [
    {
        "base_id": 216,
        "starting_budget_size": 500,
        "query_size": 250,
        "patch_size": [32, 74, 74],
        "agg_stride": 8,
    },  # AMOS
    {
        "base_id": 216,
        "starting_budget_size": 40,
        "query_size": 20,
        "patch_size": [32, 74, 74],
        "agg_stride": 8,
    },  # AMOS
    {
        "base_id": 135,
        "starting_budget_size": 500,
        "query_size": 250,
        "patch_size": [64, 64, 64],
        "agg_stride": 8,
    },  # KiTS
    {
        "base_id": 135,
        "starting_budget_size": 40,
        "query_size": 20,
        "patch_size": [64, 64, 64],
        "agg_stride": 8,
    },  # KiTS
    {
        "base_id": 27,
        "starting_budget_size": 90,
        "query_size": 45,
        "patch_size": [4, 40, 40],
        "agg_stride": [1, 8, 8],
    },  # ACDC
    {
        "base_id": 27,
        "starting_budget_size": 30,
        "query_size": 15,
        "patch_size": [4, 40, 40],
        "agg_stride": [1, 8, 8],
    },  # ACDC
]

for config in dataset_configs:
    register(
        make_config,
        base_id=config["base_id"],
        seeds=__seeds,
        uncertainties=__strategies,
        starting_budget_size=config["starting_budget_size"],
        query_size=config["query_size"],
        query_steps=9,  # run 9 loops since we use qs/2
        patch_size=config["patch_size"],
        agg_stride=config["agg_stride"],
        trainer=__standard_trainer,
        starting_budget=__standard_starting_budget,
        # TODO add revision tag to ACDC experiments. Remove this once all revisions are completed.
        pre_suffix_format=__revision_pre_suffix_format
        if config["base_id"] == 27
        else __standard_pre_suffix_format,
    )

################## Larger Query Size Ablation Experiments ################

dataset_configs: list[dict] = [
    {
        "base_id": 216,
        "starting_budget_size": 500,
        "query_size": 1000,
        "patch_size": [32, 74, 74],
        "agg_stride": 8,
    },  # AMOS
    {
        "base_id": 216,
        "starting_budget_size": 40,
        "query_size": 80,
        "patch_size": [32, 74, 74],
        "agg_stride": 8,
    },  # AMOS
    {
        "base_id": 135,
        "starting_budget_size": 500,
        "query_size": 1000,
        "patch_size": [64, 64, 64],
        "agg_stride": 8,
    },  # KiTS
    {
        "base_id": 135,
        "starting_budget_size": 40,
        "query_size": 80,
        "patch_size": [64, 64, 64],
        "agg_stride": 8,
    },  # KiTS
    {
        "base_id": 27,
        "starting_budget_size": 90,
        "query_size": 180,
        "patch_size": [4, 40, 40],
        "agg_stride": [1, 8, 8],
    },  # ACDC
    {
        "base_id": 27,
        "starting_budget_size": 30,
        "query_size": 60,
        "patch_size": [4, 40, 40],
        "agg_stride": [1, 8, 8],
    },  # ACDC
]

for config in dataset_configs:
    register(
        make_config,
        base_id=config["base_id"],
        seeds=__seeds,
        uncertainties=__strategies,
        starting_budget_size=config["starting_budget_size"],
        query_size=config["query_size"],
        query_steps=3,  # run 9 loops since we use qs*2
        patch_size=config["patch_size"],
        agg_stride=config["agg_stride"],
        trainer=__standard_trainer,
        starting_budget=__standard_starting_budget,
        # TODO add revision tag to ACDC experiments. Remove this once all revisions are completed.
        pre_suffix_format=__revision_pre_suffix_format
        if config["base_id"] == 27
        else __standard_pre_suffix_format,
    )

################## Training Length Experiments #############
pre_suffix_format = "__tr-{trainer}__patch-{patch_size}__sb-{starting_budget}__sbs-{starting_budget_size}__qs-{query_size}"

register(
    make_amos_config,
    seeds=__seeds,
    uncertainties=[s for s in __strategies if s != "random"],
    query_size=40,
    query_steps=5,
    trainer="nnActiveTrainer_500epochs",
    pre_suffix_format=pre_suffix_format,
)

register(
    make_kits_config,
    seeds=__seeds,
    uncertainties=[s for s in __strategies if s != "random"],
    query_size=200,
    query_steps=5,
    trainer="nnActiveTrainer_500epochs",
    pre_suffix_format=pre_suffix_format,
)

register(
    make_amos_config,
    seeds=__seeds,
    uncertainties=[s for s in __strategies if s != "random"],
    query_size=200,
    query_steps=5,
    trainer="nnActiveTrainer_500epochs",
    pre_suffix_format=pre_suffix_format,
)

register(
    make_kits_config,
    seeds=__seeds,
    uncertainties=[s for s in __strategies if s != "random"],
    query_size=500,
    query_steps=5,
    trainer="nnActiveTrainer_500epochs",
    pre_suffix_format=pre_suffix_format,
)

register(
    make_amos_config,
    seeds=__seeds,
    uncertainties=[s for s in __strategies if s != "random"],
    query_size=500,
    query_steps=5,
    trainer="nnActiveTrainer_500epochs",
    pre_suffix_format=pre_suffix_format,
)

################## Dataset Exploration Experiments #########
#### Entire Dataset 10-50% ####
# rounded down 10% of training set query size.

pre_suffix_format = "__patch-{patch_size}__qs-{query_size}__tr-{trainer}"

full_patch_size = [1000, 1000, 1000]

dataset_list: list[dict] = [
    {"base_id": 216, "query_size": 15},  # AMOS
    {"base_id": 137, "query_size": 93, "additional_overlap": 1.0},  # BraTS
    {"base_id": 135, "query_size": 22},  # KiTS
    {"base_id": 27, "query_size": 15},  # ACDC
]

for dataset in dataset_list:
    additional_overlap = dataset.get("additional_overlap", 0.4)
    for trainer in ["nnUNetTrainer_200epochs", "nnUNetTrainer_500epochs"]:
        register(
            make_config,
            base_id=dataset["base_id"],
            seeds=__seeds,
            uncertainties=["random"],
            patch_size=full_patch_size,
            query_size=dataset["query_size"],
            query_steps=5,
            trainer=trainer,
            starting_budget="random",
            pre_suffix_format=pre_suffix_format,
            additional_overlap=additional_overlap,
        )

####### Experiments training 500 epochs on queries from 200 epoch training ######
pre_suffix_format = "__tr-{trainer}__patch-{patch_size}__sb-{starting_budget}__sbs-{starting_budget_size}__qs-{query_size}__precomputed-queries"

dataset_list: list[dict] = [
    {"base_id": 216, "query_size": 200, "patch_size": [32, 74, 74]},  # AMOS
    {"base_id": 216, "query_size": 500, "patch_size": [32, 74, 74]},  # AMOS
    {"base_id": 135, "query_size": 200, "patch_size": [64, 64, 64]},  # KiTS
    {"base_id": 135, "query_size": 500, "patch_size": [64, 64, 64]},  # KiTS
]

for seed, uncertainty, dataset in product(__seeds, __strategies, dataset_list):
    try:
        reference_exp_name = make_config(
            seed=seed,
            uncertainty=uncertainty,
            base_id=dataset["base_id"],
            query_size=dataset["query_size"],
            query_steps=5,
            patch_size=dataset.get("patch_size", None),
        ).name()
    except RuntimeError:
        # as in the `register` function, ignore failures, i.e. when the dataset folder
        # is not present.
        continue

    register(
        make_config,
        base_id=dataset["base_id"],
        seeds=[seed],
        uncertainties=[uncertainty],
        query_size=dataset["query_size"],
        query_steps=5,
        patch_size=dataset.get("patch_size", None),
        trainer="nnActiveTrainer_500epochs",
        pre_suffix_format=pre_suffix_format,
        additional_overlap=dataset.get("additional_overlap", 0.4),
        queries_from_experiment=reference_exp_name,
    )

################## Smaller Patch Size Ablation Experiments #########
dataset_configs: list[dict] = [
    {"base_id": 216, "query_size": 40, "patch_size": [16, 32, 32]},  # AMOS
    {"base_id": 216, "query_size": 200, "patch_size": [16, 32, 32]},  # AMOS
    {"base_id": 216, "query_size": 500, "patch_size": [16, 32, 32]},  # AMOS
    {"base_id": 135, "query_size": 40, "patch_size": [32, 32, 32]},  # KiTS
    {"base_id": 135, "query_size": 200, "patch_size": [32, 32, 32]},  # KiTS
    {"base_id": 135, "query_size": 500, "patch_size": [32, 32, 32]},  # KiTS
    {"base_id": 27, "query_size": 30, "patch_size": [2, 20, 20]},  # ACDC
    {"base_id": 27, "query_size": 60, "patch_size": [2, 20, 20]},  # ACDC
    {"base_id": 27, "query_size": 90, "patch_size": [2, 20, 20]},  # ACDC
    {"base_id": 4, "query_size": 20, "patch_size": [10, 10, 10]},  # Hippocampus
    {"base_id": 4, "query_size": 40, "patch_size": [10, 10, 10]},  # Hippocampus
    {"base_id": 4, "query_size": 60, "patch_size": [10, 10, 10]},  # Hippocampus
]

for config in dataset_configs:
    register(
        make_config,
        base_id=config["base_id"],
        seeds=__seeds,
        uncertainties=__strategies,
        query_size=config["query_size"],
        query_steps=5,
        patch_size=config["patch_size"],
        trainer=__standard_trainer,
        starting_budget=__standard_starting_budget,
        pre_suffix_format=__standard_pre_suffix_format,
    )

################## Power Bald Beta Ablation Experiments #########
strategies = [
    "power_bald_b5",
    "power_bald_b10",
    "power_bald_b20",
    "power_bald_b40",
]

dataset_configs: list[dict] = [
    {
        "base_id": 216,
        "query_size": 40,
        "patch_size": [32, 74, 74],
        "agg_stride": 8,
    },  # AMOS
    {
        "base_id": 216,
        "query_size": 200,
        "patch_size": [32, 74, 74],
        "agg_stride": 8,
    },  # AMOS
    {
        "base_id": 216,
        "query_size": 500,
        "patch_size": [32, 74, 74],
        "agg_stride": 8,
    },  # AMOS
    {
        "base_id": 135,
        "query_size": 40,
        "patch_size": [64, 64, 64],
        "agg_stride": 8,
    },  # KiTS
    {
        "base_id": 135,
        "query_size": 200,
        "patch_size": [64, 64, 64],
        "agg_stride": 8,
    },  # KiTS
    {
        "base_id": 135,
        "query_size": 500,
        "patch_size": [64, 64, 64],
        "agg_stride": 8,
    },  # KiTS
    {
        "base_id": 27,
        "query_size": 30,
        "patch_size": [4, 40, 40],
        "agg_stride": [1, 8, 8],
    },  # ACDC
    {
        "base_id": 27,
        "query_size": 60,
        "patch_size": [4, 40, 40],
        "agg_stride": [1, 8, 8],
    },  # ACDC
    {
        "base_id": 27,
        "query_size": 90,
        "patch_size": [4, 40, 40],
        "agg_stride": [1, 8, 8],
    },  # ACDC
]

for config in dataset_configs:
    register(
        make_config,
        base_id=config["base_id"],
        seeds=__seeds,
        uncertainties=strategies,
        query_size=config["query_size"],
        query_steps=5,
        patch_size=config["patch_size"],
        agg_stride=config["agg_stride"],
        trainer=__standard_trainer,
        starting_budget=__standard_starting_budget,
        # TODO add revision tag to ACDC experiments. Remove this once all revisions are completed.
        pre_suffix_format=__revision_pre_suffix_format
        if config["base_id"] == 27
        else __standard_pre_suffix_format,
    )


################## Prototyping Experiments #########
register(
    make_amos_prototyping_config,
    seeds=[12345],
    uncertainties=__strategies,
)

register(
    make_kits_prototyping_config,
    seeds=[12345],
    uncertainties=__strategies,
)

################## Debug Experiments ###############
register(
    make_kits_debug_config,
    seeds=[12345],
    uncertainties=["mutual_information"],
)

register(
    make_hippocampus_debug_config,
    seeds=[12345],
    uncertainties=["mutual_information"],
)

################## Old Experiments #################
__old_pre_suffix_format = "__patch-{patch_size}__qs-{query_size}"
__old_starting_budget = "random-label-all-classes"

register(
    make_amos_config,
    seeds=__seeds,
    uncertainties=__strategies,
    starting_budget=__old_starting_budget,
    pre_suffix_format=__old_pre_suffix_format,
)

register(
    make_acdc_config,
    seeds=__seeds,
    uncertainties=__strategies,
    starting_budget=__old_starting_budget,
    pre_suffix_format=__old_pre_suffix_format,
)

register(
    make_kits_config,
    seeds=__seeds,
    uncertainties=__strategies,
    starting_budget=__old_starting_budget,
    pre_suffix_format=__old_pre_suffix_format,
)

register(
    make_hippocampus_config,
    seeds=__seeds,
    uncertainties=__strategies,
    query_size=20,
    query_steps=10,
    starting_budget=__old_starting_budget,
    pre_suffix_format=__old_pre_suffix_format,
)
