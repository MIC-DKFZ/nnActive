from typing import Union

from loguru import logger

from nnactive.config.struct import ActiveConfig
from nnactive.strategies.bald import BALD
from nnactive.strategies.base import AbstractQueryMethod
from nnactive.strategies.dice_query import ExpectedDiceQuery
from nnactive.strategies.entropy_exp import ExpectedEntropy
from nnactive.strategies.entropy_pred import PredictiveEntropy
from nnactive.strategies.random import Random, RandomAllClasses
from nnactive.strategies.randomlabel import RandomLabel, RandomLabelAllClasses
from nnactive.strategies.randomlabel2 import (
    RandomRandomLabel,
    RandomRandomLabelAllClasses,
)


def get_strategy(
    config: ActiveConfig, strategy_name: str, dataset_id: int, **kwargs
) -> AbstractQueryMethod:
    strategy = strategydict[strategy_name].init_from_dataset_id(
        config, dataset_id, **kwargs
    )
    return strategy


def init_strategy(
    strategy_name: str,
    dataset_id: int,
    query_size: int,
    patch_size: list[int],
    seed: int,
    agg_stride: Union[int, list[int]],
    trials_per_img: int,
    n_patch_per_image: int,
    file_ending: str = ".nii.gz",
    **kwargs,
) -> AbstractQueryMethod:
    strategy = strategydict[strategy_name](
        dataset_id,
        query_size=query_size,
        patch_size=patch_size,
        seed=seed,
        trials_per_img=trials_per_img,
        file_ending=file_ending,
        agg_stride=agg_stride,
        n_patch_per_image=n_patch_per_image,
        **kwargs,
    )
    logger.debug(f"Initializing query-strategy: {strategy.__class__.__name__}")
    return strategy


strategydict: dict[str, type[AbstractQueryMethod]] = {
    "mutual_information": BALD,
    "pred_entropy": PredictiveEntropy,
    "exp_entropy": ExpectedEntropy,
    "expected_dice": ExpectedDiceQuery,
    "random": Random,
    "random-label": RandomLabel,
    "random-all-classes": RandomAllClasses,
    "random-label-all-classes": RandomLabelAllClasses,
    "random-label2": RandomRandomLabel,
    "random-label2-all-classes": RandomRandomLabelAllClasses,
}
