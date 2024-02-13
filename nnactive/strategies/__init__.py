from typing import Union

from loguru import logger

from nnactive.strategies.bald import BALD
from nnactive.strategies.base import AbstractQueryMethod
from nnactive.strategies.entropy_exp import ExpectedEntropy
from nnactive.strategies.entropy_pred import PredictiveEntropy
from nnactive.strategies.random import Random
from nnactive.strategies.randomlabel import RandomLabel
from nnactive.strategies.randomlabel_all_classes import (
    RandomAllClasses,
    RandomLabelAllClasses,
)


def get_strategy(strategy_name: str, dataset_id: int, **kwargs) -> AbstractQueryMethod:
    strategy = strategydict[strategy_name].init_from_dataset_id(dataset_id, **kwargs)
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


strategydict = {
    "mutual_information": BALD,
    "pred_entropy": PredictiveEntropy,
    "exp_entropy": ExpectedEntropy,
    "random": Random,
    "random-label": RandomLabel,
    "random-all-classes": RandomAllClasses,
    "random-label-all-classes": RandomLabelAllClasses,
}
