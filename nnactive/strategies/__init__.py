from typing import Union

from nnactive.strategies.bald import BALD
from nnactive.strategies.base import AbstractQueryMethod
from nnactive.strategies.entropy_exp import ExpectedEntropy
from nnactive.strategies.entropy_pred import PredictiveEntropy
from nnactive.strategies.random import Random
from nnactive.strategies.randomlabel import RandomLabel


def get_strategy(strategy_name: str, dataset_id: int, **kwargs) -> AbstractQueryMethod:
    return strategydict[strategy_name].init_from_dataset_id(dataset_id, **kwargs)


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
    return strategydict[strategy_name](
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


strategydict = {
    "mutual_information": BALD,
    "pred_entropy": PredictiveEntropy,
    "exp_entropy": ExpectedEntropy,
    "random": Random,
    "random-label": RandomLabel,
}
