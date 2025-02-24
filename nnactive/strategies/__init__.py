from typing import Union

from loguru import logger

from nnactive.config.struct import ActiveConfig
from nnactive.strategies.bald import (
    BALD,
    PowerBALD,
    PowerBALD_b5,
    PowerBALD_b10,
    PowerBALD_b20,
    PowerBALD_b40,
    SoftRankBALD,
)
from nnactive.strategies.base import AbstractQueryMethod
from nnactive.strategies.dice_query import ExpectedDiceQuery
from nnactive.strategies.entropy_balanced import (
    ClassBalancedEntropy_33FG,
    ClassBalancedEntropy_66FG,
    ClassBalancedExpScheduledPowerEntropy_66FG,
    ClassBalancedExpScheduledPowerEntropy_33FG,
    ClassBalancedSmoothSigmoidScheduledPowerEntropy_66FG,
    ClassBalancedSmoothSigmoidScheduledPowerEntropy_33FG,
    ClassBalancedSharpSigmoidScheduledPowerEntropy_66FG,
    ClassBalancedSharpSigmoidScheduledPowerEntropy_33FG,
)
from nnactive.strategies.entropy_exp import ExpectedEntropy
from nnactive.strategies.entropy_pred import PowerPredictiveEntropy, PredictiveEntropy
from nnactive.strategies.kmeans_bald import KMeansBALD
from nnactive.strategies.precomputed_query import PrecomputedQuery
from nnactive.strategies.random import Random, RandomAllClasses
from nnactive.strategies.randomlabel import RandomLabel, RandomLabelAllClasses
from nnactive.strategies.randomlabel2 import (
    RandomRandomLabel,
    RandomRandomLabelAllClasses,
)


def get_strategy(
    strategy_name: str,
    config: ActiveConfig,
    dataset_id: int,
    loop_val: int,
    seed: int,
    **kwargs,
) -> AbstractQueryMethod:
    if config.queries_from_experiment is not None:
        # If queries_from_experiment is specified, use the pre-computed queries.
        # NOTE config.uncertainty is not overwritten to ensure correct experiment naming
        strategy_name = "precomputed-queries"
    strategy = strategydict[strategy_name].init_from_dataset_id(
        config, dataset_id, loop_val=loop_val, seed=seed, **kwargs
    )
    return strategy


# This function should be replaced with get_strategy
def init_strategy(
    strategy_name: str,
    dataset_id: int,
    query_size: int,
    patch_size: list[int],
    seed: int,
    agg_stride: Union[int, list[int]],
    n_patch_per_image: int,
    loop_val: int | None = -1,
    additional_overlap: float = 0.4,
    **kwargs,
) -> AbstractQueryMethod:
    config = ActiveConfig(
        patch_size=patch_size,
        query_size=query_size,
        n_patch_per_image=n_patch_per_image,
        seed=seed,
        agg_stride=agg_stride,
        additional_overlap=additional_overlap,
    )
    strategy = get_strategy(strategy_name, config, dataset_id, loop_val, seed, **kwargs)
    # strategy = strategydict[strategy_name](
    #     dataset_id,
    #     query_size=query_size,
    #     patch_size=patch_size,
    #     seed=seed,
    #     trials_per_img=trials_per_img,
    #     file_ending=file_ending,
    #     agg_stride=agg_stride,
    #     n_patch_per_image=n_patch_per_image,
    #     **kwargs,
    # )
    logger.debug(f"Initializing query-strategy: {strategy.__class__.__name__}")
    return strategy


strategydict: dict[str, type[AbstractQueryMethod]] = {
    "mutual_information": BALD,
    "power_bald": PowerBALD,
    "power_bald_b5": PowerBALD_b5,
    "power_bald_b10": PowerBALD_b10,
    "power_bald_b20": PowerBALD_b20,
    "power_bald_b40": PowerBALD_b40,
    "softrank_bald": SoftRankBALD,
    "kmeans_bald": KMeansBALD,
    "pred_entropy": PredictiveEntropy,
    "power_pe": PowerPredictiveEntropy,
    "exp_entropy": ExpectedEntropy,
    "expected_dice": ExpectedDiceQuery,
    "random": Random,
    "random-label": RandomLabel,
    "random-all-classes": RandomAllClasses,
    "random-label-all-classes": RandomLabelAllClasses,
    "random-label2": RandomRandomLabel,
    "random-label2-all-classes": RandomRandomLabelAllClasses,
    "precomputed-queries": PrecomputedQuery,
    "class_pe66": ClassBalancedEntropy_66FG,
    "class_pe33": ClassBalancedEntropy_33FG,
    "class_power_pe66_exp": ClassBalancedExpScheduledPowerEntropy_66FG,
    "class_power_pe33_exp": ClassBalancedExpScheduledPowerEntropy_33FG,
    "class_power_pe66_smooth_sigmoid": ClassBalancedSmoothSigmoidScheduledPowerEntropy_66FG,
    "class_power_pe33_smooth_sigmoid": ClassBalancedSmoothSigmoidScheduledPowerEntropy_33FG,
    "class_power_pe66_sharp_sigmoid": ClassBalancedSharpSigmoidScheduledPowerEntropy_66FG,
    "class_power_pe33_sharp_sigmoid": ClassBalancedSharpSigmoidScheduledPowerEntropy_33FG,
}
