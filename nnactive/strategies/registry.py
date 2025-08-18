from typing import Union

from loguru import logger

from nnactive.config.struct import ActiveConfig
from nnactive.strategies.base import AbstractQueryMethod

_STRATEGY_REGISTRY: dict[str, type[AbstractQueryMethod]] = {}


def register_strategy(name: str):
    """Decorator to register a query strategy under a given name."""

    def decorator(cls: type[AbstractQueryMethod]):
        if name in _STRATEGY_REGISTRY:
            logger.warning(f"Overwriting strategy '{name}' with {cls.__name__}")
        _STRATEGY_REGISTRY[name] = cls
        return cls

    return decorator


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
    strategy = _STRATEGY_REGISTRY[strategy_name].init_from_dataset_id(
        config, dataset_id, loop_val=loop_val, seed=seed, **kwargs
    )
    return strategy


def get_strategy_list():
    return list(_STRATEGY_REGISTRY.keys())


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
    logger.debug(f"Initializing query-strategy: {strategy.__class__.__name__}")
    return strategy
