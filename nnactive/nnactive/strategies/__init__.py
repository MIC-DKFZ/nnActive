from nnactive.strategies.registry import get_strategy, init_strategy, register_strategy

from . import (
    bald,
    dice_query,
    entropy_exp,
    entropy_pred,
    kmeans_bald,
    precomputed_query,
    random,
    randomlabel,
    randomlabel2,
)
