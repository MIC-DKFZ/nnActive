import pytest
import torch

from nnactive.config.struct import ActiveConfig
from nnactive.data import Patch
from nnactive.strategies.bald import BALD
from nnactive.strategies.entropy_pred import PredictiveEntropy
from nnactive.utils.pyutils import compute_conv_output_size


def test_strategy_entropy():
    input_size = torch.Tensor([10, 10, 10])
    patch_size = torch.Tensor([2, 2, 2])
    padding = 0
    config = ActiveConfig(patch_size=patch_size)
    agg_strides = [[1, 1, 1], [1, 2, 1], [2, 4, 5]]
    for agg_stride in agg_strides:
        config.agg_stride = agg_stride
        inputs = input_dicts()
        strategy: PredictiveEntropy = PredictiveEntropy(0, config, 0, 12345)
        file_dict, out_list = strategy.strategy(inputs, device=torch.device("cpu"))
        output_size = compute_conv_output_size(
            input_size, patch_size, torch.Tensor(config.agg_stride), padding
        )
        assert len(out_list) == torch.prod(output_size).item()


def input_probs() -> list[torch.Tensor]:
    num_classes = 5
    num_pred = 5
    size = [10, 10, 10]
    probs = torch.randn(num_pred, num_classes, *size)
    probs = probs / probs.sum(1, keepdim=True)
    probs = [probs[i] for i in range(len(probs))]
    return probs


def input_dicts() -> list[dict[str, torch.Tensor]]:
    probs = input_probs()
    return [{"probs": probs[i]} for i in range(len(probs))]
