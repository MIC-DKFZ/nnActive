import numpy as np
import pytest
import torch

from nnactive.strategies.uncertainties import Probs, ProbsFromFiles


@pytest.fixture(params=[Probs.mutual_information, ProbsFromFiles.mutual_information])
def mutual_information_func(request):
    return request.param


@pytest.fixture(params=[Probs.pred_entropy, ProbsFromFiles.pred_entropy])
def pred_entropy_func(request):
    return request.param


@pytest.fixture(params=[Probs.exp_entropy, ProbsFromFiles.exp_entropy])
def exp_entropy_func(request):
    return request.param


def probs_two_class():
    probs_start = torch.arange(0, 0.51, 0.1)  # N

    probs = (
        torch.stack([probs_start, 1 - probs_start], dim=1).unsqueeze(1).unsqueeze(-1)
    )  # N x M=1 x C x 1
    return probs, probs_start


def test_pred_entropy(pred_entropy_func):
    probs, probs_start = probs_two_class()
    entropies = torch.empty(probs.shape[0])
    for i in range(len(probs)):
        entropies[i] = pred_entropy_func(probs[i], device=torch.device("cpu"))

    assert entropies.argmax() == probs.shape[0] - 1
    assert np.isclose(entropies[0].item(), 0)
    assert np.isclose(entropies[probs_start == 0.5].item(), -np.log(0.5))

    # Ensure that when mean prob is p(Y=1)=0.5 H(p(Y))=np.log(0.5)
    probs_05 = torch.concatenate([probs, 1 - probs], dim=1)
    entropies = torch.empty(probs_05.shape[0])
    for i in range(len(probs_05)):
        entropies[i] = pred_entropy_func(probs_05[i], device=torch.device("cpu"))
        assert np.isclose(entropies[i].item(), -np.log(0.5))


def test_exp_entropy(exp_entropy_func):
    probs, probs_start = probs_two_class()
    exp_entropies = torch.empty(probs.shape[0])

    # verify that probs are identical
    probs_ident = torch.concatenate([probs] * 4, dim=1)  # N x M=4 x C=2 x 1
    for i in range(len(probs)):
        exp_entropies[i] = exp_entropy_func(probs_ident[i], device=torch.device("cpu"))
        assert np.isclose(
            exp_entropies[i].item(),
            exp_entropy_func(probs_ident[i], device=torch.device("cpu")).item(),
        ), (
            exp_entropies[i].item(),
            exp_entropy_func(probs_ident[i], device=torch.device("cpu")).item(),
        )
    # ensure values are proper according to math and max values etc. fit
    assert exp_entropies.argmax() == probs.shape[0] - 1
    assert np.isclose(exp_entropies[0].item(), 0)
    assert np.isclose(exp_entropies[probs_start == 0.5].item(), -np.log(0.5))

    # verify exp_ent < pred_ent
    probs_05 = torch.concatenate([probs, 1 - probs], dim=1)  # N x M=2 x C=2 x 1
    for i in range(len(probs_05)):
        assert (
            exp_entropy_func(probs_05[i], device=torch.device("cpu"))
            <= Probs.pred_entropy(probs_05[i], device=torch.device("cpu"))
        ).item()


def test_mutual_information(mutual_information_func):
    probs, probs_start = probs_two_class()

    # verify for identical probs over M, MI = 0
    probs_ident = torch.concatenate([probs] * 4, dim=1)  # N x M=4 x C=2 x 1
    for i in range(len(probs)):
        assert np.isclose(
            0,
            mutual_information_func(probs_ident[i], device=torch.device("cpu")).item(),
        )

    # verify for probs differing that MI behaves accordingly
    probs_05 = torch.concatenate([probs, 1 - probs], dim=1)  # N x M=2 x C=2 x 1
    mutual_informations = torch.empty(probs.shape[0])
    for i in range(len(probs_05)):
        mutual_informations[i] = mutual_information_func(
            probs_05[i], device=torch.device("cpu")
        )

    assert np.isclose(mutual_informations[0].item(), -np.log(0.5))
    assert mutual_informations.argmax() == 0
    assert np.isclose(mutual_informations[probs_start == 0.5].item(), 0)
