import torch


def log_entropy(probs: torch.Tensor):
    """Computes logarithm for the entropy function and ignores values equal to 0

    Args:
        probs (torch.Tensor): probability values
    """
    out = torch.log(probs)
    # overwrite values which are 0
    out[out == -torch.inf] = 0
    return out


def prob_pred_entropy(probs: torch.Tensor) -> torch.Tensor:
    """Compute predictive entropy.

    Args:
        probs (torch.Tensor): M x C x ...
    """
    mean_prob = probs.mean(dim=0)
    return -torch.sum(mean_prob * log_entropy(mean_prob), dim=0)


def prob_exp_entropy(probs: torch.Tensor) -> torch.Tensor:
    """Compute expected entropy.

    Args:
        probs (torch.Tensor): M x C x ...
    """
    return -torch.mean(torch.sum(probs * log_entropy(probs), dim=1), dim=0)


def prob_mutual_information(probs: torch.Tensor) -> torch.Tensor:
    """Compute mutual information.

    Args:
        probs (torch.Tensor): M x C x ...
    """
    return prob_pred_entropy(probs) - prob_exp_entropy(probs)
