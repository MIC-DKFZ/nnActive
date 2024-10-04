from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from nnactive.analyze.metrics import PairwisePenaltyMatrix


def build_eample_data() -> pd.DataFrame:
    data = [
        {"Query Method": "alg1", "num_samples": 10, "test_acc": 0.8},
        {"Query Method": "alg1", "num_samples": 10, "test_acc": 0.9},
        {"Query Method": "alg1", "num_samples": 10, "test_acc": 0.85},
        {"Query Method": "alg1", "num_samples": 20, "test_acc": 0.8},
        {"Query Method": "alg1", "num_samples": 20, "test_acc": 0.9},
        {"Query Method": "alg1", "num_samples": 20, "test_acc": 0.85},
        {"Query Method": "alg1", "num_samples": 30, "test_acc": 0.8},
        {"Query Method": "alg1", "num_samples": 30, "test_acc": 0.9},
        {"Query Method": "alg1", "num_samples": 30, "test_acc": 0.85},
    ]
    data2 = deepcopy(data)
    for d in data2:
        d["Query Method"] = "alg2"

    data3 = deepcopy(data)
    for d in data3:
        d["Query Method"] = "win_alg"
        d["test_acc"] += 0.2

    data = data + data2 + data3
    df = pd.DataFrame(data)
    return df


def test_pairwise_penalty_matrix_from_df():
    df = build_eample_data()
    pm = PairwisePenaltyMatrix.from_df(df)

    assert pm.matrix["alg1"]["win_alg"] > 0
    assert pm.matrix["alg2"]["win_alg"] > 0
    assert pm.matrix["alg1"]["alg2"] == 0


def test_pairwise_penalty_matrix_save_load(tmpdir):
    df = build_eample_data()
    pm = PairwisePenaltyMatrix.from_df(df)

    save_path = tmpdir / "ppm.json"
    pm.save(save_path)

    loaded_pm = PairwisePenaltyMatrix.load(save_path)

    assert pm.matrix == loaded_pm.matrix
    assert pm.alpha == loaded_pm.alpha
    assert pm.max_pos_ent == loaded_pm.max_pos_ent


def test_pairwise_penalty_matrix_create_merged_matrix():
    df = build_eample_data()
    pm = PairwisePenaltyMatrix.from_df(df)

    merged_pm = PairwisePenaltyMatrix.create_merged_matrix([pm, pm, pm])

    assert merged_pm.matrix["win_alg"]["alg1"] == 3 * pm.matrix["win_alg"]["alg1"]
    assert merged_pm.matrix["win_alg"]["alg2"] == 3 * pm.matrix["win_alg"]["alg2"]
    assert merged_pm.matrix["alg1"]["alg2"] == 3 * pm.matrix["alg1"]["alg2"]
