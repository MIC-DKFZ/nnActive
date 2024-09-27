from __future__ import annotations

from pprint import pprint
from typing import Any

import numpy as np
import pandas as pd
from pydantic.dataclasses import dataclass
from scipy import stats

from nnactive.utils.io import load_json, save_json
from nnactive.utils.pyutils import get_clean_dataclass_dict


def compute_auc(
    value: np.ndarray,
    x: None | np.ndarray = None,
    dx: None | float = None,
) -> float:
    """Computes Area Under the Curve using the trapezoid method so that [0.1, 0.1, 0.1]=0.1.

    E.g.
    Computes the Area Under the Budget Curve following:
    https://www.ijcai.org/proceedings/2021/0634.pdf
    Default dx makes integral go from 0 to 1.

    Args:
        performance (np.ndarray): value under which budget should be computed
        x (Union[None, np.ndarray], optional): The sample points corresponding to the y values. If x is None, the sample points are assumed to be evenly spaced dx apart. The default is None.. Defaults to None.
        dx (float, optional): difference of values. Defaults to None.

    Returns:
        float: aubc value
    """
    if x is None:
        if dx is None:
            # simulated integral goes from 0 to 1
            dx = 1 / (len(value) - 1)
    return np.trapz(value, x, dx).item()


@dataclass
class PairwiseMatrix:
    matrix: dict[str, dict[str, float]]
    alpha: float = 0.05
    max_pos_ent: int = 1
    compute_kwargs: dict[str, Any] = None

    def __post_init__(self):
        # ensure that matrix is symmetric and has proper values
        for alg1 in self.matrix:
            assert list(self.matrix[alg1].keys()) == self.algs

    @property
    def algs(self) -> list[str]:
        return list(self.matrix.keys())

    def detete_alg(self, alg: str):
        self.matrix.pop(alg)
        for key in self.matrix:
            self.matrix[key].pop(alg)

    @staticmethod
    def creat_vis_df(matrix):
        df_matrix = pd.DataFrame(matrix)

        mean_col = df_matrix.sum(axis=0) / (df_matrix.shape[0] - 1)
        df_matrix.loc["Mean"] = mean_col
        df_matrix.round(2)
        return df_matrix

    def save(self, path: str):
        """Save the pairwise penalty matrix to a json file."""
        save_dict = get_clean_dataclass_dict(self)
        save_json(save_dict, path)

    @classmethod
    def load(cls, path: str) -> PairwiseMatrix:
        """Load the pairwise penalty matrix from a json file."""
        return cls(**load_json(path))

    def print(self):
        df = PairwiseMatrix.creat_vis_df(self.matrix)
        pprint(df)

    @classmethod
    def create_merged_matrix(
        cls,
        matrices: list[PairwiseMatrix],
        strict_alg: bool = True,
        strict_alpha: bool = True,
    ) -> PairwiseMatrix:
        """Merges multiple PairwiseMatrices into a single PairwiseMatrix.
        The matrices are merged by summing the values of the matrices.

        Args:
            matrices (List[PairwiseMatrix]): List of PairwiseMatrices to be merged.

        Returns:
            PairwiseMatrix: Merged PairwiseMatrix.
        """
        matrix = {
            alg1: {alg2: 0 for alg2 in matrices[0].algs} for alg1 in matrices[0].algs
        }
        max_pos_ent = 0
        for pm in matrices:
            if strict_alg and pm.algs != matrices[0].algs:
                raise ValueError("Algorithms in matrices do not match.")
            if strict_alpha and pm.alpha != matrices[0].alpha:
                raise ValueError("Alpha values in matrices do not match.")
            max_pos_ent += pm.max_pos_ent
            for alg1 in pm.algs:
                for alg2 in pm.algs:
                    matrix[alg1][alg2] += pm.matrix[alg1][alg2]

        return cls(matrix, alpha=matrices[0].alpha, max_pos_ent=max_pos_ent)

    @staticmethod
    def plot_pairwise_matrix(
        matrix: dict[str, dict[str, float]] | PairwiseMatrix,
        title_tag: str | None = None,
        name_dict: dict[str, str] | None = None,
        max_poss_ent: int | None = 1,
        savepath: str = None,
        show: bool = False,
    ):
        """Plots or saves PairWise Matrix.
        Each row i indicates the number of settings in which algorithm i beats other algorithms
        and each column j indicates the number of settings in which algorithm j is beaten by another algorithm.

        Args:
            matrix (Dict[str, Dict[str, float]]): PPM matrix.
            title_tag (str, optional): Title of Figure. Defaults to "Test".
            name_dict (Dict[str, str], optional): {name_in_matrix: name_in_plot}. Defaults to None.
            max_poss_ent (int, optional): Maximal value obtainable, equal to #AL Settings. Defaults to 1.
            savepath (str, optional): Path to save the plot. Defaults to None.
            show (bool, optional): Whether to show the plot. Defaults to False.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Convert matrix to DataFrame for plotting
        if isinstance(matrix, PairwiseMatrix):
            matrix = matrix.matrix
        df_matrix = PairwiseMatrix.creat_vis_df(matrix)

        if max_poss_ent is None:
            max_poss_ent = df_matrix.max().max()

        # Rename columns and index if name_dict is provided
        if name_dict:
            df_matrix.rename(columns=name_dict, index=name_dict, inplace=True)

        # Plot the heatmap
        fig, axs = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            ax=axs,
            data=df_matrix,
            annot=True,
            cmap="viridis",
            cbar=True,
            vmin=0,
            vmax=max_poss_ent,
        )
        axs.set_title(f"Pairwise Penalty Matrix ({title_tag})")

        # Save the plot if savepath is provided
        if savepath:
            plt.savefig(savepath, bbox_inches="tight")

        # Show the plot if show is True
        if show:
            plt.show()
        else:
            plt.close()


class PairwisePenaltyMatrix(PairwiseMatrix):
    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        qm_key: str = "Query Method",
        budget_key: str = "num_samples",
        value_key: str = "test_acc",
        skip_first: bool = False,
        alpha: float = 0.05,
        max_pos_ent: int = 1,
    ):
        """Initializes the PairwiseMatrix class with the given DataFrame and keys for a single Experiment Setting.
        The computation does not ensure that each algorithm has the full budget available and that each algorithm has the same amount of experiments.

        Information:
        Code is adapted from: https://github.com/JordanAsh/badge/blob/master/scripts/agg_results.py
        We assume here that we are never in the saturation area.
        --> Performance <= 0.99*(full dataset performance).

        Computes pairwise penalty matrix (PPM) for active learning experiments.
        Each row i indicates the number of settings in which algorithm i beats other algorithms
        and each column j indicates the number of settings in which algorithm j is beaten by another algorithm.

        Returns:
            Dict[str, Dict[str, float]]: matrix[i][j] How often does algo i outperform algo j
        """
        compute_kwargs = {
            "qm_key": qm_key,
            "budget_key": budget_key,
            "value_key": value_key,
            "skip_first": skip_first,
        }
        algs = df[qm_key].unique()
        num_samples = df[budget_key].unique()
        num_samples.sort()
        if skip_first:
            num_samples = num_samples[1:]
        n_budgets = len(num_samples)
        matrix = {a1: {a2: 0 for a2 in algs} for a1 in algs}
        for num_sample in df[budget_key].unique():
            for alg1 in algs:
                for alg2 in algs:
                    if alg1 == alg2:
                        continue
                    res1 = df[df[qm_key] == alg1]
                    res2 = df[df[qm_key] == alg2]
                    exp1: np.ndarray = res1[res1[budget_key] == num_sample][
                        value_key
                    ].values
                    exp2: np.ndarray = res2[res2[budget_key] == num_sample][
                        value_key
                    ].values

                    if len(exp1) <= 1 or len(exp2) <= 1:
                        continue

                    update_matrix = cls._test_samples(exp1, exp2, alpha)
                    if update_matrix:
                        matrix[alg1][alg2] += 1.0 / n_budgets
        return cls(
            matrix, alpha=alpha, max_pos_ent=max_pos_ent, compute_kwargs=compute_kwargs
        )

    @staticmethod
    def _test_samples(exp1: np.ndarray, exp2: np.ndarray, alpha) -> bool:
        """Performs a t-test on two samples and returns True if mean of exp1 is significantly smaller than that of exp2."""
        n1 = len(exp1)
        n2 = len(exp2)

        n = min(n1, n2)
        z = exp1[:n] - exp2[:n]
        mu = np.mean(z)
        # TODO: check if this implementation is correct!
        # Shouldn't we perform a one-sided test?
        # e.g.
        # t, pval = stats.ttest_1samp(z, 0.0, alternative='less')
        # Jeremias agrees that this test should be one-sided.
        ######## Correction Term for significance level? ########
        # Should we correct for multiple testing?
        #
        #### Motivation for Not Correcting for Multiple Testing ####
        # We test multiple tests, but we are interested in relative values.
        # X is better than Y due to lower values.
        # But X and Y are both subject to the multiple testing issue.
        # So does it really matter?
        #
        #### Motivation for Correcting for Multiple Testing ####
        # Ranking is created based on tests.
        # Question: Do specific algorithms get better/worse results by not correcting for multiple testing?
        # If so, we should correct for multiple testing.
        # If not, we can ignore the multiple testing issue.
        #
        # Approach: correct along budget axis (e.g.10 loops, therefore correct for 10 tests)
        # Example: 3 methods X, Y, Z. GT: X better Z=0.03. Y better Z=0.1.
        # Results: X and Y have equal score against Z. (0.5) due to multiple tests (but X has much lower pval).
        # Therefore X gets a disadvantage against Y.
        # --> Correct for multiple tests along budget-axis.
        #
        # Approach: correct along algorithm axis (e.g. 4 algorithms, therefore correct for 4 tests)
        #
        ##########################################################
        # Original Test (two-sided)
        # t, pval = stats.ttest_1samp(z, 0.0)
        # Proposed Left-sided Test (one-sided)
        t, pval = stats.ttest_1samp(z, 0.0, alternative="less")
        if mu < 0 and pval < alpha:
            return True
        return False


if __name__ == "__main__":
    from copy import deepcopy

    values = np.array([0.9] * 3 + [0.8] * 6)
    print(f"values={values}")
    print(f"auc={compute_auc(values)}")

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
    pm = PairwisePenaltyMatrix.from_df(df)
    pm.print()
    pm.plot_pairwise_matrix(pm.matrix, savepath="ppm.png")
    pm.save("ppm.json")

    pm3 = PairwisePenaltyMatrix.create_merged_matrix([pm, pm, pm])
    pm3.print()
    pm3.detete_alg("win_alg")
    pm3.print()

    pm2 = PairwisePenaltyMatrix.load("ppm.json")
    # pm = PairwisePenaltyMatrix(df)
    # pm.compute_matrix()
    # pm.print()
    # pm.plot_pairwise_matrix(pm.matrix, savepath="ppm_new.png")
