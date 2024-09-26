from pprint import pprint

import numpy as np
import pandas as pd
from scipy import stats

from nnactive.utils.io import load_json, save_json


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


class PairwisePenaltyMatrix:
    def __init__(
        self,
        df: pd.DataFrame,
        value_key: str = "test_acc",
        budget_key: str = "num_samples",
        qm_key: str = "Query Method",
        alpha: float = 0.05,
    ):
        """Initializes the PairwiseMatrix class with the given DataFrame and keys for a single Experiment Setting.
        The computation does not ensure that each algorithm has the full budget available and that each algorithm has the same amount of experiments.

        Information:
        Code is adapted from: https://github.com/JordanAsh/badge/blob/master/scripts/agg_results.py
        We assume here that we are never in the saturation area.
        --> Performance <= 0.99*(full dataset performance).

        Args:
            df (pd.DataFrame): DataFrame containing ordered experiments with keys [Query Method, num_samples, value_key]
            value_key (str, optional): Value based on which PPM is computed. Defaults to "test_acc".
            budget_key (str, optional): Key for the budget. Defaults to "num_samples".
            qm_key (str, optional): Key for the query method. Defaults to "Query Method".
        """
        self.df = df
        self.value_key = value_key
        self.budget_key = budget_key
        self.qm_key = qm_key
        self.algs = df[qm_key].unique()
        self.algs.sort()
        self._matrix = {a1: {a2: 0 for a2 in self.algs} for a1 in self.algs}
        self.nBudgets = len(df[budget_key].unique())
        self.alpha_level = alpha
        self.compute_matrix()

    def compute_matrix(self):
        """Computes pairwise penalty matrix (PPM) for active learning experiments.
        Each row i indicates the number of settings in which algorithm i beats other algorithms
        and each column j indicates the number of settings in which algorithm j is beaten by another algorithm.

        Returns:
            Dict[str, Dict[str, float]]: matrix[i][j] How often does algo i outperform algo j
        """
        for num_sample in self.df[self.budget_key].unique():
            for alg1 in self.algs:
                for alg2 in self.algs:
                    if alg1 == alg2:
                        continue
                    res1 = self.df[self.df[self.qm_key] == alg1]
                    res2 = self.df[self.df[self.qm_key] == alg2]
                    exp1 = res1[res1[self.budget_key] == num_sample][
                        self.value_key
                    ].values
                    exp2 = res2[res2[self.budget_key] == num_sample][
                        self.value_key
                    ].values

                    if len(exp1) <= 1 or len(exp2) <= 1:
                        continue

                    update_matrix = self._ttest_samples(exp1, exp2)
                    if update_matrix:
                        self._matrix[alg1][alg2] += 1.0 / self.nBudgets

    def _ttest_samples(self, exp1: np.ndarray, exp2: np.ndarray):
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
        if mu < 0 and pval < self.alpha_level:
            return True

    @property
    def matrix(self) -> dict[str, dict[str, float]]:
        """Returns the pairwise penalty matrix without the mean row."""
        return self._matrix

    def save(self, path: str):
        """Save the pairwise penalty matrix to a json file."""
        save_json(self.matrix, path)

    def print(self):
        df = PairwisePenaltyMatrix.creat_vis_df(self.matrix)
        pprint(df)

    @staticmethod
    def plot_pairwise_matrix(
        matrix: dict[str, dict[str, float]],
        title_tag: str = "Test",
        name_dict: dict[str, str] = None,
        max_poss_ent: int = 1,
        savepath: str = None,
        show: bool = False,
    ):
        """Plots or saves pairwise penalty matrix (PPM).
        Each row i indicates the number of settings in which algorithm i beats other algorithms
        and each column j indicates the number of settings in which algorithm j is beaten by another algorithm.

        Args:
            matrix (Dict[str, Dict[str, float]]): PPM matrix.
            title_tag (str, optional): Title of Figure. Defaults to "Test".
            name_dict (Dict[str, str], optional): {name_in_matrix: name_in_plot}. Defaults to None.
            max_poss_ent (int, optional): Maximal value obtainable, equal to #AL experiments. Defaults to 1.
            savepath (str, optional): Path to save the plot. Defaults to None.
            show (bool, optional): Whether to show the plot. Defaults to False.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Convert matrix to DataFrame for plotting
        df_matrix = PairwisePenaltyMatrix.creat_vis_df(matrix)

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

    @staticmethod
    def creat_vis_df(matrix):
        df_matrix = pd.DataFrame(matrix)

        mean_col = df_matrix.sum(axis=0) / (df_matrix.shape[0] - 1)
        df_matrix.loc["Mean"] = mean_col
        df_matrix.round(2)
        return df_matrix


if __name__ == "__main__":
    from copy import deepcopy

    values = np.array([0.9] * 3 + [0.8] * 6)
    print(f"values={values}")
    print(f"auc={compute_auc(values)}")

    data = {
        "alg1": [0.8, 0.9, 0.85, 0.87, 0.88],
        "alg2": [0.75, 0.82, 0.80, 0.78, 0.79],
        "alg3": [0.78, 0.85, 0.83, 0.81, 0.82],
        "alg4": [0.77, 0.84, 0.82, 0.80, 0.81],
    }
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
    pm = PairwisePenaltyMatrix(df)
    pm.compute_matrix()
    pm.print()
    pm.plot_pairwise_matrix(pm.matrix, savepath="ppm_new.png")
