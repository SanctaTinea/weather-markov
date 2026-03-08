import numpy as np
import pandas as pd

from weather_markov.models.decade_base import DecadeBasedPredictor


class NonUniformMarkovChainPredictor(DecadeBasedPredictor):
    """
    Method 3: non-homogeneous Markov chain.

    Uses an individual transition matrix P(k, k+1) for each adjacent
    decade pair and predicts recursively:
    pi(k+1) = P(k, k+1)^T @ pi(k).
    """

    def __init__(
        self,
        discretizer,
        months: list[int] | None = None,
        start_label: tuple[int, int] = (2, 1),
        end_label: tuple[int, int] = (5, 1),
    ):
        super().__init__(discretizer, months)
        self.start_label = start_label
        self.end_label = end_label

        self.transition_labels: list[tuple[tuple[int, int], tuple[int, int]]] = []
        self.transition_matrices: list[pd.DataFrame] = []

    def fit(self, data: pd.DataFrame) -> "NonUniformMarkovChainPredictor":
        pairs_per_transition = self._build_transition_pairs(data)
        labels = self._get_decade_labels(data)

        if self.start_label not in labels or self.end_label not in labels:
            raise ValueError(
                f"Labels must be present in data. start={self.start_label}, end={self.end_label}"
            )

        start_idx = labels.index(self.start_label)
        end_idx = labels.index(self.end_label)
        if start_idx >= end_idx:
            raise ValueError("start_label must be earlier than end_label")

        path = labels[start_idx : end_idx + 1]
        self.transition_labels = [
            (path[i], path[i + 1]) for i in range(len(path) - 1)
        ]
        all_states = list(self.discretizer.labels)
        self.transition_matrices = [
            self._build_dense_probability_matrix(
                pairs_per_transition[label], all_states=all_states
            )
            for label in self.transition_labels
        ]

        self._is_fitted = True
        return self

    @staticmethod
    def _build_dense_probability_matrix(
        pairs: list[tuple[str, str]], all_states: list[str]
    ) -> pd.DataFrame:
        counts = pd.DataFrame(0.0, index=all_states, columns=all_states)
        for from_state, to_state in pairs:
            counts.loc[from_state, to_state] += 1.0

        row_sums = counts.sum(axis=1)
        probs = counts.div(row_sums.replace(0.0, np.nan), axis=0)

        # If state was never observed as `from_state`, use uniform row.
        uniform_row = pd.Series(1.0 / len(all_states), index=all_states)
        probs = probs.apply(lambda row: uniform_row if row.isna().all() else row, axis=1)
        return probs.fillna(0.0)

    def predict(self, state: str) -> dict[str, float]:
        self._check_fitted()

        if state not in self.discretizer.labels:
            raise ValueError(f"Unknown state: {state}")

        pi = pd.Series(0.0, index=self.discretizer.labels)
        pi.loc[state] = 1.0

        for p_k_k1 in self.transition_matrices:
            pi = p_k_k1.T.dot(pi)

        return pi.to_dict()
