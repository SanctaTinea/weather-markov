import pandas as pd

from weather_markov.markov.chain import MarkovChain
from weather_markov.markov.graph import TransitionGraph
from weather_markov.models.decade_base import DecadeBasedPredictor


class NonUniformMarkovChainPredictor(DecadeBasedPredictor):
    """
    Method 3: non-homogeneous Markov chain built from transition graphs
    for each specific adjacent decade pair (February–May).

    Transition probabilities are allowed to vary by transition step.
    """

    def __init__(self, discretizer, months=None):
        super().__init__(discretizer, months)
        self.transition_graphs: list[TransitionGraph] = []
        self.transition_labels: list[tuple[tuple[int, int], tuple[int, int]]] = []

    def fit(self, data: pd.DataFrame) -> "NonUniformMarkovChainPredictor":
        pairs_per_transition = self._build_transition_pairs(data)

        self.transition_labels = list(pairs_per_transition.keys())
        self.transition_graphs = [
            TransitionGraph.from_pairs(pairs_per_transition[label])
            for label in self.transition_labels
        ]

        self._is_fitted = True
        return self

    def predict(self, state: str) -> dict[str, float]:
        """
        Predicts the final decade distribution by chaining
        transition graphs in chronological order.
        """
        self._check_fitted()

        chain = MarkovChain(self.transition_graphs)
        return chain.predict(state)
