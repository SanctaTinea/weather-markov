import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

from weather_markov.markov.graph import TransitionGraph


def plot_transition_matrix(graph: TransitionGraph, title: str = "") -> None:
    """Heatmap of the transition probability matrix"""
    matrix = graph.get_probability_matrix()
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlOrRd")
    plt.title(title or "Transition Probability Matrix")
    plt.show()


def plot_graph_network(graph: TransitionGraph, title: str = "") -> None:
    """Network visualisation of the transition graph via networkx"""
    G = nx.DiGraph()
    for fs in graph.from_states:
        for ts, prob in graph.predict(fs).items():
            G.add_edge(fs, ts, weight=prob)
    pos = nx.spring_layout(G, seed=42)
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw(G, pos, with_labels=True, node_color="lightblue", arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title(title)
    plt.show()


def plot_prediction_distribution(
    distribution: dict[str, float], true_label: str | None = None
) -> None:
    """Bar chart of the predicted probability distribution"""
    labels, probs = zip(*sorted(distribution.items(), key=lambda x: float(x[0].split(',')[0][1:])))
    colors = ["red" if l == true_label else "steelblue" for l in labels]
    plt.bar(labels, probs, color=colors)
    plt.ylabel("Probability")
    plt.title("Predicted May Temperature Distribution")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


def compare_methods(
    predictions: dict[str, dict[str, float]], true_label: str | None = None
) -> None:
    """Compares predictions from all three methods on a single figure"""
    fig, axes = plt.subplots(1, len(predictions), figsize=(14, 4))
    for ax, (method_name, dist) in zip(axes, predictions.items()):
        labels, probs = zip(*sorted(dist.items()))
        colors = ["red" if l == true_label else "steelblue" for l in labels]
        ax.bar(labels, probs, color=colors)
        ax.set_title(method_name)
        ax.set_ylabel("Probability")
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.show()
