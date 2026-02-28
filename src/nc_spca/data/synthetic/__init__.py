"""Synthetic data utilities."""

from .generators import generate_synthetic_dataset
from .graphs import (
    GraphData,
    adjacency_to_laplacian,
    chain_graph,
    er_graph,
    grid_graph,
    knn_graph,
    random_geometric_graph,
    sbm_graph,
)

__all__ = [
    "GraphData",
    "adjacency_to_laplacian",
    "chain_graph",
    "er_graph",
    "generate_synthetic_dataset",
    "grid_graph",
    "knn_graph",
    "random_geometric_graph",
    "sbm_graph",
]
