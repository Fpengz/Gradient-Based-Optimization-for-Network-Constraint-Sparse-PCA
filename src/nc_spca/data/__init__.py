"""Dataset builders."""

from .real import build_feature_graph, load_real_dataset
from .synthetic.generators import generate_synthetic_dataset
from .synthetic.graphs import GraphData

__all__ = ["GraphData", "build_feature_graph", "generate_synthetic_dataset", "load_real_dataset"]
