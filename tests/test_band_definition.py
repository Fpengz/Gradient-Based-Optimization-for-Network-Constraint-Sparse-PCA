from grpca_gd.analysis.band import select_band


def test_select_band_returns_contiguous_block():
    rows = [
        {"lambda2": 0.04, "seed": 0, "method": "Proposed", "support_f1": 0.80, "graph_smoothness_norm": 0.20},
        {"lambda2": 0.04, "seed": 0, "method": "SparseNoGraph", "support_f1": 0.81, "graph_smoothness_norm": 0.30},
        {"lambda2": 0.05, "seed": 0, "method": "Proposed", "support_f1": 0.80, "graph_smoothness_norm": 0.18},
        {"lambda2": 0.05, "seed": 0, "method": "SparseNoGraph", "support_f1": 0.81, "graph_smoothness_norm": 0.30},
        {"lambda2": 0.10, "seed": 0, "method": "Proposed", "support_f1": 0.79, "graph_smoothness_norm": 0.17},
        {"lambda2": 0.10, "seed": 0, "method": "SparseNoGraph", "support_f1": 0.81, "graph_smoothness_norm": 0.30},
        {"lambda2": 0.20, "seed": 0, "method": "Proposed", "support_f1": 0.70, "graph_smoothness_norm": 0.15},
        {"lambda2": 0.20, "seed": 0, "method": "SparseNoGraph", "support_f1": 0.81, "graph_smoothness_norm": 0.30},
    ]
    band = select_band(
        rows,
        method="Proposed",
        baseline="SparseNoGraph",
        f1_metric="support_f1",
        smooth_metric="graph_smoothness_norm",
        f1_tolerance=0.02,
        smoothness_margin=0.05,
    )
    assert band == [0.05, 0.10]
