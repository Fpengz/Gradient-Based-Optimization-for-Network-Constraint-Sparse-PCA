import numpy as np

from src.experiments.stats import bootstrap_mean_diff_ci, paired_significance


def test_bootstrap_mean_diff_ci_returns_ordered_interval():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([0.5, 1.5, 2.5, 3.5])
    low, high = bootstrap_mean_diff_ci(a, b, n_boot=500, random_state=0)
    assert low <= high
    assert low <= 0.5 <= high


def test_paired_significance_outputs_fields():
    a = np.array([0.7, 0.8, 0.75, 0.77, 0.79])
    b = np.array([0.65, 0.72, 0.7, 0.73, 0.74])
    res = paired_significance(a, b, metric="f1", method_a="A", method_b="B")
    assert res.metric == "f1"
    assert res.method_a == "A"
    assert res.method_b == "B"
    assert res.n_pairs == 5
    assert np.isfinite(res.mean_diff)


def test_paired_significance_handles_identical_pairs_without_nan():
    a = np.array([0.7, 0.7, 0.7, 0.7])
    b = np.array([0.7, 0.7, 0.7, 0.7])
    res = paired_significance(a, b, metric="f1", method_a="A", method_b="B")
    assert res.n_pairs == 4
    assert res.mean_diff == 0.0
    assert res.p_value == 1.0
    assert res.test == "degenerate_equal"
    assert res.ci_low == 0.0
    assert res.ci_high == 0.0
