"""Statistical utilities for benchmark significance reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import stats


@dataclass
class SignificanceResult:
    metric: str
    method_a: str
    method_b: str
    n_pairs: int
    mean_diff: float
    p_value: float
    test: str
    ci_low: float
    ci_high: float


def bootstrap_mean_diff_ci(
    values_a: Iterable[float],
    values_b: Iterable[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    random_state: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for mean(a-b) on paired samples."""
    a = np.asarray(list(values_a), dtype=float)
    b = np.asarray(list(values_b), dtype=float)
    if a.shape != b.shape:
        raise ValueError("values_a and values_b must have matching shapes")
    if a.size == 0:
        return float("nan"), float("nan")
    diffs = a - b
    rng = np.random.default_rng(random_state)
    idx = rng.integers(0, diffs.size, size=(n_boot, diffs.size))
    means = diffs[idx].mean(axis=1)
    low = np.quantile(means, alpha / 2)
    high = np.quantile(means, 1 - alpha / 2)
    return float(low), float(high)


def paired_significance(
    values_a: Iterable[float],
    values_b: Iterable[float],
    metric: str,
    method_a: str,
    method_b: str,
    alpha: float = 0.05,
    random_state: int = 0,
) -> SignificanceResult:
    """Wilcoxon (fallback paired t-test) with bootstrap CI for paired samples."""
    a = np.asarray(list(values_a), dtype=float)
    b = np.asarray(list(values_b), dtype=float)
    if a.shape != b.shape:
        raise ValueError("Paired significance requires same-length arrays.")
    if a.size < 2:
        return SignificanceResult(
            metric=metric,
            method_a=method_a,
            method_b=method_b,
            n_pairs=int(a.size),
            mean_diff=float(np.mean(a - b)) if a.size else float("nan"),
            p_value=float("nan"),
            test="insufficient_pairs",
            ci_low=float("nan"),
            ci_high=float("nan"),
        )
    diff = a - b
    if np.allclose(diff, 0.0):
        return SignificanceResult(
            metric=metric,
            method_a=method_a,
            method_b=method_b,
            n_pairs=int(a.size),
            mean_diff=0.0,
            p_value=1.0,
            test="degenerate_equal",
            ci_low=0.0,
            ci_high=0.0,
        )
    try:
        stat = stats.wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
        p_value = float(stat.pvalue)
        test_name = "wilcoxon"
    except ValueError:
        stat = stats.ttest_rel(a, b, alternative="two-sided")
        p_value = float(stat.pvalue)
        test_name = "paired_ttest"
    ci_low, ci_high = bootstrap_mean_diff_ci(
        a, b, alpha=alpha, random_state=random_state
    )
    return SignificanceResult(
        metric=metric,
        method_a=method_a,
        method_b=method_b,
        n_pairs=int(a.size),
        mean_diff=float(np.mean(diff)),
        p_value=p_value,
        test=test_name,
        ci_low=ci_low,
        ci_high=ci_high,
    )
