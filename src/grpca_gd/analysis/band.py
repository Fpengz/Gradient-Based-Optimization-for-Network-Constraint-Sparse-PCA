from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _group_means(
    rows: Iterable[dict[str, Any]],
    method: str,
    lambda_key: str,
    metrics: tuple[str, ...],
) -> dict[float, dict[str, float]]:
    grouped: dict[float, dict[str, list[float]]] = defaultdict(
        lambda: {metric: [] for metric in metrics}
    )
    for row in rows:
        if row.get("method") != method:
            continue
        lam = float(row[lambda_key])
        bucket = grouped[lam]
        for metric in metrics:
            bucket[metric].append(float(row[metric]))

    return {
        lam: {metric: _mean(values) for metric, values in bucket.items()}
        for lam, bucket in grouped.items()
    }


def _longest_contiguous_block(
    candidates: list[float], ordered_lambdas: list[float]
) -> list[float]:
    if not candidates:
        return []

    candidate_set = set(candidates)

    def runs(values: list[float]) -> list[list[float]]:
        best_runs: list[list[float]] = []
        current: list[float] = []
        for lam in values:
            if lam in candidate_set:
                current.append(lam)
            elif current:
                best_runs.append(current)
                current = []
        if current:
            best_runs.append(current)
        return best_runs

    interior_lambdas = (
        ordered_lambdas[1:-1] if len(ordered_lambdas) > 2 else ordered_lambdas
    )
    interior_runs = runs(interior_lambdas)
    if interior_runs:
        return max(interior_runs, key=len)

    return max(runs(ordered_lambdas), key=len)


def select_band(
    rows: Iterable[dict[str, Any]],
    method: str,
    baseline: str,
    f1_metric: str,
    smooth_metric: str,
    f1_tolerance: float,
    smoothness_margin: float,
) -> list[float]:
    rows = list(rows)
    method_means = _group_means(rows, method, "lambda2", (f1_metric, smooth_metric))
    baseline_means = _group_means(
        rows, baseline, "lambda2", (f1_metric, smooth_metric)
    )

    ordered_lambdas = sorted({float(row["lambda2"]) for row in rows})
    candidates: list[float] = []

    for lam in ordered_lambdas:
        if lam not in method_means or lam not in baseline_means:
            continue

        method_f1 = method_means[lam][f1_metric]
        baseline_f1 = baseline_means[lam][f1_metric]
        method_smooth = method_means[lam][smooth_metric]
        baseline_smooth = baseline_means[lam][smooth_metric]

        if (
            method_f1 >= baseline_f1 - f1_tolerance
            and (baseline_smooth - method_smooth) >= smoothness_margin
        ):
            candidates.append(lam)

    return _longest_contiguous_block(candidates, ordered_lambdas)
