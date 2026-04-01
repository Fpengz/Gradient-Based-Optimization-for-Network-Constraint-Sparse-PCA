from __future__ import annotations

from typing import Any, Dict, Iterable, List


def _group_by_lambda(rows: Iterable[Dict[str, Any]], method: str) -> Dict[float, List[Dict[str, Any]]]:
    buckets: Dict[float, List[Dict[str, Any]]] = {}
    for row in rows:
        if row.get("method") != method:
            continue
        lam = float(row.get("lambda2"))
        buckets.setdefault(lam, []).append(row)
    return buckets


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _select_longest_contiguous(lams: List[float]) -> List[float]:
    if not lams:
        return []
    lams = sorted(lams)
    best = [lams[0]]
    current = [lams[0]]
    for prev, val in zip(lams, lams[1:]):
        if abs(val - prev) < 1e-9 or val > prev:
            current.append(val)
        else:
            if len(current) > len(best):
                best = current
            current = [val]
    if len(current) > len(best):
        best = current
    return best


def select_band(
    rows: Iterable[dict[str, Any]],
    method: str,
    baseline: str,
    f1_metric: str,
    smooth_metric: str,
    f1_tolerance: float,
    smoothness_margin: float,
) -> list[float]:
    method_rows = _group_by_lambda(rows, method)
    base_rows = _group_by_lambda(rows, baseline)
    candidates: List[float] = []
    for lam, mrows in method_rows.items():
        brows = base_rows.get(lam, [])
        if not brows:
            continue
        m_f1 = _mean([float(r[f1_metric]) for r in mrows])
        b_f1 = _mean([float(r[f1_metric]) for r in brows])
        m_s = _mean([float(r[smooth_metric]) for r in mrows])
        b_s = _mean([float(r[smooth_metric]) for r in brows])
        if m_f1 >= b_f1 - f1_tolerance and (b_s - m_s) >= smoothness_margin:
            candidates.append(lam)
    return _select_longest_contiguous(candidates)
