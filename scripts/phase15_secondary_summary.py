from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import fmean
from typing import Iterable, Sequence

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "results" / "trackB" / "phase1_5" / "aggregated_phase15.csv"
OUTPUT_PATH = ROOT / "results" / "trackB" / "phase1_5" / "phase15_secondary_summary.csv"
METRICS = [
    "support_f1",
    "graph_smoothness_norm",
    "connect_largest_ratio",
    "component_f1_min",
]
COMPARISONS = [
    ("Proposed", "SparseNoGraph"),
    ("Proposed", "A-ManPG"),
]
PAIR_KEYS = [
    "output_dir",
    "seed",
    "lambda2",
    "track",
    "phase",
    "graph_family",
    "support_type",
    "decoy_intensity",
]
FIELDNAMES = [
    "comparison",
    "proposed_method",
    "baseline_method",
    "metric",
    "n",
    "mean_diff",
    "t_statistic",
    "p_value",
    "holm_p_value",
    "bh_p_value",
    "holm_significant",
    "bh_significant",
]


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "" or value.lower() == "nan":
            return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(num):
        return None
    return num


def _load_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _filtered_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    return [row for row in rows if row.get("decoy_intensity") == "high"]


def _pair_key(row: dict[str, object]) -> tuple[object, ...]:
    return tuple(row.get(key) for key in PAIR_KEYS)


def _pair_rows(rows: Sequence[dict[str, object]], proposed: str, baseline: str) -> list[dict[str, object]]:
    proposed_map: dict[tuple[object, ...], dict[str, object]] = {}
    baseline_map: dict[tuple[object, ...], dict[str, object]] = {}

    for row in rows:
        method = row.get("method")
        key = _pair_key(row)
        if method == proposed:
            proposed_map[key] = row
        elif method == baseline:
            baseline_map[key] = row

    paired_keys = sorted(proposed_map.keys() & baseline_map.keys(), key=repr)
    return [
        {
            "proposed": proposed_map[key],
            "baseline": baseline_map[key],
        }
        for key in paired_keys
    ]


def _paired_values(pairs: Sequence[dict[str, dict[str, object]]], metric: str) -> tuple[list[float], list[float]]:
    proposed_values: list[float] = []
    baseline_values: list[float] = []
    for pair in pairs:
        proposed_value = _to_float(pair["proposed"].get(metric))
        baseline_value = _to_float(pair["baseline"].get(metric))
        if proposed_value is None or baseline_value is None:
            continue
        proposed_values.append(proposed_value)
        baseline_values.append(baseline_value)
    return proposed_values, baseline_values


def _mean_diff(proposed: Sequence[float], baseline: Sequence[float]) -> float:
    return fmean(a - b for a, b in zip(proposed, baseline))


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    indexed = [(idx, p) for idx, p in enumerate(p_values) if p is not None and not math.isnan(p)]
    adjusted = [math.nan] * len(p_values)
    if not indexed:
        return adjusted

    sorted_items = sorted(indexed, key=lambda item: item[1])
    m = len(sorted_items)
    running_max = 0.0
    for rank, (idx, p_value) in enumerate(sorted_items, start=1):
        value = min(1.0, (m - rank + 1) * p_value)
        running_max = max(running_max, value)
        adjusted[idx] = running_max
    return adjusted


def bh_adjust(p_values: Sequence[float]) -> list[float]:
    indexed = [(idx, p) for idx, p in enumerate(p_values) if p is not None and not math.isnan(p)]
    adjusted = [math.nan] * len(p_values)
    if not indexed:
        return adjusted

    sorted_items = sorted(indexed, key=lambda item: item[1])
    m = len(sorted_items)
    running_min = 1.0
    sorted_adjusted: list[tuple[int, float]] = []
    for rank in range(m, 0, -1):
        idx, p_value = sorted_items[rank - 1]
        value = min(1.0, (m / rank) * p_value)
        running_min = min(running_min, value)
        sorted_adjusted.append((idx, running_min))

    for idx, value in sorted_adjusted:
        adjusted[idx] = value
    return adjusted


def build_secondary_summary(input_path: Path | str = INPUT_PATH) -> list[dict[str, object]]:
    input_path = Path(input_path)
    rows = _filtered_rows(_load_rows(input_path))

    summary_rows: list[dict[str, object]] = []
    p_values: list[float] = []

    for proposed, baseline in COMPARISONS:
        pairs = _pair_rows(rows, proposed, baseline)
        for metric in METRICS:
            proposed_values, baseline_values = _paired_values(pairs, metric)
            n = len(proposed_values)
            if n == 0:
                mean_diff = math.nan
                t_statistic = math.nan
                p_value = math.nan
            else:
                mean_diff = _mean_diff(proposed_values, baseline_values)
                test = stats.ttest_rel(proposed_values, baseline_values)
                t_statistic = float(test.statistic)
                p_value = float(test.pvalue)
            summary_rows.append(
                {
                    "comparison": f"{proposed} vs {baseline}",
                    "proposed_method": proposed,
                    "baseline_method": baseline,
                    "metric": metric,
                    "n": n,
                    "mean_diff": mean_diff,
                    "t_statistic": t_statistic,
                    "p_value": p_value,
                }
            )
            p_values.append(p_value)

    holm_values = holm_adjust(p_values)
    bh_values = bh_adjust(p_values)
    for row, holm_value, bh_value in zip(summary_rows, holm_values, bh_values):
        row["holm_p_value"] = holm_value
        row["bh_p_value"] = bh_value
        row["holm_significant"] = bool(not math.isnan(holm_value) and holm_value < 0.05)
        row["bh_significant"] = bool(not math.isnan(bh_value) and bh_value < 0.05)
    return summary_rows


def write_secondary_summary(rows: Sequence[dict[str, object]], output_path: Path | str = OUTPUT_PATH) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main() -> None:
    rows = build_secondary_summary(INPUT_PATH)
    write_secondary_summary(rows, OUTPUT_PATH)


if __name__ == "__main__":
    main()
