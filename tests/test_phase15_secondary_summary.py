from __future__ import annotations

import csv
from pathlib import Path

import pytest
from scipy import stats

from scripts import phase15_secondary_summary as summary_script


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_phase15_secondary_summary_filters_high_decoy_and_adjusts_pvalues(tmp_path: Path):
    input_path = tmp_path / "aggregated_phase15.csv"
    output_path = tmp_path / "phase15_secondary_summary.csv"

    proposed_rows = [
        {"support_f1": 0.88, "graph_smoothness_norm": 0.12, "connect_largest_ratio": 0.81, "component_f1_min": 0.70},
        {"support_f1": 0.84, "graph_smoothness_norm": 0.14, "connect_largest_ratio": 0.79, "component_f1_min": 0.72},
        {"support_f1": 0.91, "graph_smoothness_norm": 0.11, "connect_largest_ratio": 0.83, "component_f1_min": 0.69},
        {"support_f1": 0.87, "graph_smoothness_norm": 0.13, "connect_largest_ratio": 0.80, "component_f1_min": 0.71},
    ]
    sparse_support_f1 = [0.845, 0.805, 0.885, 0.84]
    sparse_smoothness = [0.17, 0.18, 0.16, 0.19]
    sparse_connect = [0.79, 0.78, 0.80, 0.77]
    sparse_component = [0.69, 0.71, 0.68, 0.70]
    amanpg_support_f1 = [0.862, 0.824, 0.898, 0.851]
    amanpg_smoothness = [0.15, 0.16, 0.14, 0.17]
    amanpg_connect = [0.80, 0.79, 0.82, 0.79]
    amanpg_component = [0.68, 0.70, 0.67, 0.69]

    rows: list[dict[str, object]] = []
    for idx, proposed in enumerate(proposed_rows):
        common = {
            "experiment": "trackB_phase15",
            "output_dir": f"/runs/high_{idx}",
            "track": "B",
            "phase": 1.5,
            "graph_family": "chain",
            "support_type": "connected_disjoint",
            "decoy_intensity": "high",
            "seed": idx,
            "lambda2": 0.05,
        }
        rows.append({"method": "Proposed", **common, **proposed})
        rows.append(
            {
                "method": "SparseNoGraph",
                **common,
                "support_f1": sparse_support_f1[idx],
                "graph_smoothness_norm": sparse_smoothness[idx],
                "connect_largest_ratio": sparse_connect[idx],
                "component_f1_min": sparse_component[idx],
            }
        )
        rows.append(
            {
                "method": "A-ManPG",
                **common,
                "support_f1": amanpg_support_f1[idx],
                "graph_smoothness_norm": amanpg_smoothness[idx],
                "connect_largest_ratio": amanpg_connect[idx],
                "component_f1_min": amanpg_component[idx],
            }
        )

    rows.append(
        {
            "experiment": "trackB_phase15",
            "output_dir": "/runs/low_0",
            "track": "B",
            "phase": 1.5,
            "graph_family": "chain",
            "support_type": "connected_disjoint",
            "decoy_intensity": "low",
            "seed": 99,
            "lambda2": 0.05,
            "method": "Proposed",
            "support_f1": 0.50,
            "graph_smoothness_norm": 1.50,
            "connect_largest_ratio": 0.20,
            "component_f1_min": 0.10,
        }
    )

    _write_csv(input_path, rows)

    summary_df = summary_script.build_secondary_summary(input_path)
    summary_script.write_secondary_summary(summary_df, output_path)

    out = _read_csv(output_path)
    assert len(out) == 8
    assert {row["comparison"] for row in out} == {"Proposed vs SparseNoGraph", "Proposed vs A-ManPG"}
    assert {row["metric"] for row in out} == {
        "support_f1",
        "graph_smoothness_norm",
        "connect_largest_ratio",
        "component_f1_min",
    }

    support_rows = [
        row
        for row in out
        if row["comparison"] == "Proposed vs SparseNoGraph" and row["metric"] == "support_f1"
    ]
    assert int(support_rows[0]["n"]) == 4
    expected_mean_diff = sum(
        p - b
        for p, b in zip([0.88, 0.84, 0.91, 0.87], sparse_support_f1)
    ) / 4
    assert float(support_rows[0]["mean_diff"]) == pytest.approx(expected_mean_diff)

    expected_t = stats.ttest_rel(
        [0.88, 0.84, 0.91, 0.87],
        sparse_support_f1,
    ).statistic
    assert float(support_rows[0]["t_statistic"]) == pytest.approx(expected_t)

    pvals = [float(row["p_value"]) for row in out]
    assert [float(row["holm_p_value"]) for row in out] == pytest.approx(summary_script.holm_adjust(pvals))
    assert [float(row["bh_p_value"]) for row in out] == pytest.approx(summary_script.bh_adjust(pvals))
