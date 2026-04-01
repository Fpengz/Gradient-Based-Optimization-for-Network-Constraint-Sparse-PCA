import pytest

from grpca_gd.analysis.component_metrics import component_f1_summary


def test_component_f1_summary():
    per_component = {"0": {"f1": 0.8}, "1": {"f1": 0.6}, "2": {"f1": 0.9}}
    summary = component_f1_summary(per_component)
    assert summary["component_f1_min"] == 0.6
    assert summary["component_f1_median"] == 0.8
    assert summary["component_f1_mean"] == pytest.approx(0.7666666666666667)
