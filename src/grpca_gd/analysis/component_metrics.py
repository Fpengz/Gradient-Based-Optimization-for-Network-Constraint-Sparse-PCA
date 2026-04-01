from __future__ import annotations

from typing import Dict

import numpy as np


def component_f1_summary(per_component: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    f1_values = [float(v.get("f1", 0.0)) for v in per_component.values()]
    if not f1_values:
        return {"component_f1_min": 0.0, "component_f1_median": 0.0, "component_f1_mean": 0.0}
    values = np.array(f1_values, dtype=float)
    return {
        "component_f1_min": float(values.min()),
        "component_f1_median": float(np.median(values)),
        "component_f1_mean": float(values.mean()),
    }
