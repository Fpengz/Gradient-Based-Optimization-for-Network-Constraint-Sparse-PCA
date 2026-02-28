"""Base experiment types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ExperimentResult:
    """Standard experiment output."""

    records: list[dict[str, Any]]
    summary: dict[str, Any]
    artifact_paths: dict[str, str] = field(default_factory=dict)
