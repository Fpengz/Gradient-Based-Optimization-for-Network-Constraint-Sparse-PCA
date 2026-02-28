"""Base tracker types."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class TrackerProtocol(Protocol):
    """Minimal tracking interface."""

    run_dir: Path | None

    def start_run(self, experiment_name: str) -> Path:
        """Start a new run."""

    def log_config(self, payload: dict) -> None:
        """Persist the resolved configuration."""

    def log_metric(self, name: str, value: float, step: int, context: dict | None = None) -> None:
        """Append a metric event."""

    def log_event(self, name: str, payload: dict) -> None:
        """Append an event record."""

    def log_artifact(self, source: Path, relative_path: str | None = None) -> Path:
        """Copy an artifact into the run directory."""

    def save_checkpoint(
        self,
        name: str,
        model_state: dict,
        optimizer_state: dict,
        metadata: dict,
        checkpoint_group: str = "latest",
    ) -> Path:
        """Persist a checkpoint bundle."""

    def load_checkpoint(self, checkpoint_path: Path) -> dict:
        """Load a persisted checkpoint bundle."""

    def finalize(self, summary: dict) -> None:
        """Finalize the run and persist the summary."""
