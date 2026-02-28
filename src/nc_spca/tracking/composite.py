"""Composite tracking utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import TrackerProtocol


@dataclass
class CompositeTracker:
    """Filesystem-first tracker that optionally mirrors to remote trackers."""

    primary: TrackerProtocol
    mirrors: list[Any]

    @property
    def run_dir(self) -> Path | None:
        return self.primary.run_dir

    def start_run(self, experiment_name: str) -> Path:
        run_dir = self.primary.start_run(experiment_name)
        for tracker in self.mirrors:
            tracker.start_run(experiment_name)
        return run_dir

    def log_config(self, payload: dict) -> None:
        self.primary.log_config(payload)
        for tracker in self.mirrors:
            if hasattr(tracker, "log_config"):
                tracker.log_config(payload)

    def log_metric(self, name: str, value: float, step: int, context: dict | None = None) -> None:
        self.primary.log_metric(name, value, step, context)
        for tracker in self.mirrors:
            if hasattr(tracker, "log_metric"):
                tracker.log_metric(name, value, step, context)

    def log_event(self, name: str, payload: dict) -> None:
        self.primary.log_event(name, payload)
        for tracker in self.mirrors:
            if hasattr(tracker, "log_event"):
                tracker.log_event(name, payload)

    def log_artifact(self, source: Path, relative_path: str | None = None) -> Path:
        path = self.primary.log_artifact(source, relative_path)
        for tracker in self.mirrors:
            if hasattr(tracker, "log_artifact"):
                tracker.log_artifact(source, relative_path)
        return path

    def save_checkpoint(
        self,
        name: str,
        model_state: dict,
        optimizer_state: dict,
        metadata: dict,
        checkpoint_group: str = "latest",
    ) -> Path:
        return self.primary.save_checkpoint(
            name=name,
            model_state=model_state,
            optimizer_state=optimizer_state,
            metadata=metadata,
            checkpoint_group=checkpoint_group,
        )

    def load_checkpoint(self, checkpoint_path: Path) -> dict:
        return self.primary.load_checkpoint(checkpoint_path)

    def finalize(self, summary: dict) -> None:
        self.primary.finalize(summary)
        for tracker in self.mirrors:
            if hasattr(tracker, "finalize"):
                tracker.finalize(summary)
