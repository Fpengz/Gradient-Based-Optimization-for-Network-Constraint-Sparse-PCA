"""Optional WandB tracker adapter."""

from __future__ import annotations

import importlib
from dataclasses import dataclass

from ..config.schema import TrackingConfig


@dataclass
class WandBTracker:
    """Thin optional WandB adapter."""

    config: TrackingConfig

    def __post_init__(self) -> None:
        self.run = None

    def start_run(self, experiment_name: str):
        try:
            wandb = importlib.import_module("wandb")
        except ImportError as exc:  # pragma: no cover - optional dependency.
            raise RuntimeError(
                "WandB tracking requested but wandb is not installed."
            ) from exc
        self.run = wandb.init(
            project=self.config.wandb_project or self.config.project,
            entity=self.config.wandb_entity,
            name=experiment_name,
        )
        return self.run

    def log_config(self, payload: dict) -> None:
        if self.run is not None:
            self.run.config.update(payload, allow_val_change=True)

    def log_metric(
        self,
        name: str,
        value: float,
        step: int,
        context: dict | None = None,
    ) -> None:
        if self.run is not None:
            record = {name: value, "_step": step}
            if context:
                record.update({f"context/{key}": val for key, val in context.items()})
            self.run.log(record)

    def log_event(self, name: str, payload: dict) -> None:
        if self.run is not None:
            event_payload = {f"event/{name}/{key}": value for key, value in payload.items()}
            self.run.log(event_payload)

    def finalize(self, summary: dict) -> None:
        if self.run is not None:
            self.run.summary.update(summary)
            self.run.finish()
