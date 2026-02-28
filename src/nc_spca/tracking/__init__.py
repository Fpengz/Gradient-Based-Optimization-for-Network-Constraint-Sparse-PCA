"""Tracking and checkpointing backends."""

from ..config.schema import TrackingConfig
from .checkpoint import NumpyCheckpointIO
from .composite import CompositeTracker
from .filesystem import FilesystemTracker
from .wandb import WandBTracker


def build_tracker(config: TrackingConfig):
    primary = FilesystemTracker(config)
    if not config.enable_wandb:
        return primary
    return CompositeTracker(primary=primary, mirrors=[WandBTracker(config)])


__all__ = [
    "build_tracker",
    "CompositeTracker",
    "FilesystemTracker",
    "NumpyCheckpointIO",
    "WandBTracker",
]
