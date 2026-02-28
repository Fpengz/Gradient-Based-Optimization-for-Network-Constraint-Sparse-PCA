from __future__ import annotations

import numpy as np
from pathlib import Path


def test_filesystem_tracker_checkpoint_round_trip(tmp_path: Path) -> None:
    from nc_spca.config.schema import TrackingConfig
    from nc_spca.tracking.filesystem import FilesystemTracker

    tracker = FilesystemTracker(
        TrackingConfig(root_dir=str(tmp_path), project="checkpoint_test")
    )
    tracker.start_run("smoke")
    checkpoint_path = tracker.save_checkpoint(
        name="iter_0001",
        model_state={"weight": np.array([1.0, -2.0, 0.5])},
        optimizer_state={"iteration": np.array([1.0]), "step_size": np.array([0.1])},
        metadata={"backend": "numpy", "objective": 1.25},
        checkpoint_group="latest",
    )

    payload = tracker.load_checkpoint(checkpoint_path)

    np.testing.assert_allclose(payload["model_state"]["weight"], np.array([1.0, -2.0, 0.5]))
    np.testing.assert_allclose(payload["optimizer_state"]["step_size"], np.array([0.1]))
    assert payload["metadata"]["backend"] == "numpy"
