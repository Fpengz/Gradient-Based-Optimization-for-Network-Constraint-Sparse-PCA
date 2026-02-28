"""Filesystem-first experiment tracker."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from subprocess import run

from ..config.schema import TrackingConfig
from .checkpoint import NumpyCheckpointIO


@dataclass
class FilesystemTracker:
    """Structured local tracker.

    The filesystem is the source of truth for every run. Optional remote
    trackers are expected to mirror this state, not replace it.
    """

    config: TrackingConfig

    def __post_init__(self) -> None:
        self.root_dir = Path(self.config.root_dir)
        self.run_dir: Path | None = None
        self.checkpoint_io = NumpyCheckpointIO()

    def start_run(self, experiment_name: str) -> Path:
        base_dir = self.root_dir / self.config.project / experiment_name
        base_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        run_id = f"{stamp}_{experiment_name}"
        run_dir = base_dir / run_id
        suffix = 1
        while run_dir.exists():
            run_dir = base_dir / f"{run_id}_{suffix:02d}"
            suffix += 1
        for subdir in (
            "logs",
            "checkpoints/latest",
            "checkpoints/best",
            "artifacts",
            "artifacts/reports",
        ):
            (run_dir / subdir).mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir
        self._write_environment_manifest(run_dir)
        return run_dir

    def _write_environment_manifest(self, run_dir: Path) -> None:
        env_payload = {
            "cwd": str(Path.cwd()),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        }
        (run_dir / "env.json").write_text(
            json.dumps(env_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        git_commit = self._git_commit()
        if git_commit is not None:
            (run_dir / "git_commit.txt").write_text(f"{git_commit}\n", encoding="utf-8")

    def _git_commit(self) -> str | None:
        capture = run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if capture.returncode != 0:
            return None
        return capture.stdout.strip() or None

    def _ensure_run_dir(self) -> Path:
        if self.run_dir is None:
            raise RuntimeError("Tracker run has not been started.")
        return self.run_dir

    def log_config(self, payload: dict) -> None:
        run_dir = self._ensure_run_dir()
        (run_dir / "resolved_config.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def log_metric(self, name: str, value: float, step: int, context: dict | None = None) -> None:
        run_dir = self._ensure_run_dir()
        record = {"name": name, "value": value, "step": step, "context": context or {}}
        with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def log_event(self, name: str, payload: dict) -> None:
        run_dir = self._ensure_run_dir()
        record = {"name": name, "payload": payload}
        with (run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def log_artifact(self, source: Path, relative_path: str | None = None) -> Path:
        run_dir = self._ensure_run_dir()
        destination = (
            run_dir / "artifacts" / relative_path
            if relative_path is not None
            else run_dir / "artifacts" / source.name
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return destination

    def save_checkpoint(
        self,
        name: str,
        model_state: dict,
        optimizer_state: dict,
        metadata: dict,
        checkpoint_group: str = "latest",
    ) -> Path:
        run_dir = self._ensure_run_dir()
        checkpoint_dir = run_dir / "checkpoints" / checkpoint_group / name
        return self.checkpoint_io.save(checkpoint_dir, model_state, optimizer_state, metadata)

    def load_checkpoint(self, checkpoint_path: Path) -> dict:
        return self.checkpoint_io.load(checkpoint_path)

    def finalize(self, summary: dict) -> None:
        run_dir = self._ensure_run_dir()
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
