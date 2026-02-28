"""Experiment runner implementations."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from ..config.schema import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    ObjectiveConfig,
    OptimizerConfig,
    TrackingConfig,
)
from ..data import generate_synthetic_dataset
from ..metrics import connected_support_lcc_ratio, explained_variance, laplacian_energy, support_metrics
from ..tracking import build_tracker
from .base import ExperimentResult


@dataclass
class SyntheticExperiment:
    """Single-model repeated synthetic experiment."""

    experiment_cfg: ExperimentConfig
    data_cfg: DataConfig
    model_cfg: ModelConfig
    objective_cfg: ObjectiveConfig
    optimizer_cfg: OptimizerConfig
    tracking_cfg: TrackingConfig
    backend: Any
    model_builder: Callable[..., Any]

    def _summary_from_records(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        if not records:
            return {}
        frame = pd.DataFrame(records)
        summary = {
            "n_runs": int(len(records)),
            "f1_mean": float(frame["f1"].mean()),
            "explained_variance_mean": float(frame["explained_variance"].mean()),
            "lcc_ratio_mean": float(frame["lcc_ratio"].mean()),
            "runtime_sec_mean": float(frame["runtime_sec"].mean()),
            "converged_rate": float(frame["converged"].mean()),
        }
        return summary

    @staticmethod
    def _method_label(model: Any) -> str:
        optimizer_name = getattr(getattr(model, "optimizer", None), "name", None)
        return f"{model.name}:{optimizer_name}" if optimizer_name else str(model.name)

    def run(self) -> ExperimentResult:
        tracker = build_tracker(self.tracking_cfg)
        run_dir = tracker.start_run(self.experiment_cfg.name)
        tracker.log_config(
            {
                "experiment": asdict(self.experiment_cfg),
                "data": asdict(self.data_cfg),
                "model": asdict(self.model_cfg),
                "objective": asdict(self.objective_cfg),
                "optimizer": asdict(self.optimizer_cfg),
                "backend": self.backend.name,
            }
        )

        records: list[dict[str, Any]] = []
        seed_manifest: list[dict[str, int]] = []
        for repeat in range(self.experiment_cfg.repeats):
            seed = self.experiment_cfg.seed + repeat
            seed_manifest.append({"repeat": repeat, "seed": seed})
            dataset = generate_synthetic_dataset(self.data_cfg, seed=seed)
            model = self.model_builder(
                model_cfg=self.model_cfg,
                objective_cfg=self.objective_cfg,
                optimizer_cfg=self.optimizer_cfg,
                backend=self.backend,
            )
            started = perf_counter()
            fit = model.fit(dataset, tracker=None)
            elapsed = perf_counter() - started
            component = np.asarray(fit.components[0], dtype=float)
            est_support = np.flatnonzero(np.abs(component) > self.objective_cfg.support_threshold)
            support_scores = support_metrics(est_support, dataset["true_support"])
            record = {
                "repeat": repeat,
                "seed": seed,
                "method": self._method_label(model),
                "objective": float(fit.objective),
                "converged": bool(fit.converged),
                "n_iter": int(fit.n_iter),
                "runtime_sec": float(elapsed),
                "explained_variance": float(explained_variance(dataset["X"], component)),
                "f1": float(support_scores["f1"]),
                "precision": float(support_scores["precision"]),
                "recall": float(support_scores["recall"]),
                "lcc_ratio": float(
                    connected_support_lcc_ratio(component, dataset["graph"].adjacency)
                ),
                "laplacian_energy": float(
                    laplacian_energy(component, dataset["graph"].laplacian)
                ),
            }
            records.append(record)
            tracker.log_metric("f1", record["f1"], step=repeat, context={"repeat": repeat})
            tracker.log_metric(
                "explained_variance",
                record["explained_variance"],
                step=repeat,
                context={"repeat": repeat},
            )
            tracker.log_event(
                "repeat_finished",
                {
                    "repeat": repeat,
                    "seed": seed,
                    "objective": record["objective"],
                    "runtime_sec": record["runtime_sec"],
                },
            )
            tracker.save_checkpoint(
                name=f"repeat_{repeat:04d}",
                model_state=fit.params,
                optimizer_state={
                    "n_iter": np.asarray([fit.n_iter], dtype=float),
                    "objective": np.asarray([fit.objective], dtype=float),
                },
                metadata={"repeat": repeat, "seed": seed, "converged": fit.converged},
                checkpoint_group="latest",
            )

        summary = self._summary_from_records(records)
        tracker.finalize(summary)
        records_path = run_dir / "artifacts" / "records.json"
        summary_path = run_dir / "summary.json"
        seed_manifest_path = run_dir / "seed_manifest.json"
        records_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        seed_manifest_path.write_text(json.dumps(seed_manifest, indent=2), encoding="utf-8")
        pd.DataFrame(records).to_csv(run_dir / "artifacts" / "records.csv", index=False)

        return ExperimentResult(
            records=records,
            summary=summary,
            artifact_paths={
                "run_dir": str(run_dir),
                "summary": str(summary_path),
                "records": str(records_path),
                "seed_manifest": str(seed_manifest_path),
            },
        )
