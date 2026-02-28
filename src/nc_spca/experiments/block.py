"""Multi-component synthetic experiment runner."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, cast

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
from ..metrics import (
    component_support_metrics,
    connected_support_lcc_ratio,
    explained_variance,
    laplacian_energy,
    shared_support_metrics,
)
from ..tracking import build_tracker
from .base import ExperimentResult


@dataclass
class BlockSyntheticExperiment:
    """Repeated synthetic experiment for block NC-SPCA models."""

    experiment_cfg: ExperimentConfig
    data_cfg: DataConfig
    model_cfg: ModelConfig
    objective_cfg: ObjectiveConfig
    optimizer_cfg: OptimizerConfig
    tracking_cfg: TrackingConfig
    backend: Any
    model_builder: Callable[..., Any]

    @staticmethod
    def _method_label(model: Any) -> str:
        optimizer_name = getattr(getattr(model, "optimizer", None), "name", None)
        return f"{model.name}:{optimizer_name}" if optimizer_name else str(model.name)

    def _summary_from_records(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        if not records:
            return {}
        frame = pd.DataFrame(records)
        return {
            "n_runs": int(len(records)),
            "matched_support_f1_mean": float(frame["matched_support_f1"].mean()),
            "shared_support_f1_mean": float(frame["shared_support_f1"].mean()),
            "orthogonality_error_mean": float(frame["orthogonality_error"].mean()),
            "mean_component_lcc_ratio_mean": float(frame["mean_component_lcc_ratio"].mean()),
            "runtime_sec_mean": float(frame["runtime_sec"].mean()),
            "converged_rate": float(frame["converged"].mean()),
        }

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
            loadings = np.asarray(fit.params["loadings"], dtype=float)
            components = np.asarray(fit.components, dtype=float)
            component_metrics = component_support_metrics(
                estimated_loadings=loadings,
                true_loadings=np.asarray(dataset["V_true"], dtype=float),
                threshold=self.objective_cfg.support_threshold,
            )
            shared_metrics = shared_support_metrics(
                estimated_loadings=loadings,
                true_loadings=np.asarray(dataset["V_true"], dtype=float),
                threshold=self.objective_cfg.support_threshold,
            )
            matched_support_f1 = float(cast(float, component_metrics["mean_f1"]))
            matched_support_precision = float(cast(float, component_metrics["mean_precision"]))
            matched_support_recall = float(cast(float, component_metrics["mean_recall"]))
            shared_support_f1 = float(shared_metrics["f1"])
            shared_support_precision = float(shared_metrics["precision"])
            shared_support_recall = float(shared_metrics["recall"])
            lcc_values = [
                connected_support_lcc_ratio(
                    loadings[:, component_idx],
                    dataset["graph"].adjacency,
                    threshold=self.objective_cfg.support_threshold,
                )
                for component_idx in range(loadings.shape[1])
            ]
            record = {
                "repeat": repeat,
                "seed": seed,
                "method": self._method_label(model),
                "objective": float(fit.objective),
                "converged": bool(fit.converged),
                "n_iter": int(fit.n_iter),
                "runtime_sec": float(elapsed),
                "matched_support_f1": matched_support_f1,
                "matched_support_precision": matched_support_precision,
                "matched_support_recall": matched_support_recall,
                "shared_support_f1": shared_support_f1,
                "shared_support_precision": shared_support_precision,
                "shared_support_recall": shared_support_recall,
                "orthogonality_error": float(
                    np.linalg.norm(components @ components.T - np.eye(components.shape[0]), ord="fro")
                ),
                "mean_component_lcc_ratio": float(np.mean(lcc_values)),
                "explained_variance_total": float(
                    sum(explained_variance(dataset["X"], component) for component in components)
                ),
                "laplacian_energy_total": float(
                    sum(laplacian_energy(loadings[:, idx], dataset["graph"].laplacian) for idx in range(loadings.shape[1]))
                ),
            }
            records.append(record)
            tracker.log_metric(
                "matched_support_f1",
                record["matched_support_f1"],
                step=repeat,
                context={"repeat": repeat},
            )
            tracker.log_metric(
                "orthogonality_error",
                record["orthogonality_error"],
                step=repeat,
                context={"repeat": repeat},
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
