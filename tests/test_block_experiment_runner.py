from __future__ import annotations

import warnings

import numpy as np


def test_multi_component_generator_emits_ground_truth_matrix() -> None:
    from nc_spca.config.schema import DataConfig
    from nc_spca.data.synthetic.generators import generate_synthetic_dataset

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dataset = generate_synthetic_dataset(
            DataConfig(
                name="synthetic_grid",
                n_samples=40,
                n_features=16,
                support_size=4,
                graph_type="grid",
                random_state=13,
                n_components=2,
                support_overlap_mode="disjoint",
            ),
            seed=13,
        )

    assert dataset["V_true"].shape == (16, 2)
    assert len(dataset["true_supports"]) == 2
    assert all(len(support) == 4 for support in dataset["true_supports"])
    assert caught == []


def test_component_support_metrics_are_permutation_invariant() -> None:
    from nc_spca.metrics.support import component_support_metrics

    truth = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    estimated = truth[:, ::-1]

    metrics = component_support_metrics(
        estimated_loadings=estimated,
        true_loadings=truth,
        threshold=1e-8,
    )

    assert metrics["mean_f1"] == 1.0
    assert metrics["matched_pairs"] in ([(0, 1), (1, 0)], [(1, 0), (0, 1)])


def test_block_experiment_runner_reports_multi_component_metrics(tmp_path) -> None:
    from nc_spca.api.factory import build_backend, build_experiment
    from nc_spca.config.schema import (
        BackendConfig,
        DataConfig,
        ExperimentConfig,
        ModelConfig,
        ObjectiveConfig,
        OptimizerConfig,
        TrackingConfig,
    )

    backend = build_backend(BackendConfig(name="numpy"))
    experiment = build_experiment(
        experiment_cfg=ExperimentConfig(name="block_synth", repeats=1, seed=17),
        data_cfg=DataConfig(
            name="synthetic_grid",
            n_samples=48,
            n_features=16,
            support_size=4,
            graph_type="grid",
            random_state=17,
            n_components=2,
            support_overlap_mode="shared",
        ),
        model_cfg=ModelConfig(name="nc_spca_block", n_components=2, random_state=17),
        objective_cfg=ObjectiveConfig(
            name="nc_spca_block",
            lambda1=0.05,
            lambda2=0.1,
            sparsity_mode="l21",
            group_lambda=0.05,
        ),
        optimizer_cfg=OptimizerConfig(name="manpg", max_iter=30, learning_rate="auto"),
        tracking_cfg=TrackingConfig(root_dir=str(tmp_path), project="nc_spca_block_test"),
        backend=backend,
    )

    result = experiment.run()

    assert result.records
    record = result.records[0]
    assert "matched_support_f1" in record
    assert "orthogonality_error" in record
    assert "shared_support_f1" in record
    assert "mean_component_lcc_ratio" in record
    assert "matched_support_f1_mean" in result.summary
    assert "orthogonality_error_mean" in result.summary
