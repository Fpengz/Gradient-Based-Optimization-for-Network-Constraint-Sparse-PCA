from __future__ import annotations


def test_app_config_from_mapping_builds_typed_sections() -> None:
    from nc_spca.config.loader import app_config_from_mapping

    config = app_config_from_mapping(
        {
            "backend": {"name": "numpy"},
            "data": {"name": "synthetic_chain", "n_samples": 20, "n_features": 9},
            "objective": {"name": "nc_spca_single", "lambda1": 0.1, "lambda2": 0.2},
            "optimizer": {"name": "pg", "max_iter": 12},
            "model": {"name": "nc_spca_single", "n_components": 1},
            "experiment": {"name": "paper_core", "repeats": 2, "seed": 9},
            "tracking": {"root_dir": "outputs", "project": "nc_spca"},
        }
    )

    assert config.backend.name == "numpy"
    assert config.data.n_samples == 20
    assert config.objective.lambda2 == 0.2
    assert config.optimizer.max_iter == 12
    assert config.experiment.repeats == 2


def test_app_config_from_mapping_preserves_block_manifold_fields() -> None:
    from nc_spca.config.loader import app_config_from_mapping

    config = app_config_from_mapping(
        {
            "backend": {"name": "torch", "dtype": "float64"},
            "data": {
                "name": "synthetic_grid",
                "n_samples": 48,
                "n_features": 16,
                "support_size": 4,
                "n_components": 2,
                "support_overlap_mode": "shared",
            },
            "objective": {
                "name": "nc_spca_block",
                "lambda1": 0.05,
                "lambda2": 0.1,
                "sparsity_mode": "l21",
                "group_lambda": 0.05,
                "retraction": "qr",
            },
            "optimizer": {
                "name": "manpg",
                "max_iter": 30,
                "learning_rate": "auto",
                "grad_norm_tol": 1e-5,
            },
            "model": {"name": "nc_spca_block", "n_components": 2},
            "experiment": {"name": "block_synth_core", "repeats": 2, "seed": 3},
            "tracking": {"root_dir": "outputs", "project": "nc_spca"},
        }
    )

    assert config.data.n_components == 2
    assert config.data.support_overlap_mode == "shared"
    assert config.objective.retraction == "qr"
    assert config.optimizer.grad_norm_tol == 1e-5
