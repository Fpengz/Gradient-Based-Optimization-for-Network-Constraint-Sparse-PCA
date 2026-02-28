from __future__ import annotations

import numpy as np


def test_factory_builds_block_model_and_fits_two_components() -> None:
    from nc_spca.api.factory import build_backend, build_model
    from nc_spca.config.schema import BackendConfig, ModelConfig, ObjectiveConfig, OptimizerConfig
    from nc_spca.data.synthetic.generators import generate_synthetic_dataset
    from nc_spca.config.schema import DataConfig

    backend = build_backend(BackendConfig(name="numpy"))
    model = build_model(
        model_cfg=ModelConfig(name="nc_spca_block", n_components=2, random_state=4),
        objective_cfg=ObjectiveConfig(
            name="nc_spca_block",
            lambda1=0.05,
            lambda2=0.1,
            sparsity_mode="l21",
            group_lambda=0.05,
        ),
        optimizer_cfg=OptimizerConfig(name="manpg", max_iter=20, learning_rate="auto"),
        backend=backend,
    )
    dataset = generate_synthetic_dataset(
        DataConfig(
            name="synthetic_grid",
            n_samples=36,
            n_features=16,
            support_size=4,
            graph_type="grid",
            random_state=4,
        ),
        seed=4,
    )

    fit = model.fit(dataset)

    assert fit.components.shape == (2, 16)
    gram = fit.components @ fit.components.T
    np.testing.assert_allclose(gram, np.eye(2), atol=1e-4)

