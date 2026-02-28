from __future__ import annotations


def test_factory_builds_baseline_models() -> None:
    from nc_spca.api.factory import build_backend, build_model
    from nc_spca.config.schema import BackendConfig, ModelConfig, ObjectiveConfig, OptimizerConfig

    backend = build_backend(BackendConfig(name="numpy"))

    pca_model = build_model(
        model_cfg=ModelConfig(name="pca", n_components=1),
        objective_cfg=ObjectiveConfig(name="pca"),
        optimizer_cfg=OptimizerConfig(name="pg", max_iter=10),
        backend=backend,
    )
    l1_model = build_model(
        model_cfg=ModelConfig(name="l1_spca", n_components=1),
        objective_cfg=ObjectiveConfig(name="l1_spca", lambda1=0.1),
        optimizer_cfg=OptimizerConfig(name="pg", max_iter=10),
        backend=backend,
    )

    assert pca_model.name == "pca"
    assert l1_model.name == "l1_spca"
