from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "nc_spca_mpl_cache"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import _bootstrap  # noqa: F401

from nc_spca.api.factory import build_backend, build_model
from nc_spca.config.schema import BackendConfig, DataConfig, ModelConfig, ObjectiveConfig, OptimizerConfig
from nc_spca.data import generate_synthetic_dataset
from nc_spca.metrics import component_support_metrics, shared_support_metrics


def _make_dataset(seed: int) -> dict[str, object]:
    cfg = DataConfig(
        name="synthetic_chain",
        graph_type="chain",
        n_samples=60,
        n_features=20,
        support_size=4,
        n_components=2,
        support_overlap_mode="shared",
        random_state=seed,
    )
    return generate_synthetic_dataset(cfg, seed=seed)


def _fit_loadings(dataset: dict[str, object], sparsity_mode: str) -> np.ndarray:
    backend = build_backend(BackendConfig(name="numpy"))
    model = build_model(
        model_cfg=ModelConfig(name="nc_spca_block", n_components=2, random_state=42),
        objective_cfg=ObjectiveConfig(
            name="nc_spca_block",
            lambda1=0.15,
            lambda2=0.25,
            sparsity_mode=sparsity_mode,
            group_lambda=0.15 if sparsity_mode == "l21" else None,
            retraction="polar",
            support_threshold=1e-6,
        ),
        optimizer_cfg=OptimizerConfig(
            name="manpg",
            max_iter=400,
            learning_rate="auto",
            tol=1e-6,
            monotone_backtracking=True,
        ),
        backend=backend,
    )
    fit = model.fit(dataset)
    return np.asarray(fit.params["loadings"], dtype=float)


def _support_mask(loadings: np.ndarray, threshold: float = 1e-6) -> np.ndarray:
    return (np.abs(loadings) > threshold).astype(float)


def generate_figure(output_path: Path, seed: int = 42) -> Path:
    dataset = _make_dataset(seed)
    true_loadings = np.asarray(dataset["V_true"], dtype=float)
    l1_loadings = _fit_loadings(dataset, "l1")
    l21_loadings = _fit_loadings(dataset, "l21")

    panels = [
        ("True", true_loadings),
        ("Block ManPG ($\\ell_1$)", l1_loadings),
        ("Block ManPG ($\\ell_{2,1}$)", l21_loadings),
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12, 4.8),
        gridspec_kw={"height_ratios": [2.2, 1.0]},
        constrained_layout=True,
    )

    vmax = max(float(np.max(np.abs(arr))) for _, arr in panels)
    for col, (title, loadings) in enumerate(panels):
        top_ax = axes[0, col]
        bottom_ax = axes[1, col]
        heat = top_ax.imshow(
            np.abs(loadings.T),
            aspect="auto",
            cmap="YlGnBu",
            vmin=0.0,
            vmax=vmax,
        )
        top_ax.set_title(title)
        top_ax.set_ylabel("Component")
        top_ax.set_yticks([0, 1], labels=["1", "2"])
        top_ax.set_xticks([])

        mask = _support_mask(loadings).T
        bottom_ax.imshow(mask, aspect="auto", cmap="Greys", vmin=0.0, vmax=1.0)
        bottom_ax.set_xlabel("Feature Index")
        bottom_ax.set_yticks([0, 1], labels=["1", "2"])
        bottom_ax.set_xticks([0, 4, 8, 12, 16, 19])
        if col == 0:
            bottom_ax.set_ylabel("Support")

    l1_component = component_support_metrics(l1_loadings, true_loadings)
    l1_shared = shared_support_metrics(l1_loadings, true_loadings)
    l21_component = component_support_metrics(l21_loadings, true_loadings)
    l21_shared = shared_support_metrics(l21_loadings, true_loadings)

    fig.suptitle(
        "Shared-support chain example. "
        f"$\\ell_1$: matched F1={l1_component['mean_f1']:.3f}, shared F1={l1_shared['f1']:.3f}. "
        f"$\\ell_{{2,1}}$: matched F1={l21_component['mean_f1']:.3f}, shared F1={l21_shared['f1']:.3f}.",
        fontsize=11,
    )
    fig.colorbar(heat, ax=axes[0, :], shrink=0.8, location="right", label="|loading|")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a support-pattern figure for the block NC-SPCA manuscript section."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("doc/latex/figures/block_support_patterns.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to generate the fixed synthetic example.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = generate_figure(args.output, seed=args.seed)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
