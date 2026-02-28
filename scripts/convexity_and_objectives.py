from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def _configure_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.figsize"] = [12, 6]
    plt.rcParams["font.size"] = 12


def _plot_quadratic_form(ax: plt.Axes, matrix: np.ndarray, title: str) -> None:
    x = np.linspace(-2.0, 2.0, 50)
    y = np.linspace(-2.0, 2.0, 50)
    grid_x, grid_y = np.meshgrid(x, y)
    values = np.zeros_like(grid_x)

    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            vector = np.array([grid_x[i, j], grid_y[i, j]])
            values[i, j] = vector.T @ matrix @ vector

    ax.plot_surface(
        grid_x,
        grid_y,
        values,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=True,
        alpha=0.8,
    )
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("Q(x)")


def _spca_objective(
    weight: np.ndarray, covariance: np.ndarray, lambda1: float, lambda2: float
) -> float:
    laplacian = np.array([[1.0, -1.0], [-1.0, 1.0]])
    variance = weight.T @ covariance @ weight
    sparsity = np.sum(np.abs(weight))
    smoothness = weight.T @ laplacian @ weight
    return float(variance - lambda1 * sparsity - lambda2 * smoothness)


def _plot_spca_contours(
    ax: plt.Axes, covariance: np.ndarray, lambda1: float, lambda2: float, title: str
) -> None:
    bound = 1.3
    x = np.linspace(-bound, bound, 200)
    y = np.linspace(-bound, bound, 200)
    grid_x, grid_y = np.meshgrid(x, y)
    values = np.zeros_like(grid_x)

    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            weight = np.array([grid_x[i, j], grid_y[i, j]])
            values[i, j] = _spca_objective(weight, covariance, lambda1, lambda2)
            if np.linalg.norm(weight) > 1.0:
                values[i, j] = np.nan

    ax.contourf(grid_x, grid_y, values, 20, cmap="viridis")
    circle = plt.Circle(
        (0.0, 0.0),
        1.0,
        color="red",
        fill=False,
        linestyle="--",
        linewidth=2,
        label="||w||_2 = 1",
    )
    ax.add_artist(circle)

    valid_mask = (grid_x**2 + grid_y**2) <= 1.001
    valid_values = np.where(valid_mask, values, -np.inf)
    max_idx = np.unravel_index(np.argmax(valid_values), valid_values.shape)
    max_weight = (grid_x[max_idx], grid_y[max_idx])

    ax.scatter(max_weight[0], max_weight[1], c="red", s=100, marker="*")
    ax.plot([0.0, max_weight[0]], [0.0, max_weight[1]], "r:", linewidth=1)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlabel("w1")
    ax.set_ylabel("w2")


def _plot_spca_surface(
    ax: plt.Axes, covariance: np.ndarray, lambda1: float, lambda2: float, title: str
) -> None:
    radius = np.linspace(0.0, 1.0, 30)
    theta = np.linspace(0.0, 2.0 * np.pi, 60)
    grid_r, grid_theta = np.meshgrid(radius, theta)
    grid_x = grid_r * np.cos(grid_theta)
    grid_y = grid_r * np.sin(grid_theta)
    values = np.zeros_like(grid_x)

    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            weight = np.array([grid_x[i, j], grid_y[i, j]])
            values[i, j] = _spca_objective(weight, covariance, lambda1, lambda2)

    ax.plot_surface(
        grid_x,
        grid_y,
        values,
        cmap="plasma",
        alpha=0.9,
        linewidth=0.2,
        antialiased=True,
    )
    ax.set_title(title, pad=20)
    ax.set_xlabel("w1")
    ax.set_ylabel("w2")
    ax.set_zlabel("Obj")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False


def generate_figures(output_dir: Path) -> list[Path]:
    _configure_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    convex = np.array([[1.5, 0.0], [0.0, 1.5]])
    concave = np.array([[-1.5, 0.0], [0.0, -1.5]])
    saddle = np.array([[1.5, 0.0], [0.0, -1.5]])

    figure = plt.figure(figsize=(18, 6))
    _plot_quadratic_form(figure.add_subplot(131, projection="3d"), convex, "Positive Definite\n(Convex)")
    _plot_quadratic_form(figure.add_subplot(132, projection="3d"), concave, "Negative Definite\n(Concave)")
    _plot_quadratic_form(figure.add_subplot(133, projection="3d"), saddle, "Indefinite\n(Saddle Point)")
    figure.tight_layout()
    path = output_dir / "quadratic_forms.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    written.append(path)

    covariance = np.array([[1.0, -0.8], [-0.8, 1.0]])

    figure, axes = plt.subplots(1, 3, figsize=(18, 6))
    _plot_spca_contours(axes[0], covariance, 0.0, 0.0, "1. Standard PCA\n(Max Variance)")
    _plot_spca_contours(axes[1], covariance, 0.8, 0.0, "2. Sparse PCA\n(Variance - lambda1 Sparsity)")
    _plot_spca_contours(axes[2], covariance, 0.0, 1.0, "3. Graph PCA\n(Variance - lambda2 Smoothness)")
    figure.tight_layout()
    path = output_dir / "spca_objective_contours.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    written.append(path)

    figure = plt.figure(figsize=(20, 6))
    _plot_spca_surface(figure.add_subplot(141, projection="3d"), covariance, 0.0, 0.0, "Standard PCA")
    _plot_spca_surface(figure.add_subplot(142, projection="3d"), covariance, 0.8, 0.0, "Sparse PCA (L1)")
    _plot_spca_surface(figure.add_subplot(143, projection="3d"), covariance, 0.0, 1.0, "Graph PCA (L)")
    _plot_spca_surface(figure.add_subplot(144, projection="3d"), covariance, 0.5, 0.5, "Combined Net-SPCA")
    figure.tight_layout()
    path = output_dir / "spca_objective_surfaces.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    written.append(path)

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate convexity and NC-SPCA objective visualizations."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures") / "convexity_and_objectives",
        help="Directory where the generated figures will be written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    written = generate_figures(args.output_dir)
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
