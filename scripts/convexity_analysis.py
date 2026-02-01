"""
Convexity and Network-Constrained SPCA Objectives Analysis.

This script provides visualizations to understand:
1. Quadratic Forms: How eigenvalues determine convexity.
2. SPCA Objectives: Effects of Sparsity (L1) and Network Constraints (Laplacian).

It generates and saves plots to the 'figures' directory.
"""

import logging
import sys
from pathlib import Path
from typing import Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def setup_plotting_style() -> None:
    """Sets up matplotlib plotting style defaults."""
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['figure.figsize'] = [12, 6]
        plt.rcParams['font.size'] = 12
        logger.info("Plotting style set to 'seaborn-v0_8-whitegrid'.")
    except Exception as e:
        logger.warning(f"Could not set plot style: {e}. Falling back to default.")


def compute_quadratic_form_vectorized(
    X: np.ndarray, Y: np.ndarray, A: np.ndarray
) -> np.ndarray:
    """
    Computes the quadratic form Z = v^T A v for a grid of vectors v = [x, y].

    Args:
        X (np.ndarray): X coordinates meshgrid.
        Y (np.ndarray): Y coordinates meshgrid.
        A (np.ndarray): The matrix defining the quadratic form.

    Returns:
        np.ndarray: Z values corresponding to the grid.
    """
    # Stack X and Y to create a grid of vectors of shape (N, M, 2)
    vecs = np.stack([X, Y], axis=-1)
    
    # Compute v^T A v efficiently
    # (vecs @ A.T) results in shape (N, M, 2)
    # Element-wise multiplication with vecs and sum over the last axis
    # is equivalent to the dot product for each vector
    temp = vecs @ A.T
    Z = np.sum(temp * vecs, axis=-1)
    return Z


def plot_quadratic_form(A: np.ndarray, title: str, ax: plt.Axes) -> None:
    """
    Plots the surface z = x.T @ A @ x.

    Args:
        A (np.ndarray): The 2x2 matrix defining the quadratic form.
        title (str): Title for the plot.
        ax (plt.Axes): The 3D axes object to plot on.
    """
    try:
        x = np.linspace(-2, 2, 50)
        y = np.linspace(-2, 2, 50)
        X, Y = np.meshgrid(x, y)

        Z = compute_quadratic_form_vectorized(X, Y, A)

        ax.plot_surface(
            X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True, alpha=0.8
        )
        ax.set_title(title)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('Q(x)')
        logger.debug(f"Plotted quadratic form: {title}")

    except Exception as e:
        logger.error(f"Error plotting quadratic form '{title}': {e}")
        raise


def plot_spca_objective(
    Sigma: np.ndarray,
    lambda1: float,
    lambda2: float,
    title: str,
    ax: plt.Axes,
    L: Optional[np.ndarray] = None,
) -> None:
    """
    Plots contours of the SPCA objective function.
    Objective: w^T Sigma w - lambda1 ||w||_1 - lambda2 w^T L w

    Args:
        Sigma (np.ndarray): Covariance matrix (2x2).
        lambda1 (float): Sparsity regularization coefficient.
        lambda2 (float): Network smoothness regularization coefficient.
        title (str): Title of the plot.
        ax (plt.Axes): Axes to plot on.
        L (Optional[np.ndarray]): Laplacian matrix. Defaults to 2-node connected graph.
    """
    try:
        # 1. Define Grid
        bound = 1.3
        x = np.linspace(-bound, bound, 200)
        y = np.linspace(-bound, bound, 200)
        X, Y = np.meshgrid(x, y)

        # 2. Define Laplacian if not provided
        if L is None:
            L = np.array([[1, -1], [-1, 1]])

        # 3. Compute Objective Z (Vectorized)
        vecs = np.stack([X, Y], axis=-1)  # Shape (N, M, 2)

        # Variance term: w^T Sigma w
        # (N, M, 2) @ (2, 2) -> (N, M, 2)
        variance = np.sum((vecs @ Sigma.T) * vecs, axis=-1)

        # Sparsity term: L1 norm
        sparsity = np.sum(np.abs(vecs), axis=-1)

        # Graph term: w^T L w
        smoothness = np.sum((vecs @ L.T) * vecs, axis=-1)

        # Total Objective (Maximization form)
        Z = variance - lambda1 * sparsity - lambda2 * smoothness

        # Mask values outside the unit ball
        norm_sq = X**2 + Y**2
        Z[norm_sq > 1.0] = np.nan

        # 4. Plot Contours
        ax.contourf(X, Y, Z, 20, cmap='viridis')

        # 5. Draw Constraints (Unit Circle)
        circle = plt.Circle(
            (0, 0), 1, color='red', fill=False, linestyle='--', linewidth=2, label='||w||_2 = 1'
        )
        ax.add_artist(circle)

        # 6. Find and plot the maximum on the grid
        # Only consider valid points inside unit circle
        # Note: In original code, it checked <= 1.001
        valid_mask = norm_sq <= 1.001
        Z_valid = np.where(valid_mask, Z, -np.inf)
        
        # Handle case where all are -inf (shouldn't happen with valid grid)
        if np.all(np.isinf(Z_valid)):
             logger.warning(f"No valid points found for max search in {title}")
        else:
            max_idx = np.unravel_index(np.argmax(Z_valid), Z_valid.shape)
            max_w = (X[max_idx], Y[max_idx])

            ax.scatter(
                max_w[0], max_w[1], c='red', s=100, marker='*', label='Optimum'
            )
            ax.plot([0, max_w[0]], [0, max_w[1]], 'r:', linewidth=1)

        ax.set_title(title)
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_xlabel('w1')
        ax.set_ylabel('w2')
        logger.debug(f"Plotted SPCA objective: {title}")

    except Exception as e:
        logger.error(f"Error in plot_spca_objective '{title}': {e}")
        raise


def plot_spca_3d(
    Sigma: np.ndarray,
    lambda1: float,
    lambda2: float,
    title: str,
    ax: plt.Axes,
    L: Optional[np.ndarray] = None,
) -> None:
    """
    Plots the 3D surface of the SPCA objective over the unit disk.

    Args:
        Sigma (np.ndarray): Covariance matrix.
        lambda1 (float): Sparsity coeff.
        lambda2 (float): Smoothness coeff.
        title (str): Plot title.
        ax (plt.Axes): 3D Axes.
        L (Optional[np.ndarray]): Laplacian matrix.
    """
    try:
        # 1. Define Grid (Unit Ball)
        r = np.linspace(0, 1, 30)
        theta = np.linspace(0, 2 * np.pi, 60)
        R, Theta = np.meshgrid(r, theta)
        X = R * np.cos(Theta)
        Y = R * np.sin(Theta)

        if L is None:
            L = np.array([[1, -1], [-1, 1]])

        # 2. Compute Z (Vectorized)
        vecs = np.stack([X, Y], axis=-1)

        variance = np.sum((vecs @ Sigma.T) * vecs, axis=-1)
        sparsity = np.sum(np.abs(vecs), axis=-1)
        smoothness = np.sum((vecs @ L.T) * vecs, axis=-1)

        Z = variance - lambda1 * sparsity - lambda2 * smoothness

        # 3. Plot
        ax.plot_surface(
            X, Y, Z, cmap='plasma', alpha=0.9, linewidth=0.2, antialiased=True
        )
        ax.set_title(title, pad=20)
        ax.set_xlabel('w1')
        ax.set_ylabel('w2')
        ax.set_zlabel('Obj')
        
        # Remove panes for cleaner look
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

    except Exception as e:
        logger.error(f"Error in plot_spca_3d '{title}': {e}")
        raise


def run_analysis(output_dir: str = "figures") -> None:
    """
    Main function to run the analysis and save plots.

    Args:
        output_dir (str): Directory to save figures.
    """
    setup_plotting_style()
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # --- Part 1: Quadratic Forms ---
    logger.info("Generating Quadratic Forms visualization...")
    try:
        # Define matrices
        A_convex = np.array([[1.5, 0], [0, 1.5]])  # Positive Definite
        A_concave = np.array([[-1.5, 0], [0, -1.5]])  # Negative Definite
        A_saddle = np.array([[1.5, 0], [0, -1.5]])  # Indefinite

        fig1 = plt.figure(figsize=(18, 6))

        ax1 = fig1.add_subplot(131, projection='3d')
        plot_quadratic_form(A_convex, "Positive Definite\n(Convex)", ax1)

        ax2 = fig1.add_subplot(132, projection='3d')
        plot_quadratic_form(A_concave, "Negative Definite\n(Concave)", ax2)

        ax3 = fig1.add_subplot(133, projection='3d')
        plot_quadratic_form(A_saddle, "Indefinite\n(Saddle Point)", ax3)

        plt.tight_layout()
        save_path1 = out_path / "quadratic_forms.png"
        plt.savefig(save_path1)
        logger.info(f"Saved {save_path1}")
        plt.close(fig1)
    except Exception as e:
        logger.error(f"Failed to generate Part 1 plots: {e}")

    # --- Part 2: Network-Constrained SPCA Landscape ---
    logger.info("Generating SPCA Landscape visualization...")
    try:
        Sigma = np.array([[1, -0.8], [-0.8, 1]])

        fig2, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Case 1: Pure PCA
        plot_spca_objective(
            Sigma, lambda1=0, lambda2=0,
            title="1. Standard PCA\n(Max Variance)", ax=axes[0]
        )

        # Case 2: Sparse PCA
        plot_spca_objective(
            Sigma, lambda1=0.8, lambda2=0,
            title=r"2. Sparse PCA\n(Variance - $\lambda_1$ Sparsity)", ax=axes[1]
        )

        # Case 3: Network Constrained PCA
        plot_spca_objective(
            Sigma, lambda1=0, lambda2=1.0,
            title=r"3. Graph PCA\n(Variance - $\lambda_2$ Smoothness)", ax=axes[2]
        )

        plt.tight_layout()
        save_path2 = out_path / "spca_landscape.png"
        plt.savefig(save_path2)
        logger.info(f"Saved {save_path2}")
        plt.close(fig2)

    except Exception as e:
        logger.error(f"Failed to generate Part 2 plots: {e}")

    # --- Part 3: 3D Visualization of SPCA Objectives ---
    logger.info("Generating 3D SPCA visualization...")
    try:
        Sigma = np.array([[1, -0.8], [-0.8, 1]])
        fig3 = plt.figure(figsize=(20, 6))

        # 1. Standard
        ax1 = fig3.add_subplot(141, projection='3d')
        plot_spca_3d(Sigma, 0, 0, "Standard PCA", ax1)

        # 2. Sparse
        ax2 = fig3.add_subplot(142, projection='3d')
        plot_spca_3d(Sigma, 0.8, 0, "Sparse PCA (L1)", ax2)

        # 3. Graph
        ax3 = fig3.add_subplot(143, projection='3d')
        plot_spca_3d(Sigma, 0, 1.0, "Graph PCA (L)", ax3)

        # 4. Combined
        ax4 = fig3.add_subplot(144, projection='3d')
        plot_spca_3d(Sigma, 0.5, 0.5, "Combined Net-SPCA", ax4)

        plt.tight_layout()
        save_path3 = out_path / "spca_3d_landscape.png"
        plt.savefig(save_path3)
        logger.info(f"Saved {save_path3}")
        plt.close(fig3)
    except Exception as e:
        logger.error(f"Failed to generate Part 3 plots: {e}")


if __name__ == "__main__":
    run_analysis()
