import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .base import EstimatorStateMixin


class GeneralizedPowerMethod(BaseEstimator, TransformerMixin, EstimatorStateMixin):
    """
    Generalized Power Method for Sparse PCA.
    As proposed by Journée, Nesterov, Richtárik, and Sepulchre (2010).

    Optimizes: max || soft(X^T * z, gamma) ||_2 s.t. ||z||_2 <= 1
    where soft(u, gamma) = sgn(u) * max(|u| - gamma, 0)

    Parameters
    ----------
    n_components : int, default=2
        Number of sparse principal components.
    gamma : float, default=0.1
        Sparsity parameter (threshold).
    max_iter : int, default=100
    tol : float, default=1e-5
    verbose : bool, default=False
    """

    def __init__(
        self,
        n_components=2,
        gamma=0.1,
        max_iter=100,
        tol=1e-5,
        verbose=False,
    ):
        self.n_components = n_components
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def _soft_threshold(self, x, threshold):
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

    @staticmethod
    def _safe_normalize(x: np.ndarray) -> np.ndarray:
        if not np.all(np.isfinite(x)):
            return np.zeros_like(x)
        norm_x = np.linalg.norm(x)
        if norm_x < 1e-12 or not np.isfinite(norm_x):
            return np.zeros_like(x)
        return x / norm_x

    def fit(self, X, y=None):
        self._init_fit_state()
        X = np.asanyarray(X)
        n_samples, n_features = X.shape
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # X is (n, p)
        # Loadings B (p, d)
        # Scores Z (n, d)

        self.components_ = np.zeros((self.n_components, n_features))
        self.n_components_ = self.n_components
        X_current = X_centered.copy()
        component_iters = []
        component_converged = []

        for k in range(self.n_components):
            # Initialize z (score vector)
            z = np.random.randn(n_samples)
            z = self._safe_normalize(z)

            for i in range(self.max_iter):
                if not np.all(np.isfinite(z)):
                    break
                z_old = z.copy()

                # 1. Update x (loading vector)
                # x = soft(X^T * z, gamma)
                # Note: The paper uses a slightly different normalization.
                # In their single-unit algorithm (Section 3.1):
                # x = soft(X^T * z, gamma)
                # z = X * x / ||X * x||_2
                u = X_current.T @ z
                x = self._soft_threshold(u, self.gamma)

                x = self._safe_normalize(x)
                if np.linalg.norm(x) < 1e-12:
                    # gamma too high, everything zero
                    break

                # 2. Update z (score vector)
                z_next = X_current @ x
                z = self._safe_normalize(z_next)
                if np.linalg.norm(z) < 1e-12:
                    break

                if np.linalg.norm(z - z_old) < self.tol:
                    component_converged.append(True)
                    component_iters.append(i + 1)
                    break
            else:
                component_converged.append(False)
                component_iters.append(self.max_iter)

            # The component is x
            # Re-calculate x to ensure sparsity if we broke early
            u = X_current.T @ z
            x_final = self._soft_threshold(u, self.gamma)
            x_final = self._safe_normalize(x_final)

            self.components_[k, :] = x_final
            self._push_history(
                "component_nnz", int(np.count_nonzero(np.abs(x_final) > 1e-12))
            )

            # Deflation
            # scores = X * x
            scores = X_current @ x_final
            if np.all(np.isfinite(scores)):
                X_current = X_current - np.outer(scores, x_final)
        self.converged_ = (
            bool(all(component_converged)) if component_converged else False
        )
        self.n_iter_ = component_iters
        self.objective_ = None
        return self

    def transform(self, X):
        X_centered = X - self.mean_
        return X_centered @ self.components_.T
