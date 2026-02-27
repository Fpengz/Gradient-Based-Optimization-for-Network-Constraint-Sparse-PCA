import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet

from .base import EstimatorStateMixin


class ZouSparsePCA(BaseEstimator, TransformerMixin, EstimatorStateMixin):
    """
    Sparse Principal Component Analysis (SPCA) using the regression formulation.
    As proposed by Zou, Hastie, and Tibshirani (2006).

    Parameters
    ----------
    n_components : int, default=2
        Number of sparse principal components to extract.
    alpha : float, default=1.0
        L1 penalty parameter for each component.
    lambda_l2 : float, default=1e-3
        L2 penalty parameter (shared across components).
    max_iter : int, default=100
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for convergence.
    verbose : bool, default=False
        Whether to print progress.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Sparse loading vectors (normalized).
    explained_variance_ : ndarray of shape (n_components,)
        The amount of variance explained by each of the selected components.
    """

    def __init__(
        self,
        n_components=2,
        alpha=1.0,
        lambda_l2=1e-3,
        max_iter=100,
        tol=1e-4,
        verbose=False,
    ):
        self.n_components = n_components
        self.alpha = alpha
        self.lambda_l2 = lambda_l2
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def fit(self, X, y=None, is_covariance=False):
        """
        Fit the model to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_features, n_features)
            Training data or covariance matrix.
        y : ignored
        is_covariance : bool, default=False
            Whether X is a covariance matrix.
        """
        self._init_fit_state()
        X = np.asanyarray(X, dtype=float)
        if is_covariance:
            Sigma = X
            n_features = Sigma.shape[0]
            # Use eigenvalue decomp to get pseudo-X such that X.T @ X = Sigma
            vals, vecs = np.linalg.eigh(Sigma)
            vals = np.maximum(vals, 0)
            X_centered = vecs @ np.diag(np.sqrt(vals)) @ vecs.T
            n_samples = n_features  # Pseudo-data has n_features samples
            self.mean_ = np.zeros(n_features)
        else:
            n_samples, n_features = X.shape
            self.mean_ = np.mean(X, axis=0)
            X_centered = X - self.mean_
            # Numerical stabilization for elastic-net subproblems.
            scale = np.std(X_centered, axis=0)
            scale[scale < 1e-12] = 1.0
            X_centered = X_centered / scale
            self.scale_ = scale
            # Use einsum to avoid occasional BLAS overflow warnings on some backends.
            Sigma = np.einsum(
                "ni,nj->ij", X_centered, X_centered, optimize=True
            ) / float(n_samples)

        # 1. Initialize A with the first k ordinary PCs
        # Using Sigma for PCA
        vals, vecs = np.linalg.eigh(Sigma)
        idx = np.argsort(vals)[::-1]
        A = vecs[:, idx[: self.n_components]]
        B = np.zeros((n_features, self.n_components))

        sk_alpha = max((self.alpha / 2 + self.lambda_l2) / n_samples, 1e-12)
        l1_ratio = (
            (self.alpha / 2) / (self.alpha / 2 + self.lambda_l2)
            if (self.alpha + self.lambda_l2) > 0
            else 0
        )

        for i in range(self.max_iter):
            A_old = A.copy()
            if not np.all(np.isfinite(A)):
                self._push_history("numerical_issue", "non_finite_A_iterate")
                self.converged_ = False
                self.n_iter_ = i + 1
                break

            # 2. Fix A, solve for B (Elastic Net)
            for j in range(self.n_components):
                # Target y_j = X_centered @ A[:, j]
                # ElasticNet minimizes 1/(2n) ||y - Xw||^2 + ...
                y_j = X_centered @ A[:, j]
                enet = ElasticNet(
                    alpha=sk_alpha,
                    l1_ratio=l1_ratio,
                    fit_intercept=False,
                    max_iter=1000,
                )
                enet.fit(X_centered, y_j)
                B[:, j] = enet.coef_

            # 3. Fix B, update A (Procrustes)
            M = Sigma @ B
            if not np.all(np.isfinite(M)):
                self._push_history("numerical_issue", "non_finite_procrustes_matrix")
                self.converged_ = False
                self.n_iter_ = i + 1
                break
            U, _, Vt = np.linalg.svd(M, full_matrices=False)
            A = U @ Vt
            if not np.all(np.isfinite(A)):
                self._push_history("numerical_issue", "non_finite_A")
                self.converged_ = False
                self.n_iter_ = i + 1
                break

            # Check convergence
            diff = np.linalg.norm(A - A_old)
            self._push_history("procrustes_diff", float(diff))
            if self.verbose:
                print(f"Iteration {i}: diff = {diff:.6f}")
            if diff < self.tol:
                self.converged_ = True
                self.n_iter_ = i + 1
                break
        else:
            self.n_iter_ = self.max_iter

        # Normalize B to get components_ (loadings)
        # The paper suggests normalizing the columns of B to unit length
        for j in range(self.n_components):
            norm = np.linalg.norm(B[:, j])
            if norm > 1e-10:
                B[:, j] /= norm

        self.components_ = B.T
        self.n_components_ = self.components_.shape[0]
        self.objective_ = None
        return self

    def transform(self, X):
        X_centered = X - self.mean_
        return X_centered @ self.components_.T


class SparsePCA_L1_ProxGrad(BaseEstimator, TransformerMixin, EstimatorStateMixin):
    """Single/multi-component L1-SPCA baseline via proximal gradient + deflation."""

    def __init__(
        self,
        n_components: int = 1,
        lambda1: float = 0.1,
        max_iter: int = 1000,
        learning_rate: float | str = "auto",
        tol: float = 1e-6,
        init: str = "pca",
        verbose: bool = False,
        monotone_backtracking: bool = True,
    ):
        self.n_components = n_components
        self.lambda1 = lambda1
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.tol = tol
        self.init = init
        self.verbose = verbose
        self.monotone_backtracking = monotone_backtracking

    @staticmethod
    def _soft_threshold(w: np.ndarray, threshold: float) -> np.ndarray:
        return np.sign(w) * np.maximum(np.abs(w) - threshold, 0.0)

    @staticmethod
    def _proj_l2_ball(u: np.ndarray) -> np.ndarray:
        if not np.all(np.isfinite(u)):
            return np.zeros_like(u)
        norm_u = np.linalg.norm(u)
        if not np.isfinite(norm_u):
            return np.zeros_like(u)
        return u / norm_u if norm_u > 1.0 else u

    def _objective(self, Xc: np.ndarray, w: np.ndarray) -> float:
        n = Xc.shape[0]
        sigma_term = -(w @ (Xc.T @ (Xc @ w))) / n
        return float(sigma_term + self.lambda1 * np.linalg.norm(w, 1))

    def _estimate_lipschitz(self, Xc: np.ndarray) -> float:
        pca = PCA(n_components=1)
        pca.fit(Xc)
        sigma_max = float(pca.explained_variance_[0])
        return 2.0 * sigma_max

    def _one_component(
        self, Xc: np.ndarray
    ) -> tuple[np.ndarray, int, bool, list[float]]:
        n, p = Xc.shape
        if self.init == "pca":
            pca = PCA(n_components=1)
            pca.fit(Xc)
            w = pca.components_[0].copy()
        else:
            rng = np.random.default_rng(0)
            w = rng.normal(size=p)
            w /= np.linalg.norm(w) + 1e-12

        eta = (
            1.0 / max(self._estimate_lipschitz(Xc), 1e-12)
            if self.learning_rate == "auto"
            else float(self.learning_rate)
        )
        history: list[float] = []
        converged = False
        n_iter = self.max_iter

        for it in range(self.max_iter):
            if not np.all(np.isfinite(w)):
                return np.zeros_like(w), it + 1, False, history
            w_old = w.copy()
            grad = -2.0 / n * (Xc.T @ (Xc @ w))
            eta_trial = eta

            while True:
                v = w - eta_trial * grad
                s = self._soft_threshold(v, eta_trial * self.lambda1)
                w_trial = self._proj_l2_ball(s)
                if not np.all(np.isfinite(w_trial)):
                    w = np.zeros_like(w)
                    converged = False
                    n_iter = it + 1
                    return w, n_iter, converged, history
                if not self.monotone_backtracking:
                    w = w_trial
                    eta = eta_trial
                    break
                f_old = self._objective(Xc, w)
                f_new = self._objective(Xc, w_trial)
                if not np.isfinite(f_old) or not np.isfinite(f_new):
                    w = np.zeros_like(w)
                    converged = False
                    n_iter = it + 1
                    return w, n_iter, converged, history
                if f_new <= f_old + 1e-12 or eta_trial < 1e-16:
                    w = w_trial
                    eta = eta_trial
                    break
                eta_trial *= 0.5

            obj = self._objective(Xc, w)
            history.append(obj)
            diff = np.linalg.norm(w - w_old) / (np.linalg.norm(w_old) + 1e-12)
            if diff < self.tol:
                converged = True
                n_iter = it + 1
                break

        return w, n_iter, converged, history

    def fit(self, X, y=None):
        self._init_fit_state()
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        n, p = Xc.shape
        k = min(self.n_components, p)

        self.components_ = np.zeros((k, p), dtype=float)
        self.n_components_ = k
        self.component_converged_ = []
        self.component_n_iter_ = []
        X_work = Xc.copy()

        all_histories: list[list[float]] = []
        for comp_idx in range(k):
            w, n_iter, converged, hist = self._one_component(X_work)
            self.components_[comp_idx] = w
            self.component_converged_.append(converged)
            self.component_n_iter_.append(n_iter)
            all_histories.append(hist)

            # Sequential deflation for fair multi-component comparisons.
            w_unit = w / (np.linalg.norm(w) + 1e-12)
            if np.all(np.isfinite(w_unit)) and np.all(np.isfinite(X_work)):
                with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                    scores = X_work @ w_unit
                if np.all(np.isfinite(scores)):
                    X_work = X_work - np.outer(scores, w_unit)

        self.history_["objective_history_by_component"] = all_histories
        self.converged_ = bool(all(self.component_converged_))
        self.n_iter_ = self.component_n_iter_
        self.objective_ = float(sum(h[-1] for h in all_histories if h))
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) @ self.components_.T
