"""Network-constrained sparse PCA estimators."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA

from .base import EstimatorStateMixin


class NetworkSparsePCA(BaseEstimator, TransformerMixin, EstimatorStateMixin):
    """Single-/multi-component network-constrained sparse PCA via prox-gradient.

    Objective for each component (with deflation for multiple components):
        min_w -w^T Sigma w + lambda1 ||w||_1 + lambda2 w^T L w
        s.t. ||w||_2 <= 1

    Parameters
    ----------
    n_components : int
        Number of components (extracted sequentially with deflation).
    lambda1, lambda2 : float
        Sparsity and Laplacian smoothness regularization strengths.
    max_iter : int
        Maximum iterations per component.
    learning_rate : float | "auto"
        Fixed step size or spectral estimate based on a Lipschitz upper bound.
    tol : float
        Relative iterate tolerance.
    init : {"pca", "random"}
    verbose : bool
    monotone_backtracking : bool
        If True, uses simple monotone line search.
    algorithm : {"pg", "maspg_car"}
        Baseline prox-gradient or practical accelerated variant.
    random_state : int | None
    support_threshold : float
        Threshold for support-stability checks in the accelerated variant.
    active_set_window : int
        Number of consecutive iterations with unchanged support before refinement.
    """

    def __init__(
        self,
        n_components: int = 1,
        lambda1: float = 0.1,
        lambda2: float = 0.1,
        max_iter: int = 1000,
        learning_rate: float | str = "auto",
        tol: float = 1e-6,
        init: str = "pca",
        verbose: bool = False,
        monotone_backtracking: bool = True,
        algorithm: str = "pg",
        random_state: int | None = None,
        support_threshold: float = 1e-8,
        active_set_window: int = 10,
    ):
        self.n_components = n_components
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.tol = tol
        self.init = init
        self.verbose = verbose
        self.monotone_backtracking = monotone_backtracking
        self.algorithm = algorithm
        self.random_state = random_state
        self.support_threshold = support_threshold
        self.active_set_window = active_set_window

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

    def _estimate_lipschitz(
        self, X_centered: np.ndarray, L: sp.spmatrix | np.ndarray
    ) -> float:
        pca = PCA(n_components=1)
        pca.fit(X_centered)
        norm_sigma = float(pca.explained_variance_[0])
        if sp.issparse(L):
            from scipy.sparse.linalg import eigsh

            norm_L = float(eigsh(L, k=1, which="LM", return_eigenvectors=False)[0])
        else:
            L_dense = np.asarray(L, dtype=float)
            norm_L = float(np.linalg.norm(L_dense, 2))
        return 2.0 * norm_sigma + 2.0 * self.lambda2 * norm_L

    def _smooth_grad(
        self,
        Xc: np.ndarray,
        w: np.ndarray,
        L: sp.spmatrix | np.ndarray,
    ) -> np.ndarray:
        n = Xc.shape[0]
        grad_sigma = -2.0 / n * (Xc.T @ (Xc @ w))
        if sp.issparse(L):
            grad_L = 2.0 * self.lambda2 * (L @ w)
        else:
            L_dense = np.asarray(L, dtype=float)
            grad_L = 2.0 * self.lambda2 * (L_dense @ w)
        return grad_sigma + np.asarray(grad_L).reshape(-1)

    def _objective(
        self, Xc: np.ndarray, w: np.ndarray, L: sp.spmatrix | np.ndarray
    ) -> float:
        n = Xc.shape[0]
        sigma_term = -(w @ (Xc.T @ (Xc @ w))) / n
        if sp.issparse(L):
            lap = float(w @ (L @ w))
        else:
            L_dense = np.asarray(L, dtype=float)
            lap = float(w @ (L_dense @ w))
        return float(
            sigma_term + self.lambda1 * np.linalg.norm(w, 1) + self.lambda2 * lap
        )

    def _initialize_component(
        self, Xc: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        if self.init == "pca":
            pca = PCA(n_components=1)
            pca.fit(Xc)
            w = pca.components_[0].copy()
        elif self.init == "random":
            w = rng.normal(size=Xc.shape[1])
            w /= np.linalg.norm(w) + 1e-12
        else:
            raise ValueError(f"Unknown init={self.init!r}")
        return w

    def _accept_with_backtracking(
        self,
        Xc: np.ndarray,
        y: np.ndarray,
        L: sp.spmatrix | np.ndarray,
        eta: float,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        grad = self._smooth_grad(Xc, y, L)
        eta_trial = eta
        while True:
            v = y - eta_trial * grad
            s = self._soft_threshold(v, eta_trial * self.lambda1)
            w_trial = self._proj_l2_ball(s)
            if not np.all(np.isfinite(w_trial)):
                return np.zeros_like(y), eta_trial, grad
            if not self.monotone_backtracking:
                return w_trial, eta_trial, grad
            f_y = self._objective(Xc, y, L)
            f_trial = self._objective(Xc, w_trial, L)
            if not np.isfinite(f_y) or not np.isfinite(f_trial):
                return np.zeros_like(y), eta_trial, grad
            if f_trial <= f_y + 1e-12 or eta_trial < 1e-16:
                return w_trial, eta_trial, grad
            eta_trial *= 0.5

    def _fit_one_component(
        self,
        Xc: np.ndarray,
        L: sp.spmatrix | np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, dict[str, list[float]], bool, int]:
        w = self._initialize_component(Xc, rng)
        eta = (
            1.0 / max(self._estimate_lipschitz(Xc, L), 1e-12)
            if self.learning_rate == "auto"
            else float(self.learning_rate)
        )

        local_history: dict[str, list[float]] = {
            "objective": [],
            "step_size": [],
            "rel_change": [],
        }
        support_history: list[tuple[int, ...]] = []
        converged = False
        n_iter = self.max_iter
        w_prev = w.copy()

        for it in range(self.max_iter):
            if not np.all(np.isfinite(w)):
                return np.zeros_like(w), local_history, False, it + 1
            w_old = w.copy()
            if self.algorithm == "maspg_car" and it > 0:
                # Monotone inertial heuristic with safe cap; restart if objective worsens.
                beta = min(0.9, (it - 1) / (it + 2))
                y = w + beta * (w - w_prev)
            else:
                y = w.copy()
            if not np.all(np.isfinite(y)):
                return np.zeros_like(w), local_history, False, it + 1

            w_trial, eta, _ = self._accept_with_backtracking(Xc, y, L, eta)

            if self.algorithm == "maspg_car" and self.monotone_backtracking:
                obj_old = self._objective(Xc, w, L)
                obj_trial = self._objective(Xc, w_trial, L)
                if obj_trial > obj_old + 1e-12:
                    # Restart: drop inertia and recompute from current iterate.
                    w_trial, eta, _ = self._accept_with_backtracking(Xc, w, L, eta)
            w_prev = w.copy()
            w = w_trial

            obj = self._objective(Xc, w, L)
            rel = np.linalg.norm(w - w_old) / (np.linalg.norm(w_old) + 1e-12)
            local_history["objective"].append(float(obj))
            local_history["step_size"].append(float(eta))
            local_history["rel_change"].append(float(rel))

            support = tuple(np.flatnonzero(np.abs(w) > self.support_threshold).tolist())
            support_history.append(support)
            if (
                self.algorithm == "maspg_car"
                and len(support_history) >= self.active_set_window
                and len(set(support_history[-self.active_set_window :])) == 1
                and len(support) > 0
            ):
                # Light restricted refinement: one gradient/prox step on active coordinates only.
                active = np.array(support, dtype=int)
                y_active = w[active].copy()
                grad_full = self._smooth_grad(Xc, w, L)
                v_active = y_active - eta * grad_full[active]
                s_active = self._soft_threshold(v_active, eta * self.lambda1)
                w_refined = w.copy()
                w_refined[:] = 0.0
                w_refined[active] = s_active
                w = self._proj_l2_ball(w_refined)

            if rel < self.tol:
                converged = True
                n_iter = it + 1
                break

        return w, local_history, converged, n_iter

    def fit(self, X, L=None, graph=None, y=None):
        self._init_fit_state()
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        if graph is not None and L is None:
            L = (
                getattr(graph, "laplacian", None)
                if not isinstance(graph, dict)
                else graph.get("laplacian")
            )
        if L is None:
            L = sp.eye(n_features, format="csr")

        if self.algorithm not in {"pg", "maspg_car"}:
            raise ValueError(f"Unknown algorithm={self.algorithm!r}")

        rng = np.random.default_rng(self.random_state)
        k = min(self.n_components, n_features)
        self.components_ = np.zeros((k, n_features))
        self.n_components_ = k
        self.component_converged_ = []
        self.component_n_iter_ = []
        X_current = X_centered.copy()
        objective_by_component = []

        for comp in range(k):
            w, local_hist, converged, n_iter = self._fit_one_component(
                X_current, L, rng
            )
            self.components_[comp, :] = w
            self.component_converged_.append(converged)
            self.component_n_iter_.append(n_iter)
            objective_by_component.append(local_hist["objective"])
            self._push_history(
                "objective_history_by_component", local_hist["objective"]
            )
            self._push_history(
                "step_size_history_by_component", local_hist["step_size"]
            )
            self._push_history(
                "rel_change_history_by_component", local_hist["rel_change"]
            )

            # Sequential deflation (shared policy across Euclidean baselines).
            w_unit = w / (np.linalg.norm(w) + 1e-12)
            if np.all(np.isfinite(w_unit)) and np.all(np.isfinite(X_current)):
                scores = X_current @ w_unit
                if np.all(np.isfinite(scores)):
                    X_current = X_current - np.outer(scores, w_unit)

        self.converged_ = bool(all(self.component_converged_))
        self.n_iter_ = self.component_n_iter_
        self.objective_ = float(sum(h[-1] for h in objective_by_component if h))
        self.graph_laplacian_ = L
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        X_centered = X - self.mean_
        return X_centered @ self.components_.T


class NetworkSparsePCA_MASPG_CAR(NetworkSparsePCA):
    """Convenience wrapper for the practical accelerated variant."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("algorithm", "maspg_car")
        super().__init__(*args, **kwargs)
