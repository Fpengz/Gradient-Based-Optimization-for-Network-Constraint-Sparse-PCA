"""Network-constrained sparse PCA estimators."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from time import perf_counter
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
        qn_memory: int = 10,
        qn_orthant_aware: bool = True,
        qn_min_curvature: float = 1e-12,
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
        self.qn_memory = qn_memory
        self.qn_orthant_aware = qn_orthant_aware
        self.qn_min_curvature = qn_min_curvature

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
            L_sparse = sp.csr_matrix(L)
            if L_sparse.nnz == 0:
                norm_L = 0.0
            else:
                from scipy.sparse.linalg import ArpackError, eigsh

                try:
                    norm_L = float(
                        eigsh(
                            L_sparse, k=1, which="LM", return_eigenvectors=False
                        )[0]
                    )
                except ArpackError:
                    norm_L = float(spla.norm(L_sparse, ord=2))
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
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            grad_sigma = -2.0 / n * (Xc.T @ (Xc @ w))
        grad_sigma = np.nan_to_num(grad_sigma, nan=0.0, posinf=0.0, neginf=0.0)
        if sp.issparse(L):
            grad_L = 2.0 * self.lambda2 * (L @ w)
        else:
            L_dense = np.asarray(L, dtype=float)
            grad_L = 2.0 * self.lambda2 * (L_dense @ w)
        grad_L = np.nan_to_num(np.asarray(grad_L), nan=0.0, posinf=0.0, neginf=0.0)
        return grad_sigma + np.asarray(grad_L).reshape(-1)

    def _objective(
        self, Xc: np.ndarray, w: np.ndarray, L: sp.spmatrix | np.ndarray
    ) -> float:
        n = Xc.shape[0]
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            sigma_term = -(w @ (Xc.T @ (Xc @ w))) / n
        if sp.issparse(L):
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                lap = float(w @ (L @ w))
        else:
            L_dense = np.asarray(L, dtype=float)
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                lap = float(w @ (L_dense @ w))
        value = float(
            sigma_term + self.lambda1 * np.linalg.norm(w, 1) + self.lambda2 * lap
        )
        if not np.isfinite(value):
            return float("inf")
        return value

    def _smooth_value(
        self, Xc: np.ndarray, w: np.ndarray, L: sp.spmatrix | np.ndarray
    ) -> float:
        n = Xc.shape[0]
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            sigma_term = -(w @ (Xc.T @ (Xc @ w))) / n
        if sp.issparse(L):
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                lap = float(w @ (L @ w))
        else:
            L_dense = np.asarray(L, dtype=float)
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                lap = float(w @ (L_dense @ w))
        value = float(sigma_term + self.lambda2 * lap)
        return value if np.isfinite(value) else float("inf")

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
        use_smooth_majorization: bool = False,
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
            if use_smooth_majorization:
                f_y = self._smooth_value(Xc, y, L)
                f_trial = self._smooth_value(Xc, w_trial, L)
                quad = float(
                    np.dot(grad, (w_trial - y))
                    + (0.5 / eta_trial) * np.linalg.norm(w_trial - y) ** 2
                )
                if not np.isfinite(f_y) or not np.isfinite(f_trial) or not np.isfinite(quad):
                    return np.zeros_like(y), eta_trial, grad
                if f_trial <= f_y + quad + 1e-12 or eta_trial < 1e-16:
                    return w_trial, eta_trial, grad
            else:
                f_y = self._objective(Xc, y, L)
                f_trial = self._objective(Xc, w_trial, L)
                if not np.isfinite(f_y) or not np.isfinite(f_trial):
                    return np.zeros_like(y), eta_trial, grad
                if f_trial <= f_y + 1e-12 or eta_trial < 1e-16:
                    return w_trial, eta_trial, grad
            if eta_trial < 1e-16:
                return np.zeros_like(y), eta_trial, grad
            eta_trial *= 0.5

    def _accept_with_backtracking_direction(
        self,
        Xc: np.ndarray,
        y: np.ndarray,
        L: sp.spmatrix | np.ndarray,
        eta: float,
        direction: np.ndarray,
        orthant: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, bool]:
        eta_trial = eta
        f_y = self._objective(Xc, y, L)
        if not np.isfinite(f_y):
            return np.zeros_like(y), eta_trial, False
        while True:
            v = y + eta_trial * direction
            if orthant is not None:
                v = np.where(v * orthant < 0.0, 0.0, v)
            s = self._soft_threshold(v, eta_trial * self.lambda1)
            w_trial = self._proj_l2_ball(s)
            if not np.all(np.isfinite(w_trial)):
                return np.zeros_like(y), eta_trial, False
            if not self.monotone_backtracking:
                return w_trial, eta_trial, True
            f_trial = self._objective(Xc, w_trial, L)
            if not np.isfinite(f_y) or not np.isfinite(f_trial):
                return np.zeros_like(y), eta_trial, False
            if f_trial <= f_y + 1e-12 or eta_trial < 1e-16:
                accepted = bool(f_trial <= f_y + 1e-12)
                return w_trial, eta_trial, accepted
            eta_trial *= 0.5

    def _lbfgs_direction(
        self,
        grad: np.ndarray,
        s_hist: list[np.ndarray],
        y_hist: list[np.ndarray],
    ) -> np.ndarray:
        if not s_hist or not y_hist:
            return -grad
        q = grad.copy()
        alphas: list[float] = []
        rhos: list[float] = []
        for s, y in zip(reversed(s_hist), reversed(y_hist)):
            ys = float(np.dot(y, s))
            if not np.isfinite(ys) or ys <= 1e-16:
                return -grad
            rho = 1.0 / ys
            alpha = rho * float(np.dot(s, q))
            q = q - alpha * y
            alphas.append(alpha)
            rhos.append(rho)

        s_last = s_hist[-1]
        y_last = y_hist[-1]
        yy = float(np.dot(y_last, y_last))
        sy = float(np.dot(s_last, y_last))
        gamma = sy / yy if np.isfinite(yy) and yy > 1e-16 else 1.0
        r = gamma * q

        for i, (s, y) in enumerate(zip(s_hist, y_hist)):
            rho = rhos[-(i + 1)]
            alpha = alphas[-(i + 1)]
            beta = rho * float(np.dot(y, r))
            r = r + s * (alpha - beta)
        d = -r
        if not np.all(np.isfinite(d)) or float(np.dot(d, grad)) >= -1e-12:
            return -grad
        return d

    def _fit_one_component(
        self,
        Xc: np.ndarray,
        L: sp.spmatrix | np.ndarray,
        rng: np.random.Generator,
        warm_start: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, list[float]], bool, int]:
        if warm_start is None:
            w = self._initialize_component(Xc, rng)
        else:
            w = self._proj_l2_ball(np.asarray(warm_start, dtype=float).reshape(-1))
        eta = (
            1.0 / max(self._estimate_lipschitz(Xc, L), 1e-12)
            if self.learning_rate == "auto"
            else float(self.learning_rate)
        )

        local_history: dict[str, list[float]] = {
            "objective": [],
            "step_size": [],
            "rel_change": [],
            "pg_residual": [],
            "qn_used": [],
            "qn_accepted": [],
            "qn_fallback": [],
        }
        support_history: list[tuple[int, ...]] = []
        qn_s_hist: list[np.ndarray] = []
        qn_y_hist: list[np.ndarray] = []
        converged = False
        n_iter = self.max_iter
        w_prev = w.copy()
        grad_prev = self._smooth_grad(Xc, w, L)
        prev_obj: float | None = None

        for it in range(self.max_iter):
            if not np.all(np.isfinite(w)):
                return np.zeros_like(w), local_history, False, it + 1
            w_old = w.copy()
            grad_old = self._smooth_grad(Xc, w_old, L)

            if self.algorithm == "maspg_car" and it > 0:
                d = w_old - w_prev
                r = grad_old - grad_prev
                denom = float(np.dot(d, r))
                if np.isfinite(denom) and denom > 1e-16:
                    eta_bb = float(np.dot(d, d) / denom)
                    if np.isfinite(eta_bb) and eta_bb > 0.0:
                        eta = float(np.clip(eta_bb, 1e-8, 1e2))
            if self.algorithm == "maspg_car" and it > 0:
                # Monotone inertial heuristic with safe cap; restart if objective worsens.
                beta = min(0.9, (it - 1) / (it + 2))
                y = w + beta * (w - w_prev)
            else:
                y = w.copy()
            if not np.all(np.isfinite(y)):
                return np.zeros_like(w), local_history, False, it + 1

            qn_used = 0.0
            qn_accepted = 0.0
            qn_fallback = 0.0
            if self.algorithm == "prox_qn":
                direction = self._lbfgs_direction(grad_old, qn_s_hist, qn_y_hist)
                qn_used = 1.0
                orthant = np.sign(y)
                zero_mask = orthant == 0.0
                orthant[zero_mask] = np.sign(-grad_old[zero_mask])
                if not self.qn_orthant_aware:
                    orthant = None
                w_trial, eta, accepted = self._accept_with_backtracking_direction(
                    Xc,
                    y,
                    L,
                    eta,
                    direction=direction,
                    orthant=orthant,
                )
                qn_accepted = 1.0 if accepted else 0.0
                if not accepted:
                    qn_fallback = 1.0
                    w_trial, eta, _ = self._accept_with_backtracking(
                        Xc,
                        y,
                        L,
                        eta,
                        use_smooth_majorization=False,
                    )
            else:
                w_trial, eta, _ = self._accept_with_backtracking(
                    Xc,
                    y,
                    L,
                    eta,
                    use_smooth_majorization=(self.algorithm == "maspg_car"),
                )

            if self.algorithm == "maspg_car" and self.monotone_backtracking:
                obj_old = self._objective(Xc, w, L)
                obj_trial = self._objective(Xc, w_trial, L)
                if obj_trial > obj_old + 1e-12:
                    # Restart: drop inertia and recompute from current iterate.
                    w_trial, eta, _ = self._accept_with_backtracking(
                        Xc,
                        w,
                        L,
                        eta,
                        use_smooth_majorization=(self.algorithm == "maspg_car"),
                    )
            w_prev = w_old
            grad_prev = grad_old
            w = w_trial
            if self.algorithm == "prox_qn":
                grad_new = self._smooth_grad(Xc, w, L)
                s_k = w - w_old
                y_k = grad_new - grad_old
                sy = float(np.dot(s_k, y_k))
                if np.isfinite(sy) and sy > max(float(self.qn_min_curvature), 1e-16):
                    qn_s_hist.append(s_k.copy())
                    qn_y_hist.append(y_k.copy())
                    if len(qn_s_hist) > max(int(self.qn_memory), 1):
                        qn_s_hist.pop(0)
                        qn_y_hist.pop(0)

            obj = self._objective(Xc, w, L)
            rel = np.linalg.norm(w - w_old) / (np.linalg.norm(w_old) + 1e-12)
            rel_obj = (
                abs(obj - prev_obj) / (abs(prev_obj) + 1e-12)
                if prev_obj is not None and np.isfinite(prev_obj)
                else np.inf
            )
            residual = float(
                np.linalg.norm((y - w_trial) / max(eta, 1e-12))
            )
            local_history["objective"].append(float(obj))
            local_history["step_size"].append(float(eta))
            local_history["rel_change"].append(float(rel))
            local_history["pg_residual"].append(residual)
            local_history["qn_used"].append(float(qn_used))
            local_history["qn_accepted"].append(float(qn_accepted))
            local_history["qn_fallback"].append(float(qn_fallback))

            support = tuple(np.flatnonzero(np.abs(w) > self.support_threshold).tolist())
            support_history.append(support)
            if (
                self.algorithm in {"maspg_car", "prox_qn"}
                and len(support_history) >= self.active_set_window
                and len(set(support_history[-self.active_set_window :])) == 1
                and len(support) > 0
            ):
                # Restricted refinement: one prox step on active coordinates only.
                active = np.array(support, dtype=int)
                grad_full = self._smooth_grad(Xc, w, L)
                d_active = np.zeros_like(w)
                if self.algorithm == "prox_qn":
                    d_full = self._lbfgs_direction(grad_full, qn_s_hist, qn_y_hist)
                    d_active[active] = d_full[active]
                    if float(np.dot(d_active, grad_full)) >= -1e-12:
                        d_active[active] = -grad_full[active]
                else:
                    d_active[active] = -grad_full[active]

                orthant = np.sign(w) if self.qn_orthant_aware else None
                w_refined, _, accepted_ref = self._accept_with_backtracking_direction(
                    Xc,
                    w,
                    L,
                    eta,
                    direction=d_active,
                    orthant=orthant,
                )
                if accepted_ref and self._objective(Xc, w_refined, L) <= self._objective(Xc, w, L) + 1e-12:
                    w = self._proj_l2_ball(w_refined)

            if rel < self.tol or rel_obj < self.tol:
                converged = True
                n_iter = it + 1
                break
            prev_obj = obj

        return w, local_history, converged, n_iter

    def fit(self, X, L=None, graph=None, y=None, init_components=None):
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

        if self.algorithm not in {"pg", "maspg_car", "prox_qn"}:
            raise ValueError(f"Unknown algorithm={self.algorithm!r}")

        rng = np.random.default_rng(self.random_state)
        k = min(self.n_components, n_features)
        self.components_ = np.zeros((k, n_features))
        self.n_components_ = k
        self.component_converged_ = []
        self.component_n_iter_ = []
        X_current = X_centered.copy()
        objective_by_component = []
        warm_components = None
        if init_components is not None:
            warm_components = np.asarray(init_components, dtype=float)
            if warm_components.ndim != 2:
                raise ValueError("init_components must be 2D with shape (k, p).")
            if warm_components.shape[1] != n_features:
                raise ValueError(
                    "init_components feature dimension does not match input X."
                )

        for comp in range(k):
            warm_w = None
            if warm_components is not None and comp < warm_components.shape[0]:
                warm_w = warm_components[comp]
            w, local_hist, converged, n_iter = self._fit_one_component(
                X_current, L, rng, warm_start=warm_w
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
            self._push_history(
                "pg_residual_history_by_component", local_hist["pg_residual"]
            )
            self._push_history(
                "qn_used_history_by_component", local_hist["qn_used"]
            )
            self._push_history(
                "qn_accepted_history_by_component", local_hist["qn_accepted"]
            )
            self._push_history(
                "qn_fallback_history_by_component", local_hist["qn_fallback"]
            )

            # Sequential deflation (shared policy across Euclidean baselines).
            w_unit = w / (np.linalg.norm(w) + 1e-12)
            if np.all(np.isfinite(w_unit)) and np.all(np.isfinite(X_current)):
                with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                    scores = X_current @ w_unit
                if np.all(np.isfinite(scores)):
                    X_current = X_current - np.outer(scores, w_unit)

        self.converged_ = bool(all(self.component_converged_))
        self.n_iter_ = self.component_n_iter_
        self.objective_ = float(sum(h[-1] for h in objective_by_component if h))
        self.graph_laplacian_ = L
        return self

    def fit_path(
        self,
        X,
        L=None,
        graph=None,
        lambda1_grid: list[float] | tuple[float, ...] | None = None,
        lambda2_grid: list[float] | tuple[float, ...] | None = None,
        ordering: str = "serpentine",
    ) -> list[dict[str, object]]:
        """Fit a continuation path over (lambda1, lambda2) with warm starts."""
        l1_vals = list(lambda1_grid) if lambda1_grid is not None else [self.lambda1]
        l2_vals = list(lambda2_grid) if lambda2_grid is not None else [self.lambda2]
        if not l1_vals or not l2_vals:
            raise ValueError("lambda grids must be non-empty.")
        if ordering not in {"input", "serpentine"}:
            raise ValueError("ordering must be 'input' or 'serpentine'.")

        if ordering == "serpentine":
            l1_vals = sorted(float(v) for v in l1_vals)
            l2_base = sorted(float(v) for v in l2_vals)
            pairs: list[tuple[float, float]] = []
            for i, l1 in enumerate(l1_vals):
                l2_vals_row = l2_base if i % 2 == 0 else list(reversed(l2_base))
                for l2 in l2_vals_row:
                    pairs.append((l1, l2))
        else:
            pairs = [
                (float(l1), float(l2))
                for l1 in l1_vals
                for l2 in l2_vals
            ]

        base_params = {
            "n_components": self.n_components,
            "lambda1": self.lambda1,
            "lambda2": self.lambda2,
            "max_iter": self.max_iter,
            "learning_rate": self.learning_rate,
            "tol": self.tol,
            "init": self.init,
            "verbose": self.verbose,
            "monotone_backtracking": self.monotone_backtracking,
            "algorithm": self.algorithm,
            "random_state": self.random_state,
            "support_threshold": self.support_threshold,
            "active_set_window": self.active_set_window,
            "qn_memory": self.qn_memory,
            "qn_orthant_aware": self.qn_orthant_aware,
            "qn_min_curvature": self.qn_min_curvature,
        }
        if isinstance(self, NetworkSparsePCA_StiefelManifold):
            base_params["sparsity_mode"] = self.sparsity_mode
            base_params["group_lambda"] = self.group_lambda
        path: list[dict[str, object]] = []
        warm_components: np.ndarray | None = None
        for l1, l2 in pairs:
            params = dict(base_params)
            params["lambda1"] = float(l1)
            params["lambda2"] = float(l2)
            model = self.__class__(**params)
            tic = perf_counter()
            model.fit(X, L=L, graph=graph, init_components=warm_components)
            runtime_sec = perf_counter() - tic
            path.append(
                {
                    "lambda1": float(l1),
                    "lambda2": float(l2),
                    "model": model,
                    "warm_started": warm_components is not None,
                    "converged": bool(getattr(model, "converged_", False)),
                    "runtime_sec": float(runtime_sec),
                }
            )
            warm_components = model.components_.copy()
        return path

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        X_centered = X - self.mean_
        return X_centered @ self.components_.T


class NetworkSparsePCA_MASPG_CAR(NetworkSparsePCA):
    """Convenience wrapper for the practical accelerated variant."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("algorithm", "maspg_car")
        super().__init__(*args, **kwargs)


class NetworkSparsePCA_ProxQN(NetworkSparsePCA):
    """Convenience wrapper for proximal quasi-Newton updates."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("algorithm", "prox_qn")
        super().__init__(*args, **kwargs)


class NetworkSparsePCA_StiefelManifold(NetworkSparsePCA):
    """Multi-component manifold proximal-gradient solver on the Stiefel set."""

    def __init__(
        self,
        *args,
        sparsity_mode: str = "l1",
        group_lambda: float | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sparsity_mode = sparsity_mode
        self.group_lambda = group_lambda

    @staticmethod
    def _sym(M: np.ndarray) -> np.ndarray:
        return 0.5 * (M + M.T)

    @staticmethod
    def _retract_stiefel(Y: np.ndarray) -> np.ndarray:
        try:
            U, _, Vt = np.linalg.svd(Y, full_matrices=False)
            return U @ Vt
        except np.linalg.LinAlgError:
            Q, _ = np.linalg.qr(Y)
            return Q[:, : Y.shape[1]]

    @staticmethod
    def _row_group_soft_threshold(M: np.ndarray, threshold: float) -> np.ndarray:
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        safe = np.maximum(norms, 1e-12)
        scale = np.maximum(1.0 - threshold / safe, 0.0)
        scale[norms <= threshold] = 0.0
        return M * scale

    def _matrix_objective(
        self, Xc: np.ndarray, V: np.ndarray, L: sp.spmatrix | np.ndarray
    ) -> float:
        n = Xc.shape[0]
        XV = Xc @ V
        sigma_term = -float(np.sum(XV * XV) / n)
        if sp.issparse(L):
            LV = L @ V
        else:
            LV = np.asarray(L, dtype=float) @ V
        lap_term = float(np.sum(V * LV))
        if self.sparsity_mode == "l21":
            lam = float(self.group_lambda if self.group_lambda is not None else self.lambda1)
            penalty = lam * float(np.sum(np.linalg.norm(V, axis=1)))
        else:
            penalty = self.lambda1 * float(np.sum(np.abs(V)))
        val = sigma_term + penalty + self.lambda2 * lap_term
        return float(val) if np.isfinite(val) else float("inf")

    def _matrix_grad(
        self, Xc: np.ndarray, V: np.ndarray, L: sp.spmatrix | np.ndarray
    ) -> np.ndarray:
        n = Xc.shape[0]
        grad_sigma = -2.0 / n * (Xc.T @ (Xc @ V))
        if sp.issparse(L):
            grad_L = 2.0 * self.lambda2 * (L @ V)
        else:
            grad_L = 2.0 * self.lambda2 * (np.asarray(L, dtype=float) @ V)
        grad = grad_sigma + np.asarray(grad_L)
        return np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, X, L=None, graph=None, y=None, init_components=None):
        self._init_fit_state()
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape
        self.mean_ = np.mean(X, axis=0)
        Xc = X - self.mean_
        if graph is not None and L is None:
            L = (
                getattr(graph, "laplacian", None)
                if not isinstance(graph, dict)
                else graph.get("laplacian")
            )
        if L is None:
            L = sp.eye(n_features, format="csr")

        if self.sparsity_mode not in {"l1", "l21"}:
            raise ValueError("sparsity_mode must be 'l1' or 'l21'.")
        k = min(self.n_components, n_features)
        if init_components is not None:
            init_components = np.asarray(init_components, dtype=float)
            if init_components.ndim != 2 or init_components.shape[1] != n_features:
                raise ValueError("init_components must have shape (k, p).")
            V0 = init_components[:k].T
            V = self._retract_stiefel(V0)
        else:
            pca = PCA(n_components=k)
            pca.fit(Xc)
            V = pca.components_.T
            V = self._retract_stiefel(V)

        eta = (
            1.0 / max(self._estimate_lipschitz(Xc, L), 1e-12)
            if self.learning_rate == "auto"
            else float(self.learning_rate)
        )

        prev_obj: float | None = None
        converged = False
        local_hist_obj: list[float] = []
        local_hist_step: list[float] = []
        local_hist_rel: list[float] = []
        n_iter = self.max_iter

        for it in range(self.max_iter):
            grad = self._matrix_grad(Xc, V, L)
            riem_grad = grad - V @ self._sym(V.T @ grad)
            eta_trial = eta
            obj_old = self._matrix_objective(Xc, V, L)
            while True:
                Y = V - eta_trial * riem_grad
                if self.sparsity_mode == "l21":
                    lam = float(
                        self.group_lambda if self.group_lambda is not None else self.lambda1
                    )
                    S = self._row_group_soft_threshold(Y, eta_trial * lam)
                else:
                    S = self._soft_threshold(Y, eta_trial * self.lambda1)
                V_trial = self._retract_stiefel(S)
                if not self.monotone_backtracking:
                    break
                obj_new = self._matrix_objective(Xc, V_trial, L)
                if obj_new <= obj_old + 1e-12 or eta_trial < 1e-16:
                    break
                eta_trial *= 0.5

            rel = float(
                np.linalg.norm(V_trial - V, ord="fro")
                / (np.linalg.norm(V, ord="fro") + 1e-12)
            )
            obj = self._matrix_objective(Xc, V_trial, L)
            rel_obj = (
                abs(obj - prev_obj) / (abs(prev_obj) + 1e-12)
                if prev_obj is not None and np.isfinite(prev_obj)
                else np.inf
            )
            local_hist_obj.append(float(obj))
            local_hist_step.append(float(eta_trial))
            local_hist_rel.append(rel)
            V = V_trial
            eta = eta_trial
            if rel < self.tol or rel_obj < self.tol:
                converged = True
                n_iter = it + 1
                break
            prev_obj = obj

        self.components_ = V.T
        self.n_components_ = k
        self.converged_ = converged
        self.n_iter_ = n_iter
        self.objective_ = float(local_hist_obj[-1]) if local_hist_obj else float("nan")
        self.graph_laplacian_ = L
        self.history_["objective_history"] = local_hist_obj
        self.history_["step_size_history"] = local_hist_step
        self.history_["rel_change_history"] = local_hist_rel
        return self


class NetworkSparsePCA_StiefelStructured(NetworkSparsePCA_StiefelManifold):
    """Stiefel block solver with row-structured (L2,1) sparsity."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("sparsity_mode", "l21")
        super().__init__(*args, **kwargs)
