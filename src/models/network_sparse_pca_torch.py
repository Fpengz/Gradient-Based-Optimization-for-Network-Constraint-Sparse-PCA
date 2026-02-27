"""PyTorch/Geoopt backends for network-constrained sparse PCA."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA

from .base import EstimatorStateMixin

_TORCH: Any | None = None
_GEOOPT: Any | None = None


def _load_torch() -> Any | None:
    global _TORCH
    if _TORCH is None:
        try:
            _TORCH = importlib.import_module("torch")
        except Exception:
            return None
    return _TORCH


def _load_geoopt() -> Any | None:
    global _GEOOPT
    if _GEOOPT is None:
        try:
            _GEOOPT = importlib.import_module("geoopt")
        except Exception:
            return None
    return _GEOOPT


class TorchNetworkSparsePCA(BaseEstimator, TransformerMixin, EstimatorStateMixin):
    """Torch backend matching the NumPy PG/MASPG-CAR single-component objective."""

    def __init__(
        self,
        n_components: int = 1,
        lambda1: float = 0.1,
        lambda2: float = 0.1,
        max_iter: int = 1000,
        learning_rate: float | str = "auto",
        tol: float = 1e-6,
        init: str = "pca",
        monotone_backtracking: bool = True,
        backend: str = "pg",
        random_state: int | None = None,
        device: str = "cpu",
        dtype: str = "float64",
    ):
        self.n_components = n_components
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.tol = tol
        self.init = init
        self.monotone_backtracking = monotone_backtracking
        self.backend = backend
        self.random_state = random_state
        self.device = device
        self.dtype = dtype

    @staticmethod
    def _require_torch() -> Any:
        t = _load_torch()
        if t is None:
            raise ImportError(
                "PyTorch is required for TorchNetworkSparsePCA. Install torch first."
            )
        return t

    def _torch_dtype(self):
        t = self._require_torch()
        return t.float64 if self.dtype == "float64" else t.float32

    def _to_tensor(self, x: np.ndarray):
        t = self._require_torch()
        return t.as_tensor(x, dtype=self._torch_dtype(), device=self.device)

    @staticmethod
    def _soft_threshold_torch(x, threshold: float):
        return x.sign() * (x.abs() - threshold).clamp_min(0.0)

    @staticmethod
    def _proj_l2_ball_torch(x):
        t = _load_torch()
        if t is None:
            raise ImportError("PyTorch is required for TorchNetworkSparsePCA.")
        norm = x.norm()
        if not t.isfinite(norm):
            return t.zeros_like(x)
        if norm > 1.0:
            return x / norm
        return x

    def _estimate_lipschitz(self, Xc: np.ndarray, L: sp.spmatrix | np.ndarray) -> float:
        pca = PCA(n_components=1)
        pca.fit(Xc)
        norm_sigma = float(pca.explained_variance_[0])
        if sp.issparse(L):
            Ld = sp.csr_matrix(L).toarray()
        else:
            Ld = np.asarray(L, dtype=float)
        norm_L = float(np.linalg.norm(Ld, 2))
        return 2.0 * norm_sigma + 2.0 * self.lambda2 * norm_L

    def _prepare(self, X, L=None, graph=None):
        X = np.asarray(X, dtype=float)
        n, p = X.shape
        mean = X.mean(axis=0)
        Xc = X - mean
        if graph is not None and L is None:
            L = (
                getattr(graph, "laplacian", None)
                if not isinstance(graph, dict)
                else graph.get("laplacian")
            )
        if L is None:
            L = sp.eye(p, format="csr")
        Ld = sp.csr_matrix(L).toarray() if sp.issparse(L) else np.asarray(L, dtype=float)
        Xt = self._to_tensor(Xc)
        Lt = self._to_tensor(Ld)
        return X, Xc, Xt, Lt, mean

    def _objective_torch(self, Xt, Lt, w):
        n = Xt.shape[0]
        xw = Xt @ w
        sigma_term = -((xw * xw).sum() / n)
        lap_term = w @ (Lt @ w)
        return sigma_term + self.lambda1 * w.abs().sum() + self.lambda2 * lap_term

    def _smooth_grad_torch(self, Xt, Lt, w):
        n = Xt.shape[0]
        grad_sigma = -2.0 / n * (Xt.T @ (Xt @ w))
        grad_l = 2.0 * self.lambda2 * (Lt @ w)
        return grad_sigma + grad_l

    def _initialize_w(self, Xc: np.ndarray, p: int):
        _ = self._require_torch()
        if self.init == "pca":
            pca = PCA(n_components=1)
            pca.fit(Xc)
            w0 = pca.components_[0]
            return self._to_tensor(w0)
        rng = np.random.default_rng(self.random_state)
        w = rng.normal(size=p)
        w /= np.linalg.norm(w) + 1e-12
        return self._to_tensor(w)

    def _fit_one(self, Xt, Lt, Xc: np.ndarray, w0):
        _ = self._require_torch()
        eta = (
            1.0 / max(self._estimate_lipschitz(Xc, Lt.detach().cpu().numpy()), 1e-12)
            if self.learning_rate == "auto"
            else float(self.learning_rate)
        )
        w = w0.clone()
        w_prev = w.clone()
        grad_prev = self._smooth_grad_torch(Xt, Lt, w)
        prev_obj = None
        hist = {"objective": [], "step_size": [], "rel_change": [], "pg_residual": []}
        converged = False
        n_iter = self.max_iter

        for it in range(self.max_iter):
            w_old = w.clone()
            grad_old = self._smooth_grad_torch(Xt, Lt, w_old)

            if self.backend == "maspg_car" and it > 0:
                d = w_old - w_prev
                r = grad_old - grad_prev
                denom = float((d * r).sum().item())
                if np.isfinite(denom) and denom > 1e-16:
                    eta_bb = float(((d * d).sum() / denom).item())
                    if np.isfinite(eta_bb) and eta_bb > 0:
                        eta = float(np.clip(eta_bb, 1e-8, 1e2))

            if self.backend == "maspg_car" and it > 0:
                beta = min(0.9, (it - 1) / (it + 2))
                y = w + beta * (w - w_prev)
            else:
                y = w.clone()

            grad_y = self._smooth_grad_torch(Xt, Lt, y)
            eta_trial = eta
            while True:
                v = y - eta_trial * grad_y
                s = self._soft_threshold_torch(v, eta_trial * self.lambda1)
                w_trial = self._proj_l2_ball_torch(s)
                if not self.monotone_backtracking:
                    break
                if self.backend == "maspg_car":
                    f_y = self._objective_torch(Xt, Lt, y) - self.lambda1 * y.abs().sum()
                    f_trial = self._objective_torch(Xt, Lt, w_trial) - self.lambda1 * w_trial.abs().sum()
                    dlt = w_trial - y
                    rhs = f_y + (grad_y * dlt).sum() + 0.5 / eta_trial * (dlt.norm() ** 2)
                    if float(f_trial.item()) <= float(rhs.item()) + 1e-12 or eta_trial < 1e-16:
                        break
                else:
                    obj_old = self._objective_torch(Xt, Lt, y)
                    obj_new = self._objective_torch(Xt, Lt, w_trial)
                    if float(obj_new.item()) <= float(obj_old.item()) + 1e-12 or eta_trial < 1e-16:
                        break
                eta_trial *= 0.5

            if self.backend == "maspg_car" and self.monotone_backtracking:
                obj_curr = self._objective_torch(Xt, Lt, w)
                obj_trial = self._objective_torch(Xt, Lt, w_trial)
                if float(obj_trial.item()) > float(obj_curr.item()) + 1e-12:
                    y = w.clone()
                    grad_y = self._smooth_grad_torch(Xt, Lt, y)
                    v = y - eta_trial * grad_y
                    s = self._soft_threshold_torch(v, eta_trial * self.lambda1)
                    w_trial = self._proj_l2_ball_torch(s)

            w_prev = w_old
            grad_prev = grad_old
            w = w_trial
            eta = eta_trial

            obj = float(self._objective_torch(Xt, Lt, w).item())
            rel = float((w - w_old).norm().item() / (w_old.norm().item() + 1e-12))
            rel_obj = abs(obj - prev_obj) / (abs(prev_obj) + 1e-12) if prev_obj is not None else np.inf
            residual = float(((y - w) / max(eta, 1e-12)).norm().item())
            hist["objective"].append(obj)
            hist["step_size"].append(float(eta))
            hist["rel_change"].append(rel)
            hist["pg_residual"].append(residual)
            if rel < self.tol or rel_obj < self.tol:
                converged = True
                n_iter = it + 1
                break
            prev_obj = obj

        return w, hist, converged, n_iter

    def fit(self, X, L=None, graph=None, y=None, init_components=None):
        _ = self._require_torch()
        self._init_fit_state()
        X_raw, Xc, Xt, Lt, mean = self._prepare(X, L=L, graph=graph)
        _, p = X_raw.shape
        self.mean_ = mean
        k = min(self.n_components, p)
        self.components_ = np.zeros((k, p), dtype=float)
        self.n_components_ = k
        self.component_converged_ = []
        self.component_n_iter_ = []
        X_work = Xt.clone()

        for comp in range(k):
            w0 = self._initialize_w(Xc, p)
            if init_components is not None:
                init = np.asarray(init_components, dtype=float)
                if init.ndim == 2 and comp < init.shape[0]:
                    w0 = self._to_tensor(init[comp])
            w, hist, conv, n_iter = self._fit_one(X_work, Lt, Xc, w0)
            w_np = w.detach().cpu().numpy()
            self.components_[comp] = w_np
            self.component_converged_.append(conv)
            self.component_n_iter_.append(n_iter)
            self._push_history("objective_history_by_component", hist["objective"])
            self._push_history("step_size_history_by_component", hist["step_size"])
            self._push_history("rel_change_history_by_component", hist["rel_change"])
            self._push_history("pg_residual_history_by_component", hist["pg_residual"])

            w_unit = w / (w.norm() + 1e-12)
            scores = X_work @ w_unit
            X_work = X_work - scores.unsqueeze(1) @ w_unit.unsqueeze(0)

        self.converged_ = bool(all(self.component_converged_))
        self.n_iter_ = self.component_n_iter_
        obj_hist = self.history_.get("objective_history_by_component", [])
        self.objective_ = float(sum(h[-1] for h in obj_hist if h))
        return self

    def fit_path(
        self,
        X,
        L=None,
        graph=None,
        lambda1_grid: list[float] | tuple[float, ...] | None = None,
        lambda2_grid: list[float] | tuple[float, ...] | None = None,
    ) -> list[dict[str, object]]:
        l1_vals = list(lambda1_grid) if lambda1_grid is not None else [self.lambda1]
        l2_vals = list(lambda2_grid) if lambda2_grid is not None else [self.lambda2]
        base_params = self.get_params(deep=False)
        path: list[dict[str, object]] = []
        warm = None
        for l1 in l1_vals:
            for l2 in l2_vals:
                params = dict(base_params)
                params["lambda1"] = float(l1)
                params["lambda2"] = float(l2)
                model = self.__class__(**params)
                model.fit(X, L=L, graph=graph, init_components=warm)
                path.append(
                    {
                        "lambda1": float(l1),
                        "lambda2": float(l2),
                        "model": model,
                        "warm_started": warm is not None,
                        "converged": bool(model.converged_),
                    }
                )
                warm = model.components_.copy()
        return path

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) @ self.components_.T


class TorchNetworkSparsePCA_GeooptStiefel(BaseEstimator, TransformerMixin, EstimatorStateMixin):
    """Optional Geoopt Stiefel-manifold multi-component backend."""

    def __init__(
        self,
        n_components: int = 2,
        lambda1: float = 0.1,
        lambda2: float = 0.1,
        max_iter: int = 500,
        learning_rate: float = 1e-2,
        tol: float = 1e-6,
        monotone_backtracking: bool = True,
        random_state: int | None = None,
        device: str = "cpu",
        dtype: str = "float64",
    ):
        self.n_components = n_components
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.tol = tol
        self.monotone_backtracking = monotone_backtracking
        self.random_state = random_state
        self.device = device
        self.dtype = dtype

    def _require_deps(self):
        t = _load_torch()
        g = _load_geoopt()
        if t is None:
            raise ImportError("PyTorch is required for TorchNetworkSparsePCA_GeooptStiefel.")
        if g is None:
            raise ImportError("Geoopt is required for TorchNetworkSparsePCA_GeooptStiefel.")
        return t, g

    def _torch_dtype(self):
        t, _ = self._require_deps()
        return t.float64 if self.dtype == "float64" else t.float32

    def _to_tensor(self, x: np.ndarray):
        t, _ = self._require_deps()
        return t.as_tensor(x, dtype=self._torch_dtype(), device=self.device)

    def _objective(self, Xt, Lt, V):
        n = Xt.shape[0]
        XV = Xt @ V
        sigma = -((XV * XV).sum() / n)
        lap = (V * (Lt @ V)).sum()
        return sigma + self.lambda1 * V.abs().sum() + self.lambda2 * lap

    def _grad(self, Xt, Lt, V):
        n = Xt.shape[0]
        return -2.0 / n * (Xt.T @ (Xt @ V)) + 2.0 * self.lambda2 * (Lt @ V)

    def fit(self, X, L=None, graph=None, y=None, init_components=None):
        t, g = self._require_deps()
        self._init_fit_state()
        X = np.asarray(X, dtype=float)
        n, p = X.shape
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        if graph is not None and L is None:
            L = (
                getattr(graph, "laplacian", None)
                if not isinstance(graph, dict)
                else graph.get("laplacian")
            )
        if L is None:
            L = sp.eye(p, format="csr")
        Ld = sp.csr_matrix(L).toarray() if sp.issparse(L) else np.asarray(L, dtype=float)
        Xt = self._to_tensor(Xc)
        Lt = self._to_tensor(Ld)
        k = min(self.n_components, p)

        if init_components is not None:
            init = np.asarray(init_components, dtype=float)[:k].T
            V = self._to_tensor(init)
            q, _ = t.linalg.qr(V)
            V = q[:, :k]
        else:
            pca = PCA(n_components=k)
            pca.fit(Xc)
            V = self._to_tensor(pca.components_.T)
            q, _ = t.linalg.qr(V)
            V = q[:, :k]

        manifold = g.Stiefel()
        prev_obj = None
        hist_obj: list[float] = []
        hist_step: list[float] = []
        hist_rel: list[float] = []
        converged = False
        n_iter = self.max_iter
        eta = float(self.learning_rate)

        for it in range(self.max_iter):
            grad = self._grad(Xt, Lt, V)
            sym = 0.5 * (V.T @ grad + grad.T @ V)
            riem_grad = grad - V @ sym
            eta_trial = eta
            obj_old = self._objective(Xt, Lt, V)
            while True:
                Y = V - eta_trial * riem_grad
                S = Y.sign() * (Y.abs() - eta_trial * self.lambda1).clamp_min(0.0)
                V_trial = manifold.projx(S)
                if not self.monotone_backtracking:
                    break
                obj_new = self._objective(Xt, Lt, V_trial)
                if float(obj_new.item()) <= float(obj_old.item()) + 1e-12 or eta_trial < 1e-16:
                    break
                eta_trial *= 0.5

            rel = float((V_trial - V).norm().item() / (V.norm().item() + 1e-12))
            obj = float(self._objective(Xt, Lt, V_trial).item())
            rel_obj = abs(obj - prev_obj) / (abs(prev_obj) + 1e-12) if prev_obj is not None else np.inf
            hist_obj.append(obj)
            hist_step.append(float(eta_trial))
            hist_rel.append(rel)
            V = V_trial
            eta = eta_trial
            if rel < self.tol or rel_obj < self.tol:
                converged = True
                n_iter = it + 1
                break
            prev_obj = obj

        self.components_ = V.detach().cpu().numpy().T
        self.n_components_ = k
        self.converged_ = converged
        self.n_iter_ = n_iter
        self.objective_ = float(hist_obj[-1]) if hist_obj else float("nan")
        self.history_["objective_history"] = hist_obj
        self.history_["step_size_history"] = hist_step
        self.history_["rel_change_history"] = hist_rel
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) @ self.components_.T
