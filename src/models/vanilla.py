"""Vanilla PCA baseline with unified estimator API."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .base import EstimatorStateMixin


class PCAEstimator(BaseEstimator, TransformerMixin, EstimatorStateMixin):
    def __init__(self, n_components: int = 1):
        self.n_components = n_components

    def fit(self, X, y=None):
        self._init_fit_state()
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        _, s, vt = np.linalg.svd(Xc, full_matrices=False)
        k = min(self.n_components, vt.shape[0])
        self.components_ = vt[:k]
        self.n_components_ = k
        n = X.shape[0]
        self.explained_variance_ = (s[:k] ** 2) / max(n - 1, 1)
        self.converged_ = True
        self.n_iter_ = 1
        self.objective_ = float(np.sum(self.explained_variance_))
        self._push_history("explained_variance_sum", self.objective_)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) @ self.components_.T
