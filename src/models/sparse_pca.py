import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler


class ZouSparsePCA(BaseEstimator, TransformerMixin):
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
        X = np.asanyarray(X)
        if is_covariance:
            Sigma = X
            n_features = Sigma.shape[0]
            # Use eigenvalue decomp to get pseudo-X such that X.T @ X = Sigma
            vals, vecs = np.linalg.eigh(Sigma)
            vals = np.maximum(vals, 0)
            X_centered = vecs @ np.diag(np.sqrt(vals)) @ vecs.T
            n_samples = n_features # Pseudo-data has n_features samples
            self.mean_ = np.zeros(n_features)
        else:
            n_samples, n_features = X.shape
            self.mean_ = np.mean(X, axis=0)
            X_centered = X - self.mean_
            Sigma = (X_centered.T @ X_centered) / n_samples

        # 1. Initialize A with the first k ordinary PCs
        # Using Sigma for PCA
        vals, vecs = np.linalg.eigh(Sigma)
        idx = np.argsort(vals)[::-1]
        A = vecs[:, idx[:self.n_components]]
        B = np.zeros((n_features, self.n_components))

        sk_alpha = (self.alpha / 2 + self.lambda_l2) / n_samples
        l1_ratio = (self.alpha / 2) / (self.alpha / 2 + self.lambda_l2) if (self.alpha + self.lambda_l2) > 0 else 0

        for i in range(self.max_iter):
            A_old = A.copy()

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
            U, _, Vt = np.linalg.svd(M, full_matrices=False)
            A = U @ Vt

            # Check convergence
            diff = np.linalg.norm(A - A_old)
            if self.verbose:
                print(f"Iteration {i}: diff = {diff:.6f}")
            if diff < self.tol:
                break

        # Normalize B to get components_ (loadings)
        # The paper suggests normalizing the columns of B to unit length
        for j in range(self.n_components):
            norm = np.linalg.norm(B[:, j])
            if norm > 1e-10:
                B[:, j] /= norm

        self.components_ = B.T
        return self

    def transform(self, X):
        X_centered = X - self.mean_
        return X_centered @ self.components_.T
