import numpy as np
from scipy.optimize import brentq
from sklearn.base import BaseEstimator, TransformerMixin

class GradFPS(BaseEstimator, TransformerMixin):
    """
    Gradient-based Fantope Projection and Selection (GradFPS).
    As proposed by Qiu, Lei, and Roeder (2023).

    Optimizes: min -tr(S*X) + rho * ||X||_1,1 s.t. X in Fantope(d)
    where Fantope(d) = {X : 0 <= X <= I, tr(X) = d}

    Parameters
    ----------
    n_components : int, default=2
        Number of principal components (d in the paper).
    rho : float, default=0.1
        L1 penalty parameter for sparsity.
    learning_rate : float, default=0.01
    max_iter : int, default=100
    tol : float, default=1e-5
    verbose : bool, default=False
    """

    def __init__(
        self,
        n_components=2,
        rho=0.1,
        learning_rate=0.01,
        max_iter=100,
        tol=1e-5,
        verbose=False,
    ):
        self.n_components = n_components
        self.rho = rho
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def _fantope_projection(self, Y):
        """
        Projects Y onto the Fantope: {X : 0 <= X <= I, tr(X) = d}.
        """
        vals, vecs = np.linalg.eigh(Y)
        
        # We need to find theta such that sum(clip(vals - theta, 0, 1)) = d
        def f(theta):
            return np.sum(np.clip(vals - theta, 0, 1)) - self.n_components
        
        # g(theta) is decreasing.
        # Max value of g is when theta is very small: sum(clip(vals - theta, 0, 1)) -> p
        # Min value is when theta is very large: sum(clip(vals - theta, 0, 1)) -> 0
        # We search theta in [min(vals) - 1, max(vals)]
        low = np.min(vals) - 1.0
        high = np.max(vals)
        
        try:
            theta_star = brentq(f, low, high)
        except ValueError:
            # Fallback if brentq fails (e.g. if d is too large/small)
            theta_star = low if f(low) <= 0 else high

        new_vals = np.clip(vals - theta_star, 0, 1)
        return (vecs * new_vals) @ vecs.T

    def _soft_threshold(self, X, threshold):
        return np.sign(X) * np.maximum(np.abs(X) - threshold, 0)

    def fit(self, X, y=None):
        X = np.asanyarray(X)
        n_samples, n_features = X.shape
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        S = (X_centered.T @ X_centered) / n_samples
        
        # Initialize X as the standard PCA projection
        vals, vecs = np.linalg.eigh(S)
        idx = np.argsort(vals)[::-1]
        V_pca = vecs[:, idx[:self.n_components]]
        X_mat = V_pca @ V_pca.T
        
        for i in range(self.max_iter):
            X_old = X_mat.copy()
            
            # 1. Gradient step on the smooth part: -tr(SX)
            # Grad is -S. We do a step X + lr * S
            Y = X_mat + self.learning_rate * S
            
            # 2. Proximal step for L1: soft thresholding
            Y = self._soft_threshold(Y, self.learning_rate * self.rho)
            
            # 3. Proximal step for Fantope: projection
            X_mat = self._fantope_projection(Y)
            
            diff = np.linalg.norm(X_mat - X_old, 'fro')
            if self.verbose and i % 10 == 0:
                print(f"Iter {i}, diff: {diff:.6f}")
            if diff < self.tol:
                break
                
        self.X_ = X_mat
        
        # To get loading vectors, we take the top d eigenvectors of X_mat
        vals, vecs = np.linalg.eigh(X_mat)
        idx = np.argsort(vals)[::-1]
        self.components_ = vecs[:, idx[:self.n_components]].T
        
        return self

    def transform(self, X):
        X_centered = X - self.mean_
        return X_centered @ self.components_.T
