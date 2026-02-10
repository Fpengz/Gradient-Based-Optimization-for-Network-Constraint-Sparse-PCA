import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA

class NetworkSparsePCA(BaseEstimator, TransformerMixin):
    """
    Network-Constrained Sparse Principal Component Analysis (NC-SPCA).
    Optimizes: min -w^T Sigma w + lambda1 ||w||_1 + lambda2 w^T L w
    subject to ||w||_2 <= 1.

    Parameters
    ----------
    n_components : int, default=2
        Number of components to extract.
    lambda1 : float, default=0.1
        L1 penalty for sparsity.
    lambda2 : float, default=0.1
        Graph Laplacian penalty for smoothness.
    max_iter : int, default=1000
        Maximum number of proximal gradient steps.
    learning_rate : float or 'auto', default=0.01
        Step size for gradient descent. If 'auto', estimates from Lipschitz constant.
    tol : float, default=1e-6
        Convergence tolerance.
    init : {'pca', 'random'}, default='pca'
        Initialization method.
    verbose : bool, default=False
    """

    def __init__(
        self,
        n_components=2,
        lambda1=0.1,
        lambda2=0.1,
        max_iter=1000,
        learning_rate=0.01,
        tol=1e-6,
        init='pca',
        verbose=False,
    ):
        self.n_components = n_components
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.tol = tol
        self.init = init
        self.verbose = verbose

    def _soft_threshold(self, w, threshold):
        return np.sign(w) * np.maximum(np.abs(w) - threshold, 0)

    def _estimate_lipschitz(self, X, L):
        """Estimate Lipschitz constant Lf = ||-2*Sigma + 2*lambda2*L||_2."""
        # ||Sigma||_2 = largest eigenvalue of Sigma
        pca = PCA(n_components=1)
        pca.fit(X)
        norm_sigma = pca.explained_variance_[0]
        
        # ||L||_2
        if sp.issparse(L):
            from scipy.sparse.linalg import eigsh
            norm_L = eigsh(L, k=1, which='LM', return_eigenvalues=True)[0][0]
        else:
            norm_L = np.linalg.norm(L, 2)
            
        Lf = 2.0 * norm_sigma + 2.0 * self.lambda2 * norm_L
        return Lf

    def fit(self, X, L=None):
        """
        Fit the model.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data matrix.
        L : ndarray or sparse matrix of shape (n_features, n_features), optional
            Graph Laplacian matrix. If None, defaults to identity.
        """
        X = np.asanyarray(X)
        n_samples, n_features = X.shape
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        if L is None:
            L = sp.eye(n_features)
            
        if self.learning_rate == 'auto':
            Lr = 1.0 / self._estimate_lipschitz(X_centered, L)
            if self.verbose:
                print(f"Auto-selected learning rate: {Lr:.6f}")
        else:
            Lr = self.learning_rate
            
        self.components_ = np.zeros((self.n_components, n_features))
        X_current = X_centered.copy()
        
        for k in range(self.n_components):
            # 1. Initialization
            if self.init == 'pca':
                pca = PCA(n_components=1)
                pca.fit(X_current)
                w = pca.components_[0]
            else:
                w = np.random.randn(n_features)
                w /= (np.linalg.norm(w) + 1e-12)
            
            for i in range(self.max_iter):
                w_old = w.copy()
                
                # Optimized gradient: O(np)
                # grad_sigma = -2 * Sigma * w = -2/n * X^T * (X * w)
                Xw = X_current @ w
                grad_sigma = -2.0 / n_samples * (X_current.T @ Xw)
                
                if sp.issparse(L):
                    grad_L = 2.0 * self.lambda2 * (L @ w)
                else:
                    grad_L = 2.0 * self.lambda2 * np.dot(L, w)
                    
                grad = grad_sigma + grad_L
                
                # Gradient step
                v = w - Lr * grad
                
                # Proximal step (Soft thresholding for L1)
                u = self._soft_threshold(v, Lr * self.lambda1)
                
                # Projection onto L2 ball
                norm_u = np.linalg.norm(u)
                if norm_u > 1.0:
                    w = u / norm_u
                else:
                    w = u
                
                # Convergence check (relative change)
                diff = np.linalg.norm(w - w_old)
                if diff / (np.linalg.norm(w_old) + 1e-12) < self.tol:
                    if self.verbose:
                        print(f"Component {k} converged at iter {i}")
                    break
            
            self.components_[k, :] = w
            
            # Deflation
            w_unit = w / (np.linalg.norm(w) + 1e-12)
            scores_unit = X_current @ w_unit
            X_current = X_current - np.outer(scores_unit, w_unit)
            
        return self

    def transform(self, X):
        X_centered = X - self.mean_
        return X_centered @ self.components_.T