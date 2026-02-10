import numpy as np
from data.synthetic.synthetic_data import generate_zou_example_1
from data.pitprop import get_pitprop_correlation_matrix
from src.models.sparse_pca import ZouSparsePCA
from src.models.network_sparse_pca import NetworkSparsePCA
from sklearn.decomposition import PCA

from src.models.generalized_power_method import GeneralizedPowerMethod

def compare_on_zou_example_1():
    print("=== Benchmark: Zou Example 1 ===")
    X = generate_zou_example_1(n_samples=1000, seed=42)
    
    # 1. Standard PCA
    pca = PCA(n_components=2)
    pca.fit(X)
    print("PCA Components Sparsity:", np.mean(pca.components_ == 0))
    
    # 2. Zou SPCA
    spca = ZouSparsePCA(n_components=2, alpha=5000, lambda_l2=1e-3)
    spca.fit(X)
    print("Zou SPCA Sparsity:", np.mean(spca.components_ == 0))
    
    # 3. NC-SPCA
    nc_spca = NetworkSparsePCA(n_components=2, lambda1=20.0, lambda2=0.01, learning_rate=0.0001, max_iter=2000)
    nc_spca.fit(X)
    print("NC-SPCA Sparsity:", np.mean(nc_spca.components_ == 0))

    # 4. GPM (Journee et al.)
    gpm = GeneralizedPowerMethod(n_components=2, gamma=2.0)
    gpm.fit(X)
    print("GPM Sparsity:", np.mean(gpm.components_ == 0))

def compare_on_pitprop():
    print("=== Benchmark: Pitprop Dataset ===")
    corr = get_pitprop_correlation_matrix()
    vals, vecs = np.linalg.eigh(corr)
    vals = np.maximum(vals, 0)
    X_pseudo = vecs @ np.diag(np.sqrt(vals)) @ vecs.T
    
    # 1. Zou SPCA
    spca = ZouSparsePCA(n_components=3, alpha=0.2, lambda_l2=0.1)
    spca.fit(corr, is_covariance=True)
    print("Zou SPCA Sparsity:", np.mean(spca.components_ == 0))
    
    # 2. NC-SPCA
    nc_spca = NetworkSparsePCA(n_components=3, lambda1=0.05, lambda2=0.01, learning_rate=0.01)
    nc_spca.fit(X_pseudo)
    print("NC-SPCA Sparsity:", np.mean(nc_spca.components_ == 0))

    # 3. GPM
    gpm = GeneralizedPowerMethod(n_components=3, gamma=0.3)
    gpm.fit(X_pseudo)
    print("GPM Sparsity:", np.mean(gpm.components_ == 0))

if __name__ == "__main__":
    compare_on_zou_example_1()
    compare_on_pitprop()
