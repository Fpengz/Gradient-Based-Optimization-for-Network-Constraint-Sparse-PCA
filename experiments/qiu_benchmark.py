import numpy as np
from data.synthetic.qiu_synthetic import generate_qiu_synthetic
from src.models.sparse_pca import ZouSparsePCA
from src.models.grad_fps import GradFPS
from sklearn.decomposition import PCA

def subspace_distance(A, B):
    """
    Computes the subspace distance between two orthonormal bases A and B.
    sqrt(d - sum(svd(A'B)^2))
    """
    Q1, _ = np.linalg.qr(A)
    Q2, _ = np.linalg.qr(B)
    S = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    return np.sqrt(np.maximum(0, A.shape[1] - np.sum(S**2)))

def run_qiu_benchmark():
    print("Running Qiu et al. (2023) Benchmark (p=500, n=200)...")
    X, Sigma = generate_qiu_synthetic(n_samples=200, p=500)
    
    # True loadings
    v1 = np.zeros(500)
    v1[0:10] = 1.0 / np.sqrt(10)
    v2 = np.zeros(500)
    v2[10:20] = 1.0 / np.sqrt(10)
    V_true = np.column_stack([v1, v2])
    
    # 1. PCA
    pca = PCA(n_components=2)
    pca.fit(X)
    dist_pca = subspace_distance(pca.components_.T, V_true)
    print(f"PCA Subspace Distance: {dist_pca:.4f}")
    
    # 2. Zou SPCA
    spca = ZouSparsePCA(n_components=2, alpha=100, lambda_l2=1e-3)
    spca.fit(X)
    dist_zou = subspace_distance(spca.components_.T, V_true)
    print(f"Zou SPCA Subspace Distance: {dist_zou:.4f}")
    print(f"Zou SPCA Sparsity: {np.mean(spca.components_ == 0):.1%}")
    
    # 3. GradFPS
    # Increase max_iter and use a slightly higher learning rate
    gfps = GradFPS(n_components=2, rho=1.0, learning_rate=0.05, max_iter=500, verbose=False)
    gfps.fit(X)
    dist_gfps = subspace_distance(gfps.components_.T, V_true)
    print(f"GradFPS Subspace Distance: {dist_gfps:.4f}")
    print(f"GradFPS Sparsity: {np.mean(gfps.components_ == 0):.1%}")

if __name__ == "__main__":
    run_qiu_benchmark()