import numpy as np
from data.pitprop import get_pitprop_correlation_matrix
from src.models.sparse_pca import ZouSparsePCA
from sklearn.decomposition import PCA

def run_pitprop_experiment():
    print("Running Pitprop Dataset Experiment (Zou et al. 2006)...")
    corr = get_pitprop_correlation_matrix()
    
    # Standard PCA
    vals, vecs = np.linalg.eigh(corr)
    idx = np.argsort(vals)[::-1]
    pca_components = vecs[:, idx[:3]].T
    
    print("\nStandard PCA Components (top 3):")
    np.set_printoptions(precision=3, suppress=True)
    print(pca_components)
    
    # Sparse PCA
    # Using alpha=0.2
    spca = ZouSparsePCA(n_components=3, alpha=0.2, lambda_l2=0.1, verbose=False)
    spca.fit(corr, is_covariance=True)
    
    print("\nZou Sparse PCA Components (top 3):")
    print(spca.components_)
    
    # Sparsity
    sparsity = np.mean(spca.components_ == 0)
    print(f"\nSparsity: {sparsity:.1%}")

if __name__ == "__main__":
    run_pitprop_experiment()
