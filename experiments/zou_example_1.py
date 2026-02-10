import numpy as np
from data.synthetic.synthetic_data import generate_zou_example_1
from src.models.sparse_pca import ZouSparsePCA
from sklearn.decomposition import PCA

def run_zou_example_1():
    print("Running Zou et al. (2006) Example 1...")
    X = generate_zou_example_1(n_samples=1000, seed=42)
    
    # 1. Standard PCA
    pca = PCA(n_components=2)
    pca.fit(X)
    print("\nStandard PCA Components (top 2):")
    np.set_printoptions(precision=3, suppress=True)
    print(pca.components_)
    
    # 2. Sparse PCA (Zou et al.)
    # Let's try a very large alpha to force sparsity.
    spca = ZouSparsePCA(n_components=2, alpha=5000, lambda_l2=1e-3, verbose=False)
    spca.fit(X)
    print("\nZou Sparse PCA Components (top 2):")
    print(spca.components_)
    
    # Check sparsity
    sparsity = np.mean(spca.components_ == 0)
    print(f"\nSparsity: {sparsity:.1%}")

if __name__ == "__main__":
    run_zou_example_1()