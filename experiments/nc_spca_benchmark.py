import numpy as np
from data.synthetic.graph_data import generate_structured_spiked_covariance
from src.models.network_sparse_pca import NetworkSparsePCA
from src.models.sparse_pca import ZouSparsePCA
from sklearn.decomposition import PCA
import time

def calculate_lcc_ratio(w, L):
    """
    Calculates the ratio of the size of the largest connected component 
    to the total number of non-zero elements in w.
    """
    support = np.where(np.abs(w) > 1e-3)[0]
    if len(support) == 0:
        return 0.0
    
    # Adjacency matrix from Laplacian
    # L = D - W => W = D - L. Off-diagonal of -L is W.
    W = -L.copy()
    np.fill_diagonal(W, 0)
    W_sub = W[support, :][:, support]
    
    # Use BFS/DFS to find connected components
    visited = set()
    components = []
    
    for i in range(len(support)):
        if i not in visited:
            component = []
            stack = [i]
            while stack:
                node = stack.pop()
                if node not in visited:
                    visited.add(node)
                    component.append(node)
                    # neighbors in W_sub
                    neighbors = np.where(W_sub[node] > 0)[0]
                    for neighbor in neighbors:
                        if neighbor not in visited:
                            stack.append(neighbor)
            components.append(component)
            
    if not components:
        return 0.0
    max_size = max(len(c) for c in components)
    return max_size / len(support)

def run_nc_spca_benchmark():
    print("Running NC-SPCA Benchmark (Paper Section 7)...")
    p = 100
    X, L, w_true = generate_structured_spiked_covariance(p, n_samples=200, graph_type='chain', seed=42)
    true_support = set(np.where(w_true != 0)[0])
    
    methods = {
        "PCA": PCA(n_components=1),
        "L1-SPCA (Zou)": ZouSparsePCA(n_components=1, alpha=20, lambda_l2=1e-3),
        "Graph-PCA": NetworkSparsePCA(n_components=1, lambda1=0.0, lambda2=5.0),
        "NC-SPCA (Ours)": NetworkSparsePCA(n_components=1, lambda1=0.2, lambda2=5.0, learning_rate=0.01, max_iter=2000)
    }
    
    print(f"{'Method':<20} | {'Exp.Var':<8} | {'F1':<8} | {'LCC Ratio':<10} | {'Time(s)':<8}")
    print("-" * 65)
    
    Sigma = (X.T @ X) / X.shape[0]
    
    for name, model in methods.items():
        start = time.time()
        if name == "PCA":
            model.fit(X)
            w = model.components_[0]
        elif "Zou" in name:
            model.fit(X)
            w = model.components_[0]
        else:
            model.fit(X, L=L)
            w = model.components_[0]
        duration = time.time() - start
        
        w = w / (np.linalg.norm(w) + 1e-12)
        var_exp = w.T @ Sigma @ w
        
        support = set(np.where(np.abs(w) > 1e-3)[0])
        if len(support) > 0:
            precision = len(support & true_support) / len(support)
            recall = len(support & true_support) / len(true_support)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        else:
            f1 = 0.0
            
        lcc_ratio = calculate_lcc_ratio(w, L)
        
        print(f"{name:<20} | {var_exp:<8.3f} | {f1:<8.3f} | {lcc_ratio:<10.1%} | {duration:<8.4f}")

if __name__ == "__main__":
    run_nc_spca_benchmark()