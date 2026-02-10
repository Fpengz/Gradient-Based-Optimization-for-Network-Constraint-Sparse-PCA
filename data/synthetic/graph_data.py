import numpy as np
import scipy.sparse as sp

def get_chain_graph(p):
    """Returns the Laplacian of a chain graph with p nodes."""
    W = np.zeros((p, p))
    for i in range(p - 1):
        W[i, i + 1] = 1.0
        W[i + 1, i] = 1.0
    D = np.diag(np.sum(W, axis=1))
    return D - W

def get_grid_graph(m, n):
    """Returns the Laplacian of an m x n grid graph."""
    p = m * n
    W = np.zeros((p, p))
    for i in range(m):
        for j in range(n):
            node = i * n + j
            if i < m - 1: # connect to node below
                neighbor = (i + 1) * n + j
                W[node, neighbor] = 1.0
                W[neighbor, node] = 1.0
            if j < n - 1: # connect to node right
                neighbor = i * n + (j + 1)
                W[node, neighbor] = 1.0
                W[neighbor, node] = 1.0
    D = np.diag(np.sum(W, axis=1))
    return D - W

def get_rgg_graph(p, radius=0.2, seed=42):
    """Returns the Laplacian of a Random Geometric Graph."""
    np.random.seed(seed)
    pos = np.random.rand(p, 2)
    W = np.zeros((p, p))
    for i in range(p):
        for j in range(i + 1, p):
            dist = np.linalg.norm(pos[i] - pos[j])
            if dist <= radius:
                W[i, j] = 1.0
                W[j, i] = 1.0
    D = np.diag(np.sum(W, axis=1))
    return D - W

def generate_structured_spiked_covariance(p, n_samples=200, graph_type='chain', seed=42):
    """
    Generates data with a spiked covariance where the spike has structured support.
    """
    np.random.seed(seed)
    if graph_type == 'chain':
        L = get_chain_graph(p)
        # Structured support: first 20 nodes are a connected block in a chain
        w_true = np.zeros(p)
        w_true[0:20] = 1.0 / np.sqrt(20)
    elif graph_type == 'grid':
        m = int(np.sqrt(p))
        L = get_grid_graph(m, m)
        p = m * m
        # Structured support: a 4x5 block in the grid
        w_true = np.zeros(p)
        for i in range(4):
            for j in range(5):
                w_true[i * m + j] = 1.0
        w_true /= np.linalg.norm(w_true)
    else:
        L = np.eye(p)
        w_true = np.zeros(p)
        w_true[:20] = 1.0 / np.sqrt(20)

    # Sigma = 10 * w_true @ w_true^T + I
    Sigma = 10 * np.outer(w_true, w_true) + np.eye(p)
    X = np.random.multivariate_normal(np.zeros(p), Sigma, size=n_samples)
    
    return X, L, w_true

if __name__ == "__main__":
    X, L, w_true = generate_structured_spiked_covariance(100, graph_type='chain')
    print("X shape:", X.shape)
    print("L shape:", L.shape)
    print("w_true non-zeros:", np.sum(w_true != 0))
