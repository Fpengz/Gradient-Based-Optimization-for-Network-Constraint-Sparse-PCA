import numpy as np

def generate_qiu_synthetic(n_samples=200, p=500, seed=42):
    """
    Generates synthetic data from the simulation setting in Qiu et al. (2023).
    
    Parameters
    ----------
    n_samples : int, default=200
        Number of samples.
    p : int, default=500
        Number of features.
    seed : int, default=42
        Random seed.
        
    Returns
    -------
    X : ndarray of shape (n_samples, p)
        Generated data.
    Sigma : ndarray of shape (p, p)
        True covariance matrix.
    """
    np.random.seed(seed)
    
    v1 = np.zeros(p)
    v1[0:10] = 1.0 / np.sqrt(10)
    
    v2 = np.zeros(p)
    v2[10:20] = 1.0 / np.sqrt(10)
    
    # Spiked covariance: Sigma = 10 v1 v1^T + 5 v2 v2^T + I
    Sigma = 10 * np.outer(v1, v1) + 5 * np.outer(v2, v2) + np.eye(p)
    
    # Generate data: X ~ N(0, Sigma)
    X = np.random.multivariate_normal(np.zeros(p), Sigma, size=n_samples)
    
    return X, Sigma

if __name__ == "__main__":
    X, Sigma = generate_qiu_synthetic()
    print("X shape:", X.shape)
    print("Sigma shape:", Sigma.shape)
    print("First 10 vars v1 contribution in Sigma diag:", np.diag(Sigma)[:10])
    print("Next 10 vars v2 contribution in Sigma diag:", np.diag(Sigma)[10:20])
