import numpy as np


def generate_synthetic_data(
    n_samples=100, n_features=50, n_active=5, noise_std=0.1, seed=42
):
    """
    Generates synthetic data with a sparse ground truth component.
    :param n_samples: number of samples
    :param n_features: total number of features
    :param n_active: number of features with non-zero loading
    :param noise_std: standard deviation of Gaussian noise
    :return: X (data), w_true (ground truth sparse loadings)
    """
    np.random.seed(seed)
    # ground truth sparse component
    w_true = np.zeros(n_features)
    active_indices = np.random.choice(n_features, n_active, replace=False)
    w_true[active_indices] = np.random.randn(n_active)
    w_true /= np.linalg.norm(w_true)

    # generate data
    z = np.random.randn(n_samples, 1)  # latent component
    X = z @ w_true.reshape(1, -1) + noise_std * np.random.randn(
        n_samples, n_features
    )
    return X, w_true


def generate_zou_example_1(n_samples=1000, seed=42):
    """
    Generates the synthetic data from Example 1 of Zou et al. (2006).
    The data has 10 variables and 3 hidden factors.
    
    Parameters
    ----------
    n_samples : int, optional
        Number of samples to generate.
    seed : int, optional
        Random seed.
        
    Returns
    -------
    X : ndarray of shape (n_samples, 10)
        The generated data.
    """
    np.random.seed(seed)
    # Three hidden factors
    v1 = np.random.normal(0, np.sqrt(290), n_samples)
    v2 = np.random.normal(0, np.sqrt(300), n_samples)
    v3 = -0.3 * v1 + 0.925 * v2 + np.random.normal(0, 1, n_samples)
    
    X = np.zeros((n_samples, 10))
    # X1-X4 are v1 + noise
    for i in range(4):
        X[:, i] = v1 + np.random.normal(0, 1, n_samples)
    # X5-X8 are v2 + noise
    for i in range(4, 8):
        X[:, i] = v2 + np.random.normal(0, 1, n_samples)
    # X9-X10 are v3 + noise
    for i in range(8, 10):
        X[:, i] = v3 + np.random.normal(0, 1, n_samples)
        
    return X


# Example usage
X, w_true = generate_synthetic_data()
print("Data shape:", X.shape)
print("Non-zero entries in ground truth:", np.sum(w_true != 0))

X_zou = generate_zou_example_1()
print("Zou Example 1 shape:", X_zou.shape)
