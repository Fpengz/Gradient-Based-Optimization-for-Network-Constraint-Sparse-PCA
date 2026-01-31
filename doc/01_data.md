# Data Generation and Feature Graph Construction

## 1. Synthetic Data Setup

We generate synthetic datasets for testing PCA, SPCA, and network-constrained SPCA.

```python
import numpy as np

def generate_synthetic_data(n_samples=100, n_features=50, n_active=5, noise_std=0.1, seed=42):
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
    X = z @ w_true.reshape(1, -1) + noise_std * np.random.randn(n_samples, n_features)
    return X, w_true

# Example usage
X, w_true = generate_synthetic_data()
print("Data shape:", X.shape)
print("Non-zero entries in ground truth:", np.sum(w_true != 0))
