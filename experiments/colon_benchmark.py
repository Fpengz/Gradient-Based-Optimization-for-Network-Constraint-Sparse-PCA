import pandas as pd
import numpy as np
from src.models.sparse_pca import ZouSparsePCA
from src.models.grad_fps import GradFPS
from src.models.generalized_power_method import GeneralizedPowerMethod
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time

def run_colon_benchmark():
    print("Loading Alon et al. (1999) Colon Cancer Dataset...")
    try:
        df = pd.read_csv("data/colon_x.csv", index_col=0)
        X = df.values
        print(f"Dataset shape: {X.shape} (Samples x Genes)")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_components = 1
    results = {}

    pca = PCA(n_components=n_components)
    start = time.time()
    pca.fit(X_scaled)
    results['PCA'] = {
        'time': time.time() - start,
        'sparsity': np.mean(pca.components_ == 0)
    }

    spca = ZouSparsePCA(n_components=n_components, alpha=5000, lambda_l2=1e-3)
    start = time.time()
    spca.fit(X_scaled)
    results['Zou SPCA'] = {
        'time': time.time() - start,
        'sparsity': np.mean(spca.components_ == 0)
    }

    gfps = GradFPS(n_components=n_components, rho=5.0, learning_rate=0.01, max_iter=50)
    start = time.time()
    gfps.fit(X_scaled)
    results['GradFPS'] = {
        'time': time.time() - start,
        'sparsity': np.mean(gfps.components_ == 0)
    }

    gpm = GeneralizedPowerMethod(n_components=n_components, gamma=1.0)
    start = time.time()
    gpm.fit(X_scaled)
    results['GPM'] = {
        'time': time.time() - start,
        'sparsity': np.mean(gpm.components_ == 0)
    }

    print("\n=== Colon Cancer Benchmark Results (PC1) ===")
    print(f"{'Method':<15} | {'Sparsity':<10} | {'Time (s)':<10}")
    print("-" * 40)
    for name, res in results.items():
        print(f"{name:<15} | {res['sparsity']:<10.1%} | {res['time']:<10.4f}")

if __name__ == "__main__":
    run_colon_benchmark()