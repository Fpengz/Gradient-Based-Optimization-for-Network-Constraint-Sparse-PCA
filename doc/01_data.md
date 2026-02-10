# Data Sources and Benchmark Datasets

This project utilizes a mix of classic statistical benchmarks, real-world biological data, and structured synthetic models to evaluate Sparse PCA and Network-Constrained SPCA.

---

## 1. Classic Statistical Benchmarks

### Pitprop Dataset
- **Description**: a 13x13 correlation matrix derived from 180 observations of timber properties (pitprops) used in mining.
- **Source**: Jeffers (1967), popularized in SPCA literature by Zou et al. (2006).
- **Purpose**: Test the ability to recover interpretable, sparse components on a small, well-understood dataset.
- **Location**: `data/pitprop.py`

### Zou Example 1 (Synthetic)
- **Description**: 10 variables with 3 underlying hidden factors. Features $X_1-X_4$ depend on Factor 1, $X_5-X_8$ on Factor 2, and $X_9-X_{10}$ on Factor 3.
- **Source**: Zou, Hastie, & Tibshirani (2006).
- **Purpose**: Verify that the SPCA algorithm can isolate the primary sparse loadings (e.g., $X_1-X_4$ for PC1).
- **Location**: `data/synthetic/synthetic_data.py` (`generate_zou_example_1`)

---

## 2. Real-World Biological Data

### Colon Cancer Dataset (Alon et al. 1999)
- **Description**: Gene expression data for 2000 genes across 62 samples (40 tumor, 22 normal tissues).
- **Source**: Alon et al. (1999); sourced via the `alxiang/Colon-Cancer-Classification` repository.
- **Purpose**: High-dimensional ($p \gg n$) feature selection benchmark for biological interpretability.
- **Location**: `data/colon_x.csv` (Raw data), `experiments/colon_benchmark.py` (Preprocessing & Test).

---

## 3. High-Dimensional Synthetic Models

### Spiked Covariance Model (Qiu et al.)
- **Description**: $p=500, n=200$. The covariance matrix $\Sigma$ is constructed with two sparse dominant eigenvectors (spikes).
- **Source**: Qiu, Lei, & Roeder (2023).
- **Purpose**: Evaluate **subspace recovery** accuracy (subspace distance) in high-dimensional settings where standard PCA is inconsistent.
- **Location**: `data/synthetic/qiu_synthetic.py`

---

## 4. Structured Feature Graphs (NC-SPCA)

For **Network-Constrained SPCA**, we generate data where the sparse support is aligned with a feature graph $G$.

### Graph Families
- **Chain Graph**: Features are connected in a linear sequence. Tests contiguous support recovery.
- **2D Grid Graph**: Features are nodes on a grid. Tests spatial smoothness.
- **Random Geometric Graph (RGG)**: Connects features based on spatial proximity.
- **Stochastic Block Model (SBM)**: Connects features based on community/cluster membership.

### Data Generation Logic
1. Construct Laplacian $L$ for the chosen graph.
2. Select a connected subgraph as the ground truth support $S^\star$.
3. Generate data $X$ using a spiked covariance model with support $S^\star$.
- **Location**: `data/synthetic/graph_data.py`

---

## 5. Summary Table

| Dataset | Type | Dimension ($p$) | Samples ($n$) | Primary Test |
| :--- | :--- | :--- | :--- | :--- |
| Pitprop | Matrix | 13 | 180 | Loading Interpretability |
| Zou Ex 1 | Synthetic | 10 | 1000 | Basic Sparsity Recovery |
| Qiu Spiked | Synthetic | 500 | 200 | Subspace Recovery Accuracy |
| Alon Colon | Real | 2000 | 62 | Genomic Feature Selection |
| Chain/Grid | Synthetic | Variable | Variable | Graph-Consistent Support |