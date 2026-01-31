# Synthetic Graph & Data Generators (Publication-Ready)

This file defines reproducible synthetic benchmarks for **network-constrained SPCA**.

---

## 1. Goals of Synthetic Experiments

We want controlled evaluation of:
- **Structure recovery** (connected / community-aligned supports)
- **Feature selection accuracy**
- **Stability under noise and graph perturbations**
- **Scaling with dimension** $p$ and samples $n$

---

## 2. Graph Families (Feature Networks)

Let $p$ be the number of features/nodes.

### (G1) Chain Graph
- Nodes: $1,2,\dots,p$
- Edges: $(i,i+1)$
- Tests **contiguous support recovery**

### (G2) 2D Grid Graph
- Arrange features on a $\sqrt{p}\times\sqrt{p}$ grid
- 4-neighbor adjacency
- Tests **spatial smoothness** and local connectivity

### (G3) Random Geometric Graph (RGG)
- Sample node coordinates $x_i\in[0,1]^2$
- Add edge if $\|x_i-x_j\|_2 \le r$
- Tests **locality-biased** structure selection

### (G4) Stochastic Block Model (SBM)
- Partition nodes into $K$ blocks
- High intra-block probability $p_{in}$, low inter-block $p_{out}$
- Tests **community-consistent** feature selection

---

## 3. Laplacian Construction

Given adjacency matrix $A$:

$$
D = \mathrm{diag}(A\mathbf{1}),\quad L = D - A
$$

Optional normalized Laplacian:

$$
L_{\mathrm{sym}} = I - D^{-1/2} A D^{-1/2}
$$

---

## 4. Ground Truth Loading Vector $w^\star$

### Connected Support Sampling (recommended)
Goal: ensure the *true* support is graph-connected.

1. Choose a seed node $s$
2. BFS/DFS expand until support size $|S^\star|=k$
3. Set:

$$
w^\star_i \sim \mathcal{N}(0,1) \text{ for } i\in S^\star,\quad w^\star_i=0 \text{ otherwise}
$$

4. Normalize:

$$
\|w^\star\|_2 = 1
$$

---

## 5. Data Generation

### Option A: Rank-1 Signal + Noise (most intuitive)
Generate:

$$
X = z (w^\star)^\top + E
$$

where:
- $z\in\mathbb{R}^{n}$, $z\sim \mathcal{N}(0,I)$
- $E\in\mathbb{R}^{n\times p}$, iid $\mathcal{N}(0,\sigma^2)$

Then estimate covariance:

$$
\hat\Sigma = \frac{1}{n}X^\top X
$$

### Option B: Covariance Model (clean theory)
Construct:

$$
\Sigma = \alpha \, w^\star(w^\star)^\top + \sigma^2 I
$$

---

## 6. Experimental Sweeps (Recommended)

### Dimensions and samples
- $p\in\{200, 500, 1000, 5000\}$
- $n\in\{100, 200, 500\}$

### Sparsity and noise
- $k\in\{10, 20, 50\}$
- $\sigma\in\{0.1, 0.5, 1.0\}$

### Graph perturbations
Edge drop rate:

$$
\rho \in \{0, 0.1, 0.2\}
$$

---

## 7. What to Report

### Performance
- Explained variance
- Support precision/recall/F1
- Connectivity score (LCC ratio)
- Smoothness score $\hat w^\top L \hat w$

### Efficiency
- Runtime vs $p$
- Number of iterations to convergence
- Sensitivity to step size $\eta$

### Hyperparameter sensitivity
- $\lambda_1$ sweep (sparsity)
- $\lambda_2$ sweep (graph regularization)

---

## 8. Reproducibility Checklist
- Fix random seeds
- Log graph type + parameters
- Log all hyperparameters
- Save $w^\star$, $\hat w$, and support sets
- Save adjacency/Laplacian used in each run
