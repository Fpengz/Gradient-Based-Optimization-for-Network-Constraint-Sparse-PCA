# Baselines, Ablations, and Experimental Protocol (Publication-Ready)

This file defines the exact baselines, ablations, and reporting protocol
needed to make the paper publishable and reviewer-proof.

---

## 1. Baselines (Minimum Required)

### (B1) PCA (unconstrained)
Solve:

$$
\max_{\|w\|_2=1} w^\top \hat\Sigma w
$$

Report:
- explained variance
- runtime
- support size is full (no sparsity)

---

### (B2) L1-SPCA (no graph)
Solve:

$$
\max_{\|w\|_2 \le 1} w^\top \hat\Sigma w - \lambda_1\|w\|_1
$$

This isolates the effect of **sparsity without network constraints**.

---

### (B3) Graph-regularized PCA (no sparsity)
Solve:

$$
\max_{\|w\|_2 \le 1} w^\top \hat\Sigma w - \lambda_2 w^\top L w
$$

This isolates the effect of **network smoothing without feature selection**.

---

### (B4) Your Method: Graph + Sparsity (main model)
Solve:

$$
\max_{\|w\|_2 \le 1}
w^\top \hat\Sigma w
- \lambda_1\|w\|_1
- \lambda_2 w^\top L w
$$

---

## 2. Core Ablations (What Reviewers Want)

### (A1) Remove graph term
Set:

$$
\lambda_2 = 0
$$

Expected:
- sparsity still works
- connectivity of support decreases

---

### (A2) Remove sparsity term
Set:

$$
\lambda_1 = 0
$$

Expected:
- smoother weights
- feature selection collapses (dense support)

---

### (A3) Graph quality stress test
Perturb the graph by dropping edges with rate $\rho$:

$$
\rho \in \{0, 0.1, 0.2, 0.3\}
$$

Expected:
- graceful degradation
- still better than L1-SPCA if structure is partially preserved

---

### (A4) Initialization sensitivity
Compare:
- random init
- PCA init (top eigenvector)
- warm-start from L1-SPCA solution

Expected:
- PCA init is stable
- warm-start improves convergence speed

---

## 3. Hyperparameter Sweeps (Must-Have)

### Sparsity sweep ($\lambda_1$)
Try log grid:

$$
\lambda_1 \in \{10^{-4},10^{-3},\dots,10^1\}
$$

Report:
- support size
- F1 score
- explained variance

---

### Graph sweep ($\lambda_2$)
Try:

$$
\lambda_2 \in \{0, 10^{-3}, 10^{-2}, 10^{-1}, 1\}
$$

Report:
- LCC ratio
- smoothness score $\hat w^\top L \hat w$

---

## 4. Metrics (Report These Always)

### Explained variance
If $\|w\|_2=1$:

$$
\mathrm{VarExplained}(w) = w^\top \hat\Sigma w
$$

---

### Feature selection
Use:
- precision / recall / F1
- support size error

---

### Graph structure
Use:
- LCC ratio
- smoothness score $\hat w^\top L \hat w$

---

### Efficiency
- runtime
- iterations to convergence
- scaling with $p$

---

## 5. Experimental Protocol (Reproducible)

For each configuration:
1. sample graph $G$ and Laplacian $L$
2. sample ground truth support $S^\star$ and weights $w^\star$
3. generate data $X$
4. compute $\hat\Sigma$
5. run all baselines with matched stopping criteria
6. log metrics and save $\hat w$ and $\hat S$

---

## 6. Tables and Figures to Include (Paper Checklist)

### Table 1: Main comparison
Rows: methods (PCA / L1-SPCA / Graph-PCA / Yours)  
Columns:
- explained variance
- F1
- LCC ratio
- runtime

### Figure 1: Sparsity-performance tradeoff
x-axis: $|\hat S|$  
y-axis: explained variance or F1

### Figure 2: Connectivity vs $\lambda_2$
x-axis: $\lambda_2$  
y-axis: LCC ratio

### Figure 3: Robustness to graph perturbation
x-axis: $\rho$  
y-axis: F1 and LCC ratio

---

## 7. What You Can Claim (If Results Support It)

Safe claims:
- graph regularization improves **connected feature selection**
- better structure recovery than L1-SPCA
- stable convergence with proximal gradient
- robustness under mild graph corruption

Avoid:
- “globally optimal”
- “provably best feature selection” without strong theory
