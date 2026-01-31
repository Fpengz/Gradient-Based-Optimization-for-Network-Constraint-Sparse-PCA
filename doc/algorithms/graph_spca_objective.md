# Graph-Constrained SPCA Objective + Gradients (Publication-Ready)

This file finalizes the objective, gradient expressions, and practical optimization details
for **gradient-based network-constrained SPCA**.

---

## 1. Notation

- Data matrix: $X \in \mathbb{R}^{n \times p}$
- Empirical covariance:  

$$
\hat\Sigma = \frac{1}{n}X^\top X
$$

- Feature graph: $G=(V,E)$ with $|V|=p$
- Graph Laplacian: $L \in \mathbb{R}^{p\times p}$
- Loading vector: $w \in \mathbb{R}^p$

---

## 2. Core Objective (Single Component)

A standard network-constrained SPCA objective is:

$$
\max_{w \in \mathbb{R}^p}\;\; w^\top \hat\Sigma w
\;-\; \lambda_1 \|w\|_1
\;-\; \lambda_2 w^\top L w
\quad \text{s.t. } \|w\|_2 \le 1
$$

### Interpretation
- $w^\top \hat\Sigma w$: explained variance
- $\lambda_1\|w\|_1$: sparsity / feature selection
- $\lambda_2 w^\top L w$: graph smoothness / network structure
- $\|w\|_2\le 1$: prevents scale blow-up

---

## 3. Minimization Form (for Prox-Gradient)

To use proximal gradient descent, rewrite as minimization:

$$
\min_{\|w\|_2 \le 1}\;\;
F(w) := f(w) + g(w)
$$

with

### Smooth term
$$
f(w) := - w^\top \hat\Sigma w + \lambda_2 w^\top L w
$$

### Non-smooth + constraints term
$$
g(w) := \lambda_1\|w\|_1 + \iota_{\|w\|_2 \le 1}(w)
$$

where $\iota_{\mathcal{C}}(w)$ is the indicator function of set $\mathcal{C}$:
- $0$ if $w \in \mathcal{C}$
- $+\infty$ otherwise

---

## 4. Gradient of the Smooth Part

Since $\hat\Sigma$ and $L$ are symmetric:

$$
\nabla \left(- w^\top \hat\Sigma w\right) = -2\hat\Sigma w
$$

$$
\nabla \left(\lambda_2 w^\top L w\right) = 2\lambda_2 L w
$$

So:

$$
\nabla f(w) = -2\hat\Sigma w + 2\lambda_2 L w
$$

---

## 5. Lipschitz Constant (Step Size Selection)

We want $\nabla f$ to be $L_f$-Lipschitz:

$$
\|\nabla f(u) - \nabla f(v)\|_2 \le L_f \|u-v\|_2
$$

Because $\nabla f(w)$ is linear in $w$, a valid Lipschitz constant is:

$$
L_f = 2\|\,-\hat\Sigma + \lambda_2 L\,\|_2
$$

A safe step size is:

$$
\eta = \frac{1}{L_f}
$$

In practice, use:
- backtracking line search, or
- conservative $\eta$ for stability

---

## 6. Proximal Step (Key Implementation Detail)

The proximal gradient update is:

$$
w^{k+1} = \operatorname{prox}_{\eta g}\left(w^k - \eta \nabla f(w^k)\right)
$$

Let:

$$
v^k = w^k - \eta \nabla f(w^k)
$$

Then:

### Step 1: soft-thresholding (L1 prox)
$$
u^k = \operatorname{soft}(v^k, \eta\lambda_1)
$$

componentwise:

$$
(u^k)_i = \operatorname{sign}(v^k_i)\max(|v^k_i|-\eta\lambda_1, 0)
$$

### Step 2: projection onto $\ell_2$ ball
$$
w^{k+1} =
\begin{cases}
u^k, & \|u^k\|_2 \le 1 \\
\frac{u^k}{\|u^k\|_2}, & \text{otherwise}
\end{cases}
$$

This makes the method stable and scale-controlled.

---

## 7. Stopping Criteria (Publication Standard)

Stop when one of the following holds:

### Relative change small
$$
\frac{\|w^{k+1}-w^k\|_2}{\max(1,\|w^k\|_2)} \le \epsilon
$$

### Objective decrease small
$$
\frac{|F(w^{k+1})-F(w^k)|}{\max(1,|F(w^k)|)} \le \epsilon
$$

### Max iterations reached
$$
k \ge K_{\max}
$$

---

## 8. Multi-Component Extension (Optional)

To extract multiple sparse PCs:
- Deflation approaches, or
- block optimization with orthogonality constraints

However, for the paper’s core contribution, **single-component** is often enough
and easier to analyze + evaluate structure recovery.

---

## 9. What to Claim (Safe + Strong)

You can claim:
- scalable optimization via proximal gradient
- convergence to stationary points (standard)
- improved structured feature selection vs L1-only SPCA
- robustness to graph perturbations (if experiments confirm)

Avoid claiming global optimality.
