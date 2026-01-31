# Convergence Analysis Skeleton (Publication-Ready)

This note provides a clean theorem/proof skeleton suitable for inclusion in a paper appendix.

---

## 1. Optimization Problem

We consider the composite objective

$$
\min_{w \in \mathbb{R}^p} \; F(w) := f(w) + g(w),
$$

where we typically choose:

### Smooth part
$$
f(w) := - w^\top \Sigma w + \lambda_2 \, w^\top L w
$$

### Non-smooth part
$$
g(w) := \lambda_1 \|w\|_1
$$

> Notes:
> - The sign convention above is written for **minimization**.
> - If you prefer the maximization form, convert by negating the objective.

---

## 2. Assumptions (Standard for Prox-Gradient Theory)

**A1.** Covariance is PSD:  
$$
\Sigma \succeq 0
$$

**A2.** Graph Laplacian is PSD:  
$$
L \succeq 0
$$

**A3.** Smoothness: $f$ is differentiable and $\nabla f$ is $L_f$-Lipschitz:
$$
\|\nabla f(u) - \nabla f(v)\|_2 \le L_f \|u-v\|_2
$$

**A4.** $g$ is proper, lower semicontinuous, and convex (true for $\ell_1$).

**A5.** Step size satisfies:
$$
0 < \eta < \frac{1}{L_f}
$$

---

## 3. Proximal Gradient Method

The proximal gradient update is:

$$
w^{k+1} = \operatorname{prox}_{\eta g}\bigl(w^k - \eta \nabla f(w^k)\bigr)
$$

For $g(w)=\lambda_1\|w\|_1$, the proximal operator is **soft-thresholding**:

$$
(\operatorname{prox}_{\eta\lambda_1\|\cdot\|_1}(v))_i
= \operatorname{sign}(v_i)\max(|v_i| - \eta\lambda_1, 0)
$$

---

## 4. Stationarity (What We Can Prove)

A point $w^\star$ is a **critical/stationary point** if:

$$
0 \in \nabla f(w^\star) + \partial g(w^\star)
$$

This is the correct notion of optimality for non-smooth objectives.

---

## 5. Main Theorem (Template)

**Theorem (Convergence to stationary points).**  
Under Assumptions A1–A5, the iterates $\{w^k\}$ produced by proximal gradient descent satisfy:

### (1) Sufficient descent
There exists a constant $c>0$ such that:

$$
F(w^{k+1}) \le F(w^k) - c\,\|w^{k+1}-w^k\|_2^2
$$

### (2) Asymptotic regularity
$$
\|w^{k+1}-w^k\|_2 \to 0
$$

### (3) Stationarity of limit points
Every accumulation point $\bar w$ of $\{w^k\}$ is stationary:

$$
0 \in \nabla f(\bar w) + \partial g(\bar w)
$$

---

## 6. Proof Roadmap (Reviewer-Friendly)

### Step 1 — Smoothness inequality (descent lemma)
Using Lipschitz continuity of $\nabla f$:

$$
f(u) \le f(v) + \langle \nabla f(v), u-v \rangle + \frac{L_f}{2}\|u-v\|_2^2
$$

### Step 2 — Proximal optimality condition
From the proximal definition:

$$
0 \in \partial g(w^{k+1}) + \frac{1}{\eta}(w^{k+1}-w^k) + \nabla f(w^k)
$$

### Step 3 — Descent for the composite objective
Combine Step 1 and Step 2 to obtain sufficient descent of $F(w^k)$.

### Step 4 — Summability
Summing descent inequalities over $k$ gives:

$$
\sum_{k=0}^{\infty} \|w^{k+1}-w^k\|_2^2 < \infty
$$

which implies $\|w^{k+1}-w^k\|_2 \to 0$.

### Step 5 — Stationarity of limit points
Take limits in the optimality condition using closedness of $\partial g$.

---

## 7. Practical Notes for the Paper

### Adding constraints (common in SPCA)
If you add a constraint like $\|w\|_2 \le 1$, include it via an indicator function:

$$
g(w)=\lambda_1\|w\|_1 + \iota_{\|w\|_2 \le 1}(w)
$$

Then the proximal step becomes:
- soft-thresholding (for $\ell_1$)
- followed by projection onto the $\ell_2$ ball

---

## 8. What to Cite (Standard References)
- Beck & Teboulle (2009): ISTA/FISTA framework  
- Attouch, Bolte, Svaiter (2013): nonconvex proximal splitting convergence  
