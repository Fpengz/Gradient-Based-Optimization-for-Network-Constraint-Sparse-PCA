# Implementation Review: Gradient-Based SPCA Suite

## 1. Algorithm Correctness & Fidelity

### NC-SPCA (`NetworkSparsePCA`)
- **Mathematical Alignment**: The implementation now correctly aligns with the proximal gradient framework for the objective:
  $$\min_{\|w\|_2 \le 1} -w^	op \hat\Sigma w + \lambda_1 \|w\|_1 + \lambda_2 w^	op L w$$
- **Fidelity**: The use of soft-thresholding followed by $L_2$ projection is the correct proximal operator for the combined $L_1$ penalty and $L_2$ ball constraint.
- **Improvements Made**:
    - **Initialization**: Added PCA initialization to avoid poor local minima.
    - **Scalability**: Optimized gradient computation from $O(p^2)$ to $O(np)$ per iteration.
    - **Robustness**: Added automatic Lipschitz-based step size estimation.

### Zou-SPCA (`ZouSparsePCA`)
- **Mathematical Alignment**: Correctly implements the alternating minimization (Elastic Net regression + Procrustes projection) as described in Zou et al. (2006).
- **Fidelity**: Uses `sklearn.linear_model.ElasticNet` which is a robust solver for the subproblem.
- **Discrepancy**: The normalization of loadings $B$ follows the common practice of unit-norm vectors, but the original paper has a "v-style" and "beta-style" normalization which can differ in scale.

### GradFPS (`GradFPS`)
- **Mathematical Alignment**: Implements the Fantope Projection and Selection correctly.
- **Fidelity**: The eigenvalue clipping for Fantope projection $\{X : 0 \preceq X \preceq I, 	ext{tr}(X) = d\}$ is implemented via a root-finding method for the Lagrange multiplier, which is the standard approach.

### GPM (`GeneralizedPowerMethod`)
- **Mathematical Alignment**: Correctly implements the single-unit GPower algorithm from Journée et al. (2010).

---

## 2. Computational Complexity Analysis

| Algorithm | Per-Iteration Complexity | Scalability |
|-----------|--------------------------|-------------|
| **NC-SPCA** | $O(np + |E|)$ | High (linear in $p$ if $L$ is sparse) |
| **Zou-SPCA**| $O(n p^2)$ (due to ENET) | Medium |
| **GradFPS** | $O(p^3)$ (due to EVD) | Low |
| **GPM**     | $O(np)$ | High |

*Note: $|E|$ is the number of edges in the feature graph.*

---

## 3. Optimization Opportunities

1. **NC-SPCA Step Size**: While `auto` estimation provides a safe upper bound ($L_f \le 2\|\Sigma\|_2 + 2\lambda_2 \|L\|_2$), it can be conservative. **Backtracking line search** would allow for larger steps and faster convergence.
2. **Sparse Laplacian**: Ensure that $L$ is always passed as a `scipy.sparse` matrix for high-dimensional feature graphs to maintain $O(|E|)$ complexity.
3. **Deflation**: For multi-component SPCA, the current deflation $X \leftarrow X - Xww^	op$ does not guarantee the best sparse basis. **Projection-based deflation** (Mackey, 2009) or **simultaneous optimization** (block-coordinate descent on the Stiefel manifold) could improve results for $k > 1$.
4. **GradFPS**: For large $p$, the full EVD in every iteration is the bottleneck. If $d \ll p$, one could theoretically use a **truncated EVD**, but the trace constraint $	ext{tr}(X)=d$ technically requires knowing all eigenvalues. However, for the $L_1$-regularized case, many eigenvalues of the optimal $X$ are 0 or 1, which might be exploitable.

---

## 4. Documentation of Discrepancies

- **NC-SPCA Parameter Scaling**: In the current benchmarks, `lambda1` in NC-SPCA and `alpha` in Zou-SPCA are not on the same scale due to different normalization factors in the loss functions. This makes direct comparisons of "sparsity" vs "hyperparameter value" misleading.
- **Convergence Criteria**: NC-SPCA currently uses a relative change in the loading vector. For some ill-conditioned problems, this might trigger early stopping before the objective has fully stabilized. Adding a check for the relative change in the objective value is recommended.

## 5. Suggested Refactoring

- **Base Class**: Create a `BaseSparsePCA` class to share `_soft_threshold`, `transform`, and common validation logic.
- **Logging**: Implement a more unified logging/verbosity system across all models.
- **Warm Starts**: Allow `fit` to take an initial `w` for warm-starting optimization in streaming or cross-validation scenarios.
