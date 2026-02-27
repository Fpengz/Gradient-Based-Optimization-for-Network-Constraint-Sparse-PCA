# AUDIT_REPORT_METHOD

## Scope
Equation-to-code audit against the LaTeX source objective, PG baseline, and MASPG-CAR practical variant.

## Compliance Checks
1. **Objective signs:** compliant.
   - Max form `w^T Sigmahat w - lambda1||w||1 - lambda2 w^T L w` is implemented as minimization with negative variance and positive regularizers.
2. **Gradient formula:** compliant.
   - `∇f(w) = -2 Sigmahat w + 2 lambda2 L w` implemented exactly (implicit `Sigmahat`).
3. **Prox step:** compliant.
   - `s = soft(v, eta*lambda1)`, `w_next = s / max(1, ||s||2)`.
4. **Constraint handling:** compliant.
   - Unit l2-ball (`<=1`) projection used.
5. **Covariance scaling:** compliant.
   - Uses `X^T(Xw)/n`, matching `Sigmahat = X^T X / n`.
6. **Laplacian consistency:** compliant.
   - Optimization and metrics use same Laplacian object from graph utilities.

## Mismatches Found and Fixed
1. Missing RGG family in synthetic protocol -> **fixed** (`random_geometric_graph`, `graph_type='rgg'`).
2. MASPG-CAR backtracking lacked explicit smooth-majorization check -> **fixed**.
3. No PG residual trace for convergence diagnostics -> **fixed** (`pg_residual_history_by_component`).

## Verification
- Unit/integration tests: `27 passed`.
- No failure events in current benchmark batch.
