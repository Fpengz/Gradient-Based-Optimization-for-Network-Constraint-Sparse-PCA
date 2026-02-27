# AUDIT_REPORT_CONVERGENCE

## Baseline PG (theorem-scoped)
- Step policy: spectral `eta=1/L_hat` plus descent-preserving backtracking.
- Objective monotonicity observed in audit runs.
- Proximal-gradient residual decreases consistently.

## MASPG-CAR (practical variant)
- Correctly treated as heuristic acceleration, not theorem-certified.
- Smooth-majorization backtracking and monotone restart safeguards are implemented.

## Claims vs Reality
- **Proven in paper:** baseline PG critical-point convergence under assumptions.
- **Observed in implementation:** descent + convergence diagnostics align with assumptions on tested regimes.
- **Not formally proven here:** MASPG-CAR global convergence theorem (correctly not claimed).

## Empirical Diagnostics

| method | monotone_objective_rate | converged_rate | residual_ratio_mean | n_iter_mean |
| --- | --- | --- | --- | --- |
| NetSPCA-MASPG-CAR | 1.0000 | 1.0000 | 0.0616 | 6.0000 |
| NetSPCA-PG | 1.0000 | 1.0000 | 0.0113 | 9.5000 |