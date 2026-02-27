# ABLATION_TABLES

## Continuation Path Ablation

| naive_total_runtime_sec | warm_total_runtime_sec | runtime_speedup_x | naive_total_iters | warm_total_iters |
| --- | --- | --- | --- | --- |
| 0.2102 | 0.1470 | 1.4301 | 136.0000 | 91.0000 |

## PG vs MASPG-CAR Convergence Ablation

| method | monotone_objective_rate | converged_rate | residual_ratio_mean | n_iter_mean |
| --- | --- | --- | --- | --- |
| NetSPCA-MASPG-CAR | 1.0000 | 1.0000 | 0.0616 | 6.0000 |
| NetSPCA-PG | 1.0000 | 1.0000 | 0.0113 | 9.5000 |