# FINAL_CERTIFICATION

## 1) Method adherence score
**93/100**

## 2) Convergence compliance
- Baseline PG theorem conditions: **Yes (empirically satisfied in audit runs)**
- MASPG-CAR correctly scoped as practical heuristic: **Yes**

## 3) Stability assessment
- Failure rate in V0/V1 benchmark batches: **0%**
- Residual and objective curves are finite and monotone in tested settings.

## 4) Performance summary

### V0
| method | f1_mean | f1_topk_mean | explained_variance_mean | lcc_ratio_mean | runtime_sec_mean | converged_rate |
| --- | --- | --- | --- | --- | --- | --- |
| L1-SPCA-ProxGrad | 0.4028 | 0.8708 | 9.8574 | 0.9979 | 0.0074 | 1.0000 |
| NetSPCA-MASPG-CAR | 0.3971 | 0.8458 | 9.7973 | 0.9947 | 0.0090 | 1.0000 |
| NetSPCA-PG | 0.3967 | 0.8458 | 9.7986 | 0.9947 | 0.0094 | 1.0000 |
| PCA | 0.3371 | 0.8708 | 9.8991 | 1.0000 | 0.0027 | 1.0000 |
| Graph-PCA | 0.3355 | 0.8458 | 9.8301 | 1.0000 | 0.0092 | 1.0000 |

### V1
| method | f1_mean | f1_topk_mean | explained_variance_mean | lcc_ratio_mean | runtime_sec_mean | converged_rate |
| --- | --- | --- | --- | --- | --- | --- |
| L1-SPCA-ProxGrad | 0.4188 | 0.8208 | 9.3666 | 1.0000 | 0.0076 | 1.0000 |
| NetSPCA-MASPG-CAR | 0.4033 | 0.7917 | 9.2205 | 1.0000 | 0.0087 | 1.0000 |
| NetSPCA-PG | 0.4029 | 0.7917 | 9.2217 | 1.0000 | 0.0094 | 1.0000 |
| PCA | 0.3378 | 0.8292 | 9.4104 | 1.0000 | 0.0028 | 1.0000 |
| Graph-PCA | 0.3352 | 0.7917 | 9.2340 | 1.0000 | 0.0096 | 1.0000 |

## 5) Tuning guide
- Start with `lambda1 in [0.1, 0.3]`, `lambda2 in [0.05, 0.25]`.
- Use `fit_path(...)` continuation for grid search (measured speedup ~1.43x).
- Prefer PG for theorem-aligned analysis; MASPG-CAR for practical iteration reduction.

## 6) Remaining risks / TODOs
- Add larger-scale stress (`p >= 2000`).
- Add explicit stationarity metric export to CLI outputs.
- Extend misspecification sweep to multiple perturbation rates by default.

## Verdict
**Partially compliant (high)**: theorem-scoped PG is aligned; MASPG-CAR remains empirical by design.
