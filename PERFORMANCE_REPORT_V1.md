# PERFORMANCE_REPORT_V1

Audit artifact directory: `results/paper_audit_20260227-111514`

## Aggregated performance under graph misspecification (rate=0.1)

| method | f1_mean | f1_topk_mean | explained_variance_mean | lcc_ratio_mean | runtime_sec_mean | converged_rate |
| --- | --- | --- | --- | --- | --- | --- |
| L1-SPCA-ProxGrad | 0.4188 | 0.8208 | 9.3666 | 1.0000 | 0.0076 | 1.0000 |
| NetSPCA-MASPG-CAR | 0.4033 | 0.7917 | 9.2205 | 1.0000 | 0.0087 | 1.0000 |
| NetSPCA-PG | 0.4029 | 0.7917 | 9.2217 | 1.0000 | 0.0094 | 1.0000 |
| PCA | 0.3378 | 0.8292 | 9.4104 | 1.0000 | 0.0028 | 1.0000 |
| Graph-PCA | 0.3352 | 0.7917 | 9.2340 | 1.0000 | 0.0096 | 1.0000 |

## Before/After (V0 vs V1 aggregate)

| method | f1_mean_v0 | runtime_sec_mean_v0 | f1_mean_v1 | runtime_sec_mean_v1 | delta_f1 | delta_runtime_sec |
| --- | --- | --- | --- | --- | --- | --- |
| L1-SPCA-ProxGrad | 0.4028 | 0.0074 | 0.4188 | 0.0076 | 0.0159 | 0.0002 |
| NetSPCA-MASPG-CAR | 0.3971 | 0.0090 | 0.4033 | 0.0087 | 0.0062 | -0.0003 |
| NetSPCA-PG | 0.3967 | 0.0094 | 0.4029 | 0.0094 | 0.0062 | -0.0000 |
| PCA | 0.3371 | 0.0027 | 0.3378 | 0.0028 | 0.0007 | 0.0001 |
| Graph-PCA | 0.3355 | 0.0092 | 0.3352 | 0.0096 | -0.0002 | 0.0004 |

## Key improvement result
- Continuation warm-start speedup: **1.430x** (`136 -> 91` total iterations).
