# PERFORMANCE_REPORT_V0

Audit artifact directory: `results/paper_audit_20260227-111514`

## Aggregated core-method performance (mean across graph families)

| method | f1_mean | f1_topk_mean | explained_variance_mean | lcc_ratio_mean | runtime_sec_mean | converged_rate |
| --- | --- | --- | --- | --- | --- | --- |
| L1-SPCA-ProxGrad | 0.4028 | 0.8708 | 9.8574 | 0.9979 | 0.0074 | 1.0000 |
| NetSPCA-MASPG-CAR | 0.3971 | 0.8458 | 9.7973 | 0.9947 | 0.0090 | 1.0000 |
| NetSPCA-PG | 0.3967 | 0.8458 | 9.7986 | 0.9947 | 0.0094 | 1.0000 |
| PCA | 0.3371 | 0.8708 | 9.8991 | 1.0000 | 0.0027 | 1.0000 |
| Graph-PCA | 0.3355 | 0.8458 | 9.8301 | 1.0000 | 0.0092 | 1.0000 |

## By-family core-method performance

| graph_family | method | f1_mean | f1_topk_mean | explained_variance_mean | lcc_ratio_mean | runtime_sec_mean | converged_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| chain | L1-SPCA-ProxGrad | 0.4170 | 0.8333 | 9.9659 | 1.0000 | 0.0075 | 1.0000 |
| chain | NetSPCA-PG | 0.4113 | 0.8333 | 9.9511 | 1.0000 | 0.0107 | 1.0000 |
| chain | NetSPCA-MASPG-CAR | 0.4113 | 0.8333 | 9.9505 | 1.0000 | 0.0105 | 1.0000 |
| chain | Graph-PCA | 0.3390 | 0.8333 | 9.9935 | 1.0000 | 0.0106 | 1.0000 |
| chain | PCA | 0.3361 | 0.8333 | 10.0075 | 1.0000 | 0.0027 | 1.0000 |
| grid | L1-SPCA-ProxGrad | 0.3826 | 0.8833 | 9.7826 | 1.0000 | 0.0074 | 1.0000 |
| grid | NetSPCA-PG | 0.3748 | 0.8167 | 9.7597 | 1.0000 | 0.0083 | 1.0000 |
| grid | NetSPCA-MASPG-CAR | 0.3748 | 0.8167 | 9.7582 | 1.0000 | 0.0081 | 1.0000 |
| grid | PCA | 0.3381 | 0.8833 | 9.8243 | 1.0000 | 0.0027 | 1.0000 |
| grid | Graph-PCA | 0.3352 | 0.8167 | 9.8027 | 1.0000 | 0.0082 | 1.0000 |
| rgg | NetSPCA-MASPG-CAR | 0.4141 | 0.8833 | 9.6311 | 0.9829 | 0.0083 | 1.0000 |
| rgg | NetSPCA-PG | 0.4127 | 0.8833 | 9.6328 | 0.9829 | 0.0090 | 1.0000 |
| rgg | L1-SPCA-ProxGrad | 0.4074 | 0.9000 | 9.7401 | 1.0000 | 0.0074 | 1.0000 |
| rgg | PCA | 0.3380 | 0.9000 | 9.7817 | 1.0000 | 0.0027 | 1.0000 |
| rgg | Graph-PCA | 0.3333 | 0.8833 | 9.6772 | 1.0000 | 0.0089 | 1.0000 |
| sbm | L1-SPCA-ProxGrad | 0.4043 | 0.8667 | 9.9410 | 0.9915 | 0.0074 | 1.0000 |
| sbm | NetSPCA-PG | 0.3881 | 0.8500 | 9.8506 | 0.9959 | 0.0096 | 1.0000 |
| sbm | NetSPCA-MASPG-CAR | 0.3881 | 0.8500 | 9.8495 | 0.9959 | 0.0090 | 1.0000 |
| sbm | PCA | 0.3362 | 0.8667 | 9.9829 | 1.0000 | 0.0027 | 1.0000 |
| sbm | Graph-PCA | 0.3343 | 0.8500 | 9.8469 | 1.0000 | 0.0091 | 1.0000 |

## Failure log

No NaN/divergence/non-convergence events in this batch (`failure_log.json` is empty).
