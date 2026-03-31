# Reporting Template (Track A)

## Solver Convergence
- Objective vs iteration
- Coupling gap vs iteration
- Orthogonality error vs iteration
- Sparsity fraction vs iteration
- Laplacian energy vs iteration

## Sparsity-Structure Tradeoff
- Describe how graph smoothness interacts with sparsity and variance.
- Compare the graph-regularized Proposed method against A-ManPG, SparseNoGraph, and PCA.

## Support Recovery
- Per-component precision/recall/F1
- Union support precision/recall/F1
- Notes on alignment (matching + sign flips)

## PCA Baseline Comparison
- Explained variance
- Graph smoothness
- Support diagnostics (if reported)
- Methods tracked in the current rerun set: `PCA`, `A-ManPG`, `SparseNoGraph`, `Proposed`

## Chain robustness (seeds 0--4)
| Method | Support F1 | Graph smoothness norm | Shared explained variance |
|---|---:|---:|---:|
| PCA | 0.261 ± 0.000 | 2.099 ± 0.484 | 6.225 ± 0.406 |
| A-ManPG | 0.914 ± 0.078 | 2.111 ± 0.483 | 6.122 ± 0.408 |
| SparseNoGraph | 0.950 ± 0.033 | 2.118 ± 0.476 | 6.092 ± 0.413 |
| Proposed | 0.927 ± 0.019 | 1.871 ± 0.465 | 6.059 ± 0.415 |

Chain graphs favor local smoothness; the Proposed method is the smoothest while remaining competitive on support recovery.

## SBM robustness (seeds 0--4)
| Method | Support F1 | Graph smoothness norm | Shared explained variance |
|---|---:|---:|---:|
| PCA | 0.261 ± 0.000 | 17.085 ± 0.974 | 6.225 ± 0.406 |
| A-ManPG | 0.914 ± 0.078 | 17.134 ± 1.008 | 6.122 ± 0.408 |
| SparseNoGraph | 0.950 ± 0.033 | 17.166 ± 1.035 | 6.092 ± 0.413 |
| Proposed | 0.744 ± 0.051 | 15.764 ± 1.104 | 6.003 ± 0.384 |

SBM graphs show a clearer recovery--smoothness tradeoff: Proposed is smoother, while SparseNoGraph and A-ManPG recover supports more accurately.

## SBM lambda2 sweep (seeds 0--2)
**Figure artifact:** `../figures/sbm_lambda2_sweep_panel.png`  
This panel summarizes the SBM `lambda2` sweep (mean ± std) for support F1, graph smoothness norm, and shared explained variance, with A-ManPG as a horizontal reference line.

Light graph regularization can improve smoothness with minimal recovery loss.
Stronger graph regularization produces smoother but denser and less accurate supports on SBM.
Operational takeaway: for SBM-like graphs, `lambda2` needs topology-sensitive tuning rather than chain-tuned defaults.

## Chain lambda2 sweep (seeds 0--2)
**Figure artifact:** `../figures/chain_lambda2_sweep_panel.png`  
This panel summarizes the chain `lambda2` sweep (mean ± std) for support F1, graph smoothness norm, and shared explained variance, with A-ManPG as a horizontal reference line.

## Graph-aligned chain setting
**Figure artifact:** `../figures/graph_aligned_lambda2_sweep_panel.png`  
This panel summarizes the graph-aligned chain sweep (mean ± std) for support F1, smoothness norm, and largest connected-component ratio, showing the impact of moderate graph regularization on structural connectivity.
