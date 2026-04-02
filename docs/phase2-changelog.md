# Phase 2 Track A Changelog

## Summary
- Added real-data pipelines for TCGA-BRCA (STRING), MNIST (grid), and S&P500 (correlation) with frozen artifacts.
- Expanded misspecification experiments with controlled edge rewiring across chain, grid, SBM, and kNN.
- Updated manuscript to emphasize topology-dependent robustness and added new figures for corruption phase diagrams, summary bar plot, and unified real-data trends.

## Key Claims Updated
- Robustness to graph misspecification is topology-dependent.
- Fragile graphs (chain) degrade under corruption at high lambda2.
- Redundant graphs (grid, SBM, kNN) are more stable and can mitigate oversmoothing in some regimes.

## New Figures
- `figures/corruption_phase_panels.png`
- `figures/corruption_delta_summary.png`
- `figures/realdata/realdata_unified_panels.png`

## Numerical Warning Policy
- Runs with `runtime_warnings_count > 0` are flagged in manifests.
- Summary tables report whether flagged runs were included or excluded.
- Exclusion rules (if any) must be fixed in advance and documented with the table.
