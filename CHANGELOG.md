# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added
- Torch backend models:
  - `TorchNetworkSparsePCA`
  - `TorchNetworkSparsePCA_GeooptStiefel`
- Backend selection in experiment runners:
  - `--backend {numpy,torch,torch-geoopt}`
  - `--torch-device`, `--torch-dtype`
- Large-scale stress runner: `scripts/run_large_scale_stress.py`
- Backend comparison runner: `scripts/run_backend_comparison.py`
- Dynamic graph experiment runner: `scripts/run_dynamic_graph_experiment.py`
- Pinned manifest orchestration: `scripts/reproduce_paper_artifacts.py`
- Statistical significance utilities in `src/experiments/stats.py`
- Audit and certification reports:
  - `IMPLEMENTATION_MAP.md`
  - `AUDIT_REPORT_METHOD.md`
  - `AUDIT_REPORT_CONVERGENCE.md`
  - `PERFORMANCE_REPORT_V0.md`
  - `PERFORMANCE_REPORT_V1.md`
  - `BOTTLENECK_ANALYSIS.md`
  - `ABLATION_TABLES.md`
  - `FINAL_CERTIFICATION.md`

### Changed
- Synthetic benchmark now exports stationarity diagnostics in summary:
  - `pg_residual_last_mean`
  - `pg_residual_ratio_mean`
  - `objective_monotone_rate`
- Synthetic/real benchmark fitting now passes `graph` when estimator supports it (fixes torch graph-method wiring).
- Added random geometric graph (RGG) support to graph generators and synthetic protocol.

### Fixed
- MASPG-CAR backtracking alignment with smooth-majorization test and monotone restart safeguards.
- Dependency on `tabulate` removed from stress markdown table generation by using an internal formatter.
