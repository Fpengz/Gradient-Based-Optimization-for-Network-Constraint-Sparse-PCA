# Track A Revision Design: Graph-Smooth Sparse Orthogonal PCA Under Misspecification

## Summary
We will strengthen Track A by reframing the contribution as a controlled empirical characterization, tightening technical clarity (estimator definition, limiting cases, Laplacian normalization, convergence discussion), and substantially expanding experiments (more graph families, seeds, baselines, metrics, and real datasets). The pipeline will rely on frozen dataset artifacts, a unified runner/evaluation contract, and a canonical results schema with provenance to ensure reproducibility and comparability.

## Goals
- Reposition the paper as an empirical regime/topology characterization rather than a new algorithmic breakthrough.
- Expand the synthetic and real-data evidence base to support the regime/topology claims.
- Improve technical clarity and methodological defensibility (estimator choice, limiting cases, convergence rationale).
- Ensure experimental reproducibility through frozen artifacts, provenance fields, and canonical schemas.

## Non-Goals
- Joint graph learning or learned graph structure.
- Fully optimized or specialized implementations for each baseline.
- Exhaustive real-data benchmarking beyond the three target datasets.

## Architecture and Data Flow
### Layers
1. **Dataset preparation layer** (dataset-specific)
   - Fetch/preprocess/build graph, then freeze artifacts.
   - Outputs: `X`, graph adjacency/Laplacian, metadata, splits/seeds.
   - Produces `artifact_id`, `artifact_version`, `prep_config_hash`, `eval_protocol_id`.
2. **Experiment runner** (unified execution + evaluation contract)
   - Consumes frozen artifacts only.
   - Runs methods, logs canonical results schema.
3. **Analysis/plot layer**
   - Aggregates results and joins artifact metadata by `artifact_id`.

### Artifact Freezing
- Each artifact includes:
  - `X`, adjacency/Laplacian, metadata, split/eval protocol definitions
  - Provenance: data source, date ranges, thresholds, preprocessing config
- Runner never calls external APIs; it only consumes frozen artifacts.

## Canonical Results Schema
### Required
- **Identity/provenance:**
  - `dataset, graph_family, artifact_id, artifact_version, data_source, prep_config_hash, eval_protocol_id`
- **Method & run params:**
  - `method, method_version, seed, rank, lambda1, lambda2, rho, corruption_type, corruption_level`
- **Graph IDs:**
  - `graph_used_id, graph_reference_id`
- **Core metrics:**
  - `explained_variance, smoothness_used_graph, smoothness_reference_graph, runtime_sec, iterations`
- **Sparsity:**
  - `nnz_loadings, sparsity_ratio`
- **Optimization diagnostics:**
  - `final_objective, final_coupling_gap, final_orthogonality_defect, stop_reason, convergence_flag`
- **Synthetic-only:**
  - `support_precision, support_recall, support_f1` (explicit missing values for real data)
- **Size/shape:**
  - `n_samples, n_features`

### Optional
- **Stability metrics:**
  - `stability_jaccard, stability_cosine, stability_subspace`

### Artifact Metadata (Separate)
- Dataset-level info (ticker universe, date range, graph threshold rule, MNIST resolution, kNN parameters, etc.) stored in artifact metadata referenced by `artifact_id`.

## Datasets (Real Data)
1. **TCGA-BRCA + STRING (existing)**
   - Anchor dataset, modular/SBM-like topology.
2. **MNIST (pixel grid)**
   - Graph: 2D grid adjacency, matches synthetic grid topology.
3. **S&P500 correlation network**
   - Data source: `yfinance` (or equivalent), compute daily returns.
   - Graph: correlation graph (threshold or top-k), noisy proxy.

All three datasets will be plotted on common axes (x: `lambda2`, y: variance, smoothness, sparsity, optional stability) to support unified narrative.

## Synthetic Graph Families
- Existing: chain, grid, SBM
- Add: Erdos-Renyi, geometric/kNN, small-world
- Corruption sweeps (delete/rewire; optional weight perturbation) applied across all families
- Seeds: increase to >=10 (target 10-20)

## Baselines
- Graph-only PCA (`lambda1 = 0`)
- gLSPCA / GDSPCA-style baselines (consistent implementation)
- Graph + sparsity non-manifold baseline (same objective, no Stiefel constraints)
- Existing baselines retained for continuity

## Metrics & Comparability
- Add runtime, iterations, sparsity, convergence diagnostics
- **Runtime comparisons only within shared hardware/execution config; report as indicative, not definitive**
- Report iterations to convergence and time per iteration
- Synthetic-only support metrics; real data uses variance/smoothness/sparsity/stability

## Manuscript Updates
- Reframe abstract + contributions as empirical regime characterization
- Replace “underexplored intersection” with “not systematically evaluated under controlled misspecification”
- Add takeaway bullets in intro + conclusion
- Clarify estimator usage: `B` is primary for support/smoothness; `A` (or `QR(B)`) for variance/orthogonality
- Add limiting cases: `lambda2 = 0`, `lambda1 = 0`, `rho -> infinity`
- Normalize Laplacian and document scaling/interpretation
- Add convergence discussion citing ManPG/A-ManPG
- Add explicit Threats to Validity paragraph

## Figures
- Unified real-data panels on common axes
- Summary panel/table: MNIST robust, TCGA moderate sensitivity, S&P500 unstable at high `lambda2`
- Regime plots across graph families
- Baseline comparison plots
- Runtime/convergence diagnostics with caption caveat
- Ablation: `rho` sweep (2-3 `lambda2` values)

## Execution Plan
1. Implement dataset preparation + artifact freezing (MNIST, S&P500, TCGA updates)
2. Extend runner & schema for new fields, baselines, graphs, metrics
2.5. Sanity checks
   - `lambda2 = 0` matches SparseNoGraph
   - `lambda1 = 0` matches graph-only PCA
   - Convergence diagnostics sensible
3. Run synthetic sweeps (>=10 seeds) across graph families
4. Run real data (TCGA, MNIST, S&P500)
5. Update plots and manuscript, rebuild PDF

## Risks & Mitigations
- **Compute/runtime:** high cost for expanded sweeps
  - Mitigation: staged runs with sanity checks and smaller pilot sweeps
- **Real-data instability:** correlation graph is noisy proxy
  - Mitigation: explicit Threats to Validity + unified narrative framing
- **Baseline fairness:** implementations not equally optimized
  - Mitigation: common evaluation contract and runtime caveats
