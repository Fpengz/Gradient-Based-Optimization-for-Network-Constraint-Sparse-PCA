# Changelog: Phase 1.5 Submission → Claim-Hardening + Real-Data Checkpoint

**From:** `paper1-phase1.5-submission-v1`  
**To:** `paper1-trackB-phase1.5-claim-hardening-realdata-v1`

## Summary (One Page)
This checkpoint upgrades the Phase 1.5 submission snapshot into the first fully reproducible, claim‑hardened version that also includes a real‑data sanity check. The paper message remains narrow and defensible: graph regularization is a structural bias whose benefits are regime‑ and topology‑dependent; in the graph‑aligned high‑decoy setting, low‑to‑moderate \(\lambda_2\) yields strong smoothness gains without detectable F1 loss. The TCGA/STRING experiment strengthens plausibility but is framed as a sanity check rather than a benchmark.

## Evidence Additions
- **Real‑data sanity check:** TCGA‑BRCA expression (UCSC Xena) + STRING PPI graph; added \(\lambda_2\) sweep for Proposed vs SparseNoGraph with smoothness + explained variance. Framed as sanity check.
- **Extra topology:** grid graph sweep added for a third synthetic topology alongside chain/SBM.
- **Robustness reproducibility:** committed \(\lambda_1 \times \rho\) sweep configs (9 configurations) under `configs/trackB/robust_lambda1_rho/` so the “repeatable regime across tested settings” claim is reproducible from the tag.

## Methods / Pipeline Changes
- **Real‑data loader:** new TCGA‑BRCA + STRING pipeline in `src/grpca_gd/real_data.py` (downloads Xena expression, STRING links/aliases, builds Laplacian, intersects genes, returns dataset object).
- **Runner updates:** `src/grpca_gd/runner.py` now handles `dataset_type: real` and grid graph family; support metrics are skipped for real data (sparsity proxy retained).
- **Plotting:** new `scripts/plot_realdata_sweep.py`; `scripts/plot_lambda2_sweep.py` extended to include grid sweep.

## Manuscript Changes
- **Abstract/Conclusion:** updated to mention real‑data sanity check and grid topology; “repeatable” regime wording retained.
- **Results:** new TCGA‑BRCA + STRING subsection; new grid sweep figure/caption.
- **Discussion:** mechanism note retained; non‑claim retained (no universal recovery improvement).
- **Phase 1.5 interpretation summary:** still present and aligned with repeatable‑regime framing.

## Artifacts and Figures
- Added: `latex/figures/realdata_lambda2_sweep.png`
- Added: `figures/grid_lambda2_sweep_panel.png`
- Phase 1.5 figures unchanged (high/medium decoy, diff summary)

## Reproducibility Notes
- **Configs:**
  - Phase 1.5 robustness sweep configs committed under `configs/trackB/robust_lambda1_rho/`.
  - Real‑data sweep configs under `configs/realdata/`.
  - Grid sweep configs under `configs/grid_sweep/`.
- **Snapshot PDF:** `paper/paper1-trackB-phase1.5-v1.pdf` (built from `paper1-phase1.5-submission-v1`).

## Branch/Tag Hygiene
- **Tag:** `paper1-trackB-phase1.5-claim-hardening-realdata-v1` (this checkpoint)
- **Frozen snapshot:** `paper1-phase1.5-submission-v1` remains unchanged.

---

If needed, this changelog can serve as the basis for a rebuttal summary or advisor update.
