# V2.1 Hybrid Manuscript Design

## Summary
Produce a V2.1 hybrid main text that keeps the Version 2 framing and discipline but integrates selected Version 1 artifacts to establish a clearer method-comparison baseline and a single positive-case result. The paper’s core claim remains modest: graph regularization is not universally helpful; reliability depends on regularization strength and topology, and smoothness under a misspecified prior can be misleading. The main text stays compact; broader evidence and extended materials move to a structured appendix.

## Title
Use the sharpened title:

**Topology-Dependent Reliability of Graph Regularization under Misspecification in Sparse PCA**

## Main-Text Structure (Final Order)
1. **Chain \(\lambda_2\) sweep** (core synthetic result)
2. **Baseline comparison table** immediately after the chain sweep:
   - Use V1 table `tab:chain-summary`.
   - Purpose: establish proper method comparison before the regime narrative.
3. **Chain vs. grid misspecification comparison** (core topology contrast)
4. **Corruption phase diagram** (misspecification regime map)
5. **Graph-aligned positive-case figure** (very short subsection)
   - Use the V1 figure `fig:graph-aligned-sweep` only.
   - One short paragraph: there exists a clean aligned regime where graph regularization genuinely helps.
6. **TCGA sanity check** (small, disciplined real-data check)
7. **One-sentence broader-topology confirmation** in Results
   - Point to Appendix A (SBM/kNN) as consistent evidence.

## Abstract (Rewrite Guidelines)
Emphasize:
- Controlled misspecification setup
- Weak vs. strong regularization regimes
- Fragile vs. redundant graph topologies
- Illusory smoothness phenomenon
- Modest claims (no universal benefit, no novelty over-claim)

## Introduction (Rewrite Guidelines)
Keep the broader literature grounding from both versions, but end with:
- The question
- The regime view
- The topology mechanism
- What the paper contributes

## Results (Key Edits)
- Insert `tab:chain-summary` immediately after the chain \(\lambda_2\) sweep.
- Keep chain vs. grid misspecification and corruption phase diagram in main text.
- Add the quantified **illusory smoothness** sentence directly in Results near misspecification (after chain/grid comparison or after the phase diagram). Use existing numeric values from current results.
- Keep the **graph-aligned positive case** to one short paragraph plus the single figure.
- Keep **TCGA-BRCA** as the only main-text real-data sanity check.
- Add the one-sentence broader-topology confirmation (SBM/kNN) pointing to Appendix A.

## Discussion (Key Edits)
- Preserve the strong mechanism paragraph:
  - Chains depend on precise adjacency.
  - Grids have redundant paths.
  - Redundancy flattens the effect of perturbation.
- Interpret the illusory smoothness sentence placed in Results.
- Maintain the modest claim framing.

## Conclusion (Key Edits)
Keep compact and practical:
- Graph regularization is not universally helpful.
- Reliability depends on regularization strength and topology.
- Smoothness under the imposed graph can be misleading.

## Appendix Structure (Grouped)
Add `\appendix` and group the moved material into three blocks:

**Appendix A: Additional topology results**
- SBM and kNN misspecification
- Extended topology evidence

**Appendix B: Additional real-data results**
- MNIST and S\&P500
- TCGA subsampling/stability/coherence

**Appendix C: Implementation / extended tables**
- Extended implementation details
- Additional tables and runtime details

## Artifacts to Reuse (Source of Truth)
- **Baseline table:** V1 `tab:chain-summary` (main text).
- **Graph-aligned positive case:** V1 `fig:graph-aligned-sweep` only (main text).
- **Real-data main text:** V2 TCGA-BRCA sweep.
- **Chain/grid misspecification + corruption phase diagram:** V2 main-text figures.

## Tests / Acceptance Criteria
- Main text order matches the specified sequence.
- Only one baseline comparison table appears in main text (chain summary).
- Only one graph-aligned artifact appears in main text (figure only).
- Illusory smoothness is quantified in Results with a numeric sentence.
- TCGA is the only main-text real-data dataset.
- Appendix is grouped into A/B/C blocks with the specified content.
- Title and abstract reflect the misspecification scope and modest claims.

## Assumptions and Defaults
- Use existing results/figures for illusory smoothness quantification.
- Keep all non-core evidence in the appendix to preserve a disciplined main text.
- The appendix lives in the same LaTeX file after `\appendix`.
