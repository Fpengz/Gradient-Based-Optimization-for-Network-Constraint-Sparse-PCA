# Paper Review Checklist

This checklist applies the scientific review workflow directly to `doc/latex/main.tex`.

## 1. Objective and notation

- [ ] The single-component objective matches the implemented objective exactly:
      `max_{||w||_2 <= 1} w^T Sigmahat w - lambda1 ||w||_1 - lambda2 w^T L w`
- [ ] The minimization rewrite uses
      `f(w) = -w^T Sigmahat w + lambda2 w^T L w`
      and
      `h(w) = lambda1 ||w||_1 + I_{B2}(w)`
- [ ] The gradient is stated exactly as
      `-2 Sigmahat w + 2 lambda2 L w`
- [ ] The constraint is described as a unit `l2` ball, not a unit sphere
- [ ] The covariance scaling is stated as `Sigmahat = X^T X / n`
- [ ] The Laplacian choice is explicit and consistent across optimization and metrics

## 2. Algorithm descriptions

- [ ] Algorithm `pg` matches the implemented baseline update
- [ ] The proximal map is stated as soft-thresholding followed by radial projection
- [ ] The step-size condition matches the convergence theorem assumptions
- [ ] MASPG-CAR is clearly labeled as a practical variant
- [ ] MASPG-CAR is not presented as theorem-certified
- [ ] Active-set refinement is described as optional and empirical

## 3. Convergence claims

- [ ] Formal guarantees are restricted to baseline PG
- [ ] The theorem assumptions are sufficient and correctly scoped
- [ ] The proof sketch does not imply more than accumulation-point criticality
- [ ] No theorem-level claim is made for multi-component manifold methods
- [ ] No theorem-level claim is made for MASPG-CAR

## 4. Experimental protocol

- [ ] Graph families match the implemented generators:
      chain, grid, random geometric graph, SBM
- [ ] The spiked covariance model is stated correctly
- [ ] Reproducibility details match the scripts and reports
- [ ] The listed baselines match the benchmark harness
- [ ] Metrics match the implementation:
      explained variance, support precision/recall/F1, LCC ratio, Laplacian energy, runtime, iterations
- [ ] The paper distinguishes informative-graph and misspecified-graph regimes

## 5. Evidence and claims

- [ ] Claims about support recovery are conditional on graph quality
- [ ] Claims about wall-clock improvements are framed as empirical, not universal
- [ ] The paper does not imply uniform dominance over `l1`-SPCA
- [ ] Statements about multi-component methods are positioned as extensions, not proved contributions

## 6. Writing quality

- [ ] Long sentences are split where they hide the main point
- [ ] Definitions introduce symbols before use
- [ ] Results language is precise rather than promotional
- [ ] The conclusion reflects current measured evidence
- [ ] Related work is specific about what differs in objective, geometry, and guarantees

## 7. Final pre-submission checks

- [ ] Rebuild `doc/latex/out/main.pdf`
- [ ] Check the LaTeX log for warnings
- [ ] Reconcile any changed claims with:
      `AUDIT_REPORT_METHOD.md`
      `AUDIT_REPORT_CONVERGENCE.md`
      `PERFORMANCE_REPORT_V1.md`
- [ ] Ensure tables or placeholders do not imply unsupported numeric results
