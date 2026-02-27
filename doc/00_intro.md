# Gradient-Based Optimization for Network-Constrained Sparse PCA

## 1. Background

Principal Component Analysis (PCA) is a foundational tool for dimensionality reduction. However, classical PCA produces dense loading vectors, which limits interpretability and feature selection in high-dimensional datasets.

Sparse PCA (SPCA) addresses this by enforcing sparsity in the principal components. This project implements and benchmarks several state-of-the-art approaches to SPCA:

*   **Regression-based SPCA (Zou et al. 2006)**: Uses an Elastic Net penalty within a regression formulation.
*   **Generalized Power Method (Journée et al. 2010)**: An efficient iterative thresholding scheme.
*   **Network-Constrained SPCA (NC-SPCA)**: Our focus, which regularizes component smoothness over a feature graph $G=(V,E)$ using a graph Laplacian $L$:
    $$ \max_{w: \|w\|_2 \le 1} w^\top \Sigma w - \lambda_1 \|w\|_1 - \lambda_2 w^\top L w $$

---

## 2. Contributions

This project provides a unified framework for:

1.  **Algorithm Implementation**: High-performance implementations of Zou-SPCA, GPM, and NC-SPCA.
2.  **Diverse Benchmarking**: Evaluation across classic statistics (Pitprop), genomics (Alon Colon Cancer), and high-dimensional synthetic models.
3.  **Graph Synthesis**: Tools for generating feature networks (Chain, Grid, RGG, SBM) to test structured sparsity.
4.  **Metric Suite**: Comprehensive evaluation using explained variance, F1 support recovery, and Largest Connected Component (LCC) ratios.

---

## 3. References

*   Zou, H., Hastie, T., & Tibshirani, R. (2006). *Sparse Principal Component Analysis*. J. Comput. Graph. Statist.
*   Journée, M., Nesterov, Y., Richtárik, P., & Sepulchre, R. (2010). *Generalized Power Method for Sparse Principal Component Analysis*. JMLR.
*   Miao, R., et al. (2022). *Sparse principal component analysis based on genome network...*. Med Biol Eng Comput.
*   Alon, U., et al. (1999). *Broad patterns of gene expression revealed by clustering analysis of tumor and normal colon tissues*. PNAS.
