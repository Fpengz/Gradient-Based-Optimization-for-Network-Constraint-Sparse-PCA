# Gradient-Based Optimization for Network-Constrained Sparse PCA

## 1. Background

Principal Component Analysis (PCA) is a foundational tool for dimensionality reduction. 
Classical PCA produces dense loading vectors, limiting interpretability and feature selection in high-dimensional datasets.

Sparse PCA (SPCA) addresses this by enforcing sparsity in the principal components, typically via an $\ell_1$ penalty:
$$
\max_{w: \|w\|_2 \le 1} w^\top \Sigma w - \lambda_1 \|w\|_1
$$
where $\Sigma$ is the covariance matrix of the data, $w$ is the loading vector, and $\lambda_1$ controls sparsity.

In many applications, features are linked via **graph structures** (e.g., gene networks, sensor networks, financial factors).  
Ignoring these connections can produce fragmented or disconnected supports.  

This motivates **Network-Constrained SPCA (NC-SPCA)**, where we additionally regularize the component smoothness over a graph $G=(V,E)$:
$$
\max_{w: \|w\|_2 \le 1} w^\top \Sigma w - \lambda_1 \|w\|_1 - \lambda_2 w^\top L w
$$
where $L$ is the graph Laplacian, and $\lambda_2$ controls smoothness on the network.

---

## 2. Contributions

This project focuses on:

1. Implementing **baseline PCA and SPCA methods**.  
2. Constructing **synthetic graph-structured datasets** for evaluation.  
3. Implementing **network-constrained SPCA** using a **proximal gradient algorithm**.  
4. Evaluating **structured feature selection**, connectivity of supports, and explained variance.  

---

## 3. References

- Journée, M., Nesterov, Y., Richtárik, P., & Sepulchre, R. (2008). *Generalized Power Method for Sparse Principal Component Analysis*. arXiv:0811.4724.  
- Zou, H., & Xue, L. (2018). *A Selective Overview of Sparse Principal Component Analysis*. Proceedings of the IEEE, 106(8), 1311–1320.  
- Qiu, Y., Lei, J., & Roeder, K. (2022). *Gradient-Based Sparse Principal Component Analysis with Extensions to Online Learning*. Biometrika, 110(2), 339–360.  
- Miao, R., Dang, Q., Cai, J., et al. (2022). *Sparse Principal Component Analysis Based on Genome Network for Correcting Cell Type Heterogeneity in Epigenome-Wide Association Studies*. Med Biol Eng Comput, 60, 2601–2618.
