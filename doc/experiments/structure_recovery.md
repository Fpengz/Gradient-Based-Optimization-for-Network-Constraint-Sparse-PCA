# Structure Recovery & Feature Selection Metrics

This file defines how we evaluate **network-aware feature selection** quality.

---

## 1. Support Sets

- True support:

$$
S^\star = \{i : w^\star_i \neq 0\}
$$

- Estimated support:

$$
\hat S = \{i : \hat w_i \neq 0\}
$$

---

## 2. Feature Selection Metrics

### Precision / Recall / F1

$$
\mathrm{Precision} = \frac{|\hat S \cap S^\star|}{|\hat S|}
$$

$$
\mathrm{Recall} = \frac{|\hat S \cap S^\star|}{|S^\star|}
$$

$$
\mathrm{F1} = \frac{2\cdot \mathrm{Precision}\cdot \mathrm{Recall}}
{\mathrm{Precision}+\mathrm{Recall}}
$$

### Support Size Error

$$
\Delta k = \bigl||\hat S| - |S^\star|\bigr|
$$

---

## 3. Graph-Structure Metrics

### Largest Connected Component (LCC) Ratio
Let $\mathrm{LCC}(\hat S)$ be the size of the largest connected component
in the induced subgraph on $\hat S$:

$$
\mathrm{LCC\ Ratio} = \frac{|\mathrm{LCC}(\hat S)|}{|\hat S|}
$$

### Graph Smoothness of Loadings

$$
\hat w^\top L \hat w
$$

Lower values imply smoother signals over the graph.

---

## 4. Hypothesis (What You Want to Show)

Compared to unstructured SPCA (L1 only), network-constrained SPCA yields:
- Higher LCC ratio
- Better F1 for support recovery
- Comparable or improved explained variance
- More stable supports under graph perturbation
