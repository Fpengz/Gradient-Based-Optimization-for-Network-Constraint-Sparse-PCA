# IMPLEMENTATION_MAP

| Paper Item | Code Location(s) | Notes |
|---|---|---|
| Single-component objective `min -w^T Sigmahat w + lambda1 ||w||_1 + lambda2 w^T L w, ||w||_2<=1` | `src/models/network_sparse_pca.py:133`, `src/models/network_sparse_pca.py:146` | `_objective` computes the composite objective. |
| Smooth part `f(w)` | `src/models/network_sparse_pca.py:153` | `_smooth_value` used for majorization checks. |
| Gradient `∇f(w) = -2 Sigmahat w + 2 lambda2 L w` | `src/models/network_sparse_pca.py:115` | Implicit covariance multiplication `X^T(Xw)/n`. |
| Prox `soft(v, eta*lambda1)` then project to unit `l2` ball | `src/models/network_sparse_pca.py:77`, `src/models/network_sparse_pca.py:81`, `src/models/network_sparse_pca.py:177` | Exact composite prox operator. |
| Step-size selection (`auto` / backtracking) | `src/models/network_sparse_pca.py:89`, `src/models/network_sparse_pca.py:203`, `src/models/network_sparse_pca.py:167` | Auto from spectral bound; backtracking for descent. |
| MASPG-CAR inertial extrapolation | `src/models/network_sparse_pca.py:235` | `beta_k=min(0.9,(k-1)/(k+2))`. |
| MASPG-CAR BB initialization | `src/models/network_sparse_pca.py:227` | `eta_BB=<d,d>/<d,r>` with clipping. |
| MASPG-CAR smooth-majorization + restart | `src/models/network_sparse_pca.py:167`, `src/models/network_sparse_pca.py:246` | Smooth-majorization condition + objective restart safeguard. |
| MASPG-CAR active-set refinement | `src/models/network_sparse_pca.py:269` | Triggered after support stabilization window. |
| Continuation/warm starts | `src/models/network_sparse_pca.py:366` | Implemented by `fit_path(...)`. |
| Initialization (PCA loading) | `src/models/network_sparse_pca.py:156` | Leading PCA component, unit norm. |
| Stopping criteria | `src/models/network_sparse_pca.py:257`, `src/models/network_sparse_pca.py:286` | Relative iterate + relative objective change. |
| Graph families chain/grid/RGG/SBM | `src/utils/graph.py:66`, `src/utils/graph.py:77`, `src/utils/graph.py:102`, `src/utils/graph.py:168`, `src/utils/graph.py:120` | RGG implemented to match paper design. |
| Laplacian definition | `src/utils/graph.py:27` | Combinatorial and symmetric-normalized Laplacian supported. |
| Synthetic spiked model | `src/experiments/synthetic_benchmark.py:181` | `X = sqrt(beta) z w*^T + eps`. |
| Metrics EV/F1/LCC/Laplacian/runtime | `src/utils/metrics.py:12`, `src/utils/metrics.py:32`, `src/utils/metrics.py:75`, `src/utils/metrics.py:61`, `src/experiments/synthetic_benchmark.py:384` | Includes top-k support metrics and convergence curves. |
