# BOTTLENECK_ANALYSIS

## Ranked Issues (evidence-based)
1. Spectral Lipschitz estimation (`eigsh`) is the largest one-time per-fit cost.
2. PCA initialization is a secondary setup cost.
3. Sparse `L @ w` matvec dominates per-iteration linear algebra.
4. Objective/backtracking overhead is comparatively small.

## Evidence

```text
         50101 function calls in 0.052 seconds

   Ordered by: cumulative time
   List reduced from 276 to 30 due to restriction <30>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.052    0.052 network_sparse_pca.py:342(fit)
        1    0.000    0.000    0.052    0.052 network_sparse_pca.py:223(_fit_one_component)
        1    0.000    0.000    0.037    0.037 network_sparse_pca.py:89(_estimate_lipschitz)
        1    0.000    0.000    0.027    0.027 arpack.py:1359(eigsh)
      872    0.010    0.000    0.027    0.000 arpack.py:542(iterate)
        2    0.000    0.000    0.021    0.011 base.py:1348(wrapper)
        2    0.000    0.000    0.021    0.010 _pca.py:422(fit)
        2    0.000    0.000    0.021    0.010 _pca.py:481(_fit)
        2    0.000    0.000    0.020    0.010 _pca.py:544(_fit_full)
        2    0.019    0.009    0.019    0.009 _decomp_svd.py:13(svd)
      871    0.001    0.000    0.016    0.000 _interface.py:227(matvec)
      871    0.000    0.000    0.015    0.000 _interface.py:215(_matvec)
      871    0.001    0.000    0.014    0.000 _interface.py:333(matmat)
      871    0.000    0.000    0.012    0.000 _interface.py:824(_matmat)
      871    0.001    0.000    0.012    0.000 _base.py:481(dot)
        1    0.000    0.000    0.011    0.011 network_sparse_pca.py:169(_initialize_component)
      927    0.000    0.000    0.011    0.000 _base.py:728(__matmul__)
      927    0.001    0.000    0.008    0.000 _base.py:597(_matmul_dispatch)
      927    0.002    0.000    0.006    0.000 _compressed.py:518(_matmul_vector)
     1798    0.001    0.000    0.002    0.000 numeric.py:1927(isscalar)
      927    0.001    0.000    0.002    0.000 _sputils.py:345(isscalarlike)
       11    0.000    0.000    0.002    0.000 network_sparse_pca.py:183(_accept_with_backtracking)
      927    0.002    0.000    0.002    0.000 {built-in method scipy.sparse._sparsetools.csr_matvec}
     7513    0.001    0.000    0.002    0.000 {built-in method builtins.isinstance}
       23    0.000    0.000    0.001    0.000 network_sparse_pca.py:115(_smooth_grad)
       33    0.001    0.000    0.001    0.000 network_sparse_pca.py:133(_objective)
        2    0.000    0.000    0.001    0.000 extmath.py:895(svd_flip)
     2640    0.001    0.000    0.001    0.000 {method 'reshape' of 'numpy.ndarray' objects}
       46    0.000    0.000    0.001    0.000 _type_check_impl.py:373(nan_to_num)
```

## Tradeoff notes from results
- Graph regularization lowers Laplacian energy but may reduce support F1 under misspecification.
- High lambda1 risks degenerate sparse solutions; tune jointly with lambda2.
- Continuation warm starts materially reduce total runtime and iterations.
