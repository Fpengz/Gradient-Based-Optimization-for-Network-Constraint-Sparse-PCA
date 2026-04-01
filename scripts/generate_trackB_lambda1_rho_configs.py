from pathlib import Path
import itertools

def main():
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "configs" / "trackB" / "robust_lambda1_rho"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1.5 high-decoy robustness grid
    lambda1_values = [0.05, 0.1, 0.2]
    rho_values = [2.0, 5.0, 10.0]
    lambda2_grid = [0.00, 0.02, 0.04, 0.05, 0.06, 0.08, 0.10, 0.11, 0.12, 0.15, 0.20]
    seeds = list(range(10))

    base = {
        "n": 200,
        "p": 200,
        "r": 3,
        "support_size": 10,
        "snr": 1.0,
        "max_iters": 200,
        "tol_obj": "1.0e-06",
        "tol_gap": "1.0e-04",
        "tol_orth": "1.0e-06",
        "eta_A": 0.05,
        "baseline": "PCA",
        "graph_family": "chain",
        "support_type": "connected_disjoint",
        "track": "B",
        "phase": "1_5_robust",
        "decoy_intensity": "high",
        "decoy_count": 18,
        "decoy_variance_factor": 2.5,
    }

    for lam1, rho in itertools.product(lambda1_values, rho_values):
        for seed in seeds:
            for lam2 in lambda2_grid:
                tag = f"l1_{lam1:.2f}_rho_{rho:.1f}_seed{seed}_lambda2_{lam2:.2f}"
                tag = tag.replace('.', 'p')
                cfg = {
                    **base,
                    "seed": seed,
                    "lambda1": lam1,
                    "lambda2": lam2,
                    "rho": rho,
                    "output_dir": f"results/trackB/phase1_5/robust_lambda1_rho/l1_{lam1}/rho_{rho}/seed{seed}/lambda2_{lam2}",
                }
                path = out_dir / f"graph_aligned_{tag}_decoy_high.yaml"
                with path.open("w") as f:
                    for key, val in cfg.items():
                        f.write(f"{key}: {val}\n")

if __name__ == "__main__":
    main()
