"""
Experiment 4 — CSCV: PBO, Performance Degradation, and Probability of Loss
for Example 2 (Random Walk with Seasonal Effect).

Expected results:
  PBO ≈ 0.13  (13%)
  Prob_loss ≈ 0.13
  alpha ≈ 1.78, beta ≈ -0.35, adj_R² ≈ 0.05
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cscv import run_cscv
from src.visualization import plot_logit_histogram, plot_performance_degradation

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("=" * 60)
    print("EXP 4 — CSCV on Example 2 (seasonal effect)")
    print("=" * 60)

    M_ex2 = np.load(DATA_DIR / "M_ex2.npy")
    print(f"Loaded M_ex2: shape={M_ex2.shape}")

    t0 = time.time()
    res = run_cscv(M_ex2, S=16)
    elapsed = time.time() - t0
    print(f"CSCV completed in {elapsed:.1f}s  ({res['n_combinations']} combinations)")

    # Save arrays for downstream experiments
    np.save(DATA_DIR / "lambda_ex2.npy", res["lambda_all"])
    np.save(DATA_DIR / "R_IS_nstar_ex2.npy", res["R_IS_nstar"])
    np.save(DATA_DIR / "R_OOS_nstar_ex2.npy", res["R_OOS_nstar"])
    np.save(DATA_DIR / "R_OOS_mean_ex2.npy", res["R_OOS_mean"])

    phi = res["phi"]
    ci = res["phi_ci"]
    print(f"\n=== Example 2 CSCV Results ===")
    print(f"  PBO (phi)          = {phi:.4f}  [{ci[0]:.4f}, {ci[1]:.4f}]  (expected ≈ 0.13)")
    print(f"  Prob_loss          = {res['prob_loss']:.4f}  (expected ≈ 0.13)")
    print(f"  alpha (intercept)  = {res['intercept']:.4f}  (expected ≈ 1.78)")
    print(f"  beta  (slope)      = {res['slope']:.4f}  (expected ≈ -0.35)")
    print(f"  adj-R²             = {res['adj_r2']:.4f}  (expected ≈ 0.05)")
    print(f"  lambda mean        = {res['lambda_all'].mean():.4f}")
    print(f"  lambda std         = {res['lambda_all'].std():.4f}")

    # Plots
    plot_logit_histogram(
        lambda_all=res["lambda_all"],
        phi=phi,
        example_label="Example 2 (Seasonal Effect)",
        filename="exp4_logit_histogram_ex2.png",
    )
    plot_performance_degradation(
        R_IS_nstar=res["R_IS_nstar"],
        R_OOS_nstar=res["R_OOS_nstar"],
        intercept=res["intercept"],
        slope=res["slope"],
        adj_r2=res["adj_r2"],
        example_label="Example 2 (Seasonal Effect)",
        filename="exp4_performance_degradation_ex2.png",
    )

    print("\n✓ Exp 4 complete — arrays and plots saved.")
    return res


if __name__ == "__main__":
    main()
