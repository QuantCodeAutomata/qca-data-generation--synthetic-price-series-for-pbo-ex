"""
Experiment 3 — CSCV: PBO, Performance Degradation, and Probability of Loss
for Example 1 (Pure Random Walk).

Expected results:
  PBO ≈ 0.55  (55%)
  Prob_loss ≈ 0.53
  alpha ≈ 0.87, beta ≈ -0.75, adj_R² ≈ 0.17
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
    print("EXP 3 — CSCV on Example 1 (pure random walk)")
    print("=" * 60)

    M_ex1 = np.load(DATA_DIR / "M_ex1.npy")
    print(f"Loaded M_ex1: shape={M_ex1.shape}")

    t0 = time.time()
    res = run_cscv(M_ex1, S=16)
    elapsed = time.time() - t0
    print(f"CSCV completed in {elapsed:.1f}s  ({res['n_combinations']} combinations)")

    # Save arrays for downstream experiments
    np.save(DATA_DIR / "lambda_ex1.npy", res["lambda_all"])
    np.save(DATA_DIR / "R_IS_nstar_ex1.npy", res["R_IS_nstar"])
    np.save(DATA_DIR / "R_OOS_nstar_ex1.npy", res["R_OOS_nstar"])
    np.save(DATA_DIR / "R_OOS_mean_ex1.npy", res["R_OOS_mean"])

    phi = res["phi"]
    ci = res["phi_ci"]
    print(f"\n=== Example 1 CSCV Results ===")
    print(f"  PBO (phi)          = {phi:.4f}  [{ci[0]:.4f}, {ci[1]:.4f}]  (expected ≈ 0.55)")
    print(f"  Prob_loss          = {res['prob_loss']:.4f}  (expected ≈ 0.53)")
    print(f"  alpha (intercept)  = {res['intercept']:.4f}  (expected ≈ 0.87)")
    print(f"  beta  (slope)      = {res['slope']:.4f}  (expected ≈ -0.75)")
    print(f"  adj-R²             = {res['adj_r2']:.4f}  (expected ≈ 0.17)")
    print(f"  lambda mean        = {res['lambda_all'].mean():.4f}")
    print(f"  lambda std         = {res['lambda_all'].std():.4f}")

    # Plots
    plot_logit_histogram(
        lambda_all=res["lambda_all"],
        phi=phi,
        example_label="Example 1 (Pure Random Walk)",
        filename="exp3_logit_histogram_ex1.png",
    )
    plot_performance_degradation(
        R_IS_nstar=res["R_IS_nstar"],
        R_OOS_nstar=res["R_OOS_nstar"],
        intercept=res["intercept"],
        slope=res["slope"],
        adj_r2=res["adj_r2"],
        example_label="Example 1 (Pure Random Walk)",
        filename="exp3_performance_degradation_ex1.png",
    )

    print("\n✓ Exp 3 complete — arrays and plots saved.")
    return res


if __name__ == "__main__":
    main()
