"""
Experiment 6 — Stochastic Dominance Analysis and PSR for Example 2.

Compares:
  Distribution A = OOS SR of IS-optimal strategy (seasonal exploiter)
  Distribution B = mean OOS SR across all strategies

Expected: Distribution A FIRST-ORDER STOCHASTICALLY DOMINATES B.
PSR_stat > 2.83 (higher IS SR),  PSR ≈ 1.0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import run_stochastic_dominance_analysis
from src.visualization import (
    plot_stochastic_dominance,
    plot_combined_comparison,
)

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> dict:
    print("=" * 60)
    print("EXP 6 — Stochastic Dominance & PSR for Example 2")
    print("=" * 60)

    R_OOS_nstar = np.load(DATA_DIR / "R_OOS_nstar_ex2.npy")
    R_OOS_mean  = np.load(DATA_DIR / "R_OOS_mean_ex2.npy")
    M_ex2       = np.load(DATA_DIR / "M_ex2.npy")
    n_stars     = np.load(DATA_DIR / "n_stars.npy")
    n_star_ex2  = int(n_stars[1])

    print(f"R_OOS_nstar shape: {R_OOS_nstar.shape}")
    print(f"Globally optimal strategy index: {n_star_ex2}")

    res = run_stochastic_dominance_analysis(
        R_OOS_nstar=R_OOS_nstar,
        R_OOS_mean=R_OOS_mean,
        M=M_ex2,
        n_star_full=n_star_ex2,
    )

    np.save(DATA_DIR / "sd_x_ex2.npy", res["x_grid"])
    np.save(DATA_DIR / "sd_FA_ex2.npy", res["F_A"])
    np.save(DATA_DIR / "sd_FB_ex2.npy", res["F_B"])
    np.save(DATA_DIR / "sd_SD2_ex2.npy", res["SD2"])

    print(f"\n=== Example 2 Stochastic Dominance ===")
    print(f"  FO result: {res['FO_result']}")
    print(f"  SO result: {res['SO_result']}")
    print(f"  A dominates B (FO): {res['A_dominates_B_FO']}")
    print(f"  A dominates B (SO): {res['A_dominates_B_SO']}")

    print(f"\n=== Example 2 PSR ===")
    print(f"  SR_hat_annual      = {res['SR_hat_annual']:.4f}")
    print(f"  skewness (γ₃)      = {res['skewness']:.4f}")
    print(f"  excess kurtosis (γ₄) = {res['excess_kurtosis']:.4f}")
    print(f"  PSR_stat           = {res['PSR_stat']:.4f}")
    print(f"  PSR                = {res['PSR']:.6f}")

    plot_stochastic_dominance(
        x_grid=res["x_grid"],
        F_A=res["F_A"],
        F_B=res["F_B"],
        SD2=res["SD2"],
        fo_result=res["FO_result"],
        so_result=res["SO_result"],
        example_label="Example 2 (Seasonal Effect)",
        filename="exp6_stochastic_dominance_ex2.png",
    )

    # ---- Comparative summary ----
    lambda_ex1    = np.load(DATA_DIR / "lambda_ex1.npy")
    lambda_ex2    = np.load(DATA_DIR / "lambda_ex2.npy")
    R_IS_ex1      = np.load(DATA_DIR / "R_IS_nstar_ex1.npy")
    R_OOS_n_ex1   = np.load(DATA_DIR / "R_OOS_nstar_ex1.npy")
    R_IS_ex2      = np.load(DATA_DIR / "R_IS_nstar_ex2.npy")
    R_OOS_n_ex2   = np.load(DATA_DIR / "R_OOS_nstar_ex2.npy")

    phi_ex1 = float(np.mean(lambda_ex1 < 0))
    phi_ex2 = float(np.mean(lambda_ex2 < 0))
    prob_loss_ex1 = float(np.mean(R_OOS_n_ex1 < 0))
    prob_loss_ex2 = float(np.mean(R_OOS_n_ex2 < 0))

    # Load SD arrays and PSR scalars for Example 1 (saved by exp_5)
    sd_x_ex1  = np.load(DATA_DIR / "sd_x_ex1.npy")
    sd_FA_ex1 = np.load(DATA_DIR / "sd_FA_ex1.npy")
    sd_FB_ex1 = np.load(DATA_DIR / "sd_FB_ex1.npy")
    psr_stat_ex1 = float(np.load(DATA_DIR / "psr_stat_ex1.npy"))
    psr_ex1      = float(np.load(DATA_DIR / "psr_ex1.npy"))

    print("\n" + "=" * 60)
    print("COMPARATIVE SUMMARY (Example 1 vs Example 2)")
    print("=" * 60)
    print(f"{'Metric':<28} {'Example 1':>12} {'Example 2':>12}")
    print("-" * 54)
    print(f"{'PBO (phi)':<28} {phi_ex1:>12.4f} {phi_ex2:>12.4f}")
    print(f"{'Prob_loss':<28} {prob_loss_ex1:>12.4f} {prob_loss_ex2:>12.4f}")
    print(f"{'PSR_stat':<28} {psr_stat_ex1:>12.4f} {res['PSR_stat']:>12.4f}")
    print(f"{'PSR':<28} {psr_ex1:>12.6f} {res['PSR']:>12.6f}")

    # Combined 4-panel figure
    plot_combined_comparison(
        lambda_ex1=lambda_ex1,
        lambda_ex2=lambda_ex2,
        phi_ex1=phi_ex1,
        phi_ex2=phi_ex2,
        x_grid_ex1=sd_x_ex1,
        F_A_ex1=sd_FA_ex1,
        F_B_ex1=sd_FB_ex1,
        x_grid_ex2=res["x_grid"],
        F_A_ex2=res["F_A"],
        F_B_ex2=res["F_B"],
    )

    print("\n✓ Exp 6 complete.")
    return res


if __name__ == "__main__":
    main()
