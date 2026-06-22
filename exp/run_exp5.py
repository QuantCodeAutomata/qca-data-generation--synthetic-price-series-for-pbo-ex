"""
Experiment 5 — Stochastic Dominance Analysis and PSR for Example 1.

Compares:
  Distribution A = OOS SR of IS-optimal strategy across all CSCV combinations
  Distribution B = mean OOS SR across all strategies per combination

Expected: no FO dominance of A over B (possibly B dominates A for pure random walk).
PSR_stat ≈ 2.83,  PSR ≈ 0.997.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import run_stochastic_dominance_analysis
from src.visualization import plot_stochastic_dominance

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> dict:
    print("=" * 60)
    print("EXP 5 — Stochastic Dominance & PSR for Example 1")
    print("=" * 60)

    R_OOS_nstar = np.load(DATA_DIR / "R_OOS_nstar_ex1.npy")
    R_OOS_mean  = np.load(DATA_DIR / "R_OOS_mean_ex1.npy")
    M_ex1       = np.load(DATA_DIR / "M_ex1.npy")
    n_stars     = np.load(DATA_DIR / "n_stars.npy")
    n_star_ex1  = int(n_stars[0])

    print(f"R_OOS_nstar shape: {R_OOS_nstar.shape}")
    print(f"R_OOS_mean  shape: {R_OOS_mean.shape}")
    print(f"Globally optimal strategy index: {n_star_ex1}")

    res = run_stochastic_dominance_analysis(
        R_OOS_nstar=R_OOS_nstar,
        R_OOS_mean=R_OOS_mean,
        M=M_ex1,
        n_star_full=n_star_ex1,
    )

    # Save SD arrays for combined plot in exp_6
    np.save(DATA_DIR / "sd_x_ex1.npy", res["x_grid"])
    np.save(DATA_DIR / "sd_FA_ex1.npy", res["F_A"])
    np.save(DATA_DIR / "sd_FB_ex1.npy", res["F_B"])
    np.save(DATA_DIR / "sd_SD2_ex1.npy", res["SD2"])
    # Save PSR scalars for cross-experiment comparison in exp_6
    np.save(DATA_DIR / "psr_stat_ex1.npy", np.array(res["PSR_stat"]))
    np.save(DATA_DIR / "psr_ex1.npy", np.array(res["PSR"]))

    print(f"\n=== Example 1 Stochastic Dominance ===")
    print(f"  FO result: {res['FO_result']}")
    print(f"  SO result: {res['SO_result']}")
    print(f"  A dominates B (FO): {res['A_dominates_B_FO']}")
    print(f"  A dominates B (SO): {res['A_dominates_B_SO']}")

    print(f"\n=== Example 1 PSR ===")
    print(f"  SR_hat_annual      = {res['SR_hat_annual']:.4f}")
    print(f"  skewness (γ₃)      = {res['skewness']:.4f}")
    print(f"  excess kurtosis (γ₄) = {res['excess_kurtosis']:.4f}")
    print(f"  PSR_stat           = {res['PSR_stat']:.4f}  (expected ≈ 2.83)")
    print(f"  PSR                = {res['PSR']:.6f}  (expected ≈ 0.997)")

    plot_stochastic_dominance(
        x_grid=res["x_grid"],
        F_A=res["F_A"],
        F_B=res["F_B"],
        SD2=res["SD2"],
        fo_result=res["FO_result"],
        so_result=res["SO_result"],
        example_label="Example 1 (Pure Random Walk)",
        filename="exp5_stochastic_dominance_ex1.png",
    )

    print("\n✓ Exp 5 complete.")
    return res


if __name__ == "__main__":
    main()
