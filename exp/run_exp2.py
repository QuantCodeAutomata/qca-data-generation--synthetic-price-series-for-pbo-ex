"""
Experiment 2 — Strategy Simulation: Construction of T×N P&L Matrix M.

Loads exp_1 artifacts and simulates all 8,800 monthly seasonal strategy
configurations to produce M_ex1 and M_ex2 (each 992 × 8,800, float64).

Saves M_ex1.npy, M_ex2.npy, params.npy, SR_full_ex1.npy, SR_full_ex2.npy,
n_stars.npy to data/.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.strategy_simulation import build_pnl_matrices
from src.visualization import plot_sr_distributions

DATA_DIR = Path(__file__).parent.parent / "data"
T_ADJ = 992


def main() -> None:
    print("=" * 60)
    print("EXP 2 — Strategy Simulation  (T_adj=992, N=8,800)")
    print("=" * 60)

    # Load exp_1 artifacts
    returns_ex1 = np.load(DATA_DIR / "returns_ex1.npy")
    returns_ex2 = np.load(DATA_DIR / "returns_ex2.npy")
    calendar_df = pd.read_csv(DATA_DIR / "calendar_df.csv", index_col=0,
                              parse_dates=["date"])
    sigmas = np.load(DATA_DIR / "sigmas.npy")
    sigma_full_ex1, sigma_full_ex2 = sigmas[0], sigmas[1]

    print(f"Loaded returns shapes: ex1={returns_ex1.shape}, ex2={returns_ex2.shape}")
    print(f"Sigma_full_ex1={sigma_full_ex1:.6f},  Sigma_full_ex2={sigma_full_ex2:.6f}")

    t0 = time.time()
    result = build_pnl_matrices(
        returns_ex1=returns_ex1,
        returns_ex2=returns_ex2,
        calendar_df=calendar_df,
        sigma_full_ex1=sigma_full_ex1,
        sigma_full_ex2=sigma_full_ex2,
        T_adj=T_ADJ,
    )
    elapsed = time.time() - t0
    print(f"Simulation completed in {elapsed:.1f}s")

    M_ex1 = result["M_ex1"]
    M_ex2 = result["M_ex2"]
    params = result["params"]
    SR_full_ex1 = result["SR_full_ex1"]
    SR_full_ex2 = result["SR_full_ex2"]
    n_star_ex1 = result["n_star_ex1"]
    n_star_ex2 = result["n_star_ex2"]

    # Save
    np.save(DATA_DIR / "M_ex1.npy", M_ex1)
    np.save(DATA_DIR / "M_ex2.npy", M_ex2)
    np.save(DATA_DIR / "SR_full_ex1.npy", SR_full_ex1)
    np.save(DATA_DIR / "SR_full_ex2.npy", SR_full_ex2)
    np.save(DATA_DIR / "n_stars.npy", np.array([n_star_ex1, n_star_ex2]))

    # Save params as structured array
    import pandas as _pd
    params_df = _pd.DataFrame(params, columns=["entry_day", "holding_period",
                                                "stop_loss", "side"])
    params_df.to_csv(DATA_DIR / "params_grid.csv", index=False)

    # Report
    e1, h1, s1, d1 = params[n_star_ex1]
    e2, h2, s2, d2 = params[n_star_ex2]
    print(f"\nM_ex1 shape: {M_ex1.shape},  M_ex2 shape: {M_ex2.shape}")
    print(f"M_ex1 sparsity: {(M_ex1==0).mean()*100:.1f}% zeros")
    print(f"M_ex2 sparsity: {(M_ex2==0).mean()*100:.1f}% zeros")
    print(f"\nExample 1 globally optimal strategy (n={n_star_ex1}):")
    print(f"  Entry={e1}, Holding={h1}, StopLoss={s1}, Side={d1:+d}")
    print(f"  Full-sample SR = {SR_full_ex1[n_star_ex1]:.4f}")
    print(f"\nExample 2 globally optimal strategy (n={n_star_ex2}):")
    print(f"  Entry={e2}, Holding={h2}, StopLoss={s2}, Side={d2:+d}")
    print(f"  Full-sample SR = {SR_full_ex2[n_star_ex2]:.4f}")
    print(f"\nSR_full_ex1: mean={SR_full_ex1.mean():.4f}, "
          f"std={SR_full_ex1.std():.4f}, max={SR_full_ex1.max():.4f}")
    print(f"SR_full_ex2: mean={SR_full_ex2.mean():.4f}, "
          f"std={SR_full_ex2.std():.4f}, max={SR_full_ex2.max():.4f}")

    plot_sr_distributions(SR_full_ex1, SR_full_ex2)

    print("\n✓ Exp 2 complete — matrices saved to data/")


if __name__ == "__main__":
    main()
