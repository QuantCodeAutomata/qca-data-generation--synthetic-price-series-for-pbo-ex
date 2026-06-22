"""
Experiment 1 — Data Generation: Synthetic Price Series for PBO Experiments.

Generates and saves:
  - returns_ex1.npy, returns_ex2.npy
  - calendar_df.csv
  - prices_ex1.npy, prices_ex2.npy
  - scalars_ex1.npy (sigma_full_ex1, sigma_full_ex2)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Allow running as script from any directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generation import generate_all_artifacts
from src.visualization import plot_price_series

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("=" * 60)
    print("EXP 1 — Synthetic Data Generation")
    print("=" * 60)

    artifacts = generate_all_artifacts(
        T=1000,
        sigma=0.01,
        seed=42,
        start_date="2000-01-03",
        shift_multiple=0.25,
        max_bd=5,
    )

    returns_ex1 = artifacts["returns_ex1"]
    returns_ex2 = artifacts["returns_ex2"]
    calendar_df = artifacts["calendar_df"]
    sigma_full_ex1 = artifacts["sigma_full_ex1"]
    sigma_full_ex2 = artifacts["sigma_full_ex2"]
    prices_ex1 = artifacts["prices_ex1"]
    prices_ex2 = artifacts["prices_ex2"]
    seasonal_indices = artifacts["seasonal_indices"]

    # Save artifacts
    np.save(DATA_DIR / "returns_ex1.npy", returns_ex1)
    np.save(DATA_DIR / "returns_ex2.npy", returns_ex2)
    np.save(DATA_DIR / "prices_ex1.npy", prices_ex1)
    np.save(DATA_DIR / "prices_ex2.npy", prices_ex2)
    np.save(DATA_DIR / "sigmas.npy", np.array([sigma_full_ex1, sigma_full_ex2]))
    calendar_df.to_csv(DATA_DIR / "calendar_df.csv", index=True)

    # Diagnostics
    n_months = calendar_df["month_key"].nunique()
    n_seasonal = int(seasonal_indices.sum())

    print(f"\nreturns_ex1 : shape={returns_ex1.shape}, mean={returns_ex1.mean():.6f}, "
          f"std={returns_ex1.std(ddof=1):.6f}")
    print(f"returns_ex2 : shape={returns_ex2.shape}, mean={returns_ex2.mean():.6f}, "
          f"std={returns_ex2.std(ddof=1):.6f}")
    print(f"sigma_full_ex1 = {sigma_full_ex1:.6f}")
    print(f"sigma_full_ex2 = {sigma_full_ex2:.6f}")
    print(f"Seasonal shift magnitude = {0.25 * sigma_full_ex1:.6f}")
    print(f"Number of distinct months = {n_months}")
    print(f"Number of seasonal days   = {n_seasonal}  (expected ≈ {5*n_months})")
    print(f"prices_ex1[0] = {prices_ex1[0]:.2f}, prices_ex1[-1] = {prices_ex1[-1]:.4f}")
    print(f"prices_ex2[0] = {prices_ex2[0]:.2f}, prices_ex2[-1] = {prices_ex2[-1]:.4f}")

    print(f"\nCalendar sample (first 10 rows):\n{calendar_df.head(10).to_string()}")

    # Plot
    plot_price_series(prices_ex1, prices_ex2)

    print("\n✓ Exp 1 complete — artifacts saved to data/")


if __name__ == "__main__":
    main()
