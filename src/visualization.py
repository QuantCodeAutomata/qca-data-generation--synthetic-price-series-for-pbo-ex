"""
Visualization helpers for PBO / CSCV experiments.

All plots are saved to the results/ directory.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, filename: str) -> None:
    path = RESULTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# exp_1: price series
# ---------------------------------------------------------------------------

def plot_price_series(
    prices_ex1: np.ndarray,
    prices_ex2: np.ndarray,
    filename: str = "exp1_price_series.png",
) -> None:
    """Plot synthetic price series for both examples."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(prices_ex1, color="steelblue", linewidth=0.8)
    axes[0].set_title("Example 1 — Pure Random Walk (P₀=100)")
    axes[0].set_ylabel("Price")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(prices_ex2, color="darkorange", linewidth=0.8)
    axes[1].set_title("Example 2 — Random Walk with Monthly Seasonal Effect")
    axes[1].set_ylabel("Price")
    axes[1].set_xlabel("Trading Day")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, filename)


# ---------------------------------------------------------------------------
# exp_2: SR distributions
# ---------------------------------------------------------------------------

def plot_sr_distributions(
    SR_full_ex1: np.ndarray,
    SR_full_ex2: np.ndarray,
    filename: str = "exp2_sr_distributions.png",
) -> None:
    """Plot histograms of full-sample Sharpe ratios for both examples."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(SR_full_ex1, bins=50, color="steelblue", alpha=0.7, edgecolor="none")
    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].set_title("Example 1: Full-Sample SR Distribution (8,800 strategies)")
    axes[0].set_xlabel("Annualised Sharpe Ratio")
    axes[0].set_ylabel("Count")

    axes[1].hist(SR_full_ex2, bins=50, color="darkorange", alpha=0.7, edgecolor="none")
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_title("Example 2: Full-Sample SR Distribution (8,800 strategies)")
    axes[1].set_xlabel("Annualised Sharpe Ratio")

    fig.tight_layout()
    _save(fig, filename)


# ---------------------------------------------------------------------------
# exp_3/exp_4: logit histograms
# ---------------------------------------------------------------------------

def plot_logit_histogram(
    lambda_all: np.ndarray,
    phi: float,
    example_label: str,
    filename: str,
) -> None:
    """Plot histogram of logit λ_c values with PBO annotation."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(lambda_all, bins=40, color="steelblue", alpha=0.7, edgecolor="none")
    ax.axvline(0, color="red", linewidth=2, label="λ = 0")
    ax.set_title(f"{example_label}: Logit Distribution  (PBO = {phi:.3f})")
    ax.set_xlabel("Logit of Relative OOS Rank  λ_c")
    ax.set_ylabel("Count")
    ax.annotate(
        f"PBO = {phi:.1%}",
        xy=(0, ax.get_ylim()[1] * 0.9),
        xytext=(0.02, 0.9),
        textcoords="axes fraction",
        fontsize=12,
        color="red",
    )
    ax.legend()
    fig.tight_layout()
    _save(fig, filename)


# ---------------------------------------------------------------------------
# exp_3/exp_4: performance degradation scatter
# ---------------------------------------------------------------------------

def plot_performance_degradation(
    R_IS_nstar: np.ndarray,
    R_OOS_nstar: np.ndarray,
    intercept: float,
    slope: float,
    adj_r2: float,
    example_label: str,
    filename: str,
) -> None:
    """Scatter plot of OOS SR vs IS SR with OLS regression line."""
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(R_IS_nstar, R_OOS_nstar, s=2, alpha=0.3, color="steelblue")

    x_line = np.linspace(R_IS_nstar.min(), R_IS_nstar.max(), 200)
    y_line = intercept + slope * x_line
    ax.plot(x_line, y_line, color="red", linewidth=2,
            label=f"OLS: α={intercept:.3f}, β={slope:.3f}, adj-R²={adj_r2:.3f}")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    ax.set_title(f"{example_label}: Performance Degradation")
    ax.set_xlabel("IS Sharpe Ratio")
    ax.set_ylabel("OOS Sharpe Ratio")
    ax.legend()
    fig.tight_layout()
    _save(fig, filename)


# ---------------------------------------------------------------------------
# exp_5/exp_6: stochastic dominance CDF
# ---------------------------------------------------------------------------

def plot_stochastic_dominance(
    x_grid: np.ndarray,
    F_A: np.ndarray,
    F_B: np.ndarray,
    SD2: np.ndarray,
    fo_result: str,
    so_result: str,
    example_label: str,
    filename: str,
) -> None:
    """Two-panel plot: empirical CDFs and SD2 integral."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 9))

    axes[0].plot(x_grid, F_A, color="steelblue", label="Distribution A (IS-optimal OOS SR)")
    axes[0].plot(x_grid, F_B, color="darkorange", label="Distribution B (Mean OOS SR)")
    axes[0].set_title(f"{example_label}: Empirical CDFs\n{fo_result}")
    axes[0].set_xlabel("Sharpe Ratio")
    axes[0].set_ylabel("CDF")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_grid, SD2, color="purple", linewidth=1.5)
    axes[1].axhline(0, color="red", linewidth=1, linestyle="--")
    axes[1].fill_between(x_grid, SD2, 0, where=(SD2 >= 0),
                         alpha=0.2, color="green", label="SD2 ≥ 0 (A dominates)")
    axes[1].fill_between(x_grid, SD2, 0, where=(SD2 < 0),
                         alpha=0.2, color="red", label="SD2 < 0 (B dominates)")
    axes[1].set_title(f"SD2 Integral\n{so_result}")
    axes[1].set_xlabel("Sharpe Ratio")
    axes[1].set_ylabel("∫(F_B − F_A) dx")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, filename)


# ---------------------------------------------------------------------------
# exp_6: combined 4-panel comparison
# ---------------------------------------------------------------------------

def plot_combined_comparison(
    lambda_ex1: np.ndarray,
    lambda_ex2: np.ndarray,
    phi_ex1: float,
    phi_ex2: float,
    x_grid_ex1: np.ndarray,
    F_A_ex1: np.ndarray,
    F_B_ex1: np.ndarray,
    x_grid_ex2: np.ndarray,
    F_A_ex2: np.ndarray,
    F_B_ex2: np.ndarray,
    filename: str = "exp6_combined_comparison.png",
) -> None:
    """4-panel figure: logit histograms and CDF comparisons for both examples."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # Top-left: logit Example 1
    axes[0, 0].hist(lambda_ex1, bins=40, color="steelblue", alpha=0.7, edgecolor="none")
    axes[0, 0].axvline(0, color="red", linewidth=2)
    axes[0, 0].set_title(f"Example 1 — Logit Distribution\nPBO = {phi_ex1:.1%}")
    axes[0, 0].set_xlabel("λ_c")
    axes[0, 0].set_ylabel("Count")

    # Top-right: logit Example 2
    axes[0, 1].hist(lambda_ex2, bins=40, color="darkorange", alpha=0.7, edgecolor="none")
    axes[0, 1].axvline(0, color="red", linewidth=2)
    axes[0, 1].set_title(f"Example 2 — Logit Distribution\nPBO = {phi_ex2:.1%}")
    axes[0, 1].set_xlabel("λ_c")
    axes[0, 1].set_ylabel("Count")

    # Bottom-left: CDF Example 1
    axes[1, 0].plot(x_grid_ex1, F_A_ex1, color="steelblue", label="IS-Optimal OOS SR")
    axes[1, 0].plot(x_grid_ex1, F_B_ex1, color="darkorange", label="Mean OOS SR")
    axes[1, 0].set_title("Example 1 — Empirical CDFs")
    axes[1, 0].set_xlabel("Sharpe Ratio")
    axes[1, 0].set_ylabel("CDF")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # Bottom-right: CDF Example 2
    axes[1, 1].plot(x_grid_ex2, F_A_ex2, color="steelblue", label="IS-Optimal OOS SR")
    axes[1, 1].plot(x_grid_ex2, F_B_ex2, color="darkorange", label="Mean OOS SR")
    axes[1, 1].set_title("Example 2 — Empirical CDFs")
    axes[1, 1].set_xlabel("Sharpe Ratio")
    axes[1, 1].set_ylabel("CDF")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("CSCV Results: Example 1 (No Effect) vs Example 2 (Seasonal Effect)",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    _save(fig, filename)
