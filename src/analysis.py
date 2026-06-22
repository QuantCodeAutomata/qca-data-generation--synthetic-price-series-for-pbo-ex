"""
Stochastic dominance analysis and Probabilistic Sharpe Ratio (PSR) computation.

Implements:
  - Empirical CDF construction
  - First-order and second-order stochastic dominance checks
  - PSR-Stat calculation (Bailey et al., 2014)

Custom — Context7 found no library equivalent (paper Sections 4-5).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.stats


# ---------------------------------------------------------------------------
# Empirical CDF
# ---------------------------------------------------------------------------

def empirical_cdf(
    data: np.ndarray,
    x_grid: np.ndarray,
) -> np.ndarray:
    """Compute empirical CDF of ``data`` evaluated on ``x_grid``.

    Uses sorted-array + searchsorted for O(N log N + M log N) complexity.

    Parameters
    ----------
    data:
        1-D sample array of length N.
    x_grid:
        1-D sorted evaluation grid of length M.

    Returns
    -------
    np.ndarray of shape (M,) with values in [0, 1].
    """
    sorted_data = np.sort(data)
    return np.searchsorted(sorted_data, x_grid, side="right") / len(data)


# Public alias used by tests
compute_empirical_cdf = empirical_cdf


# ---------------------------------------------------------------------------
# First-order stochastic dominance
# ---------------------------------------------------------------------------

def assess_first_order_dominance(
    F_A: np.ndarray,
    F_B: np.ndarray,
    tol: float = 1e-6,
) -> Tuple[str, bool, bool]:
    """Assess first-order stochastic dominance from pre-computed CDFs.

    A first-order stochastically dominates B ⟺ F_A(x) ≤ F_B(x) for all x.

    Parameters
    ----------
    F_A, F_B:
        Empirical CDF arrays evaluated on the same x_grid.
    tol:
        Numerical tolerance for dominance checks.

    Returns
    -------
    Tuple (result_string, A_dominates_B, B_dominates_A).
    """
    diff_FO = F_A - F_B
    A_dom_B = bool(np.all(diff_FO <= tol))
    B_dom_A = bool(np.all(diff_FO >= -tol))

    if A_dom_B and not B_dom_A:
        result = "Distribution A first-order stochastically dominates B"
    elif B_dom_A and not A_dom_B:
        result = "Distribution B first-order stochastically dominates A"
    elif A_dom_B and B_dom_A:
        result = "Distributions are approximately equal (both FO dominate)"
    else:
        result = "No first-order stochastic dominance"

    return result, A_dom_B, B_dom_A


# ---------------------------------------------------------------------------
# Second-order stochastic dominance
# ---------------------------------------------------------------------------

def compute_second_order_dominance(
    F_A: np.ndarray,
    F_B: np.ndarray,
    x_grid: np.ndarray,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, str, bool]:
    """Compute second-order stochastic dominance integral and assess dominance.

    SD2(x) = ∫_{-∞}^{x} [F_B(t) - F_A(t)] dt ≥ 0 ∀x  ⟺  A SO dominates B.

    Parameters
    ----------
    F_A, F_B:
        Empirical CDF arrays evaluated on x_grid.
    x_grid:
        Evaluation grid (uniform spacing assumed).
    tol:
        Numerical tolerance.

    Returns
    -------
    Tuple (SD2_array, result_string, A_dominates_B).
    """
    delta_x = x_grid[1] - x_grid[0]
    SD2 = np.cumsum((F_B - F_A) * delta_x)

    A_dom_B = bool(np.all(SD2 >= -tol) and np.any(SD2 > tol))
    B_dom_A = bool(np.all(SD2 <= tol) and np.any(SD2 < -tol))

    if A_dom_B:
        result = "Distribution A second-order stochastically dominates B"
    elif B_dom_A:
        result = "Distribution B second-order stochastically dominates A"
    else:
        result = "No second-order stochastic dominance"

    return SD2, result, A_dom_B


# ---------------------------------------------------------------------------
# Stochastic dominance (combined helper)
# ---------------------------------------------------------------------------

def stochastic_dominance(
    dist_A: np.ndarray,
    dist_B: np.ndarray,
    n_grid: int = 1000,
    tol: float = 1e-6,
) -> dict:
    """Assess first- and second-order stochastic dominance between two distributions.

    Distribution A first-order stochastically dominates B  ⟺  F_A(x) ≤ F_B(x) ∀x.
    Second-order: ∫_{-∞}^{x} [F_B(t) - F_A(t)] dt ≥ 0 ∀x.

    Parameters
    ----------
    dist_A, dist_B:
        Sample arrays (e.g. R_OOS_nstar and R_OOS_mean from CSCV).
    n_grid:
        Number of evaluation points.
    tol:
        Numerical tolerance for dominance checks.

    Returns
    -------
    dict with keys:
        x_grid      — evaluation grid
        F_A, F_B    — empirical CDFs on x_grid
        SD2         — cumulative integral ∫(F_B - F_A) dx  (shape n_grid)
        FO_result   — string describing first-order dominance conclusion
        SO_result   — string describing second-order dominance conclusion
        A_dominates_B_FO — bool
        A_dominates_B_SO — bool
    """
    x_min = min(np.min(dist_A), np.min(dist_B)) - 0.1
    x_max = max(np.max(dist_A), np.max(dist_B)) + 0.1
    x_grid = np.linspace(x_min, x_max, n_grid)

    F_A = empirical_cdf(dist_A, x_grid)
    F_B = empirical_cdf(dist_B, x_grid)

    FO_result, A_dom_B_FO, _ = assess_first_order_dominance(F_A, F_B, tol)
    SD2, SO_result, A_dom_B_SO = compute_second_order_dominance(F_A, F_B, x_grid, tol)

    return {
        "x_grid": x_grid,
        "F_A": F_A,
        "F_B": F_B,
        "SD2": SD2,
        "FO_result": FO_result,
        "SO_result": SO_result,
        "A_dominates_B_FO": A_dom_B_FO,
        "A_dominates_B_SO": A_dom_B_SO,
    }


# ---------------------------------------------------------------------------
# Probabilistic Sharpe Ratio
# ---------------------------------------------------------------------------

def compute_psr(
    returns: np.ndarray,
    SR_star: float = 0.0,
    annualisation: float = 252.0,
) -> Tuple[float, float]:
    """Compute the Probabilistic Sharpe Ratio (PSR) statistic.

    Uses the Bailey & López de Prado (2012) formula accounting for non-normality
    via skewness and excess kurtosis of the return series.

    Formula (using excess kurtosis γ₄ where normal = 0):
        PSR-Stat = (SR̂_daily - SR*) × √(T-1)
                   ─────────────────────────────────────────────────
                   √[1 - γ₃ × SR̂_daily + (γ₄ + 2)/4 × SR̂_daily²]

    PSR = Φ(PSR-Stat), where Φ is the standard normal CDF.

    Parameters
    ----------
    returns:
        Daily return series (1-D array) for the strategy.
    SR_star:
        Benchmark SR (null hypothesis), default 0.
    annualisation:
        Annualisation factor (252 for daily data).

    Returns
    -------
    Tuple (PSR_stat, PSR) where PSR ∈ [0, 1].
    """
    T = len(returns)
    mu = float(np.mean(returns))
    sigma_r = float(np.std(returns, ddof=1))

    SR_hat_daily = mu / sigma_r if sigma_r > 0 else 0.0

    # scipy.stats.skew uses Fisher's definition; scipy.stats.kurtosis with
    # fisher=True returns excess kurtosis (normal = 0)
    gamma3 = float(scipy.stats.skew(returns))
    gamma4_excess = float(scipy.stats.kurtosis(returns, fisher=True))

    # PSR denominator: 1 - γ₃·SR + (γ₄_excess + 2)/4 · SR²
    denom_sq = (
        1.0
        - gamma3 * SR_hat_daily
        + (gamma4_excess + 2.0) / 4.0 * SR_hat_daily ** 2
    )
    denom_sq = max(denom_sq, 1e-12)

    PSR_stat = float((SR_hat_daily - SR_star) * np.sqrt(T - 1) / np.sqrt(denom_sq))
    PSR = float(scipy.stats.norm.cdf(PSR_stat))

    return PSR_stat, PSR


def _compute_psr_dict(
    returns: np.ndarray,
    SR_star: float = 0.0,
    annualisation: float = 252.0,
) -> dict:
    """Internal PSR computation returning a rich dict for analysis output."""
    T = len(returns)
    mu = float(np.mean(returns))
    sigma_r = float(np.std(returns, ddof=1))
    SR_hat_daily = mu / sigma_r if sigma_r > 0 else 0.0
    SR_hat_annual = SR_hat_daily * np.sqrt(annualisation)
    gamma3 = float(scipy.stats.skew(returns))
    gamma4_excess = float(scipy.stats.kurtosis(returns, fisher=True))

    PSR_stat, PSR = compute_psr(returns, SR_star=SR_star, annualisation=annualisation)

    return {
        "SR_hat_daily": SR_hat_daily,
        "SR_hat_annual": SR_hat_annual,
        "skewness": gamma3,
        "excess_kurtosis": gamma4_excess,
        "PSR_stat": PSR_stat,
        "PSR": PSR,
        "T": T,
    }


# ---------------------------------------------------------------------------
# Full analysis function
# ---------------------------------------------------------------------------

def run_stochastic_dominance_analysis(
    R_OOS_nstar: np.ndarray,
    R_OOS_mean: np.ndarray,
    M: np.ndarray,
    n_star_full: int,
    n_grid: int = 1000,
    tol: float = 1e-6,
) -> dict:
    """Run full stochastic dominance and PSR analysis.

    Parameters
    ----------
    R_OOS_nstar:
        (C,) array of OOS SR of IS-optimal strategy per CSCV combination.
    R_OOS_mean:
        (C,) array of mean OOS SR across all strategies per combination.
    M:
        (T_adj, N) full P&L matrix to extract the globally optimal strategy's
        return series.
    n_star_full:
        Column index of the globally optimal strategy in M.
    n_grid, tol:
        Passed to ``stochastic_dominance``.

    Returns
    -------
    dict merging stochastic dominance results and PSR results.
    """
    sd_results = stochastic_dominance(
        dist_A=R_OOS_nstar,
        dist_B=R_OOS_mean,
        n_grid=n_grid,
        tol=tol,
    )

    r_nstar = M[:, n_star_full]
    psr_results = _compute_psr_dict(returns=r_nstar)

    return {**sd_results, **psr_results}
