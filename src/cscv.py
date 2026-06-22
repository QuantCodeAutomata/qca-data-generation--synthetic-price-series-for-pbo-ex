"""
Combinatorially Symmetric Cross-Validation (CSCV) algorithm.

Implements the PBO methodology from Bailey et al. (2014) / López de Prado (2018):
  1. Partition the T_adj x N P&L matrix into S equal-length slices.
  2. Enumerate all C(S, S/2) IS/OOS split combinations.
  3. For each combination: compute IS and OOS Sharpe ratios for all N strategies,
     identify the IS-optimal strategy, record its OOS rank (logit λ_c).
  4. Estimate PBO, performance degradation, and probability of OOS loss.

Custom — Context7 found no library equivalent (paper Section 3).
"""

from __future__ import annotations

import itertools
from typing import Tuple

import numpy as np
import scipy.stats


# ---------------------------------------------------------------------------
# Partition helper
# ---------------------------------------------------------------------------

def partition_matrix(
    M: np.ndarray,
    S: int = 16,
) -> list:
    """Split M into S equal-length row-slices.

    Parameters
    ----------
    M:
        (T_adj, N) matrix.  T_adj must be divisible by S.
    S:
        Number of partitions.

    Returns
    -------
    List of S arrays each of shape (T_adj // S, N).
    """
    T_adj = M.shape[0]
    assert T_adj % S == 0, f"T_adj={T_adj} not divisible by S={S}"
    T_s = T_adj // S
    return [M[s * T_s: (s + 1) * T_s, :] for s in range(S)]


# ---------------------------------------------------------------------------
# Pre-compute per-partition sums for efficient SR computation
# ---------------------------------------------------------------------------

def precompute_partition_stats(
    M: np.ndarray,
    S: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-partition sums of returns and squared returns.

    Parameters
    ----------
    M:
        (T_adj, N) daily returns matrix.  T_adj must be divisible by S.
    S:
        Number of partitions.

    Returns
    -------
    sum_s   — (S, N) array: sum of returns per partition per strategy
    sumsq_s — (S, N) array: sum of squared returns per partition per strategy
    """
    T_adj, N = M.shape
    assert T_adj % S == 0, f"T_adj={T_adj} not divisible by S={S}"
    T_s = T_adj // S

    sum_s = np.zeros((S, N), dtype=np.float64)
    sumsq_s = np.zeros((S, N), dtype=np.float64)

    for s in range(S):
        block = M[s * T_s: (s + 1) * T_s, :]
        sum_s[s, :] = block.sum(axis=0)
        sumsq_s[s, :] = (block ** 2).sum(axis=0)

    return sum_s, sumsq_s


# ---------------------------------------------------------------------------
# Vectorised Sharpe ratio from aggregated statistics
# ---------------------------------------------------------------------------

def _sr_from_stats(
    sum_vals: np.ndarray,
    sumsq_vals: np.ndarray,
    n_obs: int,
    annualisation: float = 252.0,
) -> np.ndarray:
    """Compute annualised Sharpe ratio from aggregated sum/sumsq arrays.

    Parameters
    ----------
    sum_vals:
        (N,) array of summed returns.
    sumsq_vals:
        (N,) array of summed squared returns.
    n_obs:
        Total number of observations (k * T_s).
    annualisation:
        Annualisation factor.

    Returns
    -------
    np.ndarray (N,) of annualised Sharpe ratios.
    """
    mu = sum_vals / n_obs
    # Unbiased variance via biased variance * n/(n-1)
    var_biased = sumsq_vals / n_obs - mu ** 2
    var_unb = np.maximum(var_biased * n_obs / (n_obs - 1), 0.0)
    # When variance is zero (constant series) the SR sign is determined by mu.
    # Use a tiny floor so positive-mean constant returns yield positive SR.
    # The floor 1e-300 keeps values finite (no inf) but allows sign(mu) to
    # dominate when var_unb is identically zero.
    safe_var = np.where(var_unb > 0, var_unb, np.where(mu != 0, 1e-300, 1.0))
    sr = mu / np.sqrt(safe_var) * np.sqrt(annualisation)
    # Zero-mean zero-variance → SR = 0
    sr = np.where((var_unb == 0) & (mu == 0), 0.0, sr)
    return sr


# ---------------------------------------------------------------------------
# Main CSCV function
# ---------------------------------------------------------------------------

def run_cscv(
    M: np.ndarray,
    S: int = 16,
    annualisation: float = 252.0,
) -> dict:
    """Run the full CSCV algorithm on a P&L matrix.

    Parameters
    ----------
    M:
        (T_adj, N) daily strategy returns matrix.  T_adj must be divisible by S.
    S:
        Number of equal partitions (default 16 → C(16,8)=12,870 combinations).
    annualisation:
        Annualisation factor for Sharpe ratio (252 for daily data).

    Returns
    -------
    dict with keys:
        lambda_all      — (C,) logit of relative OOS rank for IS-optimal strategy
        R_IS_nstar      — (C,) IS SR of IS-optimal strategy
        R_OOS_nstar     — (C,) OOS SR of IS-optimal strategy
        R_OOS_mean      — (C,) mean OOS SR across all N strategies
        phi             — scalar PBO estimate
        phi_ci          — tuple (lower, upper) 95% CI
        prob_loss       — fraction of combinations where OOS SR < 0
        slope           — performance degradation OLS slope (beta)
        intercept       — performance degradation OLS intercept (alpha)
        r_value         — correlation coefficient
        adj_r2          — adjusted R^2
        p_value         — p-value of slope
        n_combinations  — total number of combinations C(S, S//2)
    """
    T_adj, N = M.shape
    assert T_adj % S == 0, f"T_adj={T_adj} must be divisible by S={S}"
    T_s = T_adj // S
    half = S // 2
    n_IS = half * T_s
    n_OOS = half * T_s

    # Pre-compute partition statistics
    sum_s, sumsq_s = precompute_partition_stats(M, S)

    # All C(S, S//2) IS partition combinations
    combos = list(itertools.combinations(range(S), half))
    C = len(combos)

    lambda_all = np.zeros(C, dtype=np.float64)
    R_IS_nstar = np.zeros(C, dtype=np.float64)
    R_OOS_nstar = np.zeros(C, dtype=np.float64)
    R_OOS_mean = np.zeros(C, dtype=np.float64)

    all_partition_indices = set(range(S))

    for k, combo in enumerate(combos):
        IS_idx = list(combo)
        OOS_idx = sorted(all_partition_indices - set(IS_idx))

        # Vectorised IS SR across all N strategies
        sum_IS = sum_s[IS_idx, :].sum(axis=0)
        sumsq_IS = sumsq_s[IS_idx, :].sum(axis=0)
        SR_IS = _sr_from_stats(sum_IS, sumsq_IS, n_IS, annualisation)

        # Vectorised OOS SR across all N strategies
        sum_OOS = sum_s[OOS_idx, :].sum(axis=0)
        sumsq_OOS = sumsq_s[OOS_idx, :].sum(axis=0)
        SR_OOS = _sr_from_stats(sum_OOS, sumsq_OOS, n_OOS, annualisation)

        # IS-optimal strategy
        n_star = int(np.argmax(SR_IS))
        R_IS_nstar[k] = SR_IS[n_star]
        R_OOS_nstar[k] = SR_OOS[n_star]

        # Relative OOS rank (midrank convention)
        oos_ranks = scipy.stats.rankdata(SR_OOS, method="average")
        omega_c = oos_ranks[n_star] / (N + 1)   # in (1/(N+1), N/(N+1))
        lambda_all[k] = np.log(omega_c / (1.0 - omega_c))  # logit

        R_OOS_mean[k] = float(np.mean(SR_OOS))

    # PBO estimate
    phi = float(np.mean(lambda_all < 0))
    se_phi = np.sqrt(phi * (1 - phi) / C)
    phi_ci = (phi - 1.96 * se_phi, phi + 1.96 * se_phi)

    # Probability of OOS loss
    prob_loss = float(np.mean(R_OOS_nstar < 0))

    # Performance degradation OLS
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
        R_IS_nstar, R_OOS_nstar
    )
    adj_r2 = 1.0 - (1.0 - r_value ** 2) * (C - 1) / (C - 2)

    return {
        "lambda_all": lambda_all,
        "R_IS_nstar": R_IS_nstar,
        "R_OOS_nstar": R_OOS_nstar,
        "R_OOS_mean": R_OOS_mean,
        "phi": phi,
        "phi_ci": phi_ci,
        "prob_loss": prob_loss,
        "slope": float(slope),
        "intercept": float(intercept),
        "r_value": float(r_value),
        "adj_r2": float(adj_r2),
        "p_value": float(p_value),
        "n_combinations": C,
    }


# ---------------------------------------------------------------------------
# Public functions (used by tests and external callers)
# ---------------------------------------------------------------------------

def compute_partition_stats(
    slices: list,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute sum and sum-of-squares per partition from a list of slice arrays.

    Parameters
    ----------
    slices:
        List of S arrays each of shape (T_s, N), as produced by
        :func:`partition_matrix`.

    Returns
    -------
    sum_s   — (S, N) array: sum of returns per partition per strategy
    sumsq_s — (S, N) array: sum of squared returns per partition per strategy
    """
    S = len(slices)
    N = slices[0].shape[1]
    sum_s = np.zeros((S, N), dtype=np.float64)
    sumsq_s = np.zeros((S, N), dtype=np.float64)
    for s, sl in enumerate(slices):
        sum_s[s, :] = sl.sum(axis=0)
        sumsq_s[s, :] = (sl ** 2).sum(axis=0)
    return sum_s, sumsq_s


def compute_sr_from_stats(
    sum_s: np.ndarray,
    sumsq_s: np.ndarray,
    idx: list,
    T_s: int,
    annualisation: float = 252.0,
) -> np.ndarray:
    """Compute annualised Sharpe ratios for selected partitions from aggregated stats.

    Parameters
    ----------
    sum_s:
        (S, N) array of per-partition return sums.
    sumsq_s:
        (S, N) array of per-partition squared-return sums.
    idx:
        List of partition indices to aggregate.
    T_s:
        Number of rows per partition (each partition has T_s observations).
    annualisation:
        Annualisation factor (252 for daily data).

    Returns
    -------
    (N,) array of annualised Sharpe ratios.
    """
    n_obs = len(idx) * T_s
    sum_vals = sum_s[idx, :].sum(axis=0)
    sumsq_vals = sumsq_s[idx, :].sum(axis=0)
    return _sr_from_stats(sum_vals, sumsq_vals, n_obs, annualisation)
