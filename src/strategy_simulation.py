"""
Strategy simulation for PBO experiments.

Simulates all N=8,800 monthly seasonal trading strategy configurations over
both synthetic price series to produce T_adjusted x 8,800 daily P&L matrices
M_ex1 and M_ex2.

Each strategy is parameterised by:
  entry_day     — e in {1, ..., 20}
  holding_period — h in {1, ..., 20}
  stop_loss     — s in {0, 1, ..., 10}
  side          — d in {-1, +1}

Total: 20 × 20 × 11 × 2 = 8,800 strategies.
"""

from __future__ import annotations

import itertools
from typing import List, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------

def build_param_grid() -> List[Tuple[int, int, int, int]]:
    """Return the full 8,800-element Cartesian parameter grid.

    Returns
    -------
    List of tuples (entry_day, holding_period, stop_loss, side).
    """
    params = list(
        itertools.product(
            range(1, 21),   # entry_day: 1..20
            range(1, 21),   # holding_period: 1..20
            range(0, 11),   # stop_loss: 0..10
            [-1, 1],        # side: long / short
        )
    )
    assert len(params) == 8800, f"Expected 8800 strategies, got {len(params)}"
    return params


# Public alias used by tests
build_parameter_grid = build_param_grid


# ---------------------------------------------------------------------------
# Entry-map construction
# ---------------------------------------------------------------------------

def build_entry_map(
    calendar_df: pd.DataFrame,
    T_adj: int,
) -> dict:
    """Pre-compute global day indices for each (month_key, entry_day) pair.

    Parameters
    ----------
    calendar_df:
        Business-day calendar with columns [date, month_key, bd_in_month],
        integer-indexed 0..T-1.
    T_adj:
        Trimmed length (use only calendar_df.iloc[:T_adj]).

    Returns
    -------
    dict mapping (month_key: str, entry_day: int) -> global index int or None.
    """
    cal = calendar_df.iloc[:T_adj].copy()
    entry_map: dict = {}
    for month_key, group in cal.groupby("month_key", sort=True):
        indices = group.index.tolist()  # sorted global indices for this month
        for e in range(1, 21):
            entry_map[(month_key, e)] = indices[e - 1] if len(indices) >= e else None
    return entry_map


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def simulate_strategies(
    returns: np.ndarray,
    calendar_df: pd.DataFrame,
    params: List[Tuple[int, int, int, int]],
    sigma_daily: float,
    T_adj: int = 992,
) -> np.ndarray:
    """Simulate all strategies on a single return series.

    Custom — Context7 found no library equivalent (paper Section 3).

    Parameters
    ----------
    returns:
        Daily log-returns array of length >= T_adj.
    calendar_df:
        Business-day calendar (integer index 0..T-1).
    params:
        List of (entry_day, holding_period, stop_loss, side) tuples.
    sigma_daily:
        Daily volatility scalar for stop-loss threshold.
    T_adj:
        Number of rows in the output matrix (must equal T_adj).

    Returns
    -------
    np.ndarray of shape (T_adj, N) with float64 daily strategy returns.
    """
    ret = returns[:T_adj]
    N = len(params)
    M = np.zeros((T_adj, N), dtype=np.float64)

    entry_map = build_entry_map(calendar_df, T_adj)

    # Unique months in the trimmed calendar (preserve sort order)
    cal_trimmed = calendar_df.iloc[:T_adj]
    unique_months = cal_trimmed["month_key"].unique()

    # ---------------------------------------------------------------------------
    # Group strategies by (entry_day, holding_period, stop_loss) to allow
    # side-vectorisation (long/short share the same trade windows).
    # ---------------------------------------------------------------------------
    # Pre-build a lookup: strategy index -> param tuple
    # Build arrays of params for vectorised operations
    params_arr = np.array(params, dtype=np.int32)  # (8800, 4)

    # For each unique (e, h, s) triple, find the two strategy indices (long/short)
    # This halves the simulation work.
    ehs_to_strategies: dict[Tuple[int, int, int], Tuple[int, int]] = {}
    for n, (e, h, s, d) in enumerate(params):
        key = (e, h, s)
        if key not in ehs_to_strategies:
            ehs_to_strategies[key] = []
        ehs_to_strategies[key].append(n)

    for (e, h, s), strat_indices in ehs_to_strategies.items():
        # strat_indices has exactly 2 elements: [n_short, n_long] or [n_long, n_short]
        # Find which is long and which is short
        n_long = next(n for n in strat_indices if params[n][3] == 1)
        n_short = next(n for n in strat_indices if params[n][3] == -1)

        stop_threshold = -s * sigma_daily  # negative; 0 if s==0

        for month_key in unique_months:
            t_entry = entry_map.get((month_key, e), None)
            if t_entry is None:
                continue  # this month has fewer than e business days

            t_exit = min(t_entry + h - 1, T_adj - 1)

            if s == 0:
                # No stop-loss: vectorised slice assignment
                trade_slice = ret[t_entry: t_exit + 1]
                M[t_entry: t_exit + 1, n_long] = trade_slice
                M[t_entry: t_exit + 1, n_short] = -trade_slice
            else:
                # With stop-loss: iterate day by day
                cum_long = 0.0
                cum_short = 0.0
                long_stopped = False
                short_stopped = False

                for t in range(t_entry, t_exit + 1):
                    r = ret[t]

                    if not long_stopped:
                        M[t, n_long] = r
                        cum_long += r
                        if cum_long < stop_threshold:
                            long_stopped = True

                    if not short_stopped:
                        M[t, n_short] = -r
                        cum_short += (-r)
                        if cum_short < stop_threshold:
                            short_stopped = True

                    if long_stopped and short_stopped:
                        break

    return M


# ---------------------------------------------------------------------------
# Full-sample Sharpe ratio
# ---------------------------------------------------------------------------

def compute_sharpe_ratios(
    M: np.ndarray,
    annualisation: float = 252.0,
) -> np.ndarray:
    """Compute annualised Sharpe ratios for all strategies.

    Parameters
    ----------
    M:
        (T, N) daily returns matrix.
    annualisation:
        Annualisation factor (252 for daily returns).

    Returns
    -------
    np.ndarray of shape (N,) with annualised Sharpe ratios. Zero-std columns
    receive SR=0.
    """
    mu = np.mean(M, axis=0)
    std = np.std(M, axis=0, ddof=1)
    sr = np.where(std > 0, mu / std * np.sqrt(annualisation), 0.0)
    return sr


# Public alias used by tests
compute_full_sample_sr = compute_sharpe_ratios


# ---------------------------------------------------------------------------
# Single-strategy simulation (used by tests and targeted experiments)
# ---------------------------------------------------------------------------

def simulate_strategy(
    returns: np.ndarray,
    calendar_df: pd.DataFrame,
    entry_day: int,
    holding_period: int,
    stop_loss: int,
    side: int,
    sigma_daily: float,
    T_adj: int | None = None,
    entry_map: dict | None = None,
) -> np.ndarray:
    """Simulate a single strategy and return its daily P&L series.

    Parameters
    ----------
    returns:
        Daily log-returns array.
    calendar_df:
        Business-day calendar DataFrame (integer index 0..T-1).
    entry_day:
        e ∈ {1, ..., 20} — which ordinal business day of each month to enter.
    holding_period:
        h ∈ {1, ..., 20} — number of days to hold (incl. entry day).
    stop_loss:
        s ∈ {0, ..., 10} — stop-loss multiple of sigma_daily (0 = no stop).
    side:
        +1 (long) or -1 (short).
    sigma_daily:
        Daily volatility scalar for stop-loss threshold.
    T_adj:
        If given, trim returns and calendar to this length.
    entry_map:
        Pre-built dict from :func:`build_entry_map`. If None, built internally.

    Returns
    -------
    np.ndarray of shape (T_adj,) with daily strategy returns.
    """
    if T_adj is None:
        T_adj = len(returns)
    ret = np.asarray(returns[:T_adj], dtype=np.float64)
    pnl = np.zeros(T_adj, dtype=np.float64)

    if entry_map is None:
        entry_map = build_entry_map(calendar_df, T_adj)

    cal_trimmed = calendar_df.iloc[:T_adj]
    unique_months = cal_trimmed["month_key"].unique()

    stop_threshold = -stop_loss * sigma_daily

    for month_key in unique_months:
        t_entry = entry_map.get((month_key, entry_day), None)
        if t_entry is None:
            continue
        t_exit = min(t_entry + holding_period - 1, T_adj - 1)

        if stop_loss == 0:
            trade_slice = ret[t_entry: t_exit + 1]
            pnl[t_entry: t_exit + 1] = side * trade_slice
        else:
            cum = 0.0
            for t in range(t_entry, t_exit + 1):
                r = side * ret[t]
                pnl[t] = r
                cum += r
                if cum < stop_threshold:
                    break

    return pnl


# ---------------------------------------------------------------------------
# Main builder function
# ---------------------------------------------------------------------------

def build_pnl_matrices(
    returns_ex1: np.ndarray,
    returns_ex2: np.ndarray,
    calendar_df: pd.DataFrame,
    sigma_full_ex1: float,
    sigma_full_ex2: float,
    T_adj: int = 992,
) -> dict:
    """Build M_ex1 and M_ex2 and compute global Sharpe ratio vectors.

    Parameters
    ----------
    returns_ex1, returns_ex2:
        Full-length (1000,) return arrays from exp_1.
    calendar_df:
        Full-length (1000 rows) business-day calendar.
    sigma_full_ex1, sigma_full_ex2:
        Full-sample daily volatility scalars from exp_1.
    T_adj:
        Trimmed length (default 992).

    Returns
    -------
    dict with keys:
        M_ex1         — np.ndarray (T_adj, 8800)
        M_ex2         — np.ndarray (T_adj, 8800)
        params        — list of 8800 tuples
        SR_full_ex1   — np.ndarray (8800,)
        SR_full_ex2   — np.ndarray (8800,)
        n_star_ex1    — int
        n_star_ex2    — int
        sigma_daily_ex1 — float
        sigma_daily_ex2 — float
    """
    params = build_param_grid()

    # Trim and recompute sigma on trimmed series (used for stop-loss)
    ret1 = returns_ex1[:T_adj]
    ret2 = returns_ex2[:T_adj]
    sigma_daily_ex1 = float(np.std(ret1, ddof=1))
    sigma_daily_ex2 = float(np.std(ret2, ddof=1))

    print(f"Simulating {len(params)} strategies on Example 1 (T_adj={T_adj})…")
    M_ex1 = simulate_strategies(
        returns=ret1,
        calendar_df=calendar_df,
        params=params,
        sigma_daily=sigma_daily_ex1,
        T_adj=T_adj,
    )

    print(f"Simulating {len(params)} strategies on Example 2 (T_adj={T_adj})…")
    M_ex2 = simulate_strategies(
        returns=ret2,
        calendar_df=calendar_df,
        params=params,
        sigma_daily=sigma_daily_ex2,
        T_adj=T_adj,
    )

    SR_full_ex1 = compute_sharpe_ratios(M_ex1)
    SR_full_ex2 = compute_sharpe_ratios(M_ex2)

    n_star_ex1 = int(np.argmax(SR_full_ex1))
    n_star_ex2 = int(np.argmax(SR_full_ex2))

    # Validation
    assert M_ex1.shape == (T_adj, 8800), f"M_ex1 shape {M_ex1.shape}"
    assert M_ex2.shape == (T_adj, 8800), f"M_ex2 shape {M_ex2.shape}"

    return {
        "M_ex1": M_ex1,
        "M_ex2": M_ex2,
        "params": params,
        "SR_full_ex1": SR_full_ex1,
        "SR_full_ex2": SR_full_ex2,
        "n_star_ex1": n_star_ex1,
        "n_star_ex2": n_star_ex2,
        "sigma_daily_ex1": sigma_daily_ex1,
        "sigma_daily_ex2": sigma_daily_ex2,
    }
