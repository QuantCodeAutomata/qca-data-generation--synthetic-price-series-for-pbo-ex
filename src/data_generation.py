"""
Data generation for PBO experiments.

Generates two synthetic daily return series (T=1000) used as the underlying
asset for all CSCV experiments:
  - Example 1: pure i.i.d. N(0, sigma^2) random walk
  - Example 2: random walk with a monthly seasonal effect on the first 5 BDs of
    each calendar month

All parameters are fixed for reproducibility (numpy seed = 42).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_returns_ex1(
    T: int = 1000,
    sigma: float = 0.01,
    seed: int = 42,
) -> np.ndarray:
    """Generate Example 1: pure i.i.d. N(0, sigma^2) log-return series.

    Parameters
    ----------
    T:
        Number of observations.
    sigma:
        Daily volatility (standard deviation of each draw).
    seed:
        NumPy random seed for reproducibility.

    Returns
    -------
    np.ndarray of shape (T,) with float64 log-returns.
    """
    # Custom — Context7 found no library equivalent (paper Step 1-2)
    np.random.seed(seed)
    returns = np.random.normal(loc=0.0, scale=sigma, size=T)
    return returns.astype(np.float64)


def build_business_day_calendar(
    T: int = 1000,
    start_date: str = "2000-01-03",
) -> pd.DataFrame:
    """Build a business-day calendar mapping each trading day to its position
    within its calendar month.

    Parameters
    ----------
    T:
        Number of business days to generate.
    start_date:
        First business day (ISO 8601 string).

    Returns
    -------
    pd.DataFrame with integer index 0..T-1 and columns:
        date        — pandas Timestamp
        month_key   — string "YYYY-MM"
        bd_in_month — integer (1 = first BD of that month)
    """
    # Using pd.bdate_range for weekday calendar — Context7 confirmed numpy/pandas
    dates = pd.bdate_range(start=start_date, periods=T, freq="B")
    df = pd.DataFrame({"date": dates})
    df["month_key"] = df["date"].dt.strftime("%Y-%m")
    # Ordinal position within each month (1-indexed)
    df["bd_in_month"] = df.groupby("month_key").cumcount() + 1
    df.index = range(T)
    return df


def generate_returns_ex2(
    returns_ex1: np.ndarray,
    calendar_df: pd.DataFrame,
    sigma_full_ex1: float,
    shift_multiple: float = 0.25,
    max_bd: int = 5,
) -> np.ndarray:
    """Generate Example 2: returns_ex1 plus a positive mean shift on the first
    ``max_bd`` business days of each calendar month.

    Parameters
    ----------
    returns_ex1:
        Base return series from Example 1.
    calendar_df:
        Business-day calendar DataFrame from ``build_business_day_calendar``.
    sigma_full_ex1:
        Pre-shift daily volatility of returns_ex1 (ddof=1).
    shift_multiple:
        Scalar multiplier: shift = shift_multiple * sigma_full_ex1.
    max_bd:
        Number of leading business days per month that receive the shift.

    Returns
    -------
    np.ndarray of shape (T,) with float64 log-returns.
    """
    returns_ex2 = returns_ex1.copy()
    seasonal_indices = calendar_df["bd_in_month"].values <= max_bd  # boolean mask
    shift = shift_multiple * sigma_full_ex1
    returns_ex2[seasonal_indices] += shift
    return returns_ex2.astype(np.float64)


def returns_to_prices(returns: np.ndarray, p0: float = 100.0) -> np.ndarray:
    """Convert a log-return series to a price series via cumulative exponentiation.

    P(t) = p0 * exp(sum_{s=1}^{t} r_s),  P(0) = p0.

    Parameters
    ----------
    returns:
        1-D array of log-returns of length T.
    p0:
        Initial price level.

    Returns
    -------
    np.ndarray of shape (T+1,) where index 0 is p0.
    """
    prices = np.concatenate([[p0], p0 * np.exp(np.cumsum(returns))])
    return prices.astype(np.float64)


def generate_all_artifacts(
    T: int = 1000,
    sigma: float = 0.01,
    seed: int = 42,
    start_date: str = "2000-01-03",
    shift_multiple: float = 0.25,
    max_bd: int = 5,
) -> dict:
    """Master function: generate and return all artifacts for both examples.

    Parameters
    ----------
    T, sigma, seed, start_date, shift_multiple, max_bd:
        See individual functions.

    Returns
    -------
    dict with keys:
        returns_ex1     — np.ndarray (T,)
        returns_ex2     — np.ndarray (T,)
        calendar_df     — pd.DataFrame (T rows)
        sigma_full_ex1  — float
        sigma_full_ex2  — float
        prices_ex1      — np.ndarray (T+1,)
        prices_ex2      — np.ndarray (T+1,)
        seasonal_indices — np.ndarray bool (T,)
    """
    returns_ex1 = generate_returns_ex1(T=T, sigma=sigma, seed=seed)
    sigma_full_ex1 = float(np.std(returns_ex1, ddof=1))

    calendar_df = build_business_day_calendar(T=T, start_date=start_date)

    returns_ex2 = generate_returns_ex2(
        returns_ex1=returns_ex1,
        calendar_df=calendar_df,
        sigma_full_ex1=sigma_full_ex1,
        shift_multiple=shift_multiple,
        max_bd=max_bd,
    )
    sigma_full_ex2 = float(np.std(returns_ex2, ddof=1))

    prices_ex1 = returns_to_prices(returns_ex1)
    prices_ex2 = returns_to_prices(returns_ex2)

    seasonal_indices = (calendar_df["bd_in_month"].values <= max_bd)

    # Validation assertions
    assert len(returns_ex1) == T, "returns_ex1 length mismatch"
    assert len(returns_ex2) == T, "returns_ex2 length mismatch"
    assert len(calendar_df) == T, "calendar_df length mismatch"
    assert np.isclose(prices_ex1[0], 100.0), "prices_ex1[0] must equal 100"
    assert np.isclose(prices_ex2[0], 100.0), "prices_ex2[0] must equal 100"

    return {
        "returns_ex1": returns_ex1,
        "returns_ex2": returns_ex2,
        "calendar_df": calendar_df,
        "sigma_full_ex1": sigma_full_ex1,
        "sigma_full_ex2": sigma_full_ex2,
        "prices_ex1": prices_ex1,
        "prices_ex2": prices_ex2,
        "seasonal_indices": seasonal_indices,
    }
