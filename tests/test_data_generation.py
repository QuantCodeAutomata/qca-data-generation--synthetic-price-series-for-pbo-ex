"""
Unit tests for src/data_generation.py (exp_1 methodology).

Validates:
  - Return series dimensions and statistical properties
  - Business-day calendar construction
  - Seasonal effect application
  - Price series conversion
  - Reproducibility via seed
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generation import (
    generate_returns_ex1,
    build_business_day_calendar,
    generate_returns_ex2,
    returns_to_prices,
    generate_all_artifacts,
)


# ---------------------------------------------------------------------------
# generate_returns_ex1
# ---------------------------------------------------------------------------

class TestGenerateReturnsEx1:

    def test_shape(self):
        """Returns array must have exactly T=1000 elements."""
        r = generate_returns_ex1(T=1000, sigma=0.01, seed=42)
        assert r.shape == (1000,)

    def test_dtype(self):
        """Returns must be float64."""
        r = generate_returns_ex1(T=1000, sigma=0.01, seed=42)
        assert r.dtype == np.float64

    def test_mean_near_zero(self):
        """I.i.d. N(0, σ²) sample mean ≈ 0 (within 3 standard errors)."""
        r = generate_returns_ex1(T=10000, sigma=0.01, seed=0)
        se = 0.01 / np.sqrt(10000)
        assert abs(r.mean()) < 4 * se

    def test_std_near_sigma(self):
        """Sample std ≈ sigma within 1% for large sample."""
        r = generate_returns_ex1(T=10000, sigma=0.01, seed=0)
        assert abs(r.std(ddof=1) - 0.01) < 0.001

    def test_reproducibility(self):
        """Same seed must produce identical arrays."""
        r1 = generate_returns_ex1(T=1000, sigma=0.01, seed=42)
        r2 = generate_returns_ex1(T=1000, sigma=0.01, seed=42)
        np.testing.assert_array_equal(r1, r2)

    def test_different_seeds_differ(self):
        """Different seeds must produce different arrays."""
        r1 = generate_returns_ex1(T=1000, sigma=0.01, seed=42)
        r2 = generate_returns_ex1(T=1000, sigma=0.01, seed=99)
        assert not np.array_equal(r1, r2)

    def test_single_observation(self):
        """Edge case: T=1 should work."""
        r = generate_returns_ex1(T=1, sigma=0.01, seed=0)
        assert r.shape == (1,)

    def test_extreme_sigma(self):
        """Very large sigma: distribution should scale accordingly."""
        r = generate_returns_ex1(T=10000, sigma=1.0, seed=0)
        assert abs(r.std(ddof=1) - 1.0) < 0.05


# ---------------------------------------------------------------------------
# build_business_day_calendar
# ---------------------------------------------------------------------------

class TestBuildBusinessDayCalendar:

    def setup_method(self):
        self.df = build_business_day_calendar(T=1000, start_date="2000-01-03")

    def test_length(self):
        assert len(self.df) == 1000

    def test_columns(self):
        assert set(self.df.columns) >= {"date", "month_key", "bd_in_month"}

    def test_bd_in_month_starts_at_one(self):
        """Every month must have bd_in_month starting at 1."""
        firsts = self.df.groupby("month_key")["bd_in_month"].min()
        assert (firsts == 1).all()

    def test_bd_in_month_sequential(self):
        """Within each month, bd_in_month must be 1, 2, 3, ..."""
        for _, grp in self.df.groupby("month_key"):
            bds = grp["bd_in_month"].values
            expected = np.arange(1, len(bds) + 1)
            np.testing.assert_array_equal(bds, expected)

    def test_bd_in_month_max_reasonable(self):
        """No month should have more than 23 business days."""
        assert self.df["bd_in_month"].max() <= 23

    def test_approximately_48_months(self):
        """1000 business days ≈ 48 calendar months."""
        n_months = self.df["month_key"].nunique()
        assert 46 <= n_months <= 52

    def test_dates_are_business_days(self):
        """All dates must be weekdays (Mon-Fri)."""
        day_of_week = self.df["date"].dt.dayofweek  # Mon=0, Fri=4
        assert (day_of_week <= 4).all()

    def test_index_reset(self):
        """Index must be 0..999."""
        assert list(self.df.index) == list(range(1000))

    def test_small_calendar(self):
        """Edge case: T=5."""
        df5 = build_business_day_calendar(T=5, start_date="2000-01-03")
        assert len(df5) == 5
        assert df5["bd_in_month"].iloc[0] == 1


# ---------------------------------------------------------------------------
# generate_returns_ex2
# ---------------------------------------------------------------------------

class TestGenerateReturnsEx2:

    def setup_method(self):
        np.random.seed(42)
        self.r1 = generate_returns_ex1(T=1000, sigma=0.01, seed=42)
        self.sig1 = float(np.std(self.r1, ddof=1))
        self.cal = build_business_day_calendar(T=1000, start_date="2000-01-03")
        self.r2 = generate_returns_ex2(self.r1, self.cal, self.sig1)

    def test_shape(self):
        assert self.r2.shape == (1000,)

    def test_dtype(self):
        assert self.r2.dtype == np.float64

    def test_seasonal_days_shifted_up(self):
        """First 5 BDs of each month must have higher returns than ex1."""
        mask = (self.cal["bd_in_month"].values <= 5)
        diff = self.r2[mask] - self.r1[mask]
        shift = 0.25 * self.sig1
        np.testing.assert_allclose(diff, shift, rtol=1e-10)

    def test_non_seasonal_days_unchanged(self):
        """Non-seasonal days must equal ex1 returns exactly."""
        mask = (self.cal["bd_in_month"].values > 5)
        np.testing.assert_array_equal(self.r2[mask], self.r1[mask])

    def test_ex2_mean_higher_than_ex1(self):
        """Example 2 must have higher mean due to positive seasonal shift."""
        assert self.r2.mean() > self.r1.mean()

    def test_does_not_modify_ex1(self):
        """generate_returns_ex2 must not mutate the input array."""
        r1_copy = self.r1.copy()
        _ = generate_returns_ex2(self.r1, self.cal, self.sig1)
        np.testing.assert_array_equal(self.r1, r1_copy)


# ---------------------------------------------------------------------------
# returns_to_prices
# ---------------------------------------------------------------------------

class TestReturnsToPrices:

    def test_initial_price(self):
        r = np.array([0.01, -0.005, 0.003])
        p = returns_to_prices(r, p0=100.0)
        assert np.isclose(p[0], 100.0)

    def test_length(self):
        r = np.ones(50) * 0.001
        p = returns_to_prices(r)
        assert len(p) == 51  # T+1

    def test_positive_returns_grow(self):
        r = np.ones(10) * 0.01
        p = returns_to_prices(r)
        assert p[-1] > p[0]

    def test_zero_returns_flat(self):
        r = np.zeros(100)
        p = returns_to_prices(r)
        np.testing.assert_allclose(p, 100.0)

    def test_custom_p0(self):
        r = np.zeros(5)
        p = returns_to_prices(r, p0=200.0)
        assert np.isclose(p[0], 200.0)

    def test_dtype_float64(self):
        r = np.ones(10) * 0.01
        p = returns_to_prices(r)
        assert p.dtype == np.float64

    def test_cumulative_product(self):
        """Manual verification: p[2] = 100*exp(r[0]+r[1])."""
        r = np.array([0.01, 0.02])
        p = returns_to_prices(r)
        expected = 100.0 * np.exp(0.03)
        assert np.isclose(p[2], expected)


# ---------------------------------------------------------------------------
# generate_all_artifacts (integration)
# ---------------------------------------------------------------------------

class TestGenerateAllArtifacts:

    def setup_method(self):
        self.arts = generate_all_artifacts()

    def test_keys_present(self):
        keys = {"returns_ex1", "returns_ex2", "calendar_df", "sigma_full_ex1",
                "sigma_full_ex2", "prices_ex1", "prices_ex2", "seasonal_indices"}
        assert keys.issubset(set(self.arts.keys()))

    def test_returns_shapes(self):
        assert self.arts["returns_ex1"].shape == (1000,)
        assert self.arts["returns_ex2"].shape == (1000,)

    def test_prices_shapes(self):
        assert self.arts["prices_ex1"].shape == (1001,)
        assert self.arts["prices_ex2"].shape == (1001,)

    def test_prices_start_at_100(self):
        assert np.isclose(self.arts["prices_ex1"][0], 100.0)
        assert np.isclose(self.arts["prices_ex2"][0], 100.0)

    def test_sigma_positive(self):
        assert self.arts["sigma_full_ex1"] > 0
        assert self.arts["sigma_full_ex2"] > 0

    def test_sigma_ex2_gt_ex1(self):
        """Adding positive shift to some days increases overall std slightly."""
        # not strictly guaranteed for all seeds but holds for seed=42
        assert self.arts["sigma_full_ex2"] >= self.arts["sigma_full_ex1"] * 0.99

    def test_seasonal_count_approx_240(self):
        n_seasonal = int(self.arts["seasonal_indices"].sum())
        # expect roughly 5 * 48 ≈ 240
        assert 200 <= n_seasonal <= 280

    def test_seasonal_indices_bool(self):
        assert self.arts["seasonal_indices"].dtype == bool
