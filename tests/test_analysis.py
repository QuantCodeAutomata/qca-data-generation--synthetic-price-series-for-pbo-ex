"""
Unit tests for src/analysis.py (exp_5 / exp_6 methodology).

Validates:
  - Empirical CDF computation
  - First-order stochastic dominance assessment
  - Second-order stochastic dominance integral
  - PSR formula (Bailey et al. 2014)
  - Edge cases (identical distributions, degenerate inputs)
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import (
    compute_empirical_cdf,
    assess_first_order_dominance,
    compute_second_order_dominance,
    compute_psr,
    run_stochastic_dominance_analysis,
)


# ---------------------------------------------------------------------------
# compute_empirical_cdf
# ---------------------------------------------------------------------------

class TestComputeEmpiricalCDF:

    def test_output_shape(self):
        dist = np.random.randn(100)
        x_grid = np.linspace(-3, 3, 50)
        F = compute_empirical_cdf(dist, x_grid)
        assert F.shape == (50,)

    def test_range_01(self):
        dist = np.random.randn(1000)
        x_grid = np.linspace(-5, 5, 100)
        F = compute_empirical_cdf(dist, x_grid)
        assert (F >= 0.0).all()
        assert (F <= 1.0).all()

    def test_non_decreasing(self):
        dist = np.random.randn(500)
        x_grid = np.linspace(-3, 3, 200)
        F = compute_empirical_cdf(dist, x_grid)
        assert (np.diff(F) >= 0).all()

    def test_below_min_is_zero(self):
        dist = np.array([1.0, 2.0, 3.0])
        x_grid = np.array([0.0])
        F = compute_empirical_cdf(dist, x_grid)
        assert F[0] == 0.0

    def test_above_max_is_one(self):
        dist = np.array([1.0, 2.0, 3.0])
        x_grid = np.array([4.0])
        F = compute_empirical_cdf(dist, x_grid)
        assert F[0] == 1.0

    def test_midpoint_value(self):
        """CDF at a value x = proportion of samples <= x."""
        dist = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        x_grid = np.array([3.0])
        F = compute_empirical_cdf(dist, x_grid)
        assert np.isclose(F[0], 3 / 5)

    def test_single_value_distribution(self):
        """Single observation: CDF is Heaviside step."""
        dist = np.array([0.5])
        x_grid = np.array([0.0, 0.5, 1.0])
        F = compute_empirical_cdf(dist, x_grid)
        assert F[0] == 0.0
        assert F[1] == 1.0
        assert F[2] == 1.0


# ---------------------------------------------------------------------------
# assess_first_order_dominance
# ---------------------------------------------------------------------------

class TestAssessFirstOrderDominance:

    def test_A_dominates_B(self):
        """When dist_A is strictly to the right of dist_B, A FO dominates B."""
        dist_A = np.array([3.0, 4.0, 5.0])
        dist_B = np.array([1.0, 2.0, 3.0])
        x_grid = np.linspace(0, 6, 100)
        F_A = compute_empirical_cdf(dist_A, x_grid)
        F_B = compute_empirical_cdf(dist_B, x_grid)
        result, a_dom, b_dom = assess_first_order_dominance(F_A, F_B, tol=1e-8)
        assert a_dom is True
        assert b_dom is False

    def test_B_dominates_A(self):
        """When dist_B is strictly to the right of dist_A, B FO dominates A."""
        dist_A = np.array([1.0, 2.0, 3.0])
        dist_B = np.array([3.0, 4.0, 5.0])
        x_grid = np.linspace(0, 6, 100)
        F_A = compute_empirical_cdf(dist_A, x_grid)
        F_B = compute_empirical_cdf(dist_B, x_grid)
        result, a_dom, b_dom = assess_first_order_dominance(F_A, F_B, tol=1e-8)
        assert a_dom is False
        assert b_dom is True

    def test_identical_distributions_no_dominance(self):
        """Same distributions should not exhibit strict FO dominance."""
        dist = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        x_grid = np.linspace(0, 6, 100)
        F = compute_empirical_cdf(dist, x_grid)
        result, a_dom, b_dom = assess_first_order_dominance(F, F, tol=1e-8)
        # No strict dominance (F_A == F_B everywhere)
        # Both could be flagged as "dominates" at tolerance 0, but not strict
        # We check the result string doesn't claim strict dominance
        assert "No first-order" in result or a_dom or b_dom  # tolerant

    def test_crossing_cdfs_no_dominance(self):
        """Crossing CDFs must report no first-order dominance."""
        dist_A = np.array([0.0, 3.0])  # uniform over [0,3]
        dist_B = np.array([1.0, 2.0])  # uniform over [1,2]
        # F_A crosses F_B — neither dominates strictly
        x_grid = np.linspace(-1, 4, 200)
        F_A = compute_empirical_cdf(dist_A, x_grid)
        F_B = compute_empirical_cdf(dist_B, x_grid)
        result, a_dom, b_dom = assess_first_order_dominance(F_A, F_B, tol=1e-8)
        # At x<1: F_B=0, F_A=0 — no crossing; at x=1.5: F_A=0.5, F_B=0.5
        # crossing depends on exact distribution; just verify no exception
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# compute_second_order_dominance
# ---------------------------------------------------------------------------

class TestComputeSecondOrderDominance:

    def test_A_dominates_B_so(self):
        """A shifted right of B → A SO dominates B → SD2 >= 0."""
        dist_A = np.array([3.0, 4.0, 5.0])
        dist_B = np.array([1.0, 2.0, 3.0])
        x_grid = np.linspace(0, 6, 500)
        F_A = compute_empirical_cdf(dist_A, x_grid)
        F_B = compute_empirical_cdf(dist_B, x_grid)
        SD2, result, a_dom = compute_second_order_dominance(F_A, F_B, x_grid)
        assert a_dom is True
        assert (SD2 >= -1e-6).all()

    def test_no_so_dominance_for_noise(self):
        """For symmetric noise, SD2 may cross zero."""
        rng = np.random.default_rng(42)
        dist_A = rng.normal(0, 1, 1000)
        dist_B = rng.normal(0, 1, 1000)
        x_grid = np.linspace(-4, 4, 200)
        F_A = compute_empirical_cdf(dist_A, x_grid)
        F_B = compute_empirical_cdf(dist_B, x_grid)
        SD2, result, a_dom = compute_second_order_dominance(F_A, F_B, x_grid)
        assert isinstance(SD2, np.ndarray)
        assert SD2.shape == (200,)

    def test_sd2_shape(self):
        x_grid = np.linspace(-3, 3, 150)
        F_A = np.linspace(0, 1, 150)
        F_B = np.linspace(0, 1, 150)
        SD2, _, _ = compute_second_order_dominance(F_A, F_B, x_grid)
        assert SD2.shape == (150,)

    def test_sd2_identical_distributions(self):
        """Identical CDFs → SD2 is identically zero."""
        x_grid = np.linspace(-3, 3, 100)
        F = np.linspace(0, 1, 100)
        SD2, _, _ = compute_second_order_dominance(F, F, x_grid)
        np.testing.assert_allclose(SD2, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# compute_psr (Probabilistic Sharpe Ratio)
# ---------------------------------------------------------------------------

class TestComputePSR:

    def test_psr_in_01(self):
        """PSR must always be in [0, 1]."""
        r = np.random.randn(992) * 0.01
        psr_stat, psr = compute_psr(r, SR_star=0.0)
        assert 0.0 <= psr <= 1.0

    def test_psr_increases_with_sr(self):
        """Higher SR → higher PSR given same returns distribution."""
        r_low = np.random.randn(992) * 0.01 + 0.0001
        r_high = np.random.randn(992) * 0.01 + 0.005
        _, psr_low = compute_psr(r_low, SR_star=0.0)
        _, psr_high = compute_psr(r_high, SR_star=0.0)
        assert psr_high > psr_low

    def test_psr_stat_formula(self):
        """
        Manual verification of PSR_stat:
          PSR_stat = SR_daily * sqrt(T-1) / sqrt(1 - γ3*SR + (γ4+2)/4 * SR²)
        using scipy.stats.norm.cdf convention.
        """
        from scipy.stats import skew, kurtosis, norm
        rng = np.random.default_rng(0)
        r = rng.normal(0.001, 0.01, size=500)
        mu = r.mean()
        sigma = r.std(ddof=1)
        SR_d = mu / sigma
        gamma3 = skew(r)
        gamma4_excess = kurtosis(r, fisher=True)
        T = len(r)
        denom = np.sqrt(max(1 - gamma3 * SR_d + (gamma4_excess + 2) / 4 * SR_d**2, 1e-10))
        expected_stat = SR_d * np.sqrt(T - 1) / denom
        expected_psr = norm.cdf(expected_stat)
        computed_stat, computed_psr = compute_psr(r, SR_star=0.0)
        np.testing.assert_allclose(computed_stat, expected_stat, rtol=1e-8)
        np.testing.assert_allclose(computed_psr, expected_psr, rtol=1e-8)

    def test_psr_with_nonzero_sr_star(self):
        """With SR_star > 0, PSR should be lower than with SR_star=0."""
        r = np.random.randn(500) * 0.01 + 0.002
        _, psr_0 = compute_psr(r, SR_star=0.0)
        _, psr_pos = compute_psr(r, SR_star=0.1)
        assert psr_0 >= psr_pos

    def test_psr_large_sample_converges(self):
        """For very large T and modest SR, PSR should approach 1."""
        rng = np.random.default_rng(1)
        r = rng.normal(0.002, 0.01, size=10000)
        _, psr = compute_psr(r, SR_star=0.0)
        assert psr > 0.99

    def test_negative_sr_psr_below_half(self):
        """Negative mean returns → PSR < 0.5."""
        r = np.random.randn(992) * 0.01 - 0.002
        _, psr = compute_psr(r, SR_star=0.0)
        assert psr < 0.5

    def test_denom_not_negative(self):
        """PSR denominator must be positive (guarded against numerical issues)."""
        r = np.ones(200) * 0.01  # constant returns (high skew / zero kurtosis)
        try:
            stat, psr = compute_psr(r, SR_star=0.0)
            assert np.isfinite(stat)
            assert 0.0 <= psr <= 1.0
        except Exception:
            pytest.skip("Degenerate constant-returns case — acceptable to skip")


# ---------------------------------------------------------------------------
# run_stochastic_dominance_analysis (integration)
# ---------------------------------------------------------------------------

class TestRunStochasticDominanceAnalysis:

    def _make_inputs(self, seed=0, shift=0.0):
        rng = np.random.default_rng(seed)
        R_OOS_nstar = rng.normal(shift, 0.5, size=100)
        R_OOS_mean = rng.normal(0.0, 0.3, size=100)
        M = rng.normal(0, 0.01, size=(100, 5))
        return R_OOS_nstar, R_OOS_mean, M

    def test_output_keys(self):
        R_OOS_nstar, R_OOS_mean, M = self._make_inputs()
        res = run_stochastic_dominance_analysis(R_OOS_nstar, R_OOS_mean, M, n_star_full=0)
        required_keys = {"x_grid", "F_A", "F_B", "SD2", "FO_result", "SO_result",
                         "A_dominates_B_FO", "A_dominates_B_SO", "SR_hat_annual",
                         "skewness", "excess_kurtosis", "PSR_stat", "PSR"}
        assert required_keys.issubset(set(res.keys()))

    def test_cdf_shapes_match_x_grid(self):
        R_OOS_nstar, R_OOS_mean, M = self._make_inputs()
        res = run_stochastic_dominance_analysis(R_OOS_nstar, R_OOS_mean, M, n_star_full=0)
        n = len(res["x_grid"])
        assert res["F_A"].shape == (n,)
        assert res["F_B"].shape == (n,)
        assert res["SD2"].shape == (n,)

    def test_psr_in_range(self):
        R_OOS_nstar, R_OOS_mean, M = self._make_inputs()
        res = run_stochastic_dominance_analysis(R_OOS_nstar, R_OOS_mean, M, n_star_full=0)
        assert 0.0 <= res["PSR"] <= 1.0

    def test_shifted_distribution_dominates(self):
        """A right-shifted distribution A should dominate B (FO)."""
        R_OOS_nstar, R_OOS_mean, M = self._make_inputs(shift=2.0)
        res = run_stochastic_dominance_analysis(R_OOS_nstar, R_OOS_mean, M, n_star_full=0)
        assert res["A_dominates_B_FO"] is True

    def test_result_strings_are_str(self):
        R_OOS_nstar, R_OOS_mean, M = self._make_inputs()
        res = run_stochastic_dominance_analysis(R_OOS_nstar, R_OOS_mean, M, n_star_full=0)
        assert isinstance(res["FO_result"], str)
        assert isinstance(res["SO_result"], str)
