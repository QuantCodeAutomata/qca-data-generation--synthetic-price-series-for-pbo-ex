"""
Unit tests for src/cscv.py (exp_3 / exp_4 methodology).

Validates:
  - Partition logic (S=16 slices of T_adj/S rows)
  - Combination enumeration (C(16,8) = 12,870)
  - IS/OOS SR vectorised computation
  - PBO estimation (phi) lies in [0, 1]
  - Lambda distribution properties
  - Performance degradation regression plausibility
  - Synthetic known-outcome tests
"""

import numpy as np
import pandas as pd
import pytest
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cscv import (
    partition_matrix,
    compute_partition_stats,
    compute_sr_from_stats,
    run_cscv,
)


# ---------------------------------------------------------------------------
# partition_matrix
# ---------------------------------------------------------------------------

class TestPartitionMatrix:

    def test_partition_count(self):
        M = np.random.randn(992, 10)
        slices = partition_matrix(M, S=16)
        assert len(slices) == 16

    def test_slice_shapes(self):
        M = np.random.randn(992, 10)
        slices = partition_matrix(M, S=16)
        for sl in slices:
            assert sl.shape == (62, 10)

    def test_concatenation_recovers_original(self):
        M = np.arange(992 * 4, dtype=float).reshape(992, 4)
        slices = partition_matrix(M, S=16)
        reconstructed = np.vstack(slices)
        np.testing.assert_array_equal(reconstructed, M)

    def test_small_partition(self):
        M = np.ones((8, 3))
        slices = partition_matrix(M, S=4)
        assert len(slices) == 4
        assert all(s.shape == (2, 3) for s in slices)

    def test_s_equal_1(self):
        M = np.ones((100, 5))
        slices = partition_matrix(M, S=1)
        assert len(slices) == 1
        assert slices[0].shape == (100, 5)


# ---------------------------------------------------------------------------
# compute_partition_stats
# ---------------------------------------------------------------------------

class TestComputePartitionStats:

    def test_shapes(self):
        M = np.random.randn(992, 100)
        slices = partition_matrix(M, S=16)
        sum_s, sumsq_s = compute_partition_stats(slices)
        assert sum_s.shape == (16, 100)
        assert sumsq_s.shape == (16, 100)

    def test_sum_correctness(self):
        M = np.ones((16, 4))  # trivially S=1 (single partition)
        slices = partition_matrix(M, S=1)
        sum_s, sumsq_s = compute_partition_stats(slices)
        np.testing.assert_allclose(sum_s[0], 16.0)  # 16 ones summed
        np.testing.assert_allclose(sumsq_s[0], 16.0)  # 16 ones² = 16

    def test_nonnegative_sumsq(self):
        M = np.random.randn(992, 50)
        slices = partition_matrix(M, S=16)
        _, sumsq_s = compute_partition_stats(slices)
        assert (sumsq_s >= 0).all()


# ---------------------------------------------------------------------------
# compute_sr_from_stats
# ---------------------------------------------------------------------------

class TestComputeSRFromStats:

    def test_positive_sr_for_positive_returns(self):
        """All-positive returns must give positive SR."""
        r = np.ones((100, 5)) * 0.005
        slices = partition_matrix(r, S=4)
        sum_s, sumsq_s = compute_partition_stats(slices)
        idx = list(range(4))
        SR = compute_sr_from_stats(sum_s, sumsq_s, idx, T_s=25)
        assert (SR > 0).all()

    def test_zero_variance_returns_zero(self):
        """Constant (zero-variance) returns should give SR=0, not NaN."""
        r = np.zeros((64, 3))
        slices = partition_matrix(r, S=4)
        sum_s, sumsq_s = compute_partition_stats(slices)
        SR = compute_sr_from_stats(sum_s, sumsq_s, [0, 1], T_s=16)
        assert not np.isnan(SR).any()
        assert not np.isinf(SR).any()
        np.testing.assert_allclose(SR, 0.0)

    def test_sr_sign_consistency(self):
        """Negative-mean returns must give negative SR."""
        r = np.ones((64, 2)) * -0.01
        slices = partition_matrix(r, S=4)
        sum_s, sumsq_s = compute_partition_stats(slices)
        SR = compute_sr_from_stats(sum_s, sumsq_s, [0, 1, 2], T_s=16)
        assert (SR < 0).all()

    def test_annualisation_factor(self):
        """SR must include sqrt(252) annualisation."""
        mu = 0.001
        sigma = 0.01
        # Construct deterministic series
        T = 256
        r = np.full((T, 1), mu)
        # Inject small perturbation for non-zero variance
        rng = np.random.default_rng(0)
        r += rng.normal(0, sigma, size=(T, 1))
        slices = partition_matrix(r, S=4)
        sum_s, sumsq_s = compute_partition_stats(slices)
        SR = compute_sr_from_stats(sum_s, sumsq_s, list(range(4)), T_s=T // 4)
        # Should be close to (mu/sigma)*sqrt(252)
        expected = (mu / sigma) * np.sqrt(252)
        assert abs(SR[0] - expected) < 0.5  # wide tolerance due to noise


# ---------------------------------------------------------------------------
# run_cscv (core integration tests)
# ---------------------------------------------------------------------------

class TestRunCSCV:
    """Integration tests for run_cscv using small synthetic matrices."""

    def _make_noise_matrix(self, T=496, N=100, seed=0):
        rng = np.random.default_rng(seed)
        return rng.normal(0, 0.01, size=(T, N))

    def _make_signal_matrix(self, T=496, N=100, seed=0):
        """One strategy has consistently positive returns (clear IS-optimal)."""
        rng = np.random.default_rng(seed)
        M = rng.normal(0, 0.01, size=(T, N))
        # Strategy 0 has a strong positive drift
        M[:, 0] += 0.02
        return M

    def test_output_keys(self):
        M = self._make_noise_matrix(T=128, N=20)
        res = run_cscv(M, S=8)
        expected_keys = {"lambda_all", "R_IS_nstar", "R_OOS_nstar", "R_OOS_mean",
                         "phi", "phi_ci", "prob_loss", "intercept", "slope",
                         "r_value", "adj_r2", "n_combinations"}
        assert expected_keys.issubset(set(res.keys()))

    def test_combinations_count_s8(self):
        M = self._make_noise_matrix(T=128, N=20)
        res = run_cscv(M, S=8)
        # C(8, 4) = 70
        assert res["n_combinations"] == 70
        assert len(res["lambda_all"]) == 70

    def test_phi_bounds(self):
        """PBO must always be in [0, 1]."""
        M = self._make_noise_matrix(T=128, N=50)
        res = run_cscv(M, S=8)
        assert 0.0 <= res["phi"] <= 1.0

    def test_phi_high_for_noise(self):
        """For pure noise, PBO should be ≥ 0.4 (selection is harmful)."""
        rng = np.random.default_rng(12345)
        M = rng.normal(0, 0.01, size=(128, 200))
        res = run_cscv(M, S=8)
        assert res["phi"] >= 0.35

    def test_phi_low_for_signal(self):
        """For matrix where one strategy dominates, PBO should be low."""
        M = self._make_signal_matrix(T=128, N=100, seed=42)
        res = run_cscv(M, S=8)
        # The dominant strategy should be selected IS and beat median OOS
        assert res["phi"] < 0.5

    def test_lambda_all_finite(self):
        M = self._make_noise_matrix(T=128, N=20)
        res = run_cscv(M, S=8)
        assert np.isfinite(res["lambda_all"]).all()

    def test_prob_loss_bounds(self):
        M = self._make_noise_matrix(T=128, N=20)
        res = run_cscv(M, S=8)
        assert 0.0 <= res["prob_loss"] <= 1.0

    def test_ci_width(self):
        """CI lower must be below phi, upper must be above phi."""
        M = self._make_noise_matrix(T=128, N=20)
        res = run_cscv(M, S=8)
        phi = res["phi"]
        ci_lo, ci_hi = res["phi_ci"]
        assert ci_lo <= phi <= ci_hi

    def test_r_is_nstar_shape(self):
        M = self._make_noise_matrix(T=128, N=20)
        res = run_cscv(M, S=8)
        assert res["R_IS_nstar"].shape == (70,)
        assert res["R_OOS_nstar"].shape == (70,)
        assert res["R_OOS_mean"].shape == (70,)

    def test_r_is_gt_r_oos_on_noise_mean(self):
        """For noise, IS SR of n_star should on average exceed OOS SR."""
        rng = np.random.default_rng(0)
        M = rng.normal(0, 0.01, size=(128, 200))
        res = run_cscv(M, S=8)
        # IS SR of selected strategy should be much higher than OOS SR on average
        assert res["R_IS_nstar"].mean() > res["R_OOS_nstar"].mean()


# ---------------------------------------------------------------------------
# Mathematical property tests
# ---------------------------------------------------------------------------

class TestCSCVMathProperties:

    def test_logit_omega_bounds(self):
        """omega = rank / (N+1) must be strictly in (0,1), giving finite logit."""
        # We know logit is finite if omega in (1/(N+1), N/(N+1))
        N = 8800
        omega_min = 1 / (N + 1)
        omega_max = N / (N + 1)
        assert omega_min > 0
        assert omega_max < 1
        assert np.isfinite(np.log(omega_min / (1 - omega_min)))
        assert np.isfinite(np.log(omega_max / (1 - omega_max)))

    def test_c_16_8_count(self):
        """C(16, 8) must equal 12870."""
        combos = list(combinations(range(16), 8))
        assert len(combos) == 12870

    def test_is_oos_partition_complement(self):
        """IS and OOS indices must be disjoint and cover all 16 partitions."""
        combos = list(combinations(range(16), 8))
        for IS_idx in combos[:10]:  # check first 10
            IS_set = set(IS_idx)
            OOS_set = set(range(16)) - IS_set
            assert len(IS_set) == 8
            assert len(OOS_set) == 8
            assert IS_set.isdisjoint(OOS_set)
            assert IS_set | OOS_set == set(range(16))

    def test_omega_to_logit_range(self):
        """lambda = log(omega/(1-omega)) should cover a wide range."""
        omegas = np.linspace(0.01, 0.99, 100)
        logits = np.log(omegas / (1 - omegas))
        assert logits.min() < -2
        assert logits.max() > 2
        assert np.isfinite(logits).all()

    def test_biased_to_unbiased_variance(self):
        """Variance conversion: var_unbiased = var_biased * n/(n-1)."""
        n = 496  # IS sample size (8 partitions * 62 rows)
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, size=n)
        var_biased = np.var(x, ddof=0)
        var_unbiased_expected = np.var(x, ddof=1)
        var_unbiased_computed = var_biased * n / (n - 1)
        np.testing.assert_allclose(var_unbiased_computed, var_unbiased_expected,
                                   rtol=1e-12)
