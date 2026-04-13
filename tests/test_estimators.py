"""
test_estimators.py
------------------
Unit and integration tests for gstw_pdm.

Run with:  pytest tests/ -v
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gstw_pdm import (
    rook_weights, build_twm_parametric, build_stwm, stwm_summary,
    compute_morans_i, eigvals_kronecker, logdet_kronecker,
    first_stage, aggregate_eps_hat,
    cf_2sls, cf_2sls_fit,
    cf_qmle,
    cf_gmm,
    bh_correction,
    multiplier_matrix, cross_period_effects_matrix,
    omega_A_simple, compute_sigma2_eps,
    ned_test, check_admissibility,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_dgp():
    """
    Minimal DGP: n=9 (3×3 grid), T=4, single regressor.
    Returns Y, X, W, W_S, M, h, Z_W, eps_hat, bar_eps_hat with known delta=0.3.
    """
    rng = np.random.default_rng(42)
    n_side = 3
    T      = 4
    n      = n_side ** 2
    N      = n * T

    W_S = rook_weights(n_side)
    M   = build_twm_parametric(T, rho=0.5)
    W   = build_stwm(M, W_S)

    X     = rng.standard_normal((N, 1))
    Z_W   = rng.standard_normal((T, 2))
    eps   = rng.standard_normal(T)
    h     = (Z_W @ np.array([1.0, 0.5])) + eps

    eps_hat, pi_hat = first_stage(h, Z_W)
    bar_eps_hat     = aggregate_eps_hat(eps_hat, n, T)

    delta0   = 0.3
    beta0    = 1.0
    theta0   = 0.5
    delta_c0 = 0.4

    # Generate Y from known parameters
    bar_eps_true = np.repeat(eps, n)
    xi   = rng.standard_normal(N) * 0.5
    S    = np.eye(N) - delta0 * W
    rhs  = X.ravel() * beta0 + (W @ X).ravel() * theta0 + bar_eps_true * delta_c0 + xi
    Y    = np.linalg.solve(S, rhs)

    return dict(Y=Y, X=X, W=W, W_S=W_S, M=M, h=h, Z_W=Z_W,
                eps_hat=eps_hat, bar_eps_hat=bar_eps_hat,
                n=n, T=T, N=N, n_side=n_side,
                delta0=delta0, beta0=beta0, theta0=theta0)


# ---------------------------------------------------------------------------
# Weight construction
# ---------------------------------------------------------------------------

class TestWeightConstruction:
    def test_rook_row_sum(self):
        W = rook_weights(5)
        assert_allclose(W.sum(axis=1), np.ones(25), atol=1e-10)

    def test_stwm_shape(self):
        W_S = rook_weights(4)
        M   = build_twm_parametric(5)
        W   = build_stwm(M, W_S)
        assert W.shape == (80, 80)

    def test_kronecker_structure(self):
        W_S = rook_weights(3)
        M   = build_twm_parametric(4, rho=0.6)
        W   = build_stwm(M, W_S)
        W_kron = np.kron(M, W_S)
        assert_allclose(W, W_kron, atol=1e-12)

    def test_stwm_summary_keys(self):
        W = build_stwm(build_twm_parametric(3), rook_weights(3))
        s = stwm_summary(W, 9, 3)
        for key in ["sparsity", "spectral_radius", "N"]:
            assert key in s

    def test_morans_i_random(self):
        rng = np.random.default_rng(0)
        y   = rng.standard_normal(25)
        W   = rook_weights(5)
        mi  = compute_morans_i(y, W)
        assert -1.5 < mi < 1.5

    def test_logdet_kronecker_matches_direct(self):
        W_S = rook_weights(3)
        M   = build_twm_parametric(4, rho=0.5)
        W   = build_stwm(M, W_S)
        eM, eWS = eigvals_kronecker(M, W_S)
        N  = W.shape[0]
        delta = 0.2
        ld_kron  = logdet_kronecker(delta, eM, eWS)
        _, logdet_sign = np.linalg.slogdet(np.eye(N) - delta * W)
        ld_direct = logdet_sign
        assert abs(ld_kron - ld_direct) < 1e-6


# ---------------------------------------------------------------------------
# First stage & control function
# ---------------------------------------------------------------------------

class TestFirstStage:
    def test_residuals_orthogonal_to_Z(self, small_dgp):
        dgp = small_dgp
        eps_hat = dgp["eps_hat"]
        Z_W     = dgp["Z_W"]
        assert_allclose(Z_W.T @ eps_hat, np.zeros(Z_W.shape[1]), atol=1e-8)

    def test_aggregate_length(self, small_dgp):
        dgp = small_dgp
        bar_eps = aggregate_eps_hat(dgp["eps_hat"], dgp["n"], dgp["T"])
        assert len(bar_eps) == dgp["N"]

    def test_aggregate_period_values(self, small_dgp):
        dgp = small_dgp
        bar = aggregate_eps_hat(dgp["eps_hat"], dgp["n"], dgp["T"])
        n, T = dgp["n"], dgp["T"]
        for t in range(T):
            assert_allclose(bar[t * n:(t + 1) * n],
                            dgp["eps_hat"][t] * np.ones(n), atol=1e-12)


# ---------------------------------------------------------------------------
# CF-2SLS
# ---------------------------------------------------------------------------

class TestCF2SLS:
    def test_shape(self, small_dgp):
        dgp = small_dgp
        kappa = cf_2sls(dgp["Y"], dgp["X"], dgp["W"], dgp["bar_eps_hat"])
        assert kappa.shape == (4,)   # delta, beta, theta, delta_c  (k=1)

    def test_finite(self, small_dgp):
        dgp = small_dgp
        kappa = cf_2sls(dgp["Y"], dgp["X"], dgp["W"], dgp["bar_eps_hat"])
        assert np.all(np.isfinite(kappa))

    def test_delta_sign(self, small_dgp):
        """CF-2SLS should recover positive delta (=0.3 in DGP)."""
        dgp   = small_dgp
        kappa = cf_2sls(dgp["Y"], dgp["X"], dgp["W"], dgp["bar_eps_hat"])
        assert kappa[0] > 0.0

    def test_fit_dict_keys(self, small_dgp):
        dgp    = small_dgp
        result = cf_2sls_fit(dgp["Y"], dgp["X"], dgp["W"], dgp["bar_eps_hat"])
        for key in ["kappa", "se", "t_stats", "p_values", "sigma2"]:
            assert key in result

    def test_se_positive(self, small_dgp):
        dgp    = small_dgp
        result = cf_2sls_fit(dgp["Y"], dgp["X"], dgp["W"], dgp["bar_eps_hat"])
        assert np.all(result["se"] >= 0)


# ---------------------------------------------------------------------------
# CF-QMLE
# ---------------------------------------------------------------------------

class TestCFQMLE:
    def test_delta_in_bounds(self, small_dgp):
        dgp = small_dgp
        res = cf_qmle(dgp["Y"], dgp["X"], dgp["W"], dgp["bar_eps_hat"],
                      M=dgp["M"], W_S=dgp["W_S"])
        assert -1 < res["delta_hat"] < 1

    def test_sigma2_positive(self, small_dgp):
        dgp = small_dgp
        res = cf_qmle(dgp["Y"], dgp["X"], dgp["W"], dgp["bar_eps_hat"],
                      M=dgp["M"], W_S=dgp["W_S"])
        assert res["sigma2_hat"] > 0

    def test_kronecker_vs_full(self, small_dgp):
        """Kronecker and full-eigenvalue log-det should give same δ̂."""
        dgp    = small_dgp
        res_kr = cf_qmle(dgp["Y"], dgp["X"], dgp["W"], dgp["bar_eps_hat"],
                         M=dgp["M"], W_S=dgp["W_S"])
        res_fu = cf_qmle(dgp["Y"], dgp["X"], dgp["W"], dgp["bar_eps_hat"])
        assert abs(res_kr["delta_hat"] - res_fu["delta_hat"]) < 1e-4


# ---------------------------------------------------------------------------
# CF-GMM
# ---------------------------------------------------------------------------

class TestCFGMM:
    def test_shape(self, small_dgp):
        dgp  = small_dgp
        kappa, sig2, _ = cf_gmm(dgp["Y"], dgp["X"], dgp["W"],
                                  dgp["bar_eps_hat"], max_iter=1)
        assert kappa.shape == (4,)

    def test_sigma2_positive(self, small_dgp):
        dgp  = small_dgp
        _, sig2, _ = cf_gmm(dgp["Y"], dgp["X"], dgp["W"],
                             dgp["bar_eps_hat"], max_iter=1)
        assert sig2 > 0


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

class TestInference:
    def test_multiplier_matrix_shape(self, small_dgp):
        dgp = small_dgp
        T_mat = multiplier_matrix(0.2, dgp["W"])
        assert T_mat.shape == (dgp["N"], dgp["N"])

    def test_ie_matrix_shape(self, small_dgp):
        dgp   = small_dgp
        T_mat = multiplier_matrix(0.2, dgp["W"])
        IE    = cross_period_effects_matrix(
            T_mat, 1.0, 0.5, dgp["W"], dgp["n"], dgp["T"]
        )
        assert IE.shape == (dgp["T"], dgp["T"])

    def test_bh_correction_shape(self):
        rng  = np.random.default_rng(0)
        pv   = rng.uniform(0, 1, (5, 5))
        rej, adj = bh_correction(pv, alpha=0.05)
        assert rej.shape == (5, 5)
        assert adj.shape == (5, 5)
        assert np.all((adj >= 0) & (adj <= 1))

    def test_bh_monotone(self):
        """Adjusted p-values should be ≥ raw p-values."""
        rng  = np.random.default_rng(1)
        pv   = np.sort(rng.uniform(0, 1, 20))
        _, adj = bh_correction(pv)
        # BH-adjusted ≥ raw for the individual comparison
        assert np.all(adj >= pv - 1e-10)


# ---------------------------------------------------------------------------
# Variance correction
# ---------------------------------------------------------------------------

class TestVarianceCorrection:
    def test_omega_A_shape(self, small_dgp):
        dgp    = small_dgp
        N, T   = dgp["N"], dgp["T"]
        n      = dgp["n"]
        bar    = dgp["bar_eps_hat"]
        X      = dgp["X"]
        W      = dgp["W"]
        Z_W    = dgp["Z_W"]

        WX  = W @ X
        W2X = W @ WX
        Q   = np.column_stack([X, WX, W2X, bar[:, None]])  # (N, 4)

        sig2_eps = compute_sigma2_eps(dgp["eps_hat"], Z_W)
        omega = omega_A_simple(Q, Z_W, n, T, delta_c_hat=0.3,
                               sigma2_eps=sig2_eps)
        assert omega.shape == (4, 4)

    def test_omega_A_psd(self, small_dgp):
        dgp  = small_dgp
        n, T = dgp["n"], dgp["T"]
        X, W = dgp["X"], dgp["W"]
        Z_W  = dgp["Z_W"]
        bar  = dgp["bar_eps_hat"]

        WX  = W @ X; W2X = W @ WX
        Q   = np.column_stack([X, WX, W2X, bar[:, None]])
        sig2_eps = compute_sigma2_eps(dgp["eps_hat"], Z_W)
        omega = omega_A_simple(Q, Z_W, n, T, 0.3, sig2_eps)

        eigvals = np.linalg.eigvalsh(omega)
        assert np.all(eigvals >= -1e-10), "Ω_A should be PSD"


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------

class TestUtils:
    def test_ned_test_keys(self, small_dgp):
        dgp = small_dgp
        kappa = cf_2sls(dgp["Y"], dgp["X"], dgp["W"], dgp["bar_eps_hat"])
        xi_hat = dgp["Y"] - (
            dgp["W"] @ dgp["Y"] * kappa[0]
            + dgp["X"].ravel() * kappa[1]
            + (dgp["W"] @ dgp["X"]).ravel() * kappa[2]
            + dgp["bar_eps_hat"] * kappa[3]
        )
        result = ned_test(xi_hat, dgp["n"], dgp["T"], dgp["W_S"])
        for key in ["morans_by_period", "temporal_autocorr", "summary"]:
            assert key in result

    def test_check_admissibility(self, small_dgp):
        dgp = small_dgp
        from gstw_pdm import eigvals_kronecker
        eM, eWS = eigvals_kronecker(dgp["M"], dgp["W_S"])
        assert check_admissibility(0.2, eM, eWS)
        assert not check_admissibility(100.0, eM, eWS)
