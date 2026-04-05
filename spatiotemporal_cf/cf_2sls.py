"""
cf_2sls.py
----------
Control-Function Two-Stage Least Squares (CF-2SLS) estimator for Spatial
Durbin Models with endogenous Spatio-Temporal Weight Matrices.

Structural equation
-------------------
    Y = δ·WY + X·β + WX·θ + ε̄̂·δ_c + ξ

    ⟺  Y = U_tilde · κ + ξ

where  U_tilde = [WY, X, WX, ε̄̂]   (augmented regressors)
and     Q      = [X,  WX, W²X, ε̄̂]  (instrument set).

CF-2SLS estimator
-----------------
    κ̂ = (Û'U_tilde)^{-1} Û'Y,   Û = P_Q U_tilde,   P_Q = Q(Q'Q)^{-1}Q'

Variance–covariance (corrected for generated regressor ε̄̂)
-------------------------------------------------------------
The sandwich formula depends on whether L = T or L = N:

  L = T  →  plug in Ω_A correction (see variance_correction.py)
  L = N  →  standard 2SLS sandwich  V = σ²(Û'U_tilde)^{-1}

References
----------
Paper Section 5 & Appendix B.
"""

from __future__ import annotations
import numpy as np
from scipy import linalg
from typing import Optional


# ---------------------------------------------------------------------------
# Core CF-2SLS
# ---------------------------------------------------------------------------

def cf_2sls(Y: np.ndarray,
            X: np.ndarray,
            W: np.ndarray,
            bar_eps_hat: np.ndarray) -> np.ndarray:
    """
    CF-2SLS point estimates.

    Parameters
    ----------
    Y           : (N,) dependent variable
    X           : (N, k) exogenous regressors
    W           : (N, N) spatio-temporal weight matrix
    bar_eps_hat : (N,) aggregated control function  ε̄̂

    Returns
    -------
    kappa_hat : (k+3,) = [δ̂, β̂₁…β̂_k, θ̂₁…θ̂_k, δ̂_c]
        Ordering: [δ̂,  β̂ (k,),  θ̂ (k,),  δ̂_c]
    """
    Y   = np.asarray(Y, float).ravel()
    X   = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    bar_eps_hat = np.asarray(bar_eps_hat, float).ravel()[:, None]

    WY  = (W @ Y)[:, None]
    WX  = W @ X
    W2X = W @ WX

    U_tilde = np.column_stack([WY, X, WX, bar_eps_hat])   # (N, 2k+2)
    Q       = np.column_stack([X,  WX, W2X, bar_eps_hat]) # (N, 2k+2)

    QtQ_inv = linalg.pinv(Q.T @ Q)
    P_Q_U   = Q @ (QtQ_inv @ (Q.T @ U_tilde))             # (N, 2k+2)
    P_Q_Y   = Q @ (QtQ_inv @ (Q.T @ Y))                   # (N,)

    A_mat = P_Q_U.T @ U_tilde                              # (2k+2, 2k+2)
    b_vec = P_Q_U.T @ Y                                    # (2k+2,)

    try:
        kappa_hat = linalg.solve(A_mat, b_vec)
    except linalg.LinAlgError:
        kappa_hat = np.full(A_mat.shape[0], np.nan)

    return kappa_hat


def cf_2sls_residuals(Y: np.ndarray,
                      X: np.ndarray,
                      W: np.ndarray,
                      bar_eps_hat: np.ndarray,
                      kappa_hat: np.ndarray) -> np.ndarray:
    """Return structural residuals ξ̂ = Y - U_tilde κ̂."""
    Y   = np.asarray(Y, float).ravel()
    X   = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    bar_eps_hat = np.asarray(bar_eps_hat, float).ravel()[:, None]

    WY  = (W @ Y)[:, None]
    WX  = W @ X
    U_tilde = np.column_stack([WY, X, WX, bar_eps_hat])
    return Y - U_tilde @ kappa_hat


# ---------------------------------------------------------------------------
# Asymptotic variance (homoskedastic sandwich)
# ---------------------------------------------------------------------------

def cf_2sls_avar(Y: np.ndarray,
                 X: np.ndarray,
                 W: np.ndarray,
                 bar_eps_hat: np.ndarray,
                 kappa_hat: np.ndarray,
                 sigma2: Optional[float] = None,
                 omega_A: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Asymptotic variance–covariance matrix of sqrt(N)·(κ̂ − κ₀).

    Standard formula (homoskedastic, L = N case):
        AVar = σ² · N · (Û'U)^{-1}

    With Ω_A correction (L = T case):
        The variance is augmented by an additional term from the generated
        regressor ε̄̂ = Aε̂.  Pass ``omega_A`` (from variance_correction.py)
        to include this correction.

    Parameters
    ----------
    Y, X, W, bar_eps_hat, kappa_hat : as in ``cf_2sls``
    sigma2  : error variance; estimated from residuals if None
    omega_A : (2k+2, 2k+2) variance correction matrix for generated regressors
              (only needed for L = T mode); see ``variance_correction.omega_A``

    Returns
    -------
    avar : (2k+2, 2k+2)  AVar of sqrt(N)·κ̂  (divide by N for finite-sample Var)
    """
    Y   = np.asarray(Y, float).ravel()
    X   = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    N = len(Y)
    bar_eps_hat = np.asarray(bar_eps_hat, float).ravel()[:, None]

    WY  = (W @ Y)[:, None]
    WX  = W @ X
    W2X = W @ WX

    U_tilde = np.column_stack([WY, X, WX, bar_eps_hat])
    Q       = np.column_stack([X,  WX, W2X, bar_eps_hat])

    xi_hat = Y - U_tilde @ kappa_hat
    if sigma2 is None:
        sigma2 = float(xi_hat @ xi_hat) / N

    QtQ_inv = linalg.pinv(Q.T @ Q)
    P_Q_U   = Q @ (QtQ_inv @ (Q.T @ U_tilde))

    # Bread: B = (Û'U/N)^{-1}
    bread = linalg.pinv(P_Q_U.T @ U_tilde / N)

    # Meat (standard): σ² · Û'Û/N²  — under homoskedasticity simplifies
    meat_std = sigma2 * (P_Q_U.T @ P_Q_U) / N

    avar = bread @ meat_std @ bread

    # Add Ω_A correction if provided (L = T case)
    if omega_A is not None:
        avar = avar + bread @ omega_A @ bread

    return avar


def cf_2sls_se(avar: np.ndarray, N: int) -> np.ndarray:
    """
    Standard errors = sqrt(diag(AVar / N)).

    Parameters
    ----------
    avar : (p, p) asymptotic variance of sqrt(N)·κ̂
    N    : sample size

    Returns
    -------
    se : (p,) standard errors of κ̂
    """
    return np.sqrt(np.maximum(np.diag(avar) / N, 0.0))


# ---------------------------------------------------------------------------
# Full estimation with summary
# ---------------------------------------------------------------------------

def cf_2sls_fit(Y: np.ndarray,
                X: np.ndarray,
                W: np.ndarray,
                bar_eps_hat: np.ndarray,
                sigma2: Optional[float] = None,
                omega_A: Optional[np.ndarray] = None,
                param_names: Optional[list] = None) -> dict:
    """
    CF-2SLS estimation with coefficients, SE, t-stats, p-values.

    Returns
    -------
    dict with keys:
        kappa, se, t_stats, p_values, sigma2, avar,
        residuals, param_names
    """
    from scipy import stats as scipy_stats

    Y   = np.asarray(Y, float).ravel()
    X   = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    N, k = X.shape

    kappa_hat = cf_2sls(Y, X, W, bar_eps_hat)
    xi_hat    = cf_2sls_residuals(Y, X, W, bar_eps_hat, kappa_hat)
    sigma2_   = float(xi_hat @ xi_hat) / N if sigma2 is None else sigma2
    avar      = cf_2sls_avar(Y, X, W, bar_eps_hat, kappa_hat,
                             sigma2=sigma2_, omega_A=omega_A)
    se        = cf_2sls_se(avar, N)
    t_stats   = kappa_hat / np.where(se > 1e-14, se, np.nan)
    p_values  = 2 * scipy_stats.t.sf(np.abs(t_stats), df=N - len(kappa_hat))

    if param_names is None:
        x_names = [f"x{i+1}" for i in range(k)]
        param_names = (["delta"] + x_names
                       + [f"theta_{n}" for n in x_names]
                       + ["delta_c"])

    return {
        "kappa"      : kappa_hat,
        "se"         : se,
        "t_stats"    : t_stats,
        "p_values"   : p_values,
        "sigma2"     : sigma2_,
        "avar"       : avar,
        "residuals"  : xi_hat,
        "param_names": param_names,
        "N"          : N,
    }


def cf_2sls_summary(result: dict) -> None:
    """Print a formatted CF-2SLS results table."""
    print("\n" + "=" * 65)
    print("  CF-2SLS Estimation Results")
    print("=" * 65)
    print(f"  N = {result['N']},  σ̂² = {result['sigma2']:.6f}")
    print("-" * 65)
    print(f"  {'Parameter':<14} {'Coef':>10} {'SE':>10} {'t':>8} {'p':>8}")
    print("-" * 65)
    for name, coef, se, t, p in zip(result["param_names"],
                                     result["kappa"],
                                     result["se"],
                                     result["t_stats"],
                                     result["p_values"]):
        stars = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))
        print(f"  {name:<14} {coef:>10.4f} {se:>10.4f} {t:>8.3f} {p:>8.4f} {stars}")
    print("=" * 65)
    print("  * p<0.1  ** p<0.05  *** p<0.01\n")
