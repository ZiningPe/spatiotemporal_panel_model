"""
variance_correction.py
----------------------
Variance correction for generated regressors when L = T.

When the entry equation is estimated at the period level (L = T), each of
the n structural observations within period t shares the same first-stage
residual ε̂_t.  The aggregated control function  ε̄̂ = A·ε̂,
A = I_T ⊗ ι_n ∈ ℝ^{N×T},  is therefore a *generated regressor*.

Ignoring the estimation uncertainty in ε̂ leads to under-estimation of the
variance of κ̂.  The correction term Ω_A must be added to the sandwich meat
of the CF-2SLS variance formula.

Ω_A correction (L = T case)
-----------------------------
Let  Z_W  ∈ ℝ^{T×k_Z}  be the first-stage instruments and
     Σ_ε   = E[εε']       (T×T first-stage error covariance).

Then (Appendix B of the paper):

    Ω_A = δ_c² · (1/N) · Q_A' · (M_Z ⊗ Σ_ε) · Q_A

where  Q_A  involves the instrument set Q = [X, WX, W²X, ε̄̂]  evaluated at
the true ε̄̂, and  M_Z = I_T − Z_W(Z_W'Z_W)^{-1}Z_W'.

In practice we replace Σ_ε with the estimated covariance  σ̂²_ε · I_T  under
homoskedastic first-stage errors.

Simplified implementation
--------------------------
Under the assumption that the first-stage errors are i.i.d.:
    Σ_ε ≈ σ̂²_ε · I_T

    Ω_A ≈ δ̂_c² · σ̂²_ε · (1/N) · (A'Q)' · M_Z · (A'Q) · n^{-1}

This is the default implementation.  For a heteroskedastic first stage,
pass a (T, T) estimate of Σ_ε via the ``Sigma_eps`` argument.

References
----------
Paper Appendix B, Proposition B.3.
"""

from __future__ import annotations
import numpy as np
from scipy import linalg
from typing import Optional


def omega_A(Q: np.ndarray,
            Z_W: np.ndarray,
            A: np.ndarray,
            delta_c_hat: float,
            sigma2_eps: float,
            Sigma_eps: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Ω_A correction matrix for the L = T generated-regressor problem.

    Parameters
    ----------
    Q           : (N, m) instrument matrix  [X, WX, W²X, ε̄̂]
    Z_W         : (T, k_Z) first-stage instruments
    A           : (N, T) aggregation matrix  I_T ⊗ ι_n
    delta_c_hat : estimated coefficient on ε̄̂  (= κ̂[-1])
    sigma2_eps  : estimated first-stage error variance  σ̂²_ε
    Sigma_eps   : (T, T) first-stage error covariance; defaults to σ²·I_T

    Returns
    -------
    omega : (m, m) correction matrix to be added to the sandwich meat
    """
    N, m = Q.shape
    T    = Z_W.shape[0]
    if Z_W.ndim == 1:
        Z_W = Z_W[:, None]

    # M_Z = I_T − P_Z
    ZtZ_inv = linalg.pinv(Z_W.T @ Z_W)
    P_Z     = Z_W @ ZtZ_inv @ Z_W.T     # (T, T)
    M_Z     = np.eye(T) - P_Z            # (T, T)

    # Σ_ε
    if Sigma_eps is None:
        Sigma_eps = sigma2_eps * np.eye(T)

    # (A'Q): (T, m)  — note A = I_T ⊗ ι_n, so A'Q sums n rows per period
    AtQ = A.T @ Q                         # (T, m)

    # Core: (A'Q)' · (M_Z ⊗ Sigma_eps) · (A'Q)
    # With Sigma_eps = σ²·I_T this becomes σ² · AtQ' · M_Z · AtQ
    SIG_MZ = Sigma_eps @ M_Z             # (T, T)  (scalar when iid)
    mid    = AtQ.T @ SIG_MZ @ AtQ        # (m, m)

    omega = delta_c_hat ** 2 / N * mid
    return omega


def omega_A_simple(Q: np.ndarray,
                   Z_W: np.ndarray,
                   n: int,
                   T: int,
                   delta_c_hat: float,
                   sigma2_eps: float,
                   Sigma_eps: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Ω_A correction under the block structure A = I_T ⊗ ι_n.

    Avoids materialising the (N, T) matrix A:  AtQ[t, :] = Σ_i Q[t·n+i, :].

    Assumption 6 allows a general (T, T) first-stage error covariance Σ_ε.
    The default (``Sigma_eps=None``) assumes i.i.d. errors, i.e.
    Σ_ε = σ̂²_ε · I_T.  For heteroskedastic or serially correlated first-
    stage errors, pass a (T, T) estimate as ``Sigma_eps``; in that case
    ``sigma2_eps`` is ignored.

    Parameters
    ----------
    Q           : (N, m) instrument matrix
    Z_W         : (T, k_Z) first-stage instruments
    n, T        : number of spatial units and time periods
    delta_c_hat : estimated coefficient on ε̄̂
    sigma2_eps  : first-stage error variance (i.i.d. case)
    Sigma_eps   : (T, T) error covariance; overrides sigma2_eps if provided
    """
    N, m = Q.shape
    if Z_W.ndim == 1:
        Z_W = Z_W[:, None]

    # A'Q via block structure
    Q_blk = Q.reshape(T, n, m)
    AtQ   = Q_blk.sum(axis=1)           # (T, m)

    # M_Z = I_T − P_Z
    ZtZ_inv = linalg.pinv(Z_W.T @ Z_W)
    P_Z     = Z_W @ ZtZ_inv @ Z_W.T
    M_Z     = np.eye(T) - P_Z

    if Sigma_eps is None:
        Sigma_eps = sigma2_eps * np.eye(T)

    mid   = AtQ.T @ (Sigma_eps @ M_Z) @ AtQ   # (m, m)
    omega = delta_c_hat ** 2 / N * mid
    return omega


def compute_sigma2_eps(eps_hat: np.ndarray, Z_W: np.ndarray) -> float:
    """
    Estimate first-stage error variance  σ̂²_ε = ε̂'ε̂ / (T − k_Z).

    Parameters
    ----------
    eps_hat : (T,) first-stage OLS residuals
    Z_W     : (T, k_Z) first-stage instruments

    Returns
    -------
    sigma2_eps : scalar
    """
    T  = len(eps_hat)
    kZ = Z_W.shape[1] if Z_W.ndim == 2 else 1
    return float(eps_hat @ eps_hat) / max(T - kZ, 1)


def cf_2sls_avar_corrected(Y: np.ndarray,
                            X: np.ndarray,
                            W: np.ndarray,
                            bar_eps_hat: np.ndarray,
                            eps_hat: np.ndarray,
                            Z_W: np.ndarray,
                            kappa_hat: np.ndarray,
                            n: int,
                            T: int,
                            sigma2: Optional[float] = None) -> np.ndarray:
    """
    CF-2SLS asymptotic variance with Ω_A correction (L = T case).

    Combines the standard sandwich with the generated-regressor correction.

    Returns
    -------
    avar : (2k+2, 2k+2) corrected asymptotic variance
    """
    from .cf_2sls import cf_2sls_avar

    Y   = np.asarray(Y, float).ravel()
    X   = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    N   = len(Y)

    # Build Q for the omega_A computation
    bar_eps_hat = np.asarray(bar_eps_hat, float).ravel()[:, None]
    WX  = W @ X
    W2X = W @ WX
    Q   = np.column_stack([X, WX, W2X, bar_eps_hat])

    # Standard avar (no correction)
    avar_std = cf_2sls_avar(Y, X, W, bar_eps_hat.ravel(), kappa_hat,
                            sigma2=sigma2)

    # delta_c is the last element of kappa
    delta_c_hat = float(kappa_hat[-1])

    # Aggregation matrix A = I_T ⊗ ι_n  (implicit)
    sigma2_eps = compute_sigma2_eps(eps_hat, Z_W)

    # Ω_A
    omega = omega_A_simple(Q, Z_W, n, T, delta_c_hat, sigma2_eps)

    # Bread = (Û'U/N)^{-1}  — recompute from cf_2sls_avar structure
    bar_eps_hat_vec = bar_eps_hat.ravel()
    avar_corr = cf_2sls_avar(Y, X, W, bar_eps_hat_vec, kappa_hat,
                             sigma2=sigma2, omega_A=omega)
    return avar_corr
