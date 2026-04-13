"""
cf_qmle.py
----------
Control-Function Quasi-Maximum Likelihood Estimator (CF-QMLE) for Spatial
Durbin Models with endogenous Spatio-Temporal Weight Matrices.

Model
-----
    Y = δ·WY + X·β + WX·θ + ε̄̂·δ_c + ξ,    ξ ~ N(0, σ²·I_N)

    ⟺  S(δ)·Y = R·α + ξ,
        S(δ) = I_N − δ·W,
        R    = [X, WX, ε̄̂],
        α    = (β, θ, δ_c)

Profile log-likelihood (concentrated over α, σ²)
-------------------------------------------------
    ℓ_p(δ) = ln|S(δ)| − (N/2)·ln(σ̂²(δ))

    σ̂²(δ) = ‖M_R S(δ)Y‖² / N,   M_R = I − R(R'R)^{-1}R'

For the Kronecker structure W = M ⊗ W_S, the log-determinant is computed via:

    ln|I_N − δ·W| = Σ_j Σ_k ln|1 − δ·μ_j·λ_k|

where μ_j are eigenvalues of M (T×T) and λ_k are eigenvalues of W_S (n×n).
This reduces the cost from O(N³) to O(T·n) evaluations.

For a custom (non-Kronecker) W, the log-det is computed directly from the
eigenvalues of the full N×N matrix (precomputed once).

References
----------
Paper Section 6 & Appendix C.
"""

from __future__ import annotations
import numpy as np
from scipy import linalg, optimize
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Log-determinant helpers
# ---------------------------------------------------------------------------

def logdet_kronecker(delta: float,
                     eigvals_M: np.ndarray,
                     eigvals_WS: np.ndarray) -> float:
    """
    Efficient log|I − δ·(M⊗W_S)| via Kronecker eigenvalue decomposition.

    Returns -inf if (I − δW) is singular for this δ.
    """
    total = 0.0
    for mu in eigvals_M:
        for lam in eigvals_WS:
            val = 1.0 - delta * mu * lam
            if val <= 0:
                return -np.inf
            total += np.log(abs(val))
    return total


def logdet_full(delta: float, eigvals_W: np.ndarray) -> float:
    """
    log|I − δ·W| from precomputed eigenvalues of the full W matrix.

    Use for non-Kronecker W.  Precompute eigvals_W = np.linalg.eigvals(W).real
    once before optimisation.
    """
    vals = 1.0 - delta * eigvals_W
    if np.any(vals <= 0):
        return -np.inf
    return float(np.sum(np.log(np.abs(vals))))


# ---------------------------------------------------------------------------
# Profile log-likelihood
# ---------------------------------------------------------------------------

def profile_loglik(delta: float,
                   Y: np.ndarray,
                   W: np.ndarray,
                   R: np.ndarray,
                   logdet_fn,
                   sign: float = -1.0) -> float:
    """
    Compute  sign · ℓ_p(δ)  (sign=-1 for minimisation).

    Parameters
    ----------
    delta     : candidate spatial-lag parameter
    Y         : (N,) dependent variable
    W         : (N, N) weight matrix
    R         : (N, p) augmented regressors  [X, WX, ε̄̂]
    logdet_fn : callable(delta) → log|I − δW|
    sign      : -1 for minimisation (default), +1 for likelihood value

    Returns
    -------
    scalar (sign · profile log-likelihood)
    """
    N = len(Y)
    S   = np.eye(N) - delta * W
    SY  = S @ Y

    # Concentrated OLS
    RtR_inv = linalg.pinv(R.T @ R)
    M_R_SY  = SY - R @ (RtR_inv @ (R.T @ SY))
    sigma2   = float(M_R_SY @ M_R_SY) / N

    if sigma2 <= 1e-14:
        return np.inf * sign

    ld = logdet_fn(delta)
    if not np.isfinite(ld):
        return np.inf * sign

    ll = -N / 2.0 * (1.0 + np.log(2.0 * np.pi * sigma2)) + ld
    return sign * ll


# ---------------------------------------------------------------------------
# CF-QMLE estimator
# ---------------------------------------------------------------------------

def cf_qmle(Y: np.ndarray,
            X: np.ndarray,
            W: np.ndarray,
            bar_eps_hat: np.ndarray,
            M: Optional[np.ndarray] = None,
            W_S: Optional[np.ndarray] = None,
            delta_bounds: Optional[Tuple[float, float]] = None,
            n_grid: int = 80) -> dict:
    """
    CF-QMLE estimation.

    Provide (M, W_S) for fast Kronecker log-det, or leave both None to
    fall back to the full eigenvalue decomposition of W.

    Parameters
    ----------
    Y           : (N,) dependent variable
    X           : (N, k) exogenous regressors
    W           : (N, N) spatio-temporal weight matrix  (W = M⊗W_S or custom)
    bar_eps_hat : (N,) control-function vector ε̄̂
    M           : (T, T) temporal weight matrix  (Kronecker mode)
    W_S         : (n, n) spatial weight matrix   (Kronecker mode)
    delta_bounds: (lo, hi) search interval for δ; auto-computed if None
    n_grid      : number of grid points for initial search

    Returns
    -------
    dict with keys:
        delta_hat, alpha_hat (=(beta, theta, delta_c)), sigma2_hat,
        loglik, eigvals_M, eigvals_WS (or eigvals_W), method
    """
    Y   = np.asarray(Y, float).ravel()
    X   = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    N   = len(Y)
    bar_eps_hat = np.asarray(bar_eps_hat, float).ravel()

    R = np.column_stack([X, W @ X, bar_eps_hat])  # (N, k+k+1)

    # Set up log-det function and bounds
    if M is not None and W_S is not None:
        eig_M  = linalg.eigvals(M).real
        eig_WS = linalg.eigvals(W_S).real
        logdet_fn = lambda d: logdet_kronecker(d, eig_M, eig_WS)
        method = "kronecker"
        extra = {"eigvals_M": eig_M, "eigvals_WS": eig_WS}

        if delta_bounds is None:
            from .weight_construction import delta_admissible_range
            lo, hi = delta_admissible_range(eig_M, eig_WS)
            delta_bounds = (max(lo, -0.9999), min(hi, 0.9999))
    else:
        eig_W   = linalg.eigvals(W).real
        logdet_fn = lambda d: logdet_full(d, eig_W)
        method = "full"
        extra = {"eigvals_W": eig_W}

        if delta_bounds is None:
            pos = eig_W[eig_W > 0]
            neg = eig_W[eig_W < 0]
            lo  = float(-1.0 / pos.max()) if len(pos) else -0.9999
            hi  = float(-1.0 / neg.min()) if len(neg) else  0.9999
            delta_bounds = (max(lo, -0.9999), min(hi, 0.9999))

    lo, hi = delta_bounds
    objective = lambda d: profile_loglik(d, Y, W, R, logdet_fn, sign=-1.0)

    # Grid search for a good starting interval
    grid = np.linspace(lo + 0.01 * (hi - lo), hi - 0.01 * (hi - lo), n_grid)
    ll_grid = np.array([objective(d) for d in grid])
    best_idx = int(np.argmin(ll_grid))
    d_lo = grid[max(best_idx - 2, 0)]
    d_hi = grid[min(best_idx + 2, n_grid - 1)]

    # Brent refinement
    res = optimize.minimize_scalar(
        objective,
        bounds=(d_lo, d_hi),
        method="bounded",
        options={"xatol": 1e-10, "maxiter": 500},
    )
    delta_hat = float(res.x)

    # Recover α̂ = (β̂, θ̂, δ̂_c)
    S        = np.eye(N) - delta_hat * W
    SY       = S @ Y
    RtR_inv  = linalg.pinv(R.T @ R)
    alpha_hat = RtR_inv @ (R.T @ SY)
    M_R_SY   = SY - R @ alpha_hat
    sigma2_hat = float(M_R_SY @ M_R_SY) / N

    return {
        "delta_hat" : delta_hat,
        "alpha_hat" : alpha_hat,   # (beta, theta, delta_c)
        "sigma2_hat": sigma2_hat,
        "loglik"    : float(-res.fun),
        "method"    : method,
        **extra,
    }


# ---------------------------------------------------------------------------
# Asymptotic variance of CF-QMLE  (information matrix approach)
# ---------------------------------------------------------------------------

def cf_qmle_avar(Y: np.ndarray,
                 X: np.ndarray,
                 W: np.ndarray,
                 bar_eps_hat: np.ndarray,
                 delta_hat: float,
                 alpha_hat: np.ndarray,
                 sigma2_hat: float,
                 logdet_fn,
                 h: float = 1e-5) -> np.ndarray:
    """
    Full asymptotic variance of sqrt(N)·(δ̂, α̂) via the numerical Hessian
    of the joint log-likelihood (Theorem 7).

    The joint log-likelihood at fixed σ̂² is

        L(δ, α) = log|I−δW| − (N/2)log(2πσ̂²)
                  − ‖S(δ)Y − Rα‖² / (2σ̂²)

    where S(δ) = I−δW,  R = [X, WX, ε̄̂],  α = (β, θ, δ_c).

    The asymptotic variance is

        AVar(δ̂, α̂) = N · (−∂²L/∂θ∂θ')⁻¹  at θ̂ = (δ̂, α̂).

    Step sizes are scaled to parameter magnitude for numerical stability:
        h_j = max(h, h · |θ̂_j|).

    Parameters
    ----------
    delta_hat  : estimated δ
    alpha_hat  : (k+k+1,) estimated (β, θ_coef, δ_c)
    sigma2_hat : estimated σ² (concentrated value)
    logdet_fn  : callable(delta) → log|I − δW|
    h          : base finite-difference step

    Returns
    -------
    avar : (2k+2, 2k+2) asymptotic variance matrix of sqrt(N)·(δ̂, α̂)
    """
    Y   = np.asarray(Y, float).ravel()
    X   = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    N   = len(Y)
    bar_eps_hat = np.asarray(bar_eps_hat, float).ravel()
    R   = np.column_stack([X, W @ X, bar_eps_hat])   # (N, k+k+1)

    theta_hat = np.concatenate([[delta_hat], alpha_hat])
    p = len(theta_hat)   # 2k+2

    def loglik(theta):
        delta = float(theta[0])
        alpha = theta[1:]
        ld = logdet_fn(delta)
        if not np.isfinite(ld):
            return -1e300
        S   = np.eye(N) - delta * W
        res = S @ Y - R @ alpha
        return ld - N / 2.0 * np.log(2.0 * np.pi * sigma2_hat) \
               - float(res @ res) / (2.0 * sigma2_hat)

    # Numerical Hessian via central differences
    ll0 = loglik(theta_hat)
    H   = np.zeros((p, p))
    for i in range(p):
        hi = max(h, h * abs(float(theta_hat[i])))
        for j in range(i, p):
            hj = max(h, h * abs(float(theta_hat[j])))
            if i == j:
                ei = np.zeros(p); ei[i] = hi
                H[i, i] = (loglik(theta_hat + ei)
                            + loglik(theta_hat - ei) - 2.0 * ll0) / hi ** 2
            else:
                ei = np.zeros(p); ei[i] = hi
                ej = np.zeros(p); ej[j] = hj
                H[i, j] = (loglik(theta_hat + ei + ej)
                            - loglik(theta_hat + ei - ej)
                            - loglik(theta_hat - ei + ej)
                            + loglik(theta_hat - ei - ej)) / (4.0 * hi * hj)
                H[j, i] = H[i, j]

    neg_H = -H
    try:
        avar = N * linalg.inv(neg_H)
    except linalg.LinAlgError:
        avar = N * linalg.pinv(neg_H)
    return avar   # (2k+2, 2k+2)


def qmle_static(Y: np.ndarray,
                X: np.ndarray,
                W_static: np.ndarray,
                eigvals_W_static: np.ndarray,
                delta_bounds: Tuple[float, float] = (-0.9999, 0.9999),
                n_grid: int = 60) -> float:
    """
    Misspecified QMLE using a static (exogenous) W, WITHOUT the CF augmentation.
    Useful as a baseline comparison.

    Parameters
    ----------
    W_static           : (N, N) static weight matrix (e.g. I_T ⊗ W_S)
    eigvals_W_static   : precomputed eigenvalues of W_static

    Returns
    -------
    delta_hat : scalar point estimate
    """
    Y = np.asarray(Y, float).ravel()
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]

    R = np.column_stack([X, W_static @ X])
    logdet_fn = lambda d: logdet_full(d, eigvals_W_static)
    objective = lambda d: profile_loglik(d, Y, W_static, R, logdet_fn, sign=-1.0)

    lo, hi = delta_bounds
    grid    = np.linspace(lo + 0.01, hi - 0.01, n_grid)
    ll_grid = np.array([objective(d) for d in grid])
    best_idx = int(np.argmin(ll_grid))
    d_lo = grid[max(best_idx - 2, 0)]
    d_hi = grid[min(best_idx + 2, n_grid - 1)]

    res = optimize.minimize_scalar(
        objective, bounds=(d_lo, d_hi), method="bounded",
        options={"xatol": 1e-10}
    )
    return float(res.x)
