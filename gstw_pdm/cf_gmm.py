"""
cf_gmm.py
---------
Control-Function Generalised Method of Moments (CF-GMM) estimator for
Spatial Durbin Models with endogenous Spatio-Temporal Weight Matrices.

Moment conditions
-----------------
Linear moments  (3k+1 conditions for k regressors in X):
    E[Q' ξ] = 0,    Q = [X, WX, W²X, ε̄̂]

Quadratic moments  (2 additional conditions):
    E[ξ' W  ξ] = σ² · tr(W)
    E[ξ' W'W ξ] = σ² · tr(W'W)

Combined moment vector g(κ, σ²):
    g = [Q'ξ/N,  (ξ'Wξ − σ²·tr(W))/N,  (ξ'W'Wξ − σ²·tr(W'W))/N]

Two-step GMM
------------
Step 1: identity weighting  Ψ₀ = I
Step 2: optimal weighting   Ψ̂ = [Cov(g)]^{-1}

References
----------
Paper Section 7 & Appendix D.
"""

from __future__ import annotations
import numpy as np
from scipy import linalg, optimize
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Moment construction
# ---------------------------------------------------------------------------

def _build_blocks(Y: np.ndarray,
                  X: np.ndarray,
                  W: np.ndarray,
                  bar_eps_hat: np.ndarray):
    """Build U_tilde, Q, WtW, traces — called once per estimation."""
    Y   = np.asarray(Y, float).ravel()
    X   = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    bar_eps_hat = np.asarray(bar_eps_hat, float).ravel()[:, None]

    WY  = (W @ Y)[:, None]
    WX  = W @ X
    W2X = W @ WX
    WtW = W.T @ W

    U_tilde = np.column_stack([WY, X, WX, bar_eps_hat])
    Q_mat   = np.column_stack([X,  WX, W2X, bar_eps_hat])

    trW   = float(np.trace(W))
    trWtW = float(np.trace(WtW))

    return Y, U_tilde, Q_mat, W, WtW, trW, trWtW


def moment_vector(kappa: np.ndarray,
                  sigma2: float,
                  Y: np.ndarray,
                  U_tilde: np.ndarray,
                  Q_mat: np.ndarray,
                  W: np.ndarray,
                  WtW: np.ndarray,
                  trW: float,
                  trWtW: float) -> np.ndarray:
    """
    Stacked moment vector g(κ, σ²) of length m_lin + 2.

    g_lin  = Q'ξ / N                          (linear moments)
    g_q1   = (ξ'Wξ  − σ²·tr(W))   / N       (quadratic moment 1)
    g_q2   = (ξ'W'Wξ − σ²·tr(W'W)) / N      (quadratic moment 2)
    """
    N   = len(Y)
    xi  = Y - U_tilde @ kappa
    g_lin  = Q_mat.T @ xi / N
    g_q1   = (xi @ (W @ xi) - sigma2 * trW)    / N
    g_q2   = (xi @ (WtW @ xi) - sigma2 * trWtW) / N
    return np.concatenate([g_lin, [g_q1, g_q2]])


# ---------------------------------------------------------------------------
# Optimal weighting matrix (diagonal approximation)
# ---------------------------------------------------------------------------

def _optimal_weight(kappa: np.ndarray,
                    sigma2: float,
                    Y: np.ndarray,
                    U_tilde: np.ndarray,
                    Q_mat: np.ndarray,
                    WtW: np.ndarray,
                    trWtW: float) -> np.ndarray:
    """
    Diagonal approximation to Ψ̂ = [Var(g)]^{-1}.

    Under homoskedasticity:
      Var(g_lin)  ≈ σ² I_{m_lin}
      Var(g_q1)   ≈ 2σ⁴·tr(W'W) / N²      (Isserlis' theorem)
      Var(g_q2)   ≈ 2σ⁴·tr((W'W)²) / N²
    """
    N      = len(Y)
    m_lin  = Q_mat.shape[1]
    var_lin  = sigma2 * np.ones(m_lin)
    trWtW2   = float(np.trace(WtW @ WtW))
    var_q1   = 2.0 * sigma2 ** 2 * trWtW / N
    var_q2   = 2.0 * sigma2 ** 2 * trWtW2 / N
    diag_psi = np.concatenate([var_lin,
                                [max(var_q1, 1e-12),
                                 max(var_q2, 1e-12)]])
    return np.diag(1.0 / diag_psi)


# ---------------------------------------------------------------------------
# CF-GMM estimation
# ---------------------------------------------------------------------------

def cf_gmm(Y: np.ndarray,
           X: np.ndarray,
           W: np.ndarray,
           bar_eps_hat: np.ndarray,
           sigma2_init: float = 1.0,
           max_iter: int = 2,
           tol: float = 1e-8) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Two-step (iterated) CF-GMM.

    Parameters
    ----------
    Y, X, W, bar_eps_hat : as in CF-2SLS
    sigma2_init : initial σ² for step-1 weighting
    max_iter    : number of weighting updates  (2 = two-step GMM)
    tol         : convergence tolerance for Nelder-Mead

    Returns
    -------
    kappa_hat  : (2k+2,) parameter vector [δ, β₁…βₖ, θ₁…θₖ, δ_c]
    sigma2_hat : scalar
    Psi_inv    : (m, m) final optimal weighting matrix
    """
    Y, U_tilde, Q_mat, W_, WtW, trW, trWtW = _build_blocks(Y, X, W, bar_eps_hat)
    n_kappa = U_tilde.shape[1]

    def gmm_obj(params, Psi_inv):
        kappa  = params[:n_kappa]
        sigma2 = max(params[n_kappa], 1e-8)
        g = moment_vector(kappa, sigma2, Y, U_tilde, Q_mat, W_, WtW, trW, trWtW)
        return float(g @ Psi_inv @ g)

    n_moments = Q_mat.shape[1] + 2   # 3k+3; fixes k>=2 dimension bug
    Psi_inv = np.eye(n_moments)
    x0 = np.zeros(n_kappa + 1)
    x0[-1] = sigma2_init

    for _ in range(max_iter):
        res = optimize.minimize(
            lambda p: gmm_obj(p, Psi_inv),
            x0,
            method="Nelder-Mead",
            options={"maxiter": 10_000, "xatol": tol, "fatol": tol * 1e-2},
        )
        kappa_step  = res.x[:n_kappa]
        sigma2_step = max(res.x[n_kappa], 1e-8)

        # Update weighting matrix
        Psi_inv = _optimal_weight(kappa_step, sigma2_step,
                                  Y, U_tilde, Q_mat, WtW, trWtW)
        x0 = res.x

    return kappa_step, sigma2_step, Psi_inv


# ---------------------------------------------------------------------------
# GMM asymptotic variance  (sandwich)
# ---------------------------------------------------------------------------

def cf_gmm_avar(Y: np.ndarray,
                X: np.ndarray,
                W: np.ndarray,
                bar_eps_hat: np.ndarray,
                kappa_hat: np.ndarray,
                sigma2_hat: float,
                Psi_inv: np.ndarray,
                h: float = 1e-5) -> np.ndarray:
    """
    Sandwich asymptotic variance of sqrt(N)·κ̂_GMM.

        AVar = (G'ΨG)^{-1} G'Ψ Var(√N·g) Ψ G (G'ΨG)^{-1}

    where G = ∂g/∂κ (Jacobian, approximated numerically).

    Under correct specification Var(√N·g) = Psi^{-1} and the formula
    simplifies to  AVar = (G'ΨG)^{-1}.

    Parameters
    ----------
    h : finite-difference step for numerical Jacobian

    Returns
    -------
    avar : (n_kappa, n_kappa) asymptotic variance of sqrt(N)·κ̂
    """
    Y, U_tilde, Q_mat, W_, WtW, trW, trWtW = _build_blocks(Y, X, W, bar_eps_hat)
    N = len(Y)
    n_kappa = len(kappa_hat)

    def g_of_kappa(kappa):
        return moment_vector(kappa, sigma2_hat, Y, U_tilde, Q_mat,
                             W_, WtW, trW, trWtW)

    # Numerical Jacobian ∂g/∂κ  (m × n_kappa)
    g0   = g_of_kappa(kappa_hat)
    m    = len(g0)
    G    = np.zeros((m, n_kappa))
    for j in range(n_kappa):
        e = np.zeros(n_kappa); e[j] = h
        G[:, j] = (g_of_kappa(kappa_hat + e) - g_of_kappa(kappa_hat - e)) / (2 * h)

    # Meat: empirical Var(√N·g) — diagonal from Psi_inv^{-1}
    Psi = linalg.pinv(Psi_inv)

    GtPsi  = G.T @ Psi_inv      # (n_kappa, m)
    GtPsiG = GtPsi @ G          # (n_kappa, n_kappa)
    try:
        GtPsiG_inv = linalg.pinv(GtPsiG)
    except Exception:
        GtPsiG_inv = np.full((n_kappa, n_kappa), np.nan)

    # Robust meat
    meat = GtPsi @ Psi @ Psi_inv @ G   # (n_kappa, n_kappa)
    avar = GtPsiG_inv @ meat @ GtPsiG_inv * N

    return avar


def cf_gmm_fit(Y: np.ndarray,
               X: np.ndarray,
               W: np.ndarray,
               bar_eps_hat: np.ndarray,
               sigma2_init: float = 1.0,
               max_iter: int = 2,
               param_names: Optional[list] = None) -> dict:
    """
    Full CF-GMM estimation with SE, t-stats, p-values.

    Returns
    -------
    dict with kappa, se, t_stats, p_values, sigma2, avar, param_names
    """
    from scipy import stats as scipy_stats

    Y   = np.asarray(Y, float).ravel()
    X   = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    N, k = X.shape

    kappa_hat, sigma2_hat, Psi_inv = cf_gmm(Y, X, W, bar_eps_hat,
                                             sigma2_init=sigma2_init,
                                             max_iter=max_iter)
    avar    = cf_gmm_avar(Y, X, W, bar_eps_hat, kappa_hat, sigma2_hat, Psi_inv)
    se      = np.sqrt(np.maximum(np.diag(avar) / N, 0.0))
    t_stats = kappa_hat / np.where(se > 1e-14, se, np.nan)
    p_vals  = 2 * scipy_stats.t.sf(np.abs(t_stats), df=N - len(kappa_hat))

    if param_names is None:
        x_names = [f"x{i+1}" for i in range(k)]
        param_names = (["delta"] + x_names
                       + [f"theta_{n}" for n in x_names]
                       + ["delta_c"])

    return {
        "kappa"      : kappa_hat,
        "se"         : se,
        "t_stats"    : t_stats,
        "p_values"   : p_vals,
        "sigma2"     : sigma2_hat,
        "avar"       : avar,
        "Psi_inv"    : Psi_inv,
        "param_names": param_names,
        "N"          : N,
    }
