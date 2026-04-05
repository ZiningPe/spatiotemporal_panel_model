"""
first_stage.py
--------------
OLS first-stage estimation of the entry equation

    h = Z_W π + ε

where h is the vector of period-level outcomes that drive endogenous W
(e.g., h_t = Moran's I in period t), Z_W is a matrix of instruments
excluded from the structural equation, and ε is the entry-equation
disturbance.

The residuals ε̂ are subsequently passed to control_function.py to form
the control function ε̄̂ = A ε̂.
"""

from __future__ import annotations
import numpy as np
from scipy import linalg


def first_stage(h: np.ndarray,
                Z_W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    OLS first stage:  h = Z_W π + ε.

    Parameters
    ----------
    h   : (L,) vector of entry-equation outcomes (one per time period if L=T,
          or one per observation if L=N)
    Z_W : (L, k_Z) matrix of excluded instruments

    Returns
    -------
    eps_hat : (L,) OLS residuals  ε̂ = M_{Z_W} h  (where M_{Z_W} = I − P_{Z_W})
    pi_hat  : (k_Z,) OLS coefficient estimates
    """
    h   = np.asarray(h, float).ravel()
    Z_W = np.asarray(Z_W, float)
    if Z_W.ndim == 1:
        Z_W = Z_W[:, None]

    pi_hat  = linalg.lstsq(Z_W, h, cond=None)[0]
    eps_hat = h - Z_W @ pi_hat
    return eps_hat, pi_hat


def first_stage_stats(h: np.ndarray,
                      Z_W: np.ndarray) -> dict:
    """
    Extended first-stage output with F-statistic and R².

    Returns
    -------
    dict with keys:
        eps_hat, pi_hat, fitted, sigma2,
        R2, F_stat, F_pval, n_obs, n_params
    """
    from scipy import stats as scipy_stats

    h   = np.asarray(h, float).ravel()
    Z_W = np.asarray(Z_W, float)
    if Z_W.ndim == 1:
        Z_W = Z_W[:, None]

    L, k = Z_W.shape
    eps_hat, pi_hat = first_stage(h, Z_W)
    fitted  = h - eps_hat
    sigma2  = float(eps_hat @ eps_hat) / max(L - k, 1)

    # R²
    h_mean = h.mean()
    SS_tot = float((h - h_mean) @ (h - h_mean))
    SS_res = float(eps_hat @ eps_hat)
    R2 = 1.0 - SS_res / SS_tot if SS_tot > 1e-15 else 0.0

    # F-statistic  (H₀: all π = 0)
    SS_reg = SS_tot - SS_res
    df_reg = k
    df_res = L - k
    if df_res > 0 and df_reg > 0:
        F_stat = (SS_reg / df_reg) / (SS_res / df_res)
        F_pval = float(scipy_stats.f.sf(F_stat, df_reg, df_res))
    else:
        F_stat = np.nan
        F_pval = np.nan

    return {
        "eps_hat" : eps_hat,
        "pi_hat"  : pi_hat,
        "fitted"  : fitted,
        "sigma2"  : sigma2,
        "R2"      : round(R2, 6),
        "F_stat"  : round(float(F_stat), 4) if not np.isnan(F_stat) else np.nan,
        "F_pval"  : round(F_pval, 6) if not np.isnan(F_pval) else np.nan,
        "n_obs"   : L,
        "n_params": k,
    }


def projection_matrix(Z_W: np.ndarray) -> np.ndarray:
    """
    Orthogonal projection matrix P_{Z_W} = Z_W (Z_W'Z_W)^{-1} Z_W'.

    Used to form the projected residuals P_{Z_W} ε̂ needed in some
    variance-correction formulas.

    Parameters
    ----------
    Z_W : (L, k) instrument matrix

    Returns
    -------
    P : (L, L) projection matrix
    """
    Z_W = np.asarray(Z_W, float)
    if Z_W.ndim == 1:
        Z_W = Z_W[:, None]
    ZtZ_inv = linalg.pinv(Z_W.T @ Z_W)
    return Z_W @ ZtZ_inv @ Z_W.T
