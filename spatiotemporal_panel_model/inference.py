"""
inference.py
------------
Post-estimation inference for the CF-2SLS / CF-QMLE / CF-GMM estimators:

  1. Cross-period (indirect) effects  IE_{r, t←s}
  2. Delta-method standard errors for nonlinear functions of κ̂
  3. Benjamini-Hochberg FDR control for multiple testing

Cross-period effects
--------------------
After estimating κ̂ = (δ̂, β̂, θ̂, δ̂_c), the total multiplier matrix is

    T_mat = (I − δ̂·W)^{-1}  ∈ ℝ^{N×N}

The indirect effect from spatial unit s in period τ on unit r in period t is

    IE_{(r,t) ← (s,τ)}(X) = T_mat[(r,t), (s,τ)] · (β̂ + θ̂·w_{(s,τ)})

For the aggregate (average) cross-period effect averaged over spatial units:

    IE_{t←s} = (1/n) Σ_r Σ_i T_mat[(r,t),(i,s)] · (β̂ + θ̂·w_{(i,s)})

Delta method standard errors propagate uncertainty in (δ̂, β̂, θ̂) through
the nonlinear function  IE(κ) via numerical gradients.

References
----------
Paper Section 8 & Appendix E.
LeSage, J. & Pace, R. K. (2009). Introduction to Spatial Econometrics. CRC Press.
Benjamini, Y. & Hochberg, Y. (1995). J. Royal Stat. Soc. B, 57(1), 289–300.
"""

from __future__ import annotations
import numpy as np
from scipy import linalg, stats as scipy_stats
from typing import Optional, Sequence


# ---------------------------------------------------------------------------
# 1.  Multiplier matrix
# ---------------------------------------------------------------------------

def multiplier_matrix(delta: float, W: np.ndarray) -> np.ndarray:
    """
    Compute total spatial multiplier  T = (I − δ·W)^{-1}.

    Parameters
    ----------
    delta : estimated spatial-lag parameter
    W     : (N, N) weight matrix

    Returns
    -------
    T_mat : (N, N)  — row (r,t), column (s,τ)
    """
    N = W.shape[0]
    return linalg.inv(np.eye(N) - delta * W)


def multiplier_matrix_approx(delta: float, W: np.ndarray,
                              order: int = 10) -> np.ndarray:
    """
    Neumann series approximation  T ≈ Σ_{k=0}^{order} (δW)^k.

    Faster than direct inversion for large N; accuracy depends on |δ·ρ(W)| < 1.
    """
    N = W.shape[0]
    T = np.eye(N)
    dW_k = np.eye(N)
    for _ in range(order):
        dW_k = delta * W @ dW_k
        T    = T + dW_k
    return T


# ---------------------------------------------------------------------------
# 2.  Cross-period indirect effects
# ---------------------------------------------------------------------------

def _unit_idx(unit: int, period: int, n: int) -> int:
    """Index of (unit, period) in the time-major stacked vector."""
    return period * n + unit


def cross_period_effect(T_mat: np.ndarray,
                        beta: float,
                        theta: float,
                        W: np.ndarray,
                        n: int,
                        t: int,
                        s: int,
                        r: Optional[int] = None) -> float:
    """
    Cross-period indirect effect  IE_{r,t←s}  (or average over r).

    IE_{(r,t)←s} = (1/n) Σ_i T_mat[(r,t),(i,s)] · (β + θ·W[(i,s),(i,s)]_col_sum)

    For simplicity we use the scalar version:
        IE_{r,t←s} = Σ_i T_mat[(r,t),(i,s)] · (β + θ·col_sum_W_at_(i,s))

    Parameters
    ----------
    T_mat : (N, N) multiplier matrix
    beta, theta : structural coefficients
    W     : (N, N) weight matrix
    n     : number of spatial units
    t, s  : destination and source time periods
    r     : destination unit (None → average over all r)

    Returns
    -------
    Indirect effect (scalar)
    """
    # Source column indices for period s
    src_cols = [_unit_idx(i, s, n) for i in range(n)]

    # Weight row sums for source units in period s  (used for WX term)
    # col sum of W at row (i,s) gives the spatial lag weight of X_{i,s}
    w_row_sums_s = W[src_cols, :].sum(axis=1)  # (n,) row sums of W for source period

    if r is None:
        # Average over destination units in period t
        dst_rows = [_unit_idx(ri, t, n) for ri in range(n)]
        effect = 0.0
        for col_i, i in enumerate(range(n)):
            src = src_cols[col_i]
            w_i = w_row_sums_s[col_i]
            for row in dst_rows:
                effect += T_mat[row, src] * (beta + theta * w_i)
        return effect / n
    else:
        dst = _unit_idx(r, t, n)
        effect = sum(
            T_mat[dst, src_cols[i]] * (beta + theta * w_row_sums_s[i])
            for i in range(n)
        )
        return float(effect)


def cross_period_effects_matrix(T_mat: np.ndarray,
                                beta: float,
                                theta: float,
                                W: np.ndarray,
                                n: int,
                                TT: int) -> np.ndarray:
    """
    Compute all T×T average cross-period effects  IE_{t←s}.

    Returns
    -------
    IE : (TT, TT) matrix,  IE[t, s] = average indirect effect from s to t
    """
    IE = np.zeros((TT, TT))
    for t in range(TT):
        for s in range(TT):
            IE[t, s] = cross_period_effect(T_mat, beta, theta, W, n, t, s)
    return IE


# ---------------------------------------------------------------------------
# 3.  Delta method standard errors
# ---------------------------------------------------------------------------

def delta_method_se(fn,
                    kappa: np.ndarray,
                    avar_kappa: np.ndarray,
                    N: int,
                    h: float = 1e-5) -> tuple[float, float]:
    """
    Delta method SE for a scalar function fn(κ).

    Var(fn(κ̂)) ≈ (∂fn/∂κ)' · (AVar_κ / N) · (∂fn/∂κ)

    Parameters
    ----------
    fn         : callable(kappa) → scalar
    kappa      : (p,) parameter estimates
    avar_kappa : (p, p) asymptotic variance of sqrt(N)·(κ̂ − κ₀)
    N          : sample size
    h          : finite-difference step

    Returns
    -------
    (estimate, standard_error)
    """
    f0   = float(fn(kappa))
    p    = len(kappa)
    grad = np.zeros(p)
    for j in range(p):
        e = np.zeros(p); e[j] = h
        grad[j] = (fn(kappa + e) - fn(kappa - e)) / (2 * h)

    var_f = float(grad @ (avar_kappa / N) @ grad)
    se    = np.sqrt(max(var_f, 0.0))
    return f0, se


def ie_inference(T_mat: np.ndarray,
                 kappa: np.ndarray,
                 avar_kappa: np.ndarray,
                 W: np.ndarray,
                 n: int,
                 TT: int,
                 N: int,
                 delta_idx: int = 0,
                 beta_slice: slice = None,
                 theta_slice: slice = None) -> dict:
    """
    Compute IE_{t←s} matrix together with Delta-method standard errors,
    t-statistics, and p-values.

    Assumes κ = [δ, β (k,), θ (k,), δ_c] with k=1 (single regressor case).
    For k>1 set beta_slice and theta_slice explicitly.

    Parameters
    ----------
    delta_idx   : index of δ in kappa
    beta_slice  : slice for β coefficients in kappa
    theta_slice : slice for θ coefficients in kappa

    Returns
    -------
    dict with IE (TT×TT), SE (TT×TT), t_stat (TT×TT), p_value (TT×TT)
    """
    p = len(kappa)
    k = (p - 2) // 2  # infer k from total params
    if beta_slice is None:
        beta_slice  = slice(1, 1 + k)
    if theta_slice is None:
        theta_slice = slice(1 + k, 1 + 2 * k)

    IE_mat = np.zeros((TT, TT))
    SE_mat = np.zeros((TT, TT))

    for t in range(TT):
        for s in range(TT):
            def ie_fn(kap):
                d_hat    = kap[delta_idx]
                beta_hat = kap[beta_slice].mean()  # scalar summary for k>1
                theta_hat= kap[theta_slice].mean()
                try:
                    Tm = multiplier_matrix(d_hat, W)
                except Exception:
                    return np.nan
                return cross_period_effect(Tm, beta_hat, theta_hat, W, n, t, s)

            ie_val, ie_se = delta_method_se(ie_fn, kappa, avar_kappa, N)
            IE_mat[t, s] = ie_val
            SE_mat[t, s] = ie_se

    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = np.where(SE_mat > 0, IE_mat / SE_mat, np.nan)
    p_val  = 2 * scipy_stats.norm.sf(np.abs(t_stat))

    return {
        "IE"     : IE_mat,
        "SE"     : SE_mat,
        "t_stat" : t_stat,
        "p_value": p_val,
    }


# ---------------------------------------------------------------------------
# 4.  Benjamini-Hochberg FDR control
# ---------------------------------------------------------------------------

def bh_correction(p_values: np.ndarray,
                  alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """
    Benjamini-Hochberg (1995) False Discovery Rate correction.

    Parameters
    ----------
    p_values : array of p-values (any shape; flattened internally)
    alpha    : target FDR level

    Returns
    -------
    reject      : boolean array (same shape as p_values), True = reject H₀
    adjusted_p  : BH-adjusted p-values (same shape)
    """
    shape = p_values.shape
    pv    = p_values.ravel()
    m     = len(pv)

    order   = np.argsort(pv)
    pv_sort = pv[order]
    thresholds = alpha * np.arange(1, m + 1) / m

    reject_sort = pv_sort <= thresholds
    # All hypotheses up to the last rejected one are also rejected
    if reject_sort.any():
        last_reject = np.where(reject_sort)[0].max()
        reject_sort[:last_reject + 1] = True

    # BH-adjusted p-values  (Yekutieli & Benjamini 1999 formula)
    adj = np.minimum.accumulate((m / np.arange(m, 0, -1)) * pv_sort[::-1])[::-1]
    adj = np.minimum(adj, 1.0)

    reject_out = np.empty(m, dtype=bool)
    adj_out    = np.empty(m)
    reject_out[order] = reject_sort
    adj_out[order]    = adj

    return reject_out.reshape(shape), adj_out.reshape(shape)


def multiple_testing_summary(IE: np.ndarray,
                             SE: np.ndarray,
                             p_values: np.ndarray,
                             alpha: float = 0.05,
                             fdr_alpha: float = 0.05) -> dict:
    """
    Summary table of IE_{t←s} with unadjusted and BH-adjusted decisions.

    Returns
    -------
    dict with arrays: IE, SE, p_raw, p_bh, reject_raw, reject_bh
    """
    reject_raw = p_values < alpha
    reject_bh, p_bh = bh_correction(p_values, alpha=fdr_alpha)

    return {
        "IE"        : IE,
        "SE"        : SE,
        "p_raw"     : p_values,
        "p_bh"      : p_bh,
        "reject_raw": reject_raw,
        "reject_bh" : reject_bh,
        "n_reject_raw": int(reject_raw.sum()),
        "n_reject_bh" : int(reject_bh.sum()),
    }


def print_ie_table(result: dict,
                   period_labels: Optional[Sequence] = None) -> None:
    """Print cross-period IE results with FDR-adjusted significance markers."""
    IE  = result["IE"]
    SE  = result["SE"]
    p   = result.get("p_bh", result.get("p_raw"))
    rej = result.get("reject_bh", result.get("reject_raw"))
    TT  = IE.shape[0]
    labs = period_labels if period_labels else list(range(TT))

    print("\n" + "=" * 70)
    print("  Cross-period Indirect Effects  IE_{t←s}")
    print("  (SE in parentheses, * FDR-corrected significant at 5%)")
    print("=" * 70)
    header = "  t\\s  " + "".join(f"{str(labs[s]):>12}" for s in range(TT))
    print(header)
    print("-" * len(header))
    for t in range(TT):
        row = f"  {str(labs[t]):<5}"
        for s in range(TT):
            star = "*" if rej[t, s] else " "
            row += f"  {IE[t,s]:>6.3f}({SE[t,s]:.3f}){star}"
        print(row)
    print("=" * 70)
