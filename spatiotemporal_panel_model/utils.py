"""
utils.py
--------
Miscellaneous utility functions used across the spatiotemporal_panel_model package.

Contents
--------
1. NED (Near Epoch Dependence) diagnostic
2. Eigenvalue decomposition helpers
3. Panel data reshaping (time-major ↔ unit-major)
4. Condition number and stability checks
5. Pretty-print helpers
"""

from __future__ import annotations
import numpy as np
from scipy import linalg
from typing import Optional


# ---------------------------------------------------------------------------
# 1.  NED diagnostic
# ---------------------------------------------------------------------------

def ned_test(xi_hat: np.ndarray,
             n: int,
             T: int,
             W_S: np.ndarray,
             max_lag: int = 3) -> dict:
    """
    Informal NED (Near Epoch Dependence) diagnostic for structural residuals.

    Tests whether the spatial and temporal auto-correlation of ξ̂ decays
    geometrically, consistent with the NED assumption maintained in the paper.

    Approach
    --------
    1. Spatial:  compute Moran's I of ξ̂_t for each period t and check decay.
    2. Temporal: compute cross-period correlation  corr(ξ̂_t, ξ̂_{t-k})  for
       lags k = 1…max_lag and check geometric decay.

    Parameters
    ----------
    xi_hat  : (N,) structural residuals
    n, T    : spatial units and time periods
    W_S     : (n, n) spatial weight matrix
    max_lag : maximum temporal lag to check

    Returns
    -------
    dict with morans_by_period, temporal_autocorr, ned_spatial_ok,
    ned_temporal_ok, summary
    """
    from .weight_construction import compute_morans_i

    xi = xi_hat.reshape(T, n)  # (T, n) — row t = period t

    # Spatial: Moran's I per period
    morans = np.array([compute_morans_i(xi[t], W_S) for t in range(T)])

    # Temporal: autocorrelations
    temporal_ac = {}
    xi_flat = xi.T.ravel()  # unit-major for autocorr
    for lag in range(1, min(max_lag + 1, T)):
        # Correlate period-level mean residuals at lag
        mean_by_period = xi.mean(axis=1)   # (T,)
        if T - lag > 1:
            ac = float(np.corrcoef(mean_by_period[:-lag],
                                   mean_by_period[lag:])[0, 1])
        else:
            ac = np.nan
        temporal_ac[lag] = round(ac, 4)

    # NED checks (heuristic)
    ned_spatial_ok  = bool(np.max(np.abs(morans)) < 0.5)
    ned_temporal_ok = all(
        abs(v) < 0.8 for v in temporal_ac.values() if not np.isnan(v)
    )

    return {
        "morans_by_period": {t: round(float(m), 4) for t, m in enumerate(morans)},
        "temporal_autocorr": temporal_ac,
        "ned_spatial_ok"  : ned_spatial_ok,
        "ned_temporal_ok" : ned_temporal_ok,
        "summary": (
            "NED assumption plausible (weak spatial and temporal dependence)."
            if ned_spatial_ok and ned_temporal_ok
            else "Warning: strong dependence detected — NED assumption may be violated."
        ),
    }


# ---------------------------------------------------------------------------
# 2.  Eigenvalue helpers
# ---------------------------------------------------------------------------

def eigs_sorted(A: np.ndarray) -> np.ndarray:
    """Real eigenvalues of A, sorted descending by absolute value."""
    ev = linalg.eigvals(A).real
    return ev[np.argsort(-np.abs(ev))]


def spectral_radius(A: np.ndarray) -> float:
    """Largest absolute eigenvalue of A."""
    return float(np.max(np.abs(linalg.eigvals(A))))


def check_admissibility(delta: float,
                        eigvals_M: np.ndarray,
                        eigvals_WS: np.ndarray) -> bool:
    """
    Check whether I - delta·W is non-singular (i.e. all 1 - δ·μ·λ > 0).
    """
    products = np.outer(eigvals_M, eigvals_WS).ravel()
    return bool(np.all(1.0 - delta * products > 0))


# ---------------------------------------------------------------------------
# 3.  Panel reshaping
# ---------------------------------------------------------------------------

def to_time_major(Y_unit: np.ndarray, n: int, T: int) -> np.ndarray:
    """
    Convert unit-major stacking to time-major stacking.

    Unit-major:  [y_{1,1}, y_{1,2}, …, y_{1,T},  y_{2,1}, …, y_{n,T}]
    Time-major:  [y_{1,1}, y_{2,1}, …, y_{n,1},  y_{1,2}, …, y_{n,T}]

    Parameters
    ----------
    Y_unit : (N,) unit-major panel vector
    n, T   : spatial units, time periods

    Returns
    -------
    Y_time : (N,) time-major panel vector
    """
    return Y_unit.reshape(n, T).T.ravel()


def to_unit_major(Y_time: np.ndarray, n: int, T: int) -> np.ndarray:
    """Convert time-major stacking to unit-major stacking."""
    return Y_time.reshape(T, n).T.ravel()


def panel_demean(Y: np.ndarray, n: int, T: int,
                 within: str = "unit") -> np.ndarray:
    """
    Within-group demeaning for panel data (time-major stacking).

    Parameters
    ----------
    within : 'unit' (remove unit fixed effects)
             'time' (remove time fixed effects)
             'both' (two-way demeaning)
    """
    Y_mat = Y.reshape(T, n)          # (T, n)
    if within == "unit":
        Y_out = Y_mat - Y_mat.mean(axis=0, keepdims=True)
    elif within == "time":
        Y_out = Y_mat - Y_mat.mean(axis=1, keepdims=True)
    elif within == "both":
        grand_mean = Y_mat.mean()
        Y_out = (Y_mat
                 - Y_mat.mean(axis=0, keepdims=True)
                 - Y_mat.mean(axis=1, keepdims=True)
                 + grand_mean)
    else:
        raise ValueError(f"Unknown within='{within}'.")
    return Y_out.ravel()


# ---------------------------------------------------------------------------
# 4.  Condition number / stability checks
# ---------------------------------------------------------------------------

def matrix_condition(A: np.ndarray) -> float:
    """Condition number of matrix A (2-norm)."""
    sv = linalg.svdvals(A)
    return float(sv.max() / sv.min()) if sv.min() > 0 else np.inf


def check_instrument_strength(Q: np.ndarray,
                               U_tilde: np.ndarray,
                               threshold: float = 10.0) -> dict:
    """
    First-stage relevance check for the IV estimator.

    Computes the concentration parameter  μ² = (P_Q U_tilde)'(P_Q U_tilde) / N
    as a proxy for instrument strength.  A small value signals weak instruments.

    Returns dict with condition_number and a warning if instruments are weak.
    """
    N = Q.shape[0]
    QtQ_inv = linalg.pinv(Q.T @ Q)
    P_Q_U   = Q @ (QtQ_inv @ (Q.T @ U_tilde))
    conc    = float(np.trace(P_Q_U.T @ P_Q_U)) / N

    return {
        "concentration_parameter": round(conc, 4),
        "weak_instruments"       : conc < threshold,
        "note": ("Instrument set appears strong." if conc >= threshold
                 else f"Warning: concentration parameter {conc:.2f} < {threshold} "
                      f"— potential weak instrument problem."),
    }


# ---------------------------------------------------------------------------
# 5.  Pretty-print helpers
# ---------------------------------------------------------------------------

def print_dict(d: dict, title: str = "", indent: int = 2) -> None:
    """Recursively print a results dictionary."""
    pad = " " * indent
    if title:
        print(f"\n{pad}{title}")
        print(pad + "─" * len(title))
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"{pad}  {k}:")
            print_dict(v, indent=indent + 4)
        elif isinstance(v, np.ndarray):
            print(f"{pad}  {k}: array{v.shape}")
        else:
            print(f"{pad}  {k}: {v}")


def summarise_results(cf2sls: Optional[dict] = None,
                      cfqmle: Optional[dict] = None,
                      cfgmm: Optional[dict] = None) -> None:
    """Side-by-side comparison of CF-2SLS / CF-QMLE / CF-GMM estimates."""
    print("\n" + "=" * 75)
    print("  Estimation Results Comparison")
    print("=" * 75)
    print(f"  {'Parameter':<12} {'CF-2SLS':>14} {'CF-QMLE':>14} {'CF-GMM':>14}")
    print("-" * 75)

    if cf2sls is not None:
        names  = cf2sls["param_names"]
        k2sls  = cf2sls["kappa"]
        s2sls  = cf2sls["se"]
    else:
        names  = []
        k2sls  = s2sls  = []

    # CF-QMLE packs (delta, alpha) differently
    if cfqmle is not None:
        d_q   = cfqmle["delta_hat"]
        a_q   = cfqmle["alpha_hat"]
        kqmle = np.concatenate([[d_q], a_q])
    else:
        kqmle = None

    kgmm = cfgmm["kappa"] if cfgmm is not None else None

    for i, name in enumerate(names):
        v2 = f"{k2sls[i]:>8.4f}({s2sls[i]:.4f})" if cf2sls else "    —"
        vq = f"{kqmle[i]:>8.4f}" if (kqmle is not None and i < len(kqmle)) else "    —"
        vg = f"{kgmm[i]:>8.4f}({cfgmm['se'][i]:.4f})" if (kgmm is not None and i < len(kgmm)) else "    —"
        print(f"  {name:<12} {v2:>14} {vq:>14} {vg:>14}")
    print("=" * 75)
    print("  Coef(SE) shown where SE available.\n")
