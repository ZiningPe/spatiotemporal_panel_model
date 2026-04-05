"""
weight_construction.py
----------------------
Build the spatio-temporal weight matrix W = M ⊗ W_S and the temporal weight
matrix (TWM) M from data-driven statistics (Moran's I, Geary's C, Getis-Ord G,
Spatial Gini, or parametric decay).

Stacking convention (time-major)
---------------------------------
Panel vector Y of length N = n·T is ordered as

    Y = [y_{1,1}, …, y_{n,1},   ← period 1
         y_{1,2}, …, y_{n,2},   ← period 2
         …
         y_{1,T}, …, y_{n,T}]   ← period T

so that np.kron(M, W_S) gives the correct N×N weight matrix.

References
----------
Moran, P. A. P. (1950). Biometrika, 37(1/2), 17–23.
Geary, R. C. (1954). The Incorporated Statistician, 5(3), 115–145.
Getis, A. & Ord, J. K. (1992). Geographical Analysis, 24(3), 189–206.
"""

from __future__ import annotations
import numpy as np
from scipy import linalg
from typing import Literal, Sequence


# ---------------------------------------------------------------------------
# 1.  Spatial weight matrices
# ---------------------------------------------------------------------------

def rook_weights(n_side: int) -> np.ndarray:
    """
    Row-normalised rook-contiguity weight matrix for an n_side×n_side grid.

    Parameters
    ----------
    n_side : side length of the square grid  (n = n_side²)

    Returns
    -------
    W_S : (n, n) row-normalised weight matrix
    """
    n = n_side ** 2
    W = np.zeros((n, n))
    for i in range(n):
        r, c = divmod(i, n_side)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n_side and 0 <= nc < n_side:
                W[i, nr * n_side + nc] = 1.0
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return W / row_sums


def queen_weights(n_side: int) -> np.ndarray:
    """
    Row-normalised queen-contiguity weight matrix for an n_side×n_side grid.
    """
    n = n_side ** 2
    W = np.zeros((n, n))
    for i in range(n):
        r, c = divmod(i, n_side)
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < n_side and 0 <= nc < n_side:
                    W[i, nr * n_side + nc] = 1.0
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return W / row_sums


def inverse_distance_weights(coords: np.ndarray,
                              power: float = 1.0,
                              row_standardize: bool = True) -> np.ndarray:
    """
    Inverse-distance weight matrix.

    Parameters
    ----------
    coords         : (n, 2) array of (lon, lat) or (x, y) coordinates
    power          : distance decay exponent
    row_standardize: whether to row-normalise

    Returns
    -------
    W : (n, n) weight matrix (zero diagonal)
    """
    n = len(coords)
    diff = coords[:, None, :] - coords[None, :, :]          # (n, n, 2)
    dist = np.sqrt((diff ** 2).sum(axis=-1))                  # (n, n)
    np.fill_diagonal(dist, np.inf)
    W = 1.0 / dist ** power
    np.fill_diagonal(W, 0.0)
    if row_standardize:
        rs = W.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        W = W / rs
    return W


def knn_weights(coords: np.ndarray, k: int,
                row_standardize: bool = True) -> np.ndarray:
    """
    k-nearest-neighbour binary weight matrix.
    """
    n = len(coords)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)
    W = np.zeros((n, n))
    for i in range(n):
        idx = np.argsort(dist[i])[:k]
        W[i, idx] = 1.0
    if row_standardize:
        rs = W.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        W = W / rs
    return W


# ---------------------------------------------------------------------------
# 2.  Temporal statistics used to build M (TWM)
# ---------------------------------------------------------------------------

def compute_morans_i(y: np.ndarray, W_S: np.ndarray) -> float:
    """
    Global Moran's I statistic for a single cross-section.

    I = (n / S_0) · (y'Wy) / (y'y),   S_0 = sum of all weights.

    Parameters
    ----------
    y   : (n,) cross-sectional observations (mean-demeaned internally)
    W_S : (n, n) spatial weight matrix

    Returns
    -------
    Moran's I statistic
    """
    y = np.asarray(y, float)
    z = y - y.mean()
    n = len(z)
    S0 = W_S.sum()
    if S0 == 0 or z @ z == 0:
        return 0.0
    return float(n / S0 * (z @ W_S @ z) / (z @ z))


def compute_gearys_c(y: np.ndarray, W_S: np.ndarray) -> float:
    """Geary's C statistic for a single cross-section."""
    y = np.asarray(y, float)
    n = len(y)
    S0 = W_S.sum()
    var_y = np.var(y)
    if S0 == 0 or var_y == 0:
        return 1.0
    num = 0.0
    for i in range(n):
        for j in range(n):
            num += W_S[i, j] * (y[i] - y[j]) ** 2
    return float((n - 1) * num / (2 * S0 * n * var_y))


def compute_getis_ord_g(y: np.ndarray, W_S: np.ndarray) -> float:
    """Global Getis-Ord G statistic (assumes non-negative y)."""
    y = np.asarray(y, float)
    n = len(y)
    num = 0.0
    den = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                num += W_S[i, j] * y[i] * y[j]
                den += y[i] * y[j]
    return float(num / den) if den != 0 else 0.0


def compute_spatial_gini(y: np.ndarray, W_S: np.ndarray) -> float:
    """
    Spatial Gini coefficient measuring regional inequality with spatial structure.
    """
    y = np.asarray(y, float)
    n = len(y)
    mean_y = y.mean()
    if mean_y == 0:
        return 0.0
    num = sum(W_S[i, j] * abs(y[i] - y[j])
              for i in range(n) for j in range(n))
    return float(num / (2 * n ** 2 * mean_y))


# ---------------------------------------------------------------------------
# 3.  Build temporal weight matrix M from statistics
# ---------------------------------------------------------------------------

_DECAY_REGISTRY: dict[str, callable] = {}


def _norm_row(M: np.ndarray) -> np.ndarray:
    rs = M.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return M / rs


def build_twm_from_stats(stats: Sequence[float],
                          method: Literal["moran", "geary", "getis", "gini",
                                          "raw"] = "moran",
                          row_standardize: bool = True) -> np.ndarray:
    """
    Build a T×T temporal weight matrix M from a sequence of T spatial
    statistics (one per period).

    The (t, s) entry of M is proportional to the *similarity* of period t
    and period s as measured by their statistics:

        M_raw[t, s] = exp(-|stat_t - stat_s|)   (Gaussian-like kernel)

    Zero diagonal is enforced before row-normalisation.

    Parameters
    ----------
    stats          : length-T sequence of per-period statistics
    method         : label used for display only (all methods use the same
                     kernel — pass ``method='raw'`` if stats are pre-computed)
    row_standardize: row-normalise M (recommended)

    Returns
    -------
    M : (T, T) temporal weight matrix
    """
    s = np.asarray(stats, float)
    T = len(s)
    diff = np.abs(s[:, None] - s[None, :])
    M = np.exp(-diff)
    np.fill_diagonal(M, 0.0)
    if row_standardize:
        M = _norm_row(M)
    return M


def build_twm_parametric(T: int,
                          rho: float = 0.6,
                          form: Literal["exponential", "power",
                                        "linear"] = "exponential",
                          row_standardize: bool = True) -> np.ndarray:
    """
    Parametric temporal weight matrix with M_{ts} ∝ decay(|t - s|).

    Parameters
    ----------
    T    : number of periods
    rho  : decay parameter
    form : 'exponential' → rho^|t-s|,
           'power'       → |t-s|^{-rho},
           'linear'      → max(0, 1 - rho*|t-s|)
    """
    M = np.zeros((T, T))
    for t in range(T):
        for s in range(T):
            if t == s:
                continue
            d = abs(t - s)
            if form == "exponential":
                M[t, s] = rho ** d
            elif form == "power":
                M[t, s] = d ** (-rho)
            elif form == "linear":
                M[t, s] = max(0.0, 1.0 - rho * d)
    if row_standardize:
        M = _norm_row(M)
    return M


# ---------------------------------------------------------------------------
# 4.  Assemble W = M ⊗ W_S (or a custom non-Kronecker matrix)
# ---------------------------------------------------------------------------

def build_stwm(M: np.ndarray,
               W_S: np.ndarray,
               row_standardize: bool = False) -> np.ndarray:
    """
    Build the full N×N spatio-temporal weight matrix W = M ⊗ W_S.

    If M and W_S are already row-normalised, the Kronecker product is also
    row-normalised and ``row_standardize`` can stay False.

    Parameters
    ----------
    M              : (T, T) temporal weight matrix (row-normalised recommended)
    W_S            : (n, n) spatial weight matrix  (row-normalised recommended)
    row_standardize: apply an additional row-normalisation pass after kron

    Returns
    -------
    W : (n*T, n*T) = (N, N) spatio-temporal weight matrix
    """
    W = np.kron(M, W_S)
    if row_standardize:
        rs = W.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        W = W / rs
    return W


def stwm_summary(W: np.ndarray, n: int, T: int) -> dict:
    """
    Key properties of an assembled STWM.

    Returns
    -------
    dict with shape, sparsity, spectral_radius, row_sum stats,
    and eigenvalue decomposition info.
    """
    assert W.shape == (n * T, n * T), \
        f"Expected ({n * T}, {n * T}), got {W.shape}."
    nnz = np.count_nonzero(W)
    sparsity = 1.0 - nnz / W.size
    eigvals = np.linalg.eigvals(W).real
    row_sums = W.sum(axis=1)
    off = W[np.eye(n * T, dtype=bool) == False]
    off = off[off != 0]
    return {
        "shape": W.shape,
        "n_spatial": n,
        "T_temporal": T,
        "N": n * T,
        "sparsity": round(float(sparsity), 4),
        "spectral_radius": round(float(np.max(np.abs(eigvals))), 6),
        "row_sum_min": float(row_sums.min()),
        "row_sum_max": float(row_sums.max()),
        "row_sum_mean": float(row_sums.mean()),
        "weight_min": float(off.min()) if len(off) else 0.0,
        "weight_max": float(off.max()) if len(off) else 0.0,
        "weight_mean": float(off.mean()) if len(off) else 0.0,
        "eigenvalue_range": (float(eigvals.min()), float(eigvals.max())),
    }


# ---------------------------------------------------------------------------
# 5.  Eigenvalue helpers (used by CF-QMLE log-det)
# ---------------------------------------------------------------------------

def eigvals_kronecker(M: np.ndarray,
                      W_S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute eigenvalues of M and W_S separately for fast Kronecker log-det.

    The eigenvalues of M ⊗ W_S are {μ_j · λ_k} for all j, k.

    Returns
    -------
    eigvals_M  : (T,) real part of eigenvalues of M
    eigvals_WS : (n,) real part of eigenvalues of W_S
    """
    return linalg.eigvals(M).real, linalg.eigvals(W_S).real


def logdet_kronecker(delta: float,
                     eigvals_M: np.ndarray,
                     eigvals_WS: np.ndarray) -> float:
    """
    Fast log|I_N - delta·W| exploiting Kronecker structure:

        ln|I - δ·(M⊗W_S)| = Σ_j Σ_k ln|1 - δ·μ_j·λ_k|

    Parameters
    ----------
    delta     : spatial-lag coefficient
    eigvals_M : (T,) eigenvalues of M
    eigvals_WS: (n,) eigenvalues of W_S

    Returns
    -------
    log-determinant (float, -inf if the matrix is singular for this delta)
    """
    total = 0.0
    for mu in eigvals_M:
        for lam in eigvals_WS:
            val = 1.0 - delta * mu * lam
            if val <= 0:
                return -np.inf
            total += np.log(abs(val))
    return total


def delta_admissible_range(eigvals_M: np.ndarray,
                            eigvals_WS: np.ndarray) -> tuple[float, float]:
    """
    Compute the admissible interval for delta so that I - delta·W is
    non-singular, i.e. all 1 - delta·μ_j·λ_k > 0.

    Returns (delta_min, delta_max) such that delta ∈ (delta_min, delta_max)
    guarantees |I - delta·W| ≠ 0.
    """
    products = np.outer(eigvals_M, eigvals_WS).ravel()
    pos = products[products > 0]
    neg = products[products < 0]
    lo = float(-1.0 / pos.max()) if len(pos) else -np.inf
    hi = float(-1.0 / neg.min()) if len(neg) else np.inf
    # clamp to (-1, 1) for practical stability
    return max(lo, -0.9999), min(hi, 0.9999)
