"""
run_all_tables.py
=================
Complete, self-contained Monte Carlo simulation reproducing Tables 1–7.

Usage
-----
    python run_all_tables.py                     # all tables, 500 reps
    python run_all_tables.py --reps 50           # quick smoke test
    python run_all_tables.py --tables 1,3,7      # selected tables only

Output
------
Tables are printed to stdout.
NPZ result files are saved to  ./mc_results/table{N}.npz

Tables
------
  1  Estimator bias/RMSE – exogenous (δ_c=0) vs endogenous (δ_c=0.5),
     two sample sizes (n=49, n=100), four estimators
  2  Sensitivity to δ_c ∈ {0, 0.3, 0.5, 0.8} – fixed n=49
  3  SE coverage: uncorrected vs Ω_A-corrected, L=T vs L=N entry equation
  4  Fixed-effects (within) transformation + CF estimators
  5  Cross-period indirect effects: Delta-method coverage + BH-FDR
  6  Robustness: varying ρ_M and spatial graph (Rook / Queen / KNN-4)
  7  Computation time: Kronecker log-det vs full-matrix log-det
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import argparse, os, sys, time, warnings
warnings.filterwarnings("ignore")

# ── scientific ────────────────────────────────────────────────────────────────
import numpy as np
from scipy import linalg, optimize, stats as scipy_stats

os.makedirs("mc_results", exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  (edit here or pass --reps / --tables on CLI)
# ═════════════════════════════════════════════════════════════════════════════
REPS      = 500
T         = 5
RHO_M     = 0.6
DELTA0    = 0.3
BETA0     = 1.0
THETA0    = 0.5
SIGMA_XI2 = 1.0
ALPHA     = 0.05          # nominal size / FDR level
N_SIDE_S  = 7             # small grid: n = 49
N_SIDE_L  = 10            # large grid: n = 100

# ═════════════════════════════════════════════════════════════════════════════
# §1  WEIGHT MATRICES
# ═════════════════════════════════════════════════════════════════════════════

def _row_norm(W):
    rs = W.sum(1, keepdims=True); rs[rs == 0] = 1.
    return W / rs

def rook_weights(n_side):
    n = n_side ** 2
    W = np.zeros((n, n))
    for i in range(n):
        r, c = divmod(i, n_side)
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < n_side and 0 <= nc < n_side:
                W[i, nr*n_side+nc] = 1.
    return _row_norm(W)

def queen_weights(n_side):
    n = n_side ** 2
    W = np.zeros((n, n))
    for i in range(n):
        r, c = divmod(i, n_side)
        for dr in [-1,0,1]:
            for dc in [-1,0,1]:
                if dr == dc == 0: continue
                nr, nc = r+dr, c+dc
                if 0 <= nr < n_side and 0 <= nc < n_side:
                    W[i, nr*n_side+nc] = 1.
    return _row_norm(W)

def knn_weights(n_side, k=4):
    n      = n_side ** 2
    coords = np.array([[r,c] for r in range(n_side) for c in range(n_side)], float)
    diff   = coords[:,None,:] - coords[None,:,:]
    dist   = np.sqrt((diff**2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    W = np.zeros((n, n))
    for i in range(n):
        W[i, np.argsort(dist[i])[:k]] = 1.
    return _row_norm(W)

def temporal_matrix(TT, rho_M=0.6):
    M = np.zeros((TT, TT))
    for t in range(TT):
        for s in range(TT):
            if t != s:
                M[t, s] = rho_M ** abs(t - s)
    return _row_norm(M)

def build_stwm(M, W_S):
    return np.kron(M, W_S)

# ═════════════════════════════════════════════════════════════════════════════
# §2  DATA-GENERATING PROCESSES
# ═════════════════════════════════════════════════════════════════════════════

def _solve_Y(W, X, delta0, beta0, theta0, V):
    N   = len(V)
    rhs = X.ravel()*beta0 + (W@X).ravel()*theta0 + V
    return linalg.solve(np.eye(N) - delta0*W, rhs)


def simulate_L_T(n_side, delta0, beta0, theta0, sigma_xi2, delta_c, rho_M, seed):
    """
    DGP with L = T  (period-level entry equation, one ε_t shared by n units).

    Returns
    -------
    Y, X, W, W_S, M, h, Z_W, eps, bar_eps, xi
    """
    rng = np.random.default_rng(seed)
    n   = n_side ** 2;  N = n * T
    W_S = rook_weights(n_side)
    M   = temporal_matrix(T, rho_M)
    W   = build_stwm(M, W_S)
    X   = rng.standard_normal((N, 1))
    Z_W = rng.standard_normal((T, 2))
    eps = rng.standard_normal(T)
    h   = Z_W @ np.array([1.0, 0.8]) + eps
    bar_eps = np.repeat(eps, n)                  # A = I_T ⊗ ι_n
    xi      = rng.standard_normal(N) * np.sqrt(sigma_xi2)
    V       = delta_c * bar_eps + xi
    Y       = _solve_Y(W, X, delta0, beta0, theta0, V)
    return Y, X, W, W_S, M, h, Z_W, eps, bar_eps, xi


def simulate_L_N(n_side, delta0, beta0, theta0, sigma_xi2, delta_c, rho_M, seed):
    """
    DGP with L = N  (unit-level entry equation, separate ε_p per observation).

    Returns
    -------
    Y, X, W, W_S, M, h, Z_W, eps, bar_eps, xi
    """
    rng = np.random.default_rng(seed)
    n   = n_side ** 2;  N = n * T
    W_S = rook_weights(n_side)
    M   = temporal_matrix(T, rho_M)
    W   = build_stwm(M, W_S)
    X   = rng.standard_normal((N, 1))
    Z_W = rng.standard_normal((N, 2))            # one instrument row per obs
    eps = rng.standard_normal(N)
    h   = Z_W @ np.array([1.0, 0.8]) + eps
    bar_eps = eps.copy()                          # A = I_N
    xi      = rng.standard_normal(N) * np.sqrt(sigma_xi2)
    V       = delta_c * bar_eps + xi
    Y       = _solve_Y(W, X, delta0, beta0, theta0, V)
    return Y, X, W, W_S, M, h, Z_W, eps, bar_eps, xi


def simulate_FE(n_side, delta0, beta0, theta0, sigma_xi2, delta_c, rho_M, seed):
    """
    DGP with additive unit fixed effects α_i  (L = T entry equation).

    Returns
    -------
    Y, X, W, W_S, M, h, Z_W, eps, bar_eps, xi, alpha_i
    """
    rng = np.random.default_rng(seed)
    n   = n_side ** 2;  N = n * T
    W_S = rook_weights(n_side)
    M   = temporal_matrix(T, rho_M)
    W   = build_stwm(M, W_S)
    alpha_i = rng.standard_normal(n) * 0.5      # unit FE ~ N(0, 0.25)
    fe_vec  = np.tile(alpha_i, T)                # time-major stacking
    X   = rng.standard_normal((N, 1))
    Z_W = rng.standard_normal((T, 2))
    eps = rng.standard_normal(T)
    h   = Z_W @ np.array([1.0, 0.8]) + eps
    bar_eps = np.repeat(eps, n)
    xi      = rng.standard_normal(N) * np.sqrt(sigma_xi2)
    V       = delta_c * bar_eps + xi
    rhs = fe_vec + X.ravel()*beta0 + (W@X).ravel()*theta0 + V
    Y   = linalg.solve(np.eye(N) - delta0*W, rhs)
    return Y, X, W, W_S, M, h, Z_W, eps, bar_eps, xi, alpha_i

# ═════════════════════════════════════════════════════════════════════════════
# §3  ESTIMATORS
# ═════════════════════════════════════════════════════════════════════════════

def first_stage(h, Z_W):
    """OLS first stage. Returns eps_hat (shape = h.shape)."""
    h   = np.asarray(h, float).ravel()
    Z_W = np.atleast_2d(np.asarray(Z_W, float))
    if Z_W.shape[0] != len(h):
        Z_W = Z_W.T
    pi = linalg.lstsq(Z_W, h, cond=None)[0]
    return h - Z_W @ pi


def aggregate_eps(eps_hat, n, mode="L_eq_T"):
    """
    ε̄̂ = A · ε̂.

    mode='L_eq_T': A = I_T ⊗ ι_n  →  repeat each ε̂_t for n units
    mode='L_eq_N': A = I_N          →  identity (return as-is)
    """
    eps_hat = np.asarray(eps_hat, float).ravel()
    if mode == "L_eq_T":
        return np.repeat(eps_hat, n)
    return eps_hat.copy()


def unadj_2sls(Y, X, W):
    """
    Unadjusted 2SLS (no control function).
    Instruments Q = [X, WX, W²X].
    Returns kappa = (delta, beta, theta).
    """
    N   = len(Y)
    WY  = (W @ Y)[:, None]
    WX  = W @ X
    W2X = W @ WX
    U   = np.column_stack([WY, X, WX])
    Q   = np.column_stack([X, WX, W2X])
    Qi  = linalg.pinv(Q.T @ Q)
    PU  = Q @ (Qi @ (Q.T @ U))
    try:
        k = linalg.solve(PU.T @ U, PU.T @ Y)
    except linalg.LinAlgError:
        k = np.full(3, np.nan)
    return k   # (delta, beta, theta)


def cf_2sls(Y, X, W, beh):
    """
    CF-2SLS.  Instruments Q = [X, WX, W²X, ε̄̂].
    Returns kappa = (delta, beta, theta, delta_c).
    """
    beh = np.asarray(beh, float)
    WY  = (W @ Y)[:, None]
    WX  = W @ X
    W2X = W @ WX
    U   = np.column_stack([WY, X, WX, beh])
    Q   = np.column_stack([X, WX, W2X, beh])
    Qi  = linalg.pinv(Q.T @ Q)
    PU  = Q @ (Qi @ (Q.T @ U))
    try:
        k = linalg.solve(PU.T @ U, PU.T @ Y)
    except linalg.LinAlgError:
        k = np.full(4, np.nan)
    return k   # (delta, beta, theta, delta_c)


def cf_2sls_avar(Y, X, W, beh, kappa, sigma2=None, omega_A=None):
    """
    Asymptotic variance of sqrt(N)·κ̂  (sandwich, homoskedastic).
    Optionally adds the Ω_A correction matrix for the L=T generated regressor.
    Returns (p, p) matrix.
    """
    N   = len(Y)
    beh = np.asarray(beh, float)
    WY  = (W @ Y)[:, None]
    WX  = W @ X
    W2X = W @ WX
    U   = np.column_stack([WY, X, WX, beh])
    Q   = np.column_stack([X, WX, W2X, beh])
    xi  = Y - U @ kappa
    if sigma2 is None:
        sigma2 = float(xi @ xi) / N
    Qi    = linalg.pinv(Q.T @ Q)
    PU    = Q @ (Qi @ (Q.T @ U))
    bread = linalg.pinv(PU.T @ U / N)
    meat  = sigma2 * (PU.T @ PU) / N
    avar  = bread @ meat @ bread
    if omega_A is not None:
        avar = avar + bread @ omega_A @ bread
    return avar


def cf_2sls_se(Y, X, W, beh, kappa, sigma2=None, omega_A=None):
    """SE vector = sqrt(diag(AVar / N))."""
    N    = len(Y)
    avar = cf_2sls_avar(Y, X, W, beh, kappa, sigma2=sigma2, omega_A=omega_A)
    return np.sqrt(np.maximum(np.diag(avar) / N, 0.))


def _omega_A(Q, Z_W, n, delta_c_hat, sigma2_eps):
    """
    Ω_A correction for L=T generated regressor (i.i.d. first-stage errors).

        Ω_A = δ̂_c² · σ̂²_ε · (1/N) · (A'Q)' · M_Z · (A'Q)

    Uses the block structure A = I_T ⊗ ι_n to avoid materialising A.
    """
    N, m = Q.shape
    TT   = Z_W.shape[0] if Z_W.ndim == 2 else len(Z_W)
    Z_W  = np.atleast_2d(Z_W)
    Qi   = linalg.pinv(Z_W.T @ Z_W)
    MZ   = np.eye(TT) - Z_W @ Qi @ Z_W.T   # (T, T)
    # A'Q: sum the n rows in each period block → (T, m)
    AtQ  = Q.reshape(TT, n, m).sum(axis=1)
    mid  = AtQ.T @ MZ @ AtQ                 # (m, m)
    return delta_c_hat**2 * sigma2_eps / N * mid


def logdet_kron(delta, eM, eWS):
    """ln|I - δ·(M⊗W_S)| via Kronecker eigenvalues.  Returns -inf if singular."""
    total = 0.
    for mu in eM:
        for lam in eWS:
            v = 1. - delta * mu * lam
            if v <= 0:
                return -np.inf
            total += np.log(abs(v))
    return total


def _profile_ll(delta, Y, W, R, logdet_fn):
    """Negative profile log-likelihood at delta (for minimisation)."""
    N  = len(Y)
    S  = np.eye(N) - delta * W
    SY = S @ Y
    Ri = linalg.pinv(R.T @ R)
    r  = SY - R @ (Ri @ (R.T @ SY))
    s2 = r @ r / N
    if s2 <= 1e-14:
        return np.inf
    ld = logdet_fn(delta)
    if not np.isfinite(ld):
        return np.inf
    return N/2*(1 + np.log(2*np.pi*s2)) - ld


def cf_qmle(Y, X, W, W_S, M, beh, n_grid=60):
    """
    CF-QMLE via Kronecker-structure profile likelihood.
    Returns kappa = (delta, beta, theta, delta_c, sigma2).
    """
    N   = len(Y)
    beh = np.asarray(beh, float)
    R   = np.column_stack([X, W @ X, beh])      # (N, 3)

    eM  = linalg.eigvals(M).real
    eWS = linalg.eigvals(W_S).real
    logdet_fn = lambda d: logdet_kron(d, eM, eWS)

    # Admissible bounds for delta
    prod = np.outer(eM, eWS).ravel()
    pos  = prod[prod > 0]; neg = prod[prod < 0]
    lo   = max(-1./pos.max() if len(pos) else -0.9999, -0.9999)
    hi   = min(-1./neg.min() if len(neg) else  0.9999,  0.9999)
    lo  += 1e-4*(hi-lo); hi -= 1e-4*(hi-lo)

    # Grid search + Brent refinement
    grid  = np.linspace(lo, hi, n_grid)
    ll_g  = [_profile_ll(d, Y, W, R, logdet_fn) for d in grid]
    bi    = int(np.argmin(ll_g))
    d_lo  = grid[max(bi-2, 0)];  d_hi = grid[min(bi+2, n_grid-1)]
    res   = optimize.minimize_scalar(
        lambda d: _profile_ll(d, Y, W, R, logdet_fn),
        bounds=(d_lo, d_hi), method="bounded",
        options={"xatol": 1e-9, "maxiter": 500})
    dh = float(res.x)

    # Recover alpha = (beta, theta, delta_c) by concentrated OLS
    S  = np.eye(N) - dh * W
    SY = S @ Y
    Ri = linalg.pinv(R.T @ R)
    ah = Ri @ (R.T @ SY)
    r  = SY - R @ ah
    s2 = float(r @ r) / N
    return np.concatenate([[dh], ah, [s2]])   # (delta, beta, theta, delta_c, sigma2)


def cf_gmm_step(Y, X, W, beh, max_iter=2, sigma2_init=1.):
    """
    Two-step CF-GMM with linear + quadratic moments.
    Returns kappa = (delta, beta, theta, delta_c).
    """
    N   = len(Y)
    beh = np.asarray(beh, float)
    WY  = (W @ Y)[:, None]
    WX  = W @ X;  W2X = W @ WX
    WtW = W.T @ W
    U   = np.column_stack([WY, X, WX, beh])
    Q_m = np.column_stack([X, WX, W2X, beh])
    trW    = float(np.trace(W))
    trWtW  = float(np.trace(WtW))
    trWtW2 = float(np.trace(WtW @ WtW))

    def g(kap, s2):
        xi  = Y - U @ kap
        gl  = Q_m.T @ xi / N
        gq1 = (xi @ (W   @ xi) - s2*trW)    / N
        gq2 = (xi @ (WtW @ xi) - s2*trWtW)  / N
        return np.concatenate([gl, [gq1, gq2]])

    Psi_inv = np.eye(6)
    x0      = np.zeros(5); x0[-1] = sigma2_init

    for _ in range(max_iter):
        res_ = optimize.minimize(
            lambda p: float(g(p[:4], max(p[4], 1e-8)) @ Psi_inv
                           @ g(p[:4], max(p[4], 1e-8))),
            x0, method="Nelder-Mead",
            options={"maxiter": 8000, "xatol": 1e-6, "fatol": 1e-8})
        k_  = res_.x[:4];  s2_ = max(res_.x[4], 1e-8)
        dv  = np.array([s2_]*4 +
                       [max(2*s2_**2*trWtW  / N, 1e-10),
                        max(2*s2_**2*trWtW2 / N, 1e-10)])
        Psi_inv = np.diag(1./dv)
        x0 = res_.x

    return k_   # (delta, beta, theta, delta_c)


def within_transform(Y, X, n, beh=None):
    """
    Within (unit) demeaning for time-major stacked data.
    Returns Y_dm (N,), X_dm (N,k), and optionally beh_dm (N,).
    """
    TT   = len(Y) // n
    Y_dm = (Y.reshape(TT, n) - Y.reshape(TT, n).mean(0)).ravel()
    k    = X.shape[1] if X.ndim == 2 else 1
    X_dm = (X.reshape(TT, n, k) - X.reshape(TT, n, k).mean(0)).reshape(-1, k)
    if beh is not None:
        b_dm = (beh.reshape(TT, n) - beh.reshape(TT, n).mean(0)).ravel()
        return Y_dm, X_dm, b_dm
    return Y_dm, X_dm

# ═════════════════════════════════════════════════════════════════════════════
# §4  FORMATTING HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def compute_summary(arr, true_val):
    """Return (mean, std, rmse) ignoring NaN/Inf."""
    a = arr[np.isfinite(arr)]
    if len(a) == 0:
        return np.nan, np.nan, np.nan
    return float(a.mean()), float(a.std(ddof=1)), float(np.sqrt(np.mean((a-true_val)**2)))


def format_cell(mean, std, rmse, w=9):
    """3-line cell: mean / (std) / [rmse]."""
    if np.isnan(mean):
        pad = " " * w
        return pad, pad, pad
    return (f"{mean:+{w}.4f}",
            f"({std:{w-2}.4f})",
            f"[{rmse:{w-2}.4f}]")


def hline(ch="─", width=112):
    print(ch * width)


def _banner(title):
    hline("═")
    print(f"  {title}")
    hline("═")

# ═════════════════════════════════════════════════════════════════════════════
# §5  TABLE 1
# Estimator comparison: Unadj-2SLS | CF-2SLS | CF-GMM | CF-QMLE
# Panels A (δ_c=0) and B (δ_c=0.5); samples WS (n=49) and WO (n=100)
# ═════════════════════════════════════════════════════════════════════════════

def one_rep_table1(n_side, delta_c, seed):
    Y, X, W, W_S, M, h, Z_W, eps, bar_eps, xi = simulate_L_T(
        n_side, DELTA0, BETA0, THETA0, SIGMA_XI2, delta_c, RHO_M, seed)
    n       = n_side ** 2
    eps_hat = first_stage(h, Z_W)
    beh     = aggregate_eps(eps_hat, n)
    ku      = unadj_2sls(Y, X, W)
    k2      = cf_2sls(Y, X, W, beh)
    kg      = cf_gmm_step(Y, X, W, beh)
    kq      = cf_qmle(Y, X, W, W_S, M, beh)
    return {"un": ku, "2sls": k2, "gmm": kg, "qmle": kq[:4]}


def run_table1_serial(n_side, delta_c, seed_offset=0):
    results = {"un": [], "2sls": [], "gmm": [], "qmle": []}
    rng   = np.random.default_rng(1000 + seed_offset)
    seeds = rng.integers(0, 2**31, size=REPS)
    for r in range(REPS):
        try:
            out = one_rep_table1(n_side, delta_c, int(seeds[r]))
            for k in results:
                results[k].append(out[k])
        except Exception:
            pass
    return {k: np.array(v) for k, v in results.items()}


def print_table1(res_ws0, res_wo0, res_ws05, res_wo05):
    _banner("TABLE 1  Bias / Std / RMSE — Unadj-2SLS | CF-2SLS | CF-GMM | CF-QMLE")
    param_names = ["δ", "β", "θ", "δ_c"]
    ests        = ["un", "2sls", "gmm", "qmle"]
    est_labels  = ["Unadj-2SLS", "CF-2SLS   ", "CF-GMM    ", "CF-QMLE   "]

    for panel_label, res_ws, res_wo, dc in [
            ("Panel A:  δ_c = 0.0  (exogenous W)", res_ws0,  res_wo0,  0.0),
            ("Panel B:  δ_c = 0.5  (endogenous W)", res_ws05, res_wo05, 0.5)]:

        true_vals = [DELTA0, BETA0, THETA0, dc]
        print(f"\n  {panel_label}")

        for sample_label, res in [("WS  n=49,  N=245", res_ws),
                                   ("WO  n=100, N=500", res_wo)]:
            print(f"\n    [{sample_label}]")
            hline(width=100)
            hdr = f"    {'':4s}  " + "  ".join(f"{el:^28s}" for el in est_labels)
            print(hdr)
            hline(width=100)

            # 3 lines per parameter
            for pi, (pname, tv) in enumerate(zip(param_names, true_vals)):
                lines = ["", "", ""]
                for est in ests:
                    arr = res[est]
                    col = arr[:, pi] if pi < arr.shape[1] else np.full(len(arr), np.nan)
                    L0, L1, L2 = format_cell(*compute_summary(col, tv), w=8)
                    lines[0] += f"  {L0:^28s}"
                    lines[1] += f"  {L1:^28s}"
                    lines[2] += f"  {L2:^28s}"
                prefix = [f"    {pname:>4s}", "      ", "      "]
                for i in range(3):
                    print(prefix[i] + lines[i])
                print()
            hline(width=100)


def run_table1(verbose=True):
    configs = [
        (N_SIDE_S, 0.0,   0, "WS δ_c=0"),
        (N_SIDE_L, 0.0, 100, "WO δ_c=0"),
        (N_SIDE_S, 0.5, 200, "WS δ_c=0.5"),
        (N_SIDE_L, 0.5, 300, "WO δ_c=0.5"),
    ]
    results = {}
    for ns, dc, offset, tag in configs:
        if verbose: print(f"    {tag} ...", flush=True)
        results[(ns, dc)] = run_table1_serial(ns, dc, seed_offset=offset)

    res_ws0  = results[(N_SIDE_S, 0.0)]
    res_wo0  = results[(N_SIDE_L, 0.0)]
    res_ws05 = results[(N_SIDE_S, 0.5)]
    res_wo05 = results[(N_SIDE_L, 0.5)]
    print_table1(res_ws0, res_wo0, res_ws05, res_wo05)
    np.savez("mc_results/table1.npz",
             res_ws0_un=res_ws0["un"],   res_ws0_2sls=res_ws0["2sls"],
             res_ws0_gmm=res_ws0["gmm"], res_ws0_qmle=res_ws0["qmle"],
             res_wo0_un=res_wo0["un"],   res_wo0_2sls=res_wo0["2sls"],
             res_wo0_gmm=res_wo0["gmm"], res_wo0_qmle=res_wo0["qmle"],
             res_ws05_un=res_ws05["un"],   res_ws05_2sls=res_ws05["2sls"],
             res_ws05_gmm=res_ws05["gmm"], res_ws05_qmle=res_ws05["qmle"],
             res_wo05_un=res_wo05["un"],   res_wo05_2sls=res_wo05["2sls"],
             res_wo05_gmm=res_wo05["gmm"], res_wo05_qmle=res_wo05["qmle"])
    return results

# ═════════════════════════════════════════════════════════════════════════════
# §6  TABLE 2
# Vary δ_c ∈ {0, 0.3, 0.5, 0.8}; fixed n=49
# Reports mean(SE)[RMSE] of all four estimators for each parameter
# ═════════════════════════════════════════════════════════════════════════════

def run_table2(verbose=True):
    dc_vals = [0.0, 0.3, 0.5, 0.8]

    _banner("TABLE 2  Sensitivity to δ_c — n=49, N=245, True δ=0.3, β=1.0, θ=0.5")
    param_names = ["δ", "β", "θ", "δ_c"]
    ests        = ["un", "2sls", "gmm", "qmle"]
    est_labels  = ["Unadj-2SLS", "CF-2SLS", "CF-GMM", "CF-QMLE"]
    store = {}

    for dc in dc_vals:
        if verbose: print(f"    δ_c = {dc} ...", flush=True)
        res = run_table1_serial(N_SIDE_S, dc, seed_offset=int(dc*1000) + 2000)
        store[dc] = res

    print()
    for dc in dc_vals:
        res = store[dc]
        true_vals = [DELTA0, BETA0, THETA0, dc]
        print(f"  δ_c = {dc:.1f}")
        hline(width=100)
        print(f"  {'':10s}" + "".join(f"  {en:^28s}" for en in est_labels))
        hline(width=100)
        for pi, (pname, tv) in enumerate(zip(param_names, true_vals)):
            lines = ["", "", ""]
            for est in ests:
                arr = res[est]
                col = arr[:, pi] if pi < arr.shape[1] else np.full(len(arr), np.nan)
                L0, L1, L2 = format_cell(*compute_summary(col, tv), w=8)
                lines[0] += f"  {L0:^28s}"
                lines[1] += f"  {L1:^28s}"
                lines[2] += f"  {L2:^28s}"
            pref = [f"  {pname:>8s}", "          ", "          "]
            for i in range(3):
                print(pref[i] + lines[i])
            print()
        hline(width=100)
        print()

    np.savez("mc_results/table2.npz",
             **{f"dc{str(dc).replace('.','')}_2sls": store[dc]["2sls"] for dc in dc_vals})
    return store

# ═════════════════════════════════════════════════════════════════════════════
# §7  TABLE 3
# Coverage of 95% CI for δ: uncorrected SE vs Ω_A-corrected SE
# Rows: L=T (n=49), L=T (n=100), L=N (n=49), L=N (n=100)
# δ_c = 0.5 (endogenous W, so Ω_A correction matters)
# ═════════════════════════════════════════════════════════════════════════════

def _one_rep_t3_LT(ns, seed):
    """One rep for Table 3, L=T entry equation."""
    Y, X, W, W_S, M, h, Z_W, eps, _, xi = simulate_L_T(
        ns, DELTA0, BETA0, THETA0, SIGMA_XI2, 0.5, RHO_M, seed)
    n   = ns**2; N = n * T
    eh  = first_stage(h, Z_W)
    beh = aggregate_eps(eh, n, "L_eq_T")
    kap = cf_2sls(Y, X, W, beh)
    if not np.all(np.isfinite(kap)):
        return np.nan, np.nan, np.nan

    # Uncorrected SE
    se_unc = cf_2sls_se(Y, X, W, beh, kap)

    # Ω_A-corrected SE
    WX  = W @ X; W2X = W @ WX
    Q   = np.column_stack([X, WX, W2X, beh])
    s2    = float((Y - np.column_stack([(W@Y), X, (W@X), beh]) @ kap) ** 2).mean() \
            if False else None   # recompute below
    xi_h  = Y - np.column_stack([(W@Y)[:,None], X, W@X, beh[:,None]]).squeeze() @ kap \
            if False else None
    # correct residual
    U     = np.column_stack([(W@Y), X, W@X, beh])
    xi_h  = Y - U @ kap
    s2    = float(xi_h @ xi_h) / N
    s2eps = float(eh @ eh) / max(T - Z_W.shape[1], 1)
    oA    = _omega_A(Q, Z_W, n, float(kap[3]), s2eps)
    se_cor = cf_2sls_se(Y, X, W, beh, kap, sigma2=s2, omega_A=oA)

    return float(kap[0]), float(se_unc[0]), float(se_cor[0])


def _one_rep_t3_LN(ns, seed):
    """One rep for Table 3, L=N entry equation (no Ω_A needed)."""
    Y, X, W, W_S, M, h, Z_W, eps, _, xi = simulate_L_N(
        ns, DELTA0, BETA0, THETA0, SIGMA_XI2, 0.5, RHO_M, seed)
    n   = ns**2
    eh  = first_stage(h, Z_W)
    beh = aggregate_eps(eh, n, "L_eq_N")
    kap = cf_2sls(Y, X, W, beh)
    if not np.all(np.isfinite(kap)):
        return np.nan, np.nan, np.nan
    se = cf_2sls_se(Y, X, W, beh, kap)
    return float(kap[0]), float(se[0]), float(se[0])  # corr = uncorr for L=N


def run_table3(verbose=True):
    configs = [
        (N_SIDE_S, "L=T", _one_rep_t3_LT, 3000),
        (N_SIDE_L, "L=T", _one_rep_t3_LT, 3100),
        (N_SIDE_S, "L=N", _one_rep_t3_LN, 3200),
        (N_SIDE_L, "L=N", _one_rep_t3_LN, 3300),
    ]
    _banner("TABLE 3  SE Coverage (nominal 95%) — δ_c=0.5, True δ=0.3")
    print(f"  {'Setting':<18s}  {'Bias':>8s}  {'Std':>8s}  {'RMSE':>8s}"
          f"  {'Cov(Uncorr)':>12s}  {'Cov(Ω_A corr)':>14s}  {'n_ok':>6s}")
    hline(width=88)

    store = {}
    for (ns, mode, fn, offset) in configs:
        label = f"{mode}, n={ns**2}"
        if verbose: print(f"    {label} ...", flush=True)
        rng   = np.random.default_rng(offset)
        seeds = rng.integers(0, 2**31, size=REPS)
        d_hat = []; se_unc = []; se_cor = []
        for r in range(REPS):
            try:
                d, su, sc = fn(ns, int(seeds[r]))
                if np.isfinite(d):
                    d_hat.append(d); se_unc.append(su); se_cor.append(sc)
            except Exception:
                pass
        d_hat  = np.array(d_hat); se_unc = np.array(se_unc); se_cor = np.array(se_cor)
        n_ok   = len(d_hat)
        bias, std, rmse = compute_summary(d_hat, DELTA0)
        cov_unc = float(np.mean(np.abs(d_hat - DELTA0) <= 1.96 * se_unc))
        cov_cor = float(np.mean(np.abs(d_hat - DELTA0) <= 1.96 * se_cor))
        store[label] = dict(d_hat=d_hat, se_unc=se_unc, se_cor=se_cor,
                            bias=bias, std=std, rmse=rmse,
                            cov_unc=cov_unc, cov_cor=cov_cor)
        print(f"  {label:<18s}  {bias:>+8.4f}  {std:>8.4f}  {rmse:>8.4f}"
              f"  {cov_unc:>12.3f}  {cov_cor:>14.3f}  {n_ok:>6d}")
    hline(width=88)
    print("  Note: Cov(Uncorr) ignores first-stage estimation error.")
    print("        Cov(Ω_A corr) adds the Ω_A term (Prop. B.3 in Appendix).")
    print("        For L=N the two SEs coincide (no generated-regressor bias).")

    np.savez("mc_results/table3.npz",
             **{k.replace("=","_").replace(",","_").replace(" ",""):
                np.array([v["d_hat"], v["se_unc"], v["se_cor"]]) for k,v in store.items()})
    return store

# ═════════════════════════════════════════════════════════════════════════════
# §8  TABLE 4
# Fixed-effects (within-unit) transformation + CF estimators
# Pooled CF (ignores FE, biased) vs Within CF-2SLS vs Within CF-QMLE
# ═════════════════════════════════════════════════════════════════════════════

def _one_rep_t4(ns, dc, seed):
    Y, X, W, W_S, M, h, Z_W, eps, _, xi, _ = simulate_FE(
        ns, DELTA0, BETA0, THETA0, SIGMA_XI2, dc, RHO_M, seed)
    n   = ns**2
    eh  = first_stage(h, Z_W)
    beh = aggregate_eps(eh, n)

    # 1) Pooled CF-2SLS (no FE correction — biased benchmark)
    k_pool = cf_2sls(Y, X, W, beh)

    # 2) Within-transformed CF-2SLS
    Y_dm, X_dm, b_dm = within_transform(Y, X, n, beh)
    k_fe2 = cf_2sls(Y_dm, X_dm, W, b_dm)

    # 3) Within-transformed CF-QMLE
    kq_fe = cf_qmle(Y_dm, X_dm, W, W_S, M, b_dm)

    return k_pool[0], k_fe2[0], kq_fe[0]


def run_table4(verbose=True):
    _banner("TABLE 4  Fixed-Effects + CF Estimators — True δ=0.3, unit FE present")
    print(f"  {'':18s}  {'Pooled CF-2SLS':^26s}  {'Within CF-2SLS':^26s}  {'Within CF-QMLE':^26s}")
    hline(width=104)

    store = {}
    for dc in [0.0, 0.5]:
        print(f"  δ_c = {dc:.1f}")
        for (label, ns) in [("n=49, N=245", N_SIDE_S), ("n=100, N=500", N_SIDE_L)]:
            if verbose: print(f"    {label}, δ_c={dc} ...", flush=True)
            rng   = np.random.default_rng(4000 + ns + int(dc*100))
            seeds = rng.integers(0, 2**31, size=REPS)
            pool=[]; fe2=[]; feq=[]
            for r in range(REPS):
                try:
                    kp, k2, kq = _one_rep_t4(ns, dc, int(seeds[r]))
                    if np.isfinite(kp): pool.append(kp)
                    if np.isfinite(k2): fe2.append(k2)
                    if np.isfinite(kq): feq.append(kq)
                except Exception:
                    pass
            pool = np.array(pool); fe2 = np.array(fe2); feq = np.array(feq)
            store[(ns, dc)] = dict(pool=pool, fe2=fe2, feq=feq)

            def _fmt3(arr):
                bias, std, rmse = compute_summary(arr, DELTA0)
                if np.isnan(bias): return "  —  ", "  —  ", "  —  "
                return format_cell(bias, std, rmse, w=7)

            L0p,L1p,L2p = _fmt3(pool)
            L0f,L1f,L2f = _fmt3(fe2)
            L0q,L1q,L2q = _fmt3(feq)
            pref = [f"  {label:<18s}", "  "+" "*18, "  "+" "*18]
            rows = [(L0p,L0f,L0q), (L1p,L1f,L1q), (L2p,L2f,L2q)]
            for i,(a,b,c) in enumerate(rows):
                print(pref[i] + f"  {a:^26s}  {b:^26s}  {c:^26s}")
            print()
        hline(width=104)
        print()

    print("  Bias/Std/RMSE for δ̂.  Pooled ignores unit FE → biased.")
    print("  Within-CF uses unit demeaning before CF-2SLS / CF-QMLE.")
    np.savez("mc_results/table4.npz",
             **{f"ns{ns}_dc{str(dc).replace('.','')}_fe2": store[(ns,dc)]["fe2"]
                for ns in [N_SIDE_S, N_SIDE_L] for dc in [0.0, 0.5]})
    return store

# ═════════════════════════════════════════════════════════════════════════════
# §9  TABLE 5
# Cross-period indirect effects IE_{t←s}:
#   (a) true IE values, (b) Delta-method coverage, (c) BH-FDR rejection rate
# ═════════════════════════════════════════════════════════════════════════════

def _multiplier(delta, W):
    return linalg.inv(np.eye(W.shape[0]) - delta * W)


def _ie_ts(Tm, beta, theta, W, n, t, s):
    """Average IE: effect of a unit-wide shock in period s on period-t outcomes."""
    src  = [s*n + i for i in range(n)]
    wrs  = W[np.ix_(src, list(range(W.shape[1])))].sum(1)  # row sums at source
    dst  = [t*n + r for r in range(n)]
    val  = sum(Tm[d, src[i]] * (beta + theta*wrs[i])
               for i in range(n) for d in dst)
    return float(val) / n


def _grad_ie(kap, W, n, t, s, h=2e-5):
    """Numerical gradient ∂IE_{t←s}/∂κ (κ = [δ, β, θ, δ_c])."""
    p  = len(kap); g = np.zeros(p)
    for j in range(p):
        e = np.zeros(p); e[j] = h
        def ie(k):
            Tm = _multiplier(k[0], W)
            return _ie_ts(Tm, k[1], k[2], W, n, t, s)
        g[j] = (ie(kap+e) - ie(kap-e)) / (2*h)
    return g


def _bh(pvals, alpha=0.05):
    """Benjamini-Hochberg step-up procedure. Returns boolean reject array."""
    m   = len(pvals)
    ord = np.argsort(pvals)
    thr = alpha * np.arange(1, m+1) / m
    rej = pvals[ord] <= thr
    if rej.any():
        rej[:rej.nonzero()[0].max()+1] = True
    out = np.zeros(m, bool); out[ord] = rej
    return out


def run_table5(verbose=True):
    ns  = N_SIDE_S; n = ns**2; N = n*T; dc = 0.5

    _banner("TABLE 5  Cross-Period Indirect Effects IE_{t←s} — n=49, δ_c=0.5")

    # Oracle true IE matrix
    W_S_true = rook_weights(ns); M_true = temporal_matrix(T, RHO_M)
    W_true   = build_stwm(M_true, W_S_true)
    Tm_true  = _multiplier(DELTA0, W_true)
    IE_true  = np.zeros((T, T))
    for t_ in range(T):
        for s_ in range(T):
            IE_true[t_, s_] = _ie_ts(Tm_true, BETA0, THETA0, W_true, n, t_, s_)

    print("\n  True IE matrix (oracle):")
    print("         " + "".join(f"  s={s_:d}  " for s_ in range(T)))
    for t_ in range(T):
        print(f"  t={t_:d}    " + "".join(f"  {IE_true[t_,s_]:+.4f}" for s_ in range(T)))

    # Monte Carlo
    rng    = np.random.default_rng(5000)
    seeds  = rng.integers(0, 2**31, size=REPS)
    cov_mat = np.zeros((T, T)); nrep_ok = 0
    bh_tp_arr=[]; bh_fp_arr=[]

    for r in range(REPS):
        try:
            Y, X, W, W_S, M, h, Z_W, eps, _, xi = simulate_L_T(
                ns, DELTA0, BETA0, THETA0, SIGMA_XI2, dc, RHO_M, int(seeds[r]))
            eh  = first_stage(h, Z_W)
            beh = aggregate_eps(eh, n)
            kap = cf_2sls(Y, X, W, beh)
            if not np.all(np.isfinite(kap)): continue

            # Avar of kap
            U    = np.column_stack([(W@Y), X, (W@X), beh])
            Q_m  = np.column_stack([X, (W@X), (W@(W@X)), beh])
            xi_h = Y - U @ kap
            s2   = float(xi_h @ xi_h) / N
            Qi   = linalg.pinv(Q_m.T @ Q_m)
            PU   = Q_m @ (Qi @ (Q_m.T @ U))
            br   = linalg.pinv(PU.T @ U / N)
            me   = s2 * (PU.T @ PU) / N
            avar = br @ me @ br

            Tm   = _multiplier(kap[0], W)
            pvals = []
            for t_ in range(T):
                for s_ in range(T):
                    ie_h = _ie_ts(Tm, kap[1], kap[2], W, n, t_, s_)
                    gr   = _grad_ie(kap, W, n, t_, s_)
                    se_ie = np.sqrt(max(float(gr @ (avar/N) @ gr), 0.))
                    if se_ie > 1e-10:
                        cov_mat[t_, s_] += abs(ie_h - IE_true[t_, s_]) <= 1.96*se_ie
                        z = (ie_h - IE_true[t_, s_]) / se_ie
                        pvals.append(2*scipy_stats.norm.sf(abs(z)))
                    else:
                        pvals.append(1.)

            pvals  = np.array(pvals)
            rej    = _bh(pvals, ALPHA)
            # True H₀: IE = 0 → we use |IE_true| < 0.001 as "effectively zero"
            true_h0 = np.abs(IE_true.ravel()) < 1e-3
            bh_tp_arr.append(int((rej & ~true_h0).sum()))
            bh_fp_arr.append(int((rej &  true_h0).sum()))
            nrep_ok += 1
        except Exception:
            pass

    cov_mat /= max(nrep_ok, 1)

    print(f"\n  Coverage of 95% Delta-method CI  (n_ok = {nrep_ok}/{REPS}):")
    print("         " + "".join(f"  s={s_:d}  " for s_ in range(T)))
    for t_ in range(T):
        print(f"  t={t_:d}    " + "".join(f"  {cov_mat[t_,s_]:.3f} " for s_ in range(T)))

    avg_tp = float(np.mean(bh_tp_arr)) if bh_tp_arr else np.nan
    avg_fp = float(np.mean(bh_fp_arr)) if bh_fp_arr else np.nan
    print(f"\n  BH-FDR (5%):  avg true positives = {avg_tp:.2f} / {T*T}"
          f",  avg false positives = {avg_fp:.2f}")
    hline()

    np.savez("mc_results/table5.npz", cov_mat=cov_mat, IE_true=IE_true,
             bh_tp=np.array(bh_tp_arr), bh_fp=np.array(bh_fp_arr))
    return dict(cov_mat=cov_mat, IE_true=IE_true, n_ok=nrep_ok,
                bh_tp=avg_tp, bh_fp=avg_fp)

# ═════════════════════════════════════════════════════════════════════════════
# §10 TABLE 6
# Robustness to ρ_M ∈ {0.1, 0.3, 0.5, 0.7} and graph structure
# Bias / Std / RMSE of δ̂ (CF-2SLS and CF-QMLE)
# ═════════════════════════════════════════════════════════════════════════════

def run_table6(verbose=True):
    rho_vals = [0.1, 0.3, 0.5, 0.7]
    graphs   = [("Rook",  rook_weights),
                ("Queen", queen_weights),
                ("KNN-4", knn_weights)]
    ns = N_SIDE_S; n = ns**2; dc = 0.5

    _banner("TABLE 6  Robustness: ρ_M and Graph Structure — n=49, δ_c=0.5, True δ=0.3")
    store = {}

    for gname, WS_fn in graphs:
        print(f"\n  Graph: {gname}")
        hline(width=80)
        print(f"  {'ρ_M':>5s}  {'CF-2SLS  Bias(Std)[RMSE]':^32s}  "
              f"{'CF-QMLE  Bias(Std)[RMSE]':^32s}  n_ok")
        hline(width=80)

        for rho in rho_vals:
            if verbose: print(f"    {gname}, ρ_M={rho} ...", end="\r", flush=True)
            rng   = np.random.default_rng(6000 + int(rho*100) + abs(hash(gname)) % 1000)
            seeds = rng.integers(0, 2**31, size=REPS)
            t2=[]; qm=[]

            for r in range(REPS):
                try:
                    rng2 = np.random.default_rng(int(seeds[r]))
                    N_   = n * T
                    W_S  = WS_fn(ns)
                    M_   = temporal_matrix(T, rho)
                    W_   = build_stwm(M_, W_S)
                    X_   = rng2.standard_normal((N_, 1))
                    Z_W_ = rng2.standard_normal((T, 2))
                    eps_ = rng2.standard_normal(T)
                    h_   = Z_W_ @ np.array([1., .8]) + eps_
                    baru_= np.repeat(eps_, n)
                    xi_  = rng2.standard_normal(N_)
                    rhs_ = X_.ravel()*BETA0 + (W_@X_).ravel()*THETA0 + dc*baru_ + xi_
                    Y_   = linalg.solve(np.eye(N_) - DELTA0*W_, rhs_)
                    eh_  = first_stage(h_, Z_W_)
                    beh_ = np.repeat(eh_, n)
                    k2_  = cf_2sls(Y_, X_, W_, beh_)
                    kq_  = cf_qmle(Y_, X_, W_, W_S, M_, beh_)
                    if np.isfinite(k2_[0]): t2.append(k2_[0])
                    if np.isfinite(kq_[0]): qm.append(kq_[0])
                except Exception:
                    pass

            t2 = np.array(t2); qm = np.array(qm)
            n_ok = min(len(t2), len(qm))
            store[(gname, rho)] = dict(t2=t2, qm=qm)

            b2,s2,r2 = compute_summary(t2, DELTA0)
            bq,sq,rq = compute_summary(qm, DELTA0)

            def _c(b,s,r): return f"{b:+.3f} ({s:.3f}) [{r:.3f}]" if not np.isnan(b) else "—"
            print(f"  {rho:>5.1f}  {_c(b2,s2,r2):^32s}  {_c(bq,sq,rq):^32s}  {n_ok}")

        hline(width=80)

    np.savez("mc_results/table6.npz",
             **{f"{g}_{str(r).replace('.','')}_t2": store[(g,r)]["t2"]
                for g,_ in graphs for r in rho_vals})
    return store

# ═════════════════════════════════════════════════════════════════════════════
# §11 TABLE 7
# Computation time: Kronecker log-det vs full-matrix log-det
# ═════════════════════════════════════════════════════════════════════════════

def run_table7(n_reps=200):
    import time as _time
    configs = [
        (4,  3),  (4,  5),  (4,  10),
        (7,  3),  (7,  5),  (7,  10),
        (10, 3),  (10, 5),  (10, 10),
        (15, 5),  (20, 5),
    ]
    delta = DELTA0

    _banner("TABLE 7  Computation Time for ln|I − δW|  (averaged over evaluations)")
    print(f"  {'n_side':>7s}  {'T':>4s}  {'N':>6s}  "
          f"{'Kron. eig (ms)':>16s}  {'Full eig (ms)':>15s}  "
          f"{'Kron. eval (ms)':>17s}  {'Full eval (ms)':>16s}  {'Speedup':>8s}")
    hline(width=100)

    store = {}
    for (ns, TT) in configs:
        n  = ns**2; N = n*TT
        W_S = rook_weights(ns)
        M_  = temporal_matrix(TT, RHO_M)
        W_  = build_stwm(M_, W_S)

        # Precompute eigenvalues (one-time cost)
        t0   = _time.perf_counter()
        eM   = linalg.eigvals(M_).real
        eWS  = linalg.eigvals(W_S).real
        t_ke = (_time.perf_counter() - t0) * 1000        # eig time Kron (ms)

        t0   = _time.perf_counter()
        eW   = linalg.eigvals(W_).real
        t_fe = (_time.perf_counter() - t0) * 1000        # eig time Full (ms)

        # Evaluate log-det (many times)
        t0 = _time.perf_counter()
        for _ in range(n_reps):
            logdet_kron(delta, eM, eWS)
        t_kv = (_time.perf_counter() - t0) / n_reps * 1000   # per-eval ms

        t0 = _time.perf_counter()
        for _ in range(n_reps):
            vals = 1. - delta * eW
            if np.all(vals > 0): float(np.sum(np.log(vals)))
        t_fv = (_time.perf_counter() - t0) / n_reps * 1000   # per-eval ms

        speedup = t_fv / t_kv if t_kv > 1e-9 else np.inf
        store[(ns,TT)] = dict(t_ke=t_ke, t_fe=t_fe, t_kv=t_kv, t_fv=t_fv, speedup=speedup)
        print(f"  {ns:>7d}  {TT:>4d}  {N:>6d}  "
              f"{t_ke:>16.3f}  {t_fe:>15.3f}  "
              f"{t_kv:>17.4f}  {t_fv:>16.4f}  {speedup:>8.1f}×")

    hline(width=100)
    print("  Kron. eig = eigenvalue precomputation for M and W_S separately (O(T³+n³)).")
    print("  Full eig  = eigenvalue precomputation for full W = M⊗W_S (O(N³)).")
    print("  Kron. eval = per-call Kronecker log-det (O(T·n) multiplications).")
    print("  Full eval  = per-call log-det from full eigvals (O(N) log operations).")
    print("  Note: Kron. eig is faster than Full eig AND each subsequent call is O(T·n)")
    print("        vs O(N=n·T) — the Kronecker approach dominates for many likelihood evals.")

    np.savez("mc_results/table7.npz",
             n_sides=np.array([c[0] for c in configs]),
             Ts=np.array([c[1] for c in configs]),
             t_kv=np.array([store[c]["t_kv"] for c in configs]),
             t_fv=np.array([store[c]["t_fv"] for c in configs]),
             speedup=np.array([store[c]["speedup"] for c in configs]))
    return store

# ═════════════════════════════════════════════════════════════════════════════
# §12 MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Monte Carlo Tables 1–7  (STWM-CF paper)")
    parser.add_argument("--reps",   type=int, default=500,
                        help="Monte Carlo replications (default 500; use 50 for quick test)")
    parser.add_argument("--tables", default="all",
                        help="Tables to run, e.g. '1,3,7' or 'all' (default)")
    args = parser.parse_args()

    global REPS
    REPS = args.reps
    run_all = (args.tables == "all")
    sel     = set(args.tables.split(",")) if not run_all else set()

    def want(n): return run_all or str(n) in sel

    t_global = time.time()
    print(f"\n{'━'*60}")
    print(f"  STWM-CF  Monte Carlo  —  REPS={REPS}")
    print(f"  δ₀={DELTA0}  β₀={BETA0}  θ₀={THETA0}  T={T}  ρ_M={RHO_M}  σ²={SIGMA_XI2}")
    print(f"  Results saved to ./mc_results/")
    print(f"{'━'*60}\n")

    results = {}

    for num, label, fn in [
        (1, "Table 1  (estimator comparison)",    lambda: run_table1()),
        (2, "Table 2  (vary δ_c)",                lambda: run_table2()),
        (3, "Table 3  (SE coverage, Ω_A)",        lambda: run_table3()),
        (4, "Table 4  (fixed effects + CF)",      lambda: run_table4()),
        (5, "Table 5  (cross-period IE + BH-FDR)",lambda: run_table5()),
        (6, "Table 6  (robustness)",              lambda: run_table6()),
        (7, "Table 7  (computation time)",        lambda: run_table7()),
    ]:
        if not want(num):
            continue
        print(f"\n[{label}]")
        t0 = time.time()
        results[num] = fn()
        print(f"\n  → Done in {time.time()-t0:.1f}s\n")

    print(f"\n{'━'*60}")
    print(f"  All selected tables finished in {time.time()-t_global:.1f}s")
    print(f"  NPZ files written to  ./mc_results/")
    print(f"{'━'*60}\n")
    return results


if __name__ == "__main__":
    main()
