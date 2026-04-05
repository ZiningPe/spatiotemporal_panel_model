"""
moran_example.py
----------------
Demonstrates building the temporal weight matrix M from Moran's I statistics
and estimating the Spatial Durbin Model via CF-2SLS, CF-QMLE, and CF-GMM.

DGP
---
- 25 spatial units on a 5×5 rook-contiguity grid
- T = 6 time periods
- M built from simulated Moran's I statistics
- W = M ⊗ W_S (Kronecker structure)
- True parameters: δ=0.3, β=1.0, θ=0.5, δ_c=0.4
"""

import numpy as np
from spatiotemporal_cf import (
    rook_weights, compute_morans_i, build_twm_from_stats, build_stwm,
    stwm_summary, eigvals_kronecker,
    first_stage, first_stage_stats, aggregate_eps_hat,
    cf_2sls_fit, cf_2sls_summary,
    cf_qmle,
    cf_gmm_fit,
    multiplier_matrix, cross_period_effects_matrix,
    ie_inference, bh_correction, print_ie_table, multiple_testing_summary,
    omega_A_simple, compute_sigma2_eps, cf_2sls_avar_corrected,
    summarise_results, ned_test,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Setup
# ─────────────────────────────────────────────────────────────────────────────

rng    = np.random.default_rng(2025)
n_side = 5
T      = 6
n      = n_side ** 2
N      = n * T

# True parameters
delta0   = 0.3
beta0    = 1.0
theta0   = 0.5
delta_c0 = 0.4

# Spatial weight matrix (rook, row-normalised)
W_S = rook_weights(n_side)
print(f"W_S shape: {W_S.shape}, row-sums ∈ [{W_S.sum(1).min():.2f}, {W_S.sum(1).max():.2f}]")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Simulate panel data and compute per-period Moran's I
# ─────────────────────────────────────────────────────────────────────────────

# Simulate cross-sectional data for each period to compute Moran's I
y_panels = [rng.standard_normal(n) for _ in range(T)]
morans   = [compute_morans_i(y, W_S) for y in y_panels]
print(f"\nPer-period Moran's I: {[round(m, 3) for m in morans]}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Build M (TWM) and W = M ⊗ W_S
# ─────────────────────────────────────────────────────────────────────────────

M = build_twm_from_stats(morans, method="moran")
W = build_stwm(M, W_S)

print(f"\nM shape: {M.shape}")
print("M (temporal weight matrix):")
print(np.round(M, 3))

info = stwm_summary(W, n, T)
print(f"\nSTWM summary: N={info['N']}, sparsity={info['sparsity']}, "
      f"spectral_radius={info['spectral_radius']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Simulate DGP
# ─────────────────────────────────────────────────────────────────────────────

X = rng.standard_normal((N, 1))

# Entry equation: h_t = Z_W π + ε_t  (L = T)
Z_W = rng.standard_normal((T, 2))
eps = rng.standard_normal(T)
h   = Z_W @ np.array([1.0, 0.8]) + eps

# Aggregate ε̄̂ (true, oracle version for DGP)
bar_eps_true = np.repeat(eps, n)

# Structural errors
xi  = rng.standard_normal(N) * 0.5
S   = np.eye(N) - delta0 * W
rhs = (X.ravel() * beta0
       + (W @ X).ravel() * theta0
       + bar_eps_true * delta_c0
       + xi)
Y   = np.linalg.solve(S, rhs)

print(f"\nSimulated: N={N}, n={n}, T={T}")
print(f"Y ~ N({Y.mean():.3f}, {Y.std():.3f})")

# ─────────────────────────────────────────────────────────────────────────────
# 5. First stage
# ─────────────────────────────────────────────────────────────────────────────

fs   = first_stage_stats(h, Z_W)
print(f"\nFirst stage: F={fs['F_stat']:.2f}, R²={fs['R2']:.4f}")

eps_hat     = fs["eps_hat"]
bar_eps_hat = aggregate_eps_hat(eps_hat, n, T, mode="L_eq_T")

# ─────────────────────────────────────────────────────────────────────────────
# 6. CF-2SLS
# ─────────────────────────────────────────────────────────────────────────────

res_2sls = cf_2sls_fit(Y, X, W, bar_eps_hat)
cf_2sls_summary(res_2sls)

# With Ω_A variance correction (L = T case)
sig2_eps = compute_sigma2_eps(eps_hat, Z_W)
WX   = W @ X; W2X = W @ WX
Q    = np.column_stack([X, WX, W2X, bar_eps_hat[:, None]])
omega = omega_A_simple(Q, Z_W, n, T,
                        delta_c_hat=res_2sls["kappa"][-1],
                        sigma2_eps=sig2_eps)
from spatiotemporal_cf import cf_2sls_avar, cf_2sls_se
avar_corr = cf_2sls_avar(Y, X, W, bar_eps_hat,
                          res_2sls["kappa"], omega_A=omega)
se_corr   = cf_2sls_se(avar_corr, N)
print("CF-2SLS SE with Ω_A correction:", np.round(se_corr, 4))

# ─────────────────────────────────────────────────────────────────────────────
# 7. CF-QMLE
# ─────────────────────────────────────────────────────────────────────────────

res_qmle = cf_qmle(Y, X, W, bar_eps_hat, M=M, W_S=W_S)
print(f"\nCF-QMLE:  δ̂={res_qmle['delta_hat']:.4f},  "
      f"α̂={np.round(res_qmle['alpha_hat'], 4)},  "
      f"σ̂²={res_qmle['sigma2_hat']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. CF-GMM
# ─────────────────────────────────────────────────────────────────────────────

res_gmm = cf_gmm_fit(Y, X, W, bar_eps_hat)
print(f"\nCF-GMM:   δ̂={res_gmm['kappa'][0]:.4f},  "
      f"β̂={res_gmm['kappa'][1]:.4f},  σ̂²={res_gmm['sigma2']:.4f}")

# Side-by-side comparison
summarise_results(cf2sls=res_2sls, cfqmle=res_qmle, cfgmm=res_gmm)

# ─────────────────────────────────────────────────────────────────────────────
# 9. Cross-period indirect effects with Delta method + BH correction
# ─────────────────────────────────────────────────────────────────────────────

T_mat = multiplier_matrix(res_2sls["kappa"][0], W)
ie_res = ie_inference(
    T_mat=T_mat,
    kappa=res_2sls["kappa"],
    avar_kappa=res_2sls["avar"],
    W=W, n=n, TT=T, N=N,
)
mt = multiple_testing_summary(
    ie_res["IE"], ie_res["SE"], ie_res["p_value"],
    alpha=0.05, fdr_alpha=0.05
)
print(f"\n  Significant cross-period effects (raw 5%): {mt['n_reject_raw']}")
print(f"  Significant cross-period effects (BH 5%):  {mt['n_reject_bh']}")
print_ie_table({**ie_res, **mt},
               period_labels=[f"t={t}" for t in range(T)])

# ─────────────────────────────────────────────────────────────────────────────
# 10. NED diagnostic
# ─────────────────────────────────────────────────────────────────────────────

xi_hat = Y - (
    (W @ Y) * res_2sls["kappa"][0]
    + X.ravel() * res_2sls["kappa"][1]
    + (W @ X).ravel() * res_2sls["kappa"][2]
    + bar_eps_hat * res_2sls["kappa"][3]
)
ned = ned_test(xi_hat, n, T, W_S)
print(f"\nNED check: {ned['summary']}")
print(f"  Moran's I by period: {ned['morans_by_period']}")
