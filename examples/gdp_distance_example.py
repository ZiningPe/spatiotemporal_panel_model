"""
gdp_distance_example.py
-----------------------
Demonstrates using GDP growth-distance as the basis for the temporal
weight matrix M, following the empirical application in the paper.

In the paper's empirical application:
  - Spatial units: Chinese provinces (n = 30)
  - Time periods:  T years of panel data
  - Temporal weights: inverse GDP-distance between periods
  - Spatial weights:  contiguity or inverse geographic distance

This script uses *simulated* GDP-like data to illustrate the workflow
without requiring proprietary data.
"""

import numpy as np
from spatiotemporal_cf import (
    rook_weights, inverse_distance_weights,
    build_twm_from_stats, build_twm_parametric, build_stwm, stwm_summary,
    compute_morans_i,
    first_stage, aggregate_eps_hat,
    cf_2sls_fit, cf_2sls_summary,
    cf_qmle,
    cf_gmm_fit,
    multiplier_matrix, ie_inference,
    bh_correction, multiple_testing_summary, print_ie_table,
    summarise_results,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Setup (mimicking Chinese provincial panel)
# ─────────────────────────────────────────────────────────────────────────────

rng = np.random.default_rng(2024)
n   = 30   # provinces
T   = 10   # years
N   = n * T

# True DGP parameters
delta0   = 0.25
beta0    = 0.8
theta0   = 0.3
delta_c0 = 0.5
sigma2_0 = 1.0

# ─────────────────────────────────────────────────────────────────────────────
# 2. Spatial weight matrix W_S (contiguity proxy via random adjacency)
# ─────────────────────────────────────────────────────────────────────────────

# Simulate geographic coordinates for 30 provinces
coords = rng.uniform(0, 10, (n, 2))
W_S    = inverse_distance_weights(coords, power=1.0, row_standardize=True)
print(f"W_S: {n}×{n}, row-sums ∈ [{W_S.sum(1).min():.3f}, {W_S.sum(1).max():.3f}]")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Build M from GDP-growth similarity (simulated)
# ─────────────────────────────────────────────────────────────────────────────

# Simulate per-period GDP growth rates (national average)
gdp_growth = rng.normal(0.07, 0.02, T)    # ~7% mean with fluctuations

# Temporal weight:  similar GDP growth → stronger weight
# We use the inverse-GDP-distance kernel (as in the paper)
M = build_twm_from_stats(gdp_growth, method="raw")

print(f"\nGDP growth rates: {np.round(gdp_growth, 3)}")
print(f"M (T×T temporal weight matrix):")
print(np.round(M, 3))

# Full STWM
W    = build_stwm(M, W_S)
info = stwm_summary(W, n, T)
print(f"\nSTWM: N={info['N']}, sparsity={info['sparsity']:.3f}, "
      f"spectral_radius={info['spectral_radius']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Simulate panel
# ─────────────────────────────────────────────────────────────────────────────

# Regressors: log GDP per capita (normalised), FDI inflow
X     = rng.standard_normal((N, 1))           # single regressor for clarity

# Entry equation at period level (L = T)
# Instrument: lagged national policy index, trade openness
Z_W   = rng.standard_normal((T, 2))
pi0   = np.array([0.9, -0.6])
eps   = rng.standard_normal(T) * 0.5
h     = Z_W @ pi0 + eps                       # Moran's I or similar

bar_eps_true = np.repeat(eps, n)

# Structural equation errors (correlated with ε via δ_c)
xi    = rng.standard_normal(N)
V     = delta_c0 * bar_eps_true + xi

S    = np.eye(N) - delta0 * W
rhs  = X.ravel() * beta0 + (W @ X).ravel() * theta0 + V
Y    = np.linalg.solve(S, rhs)

print(f"\nSimulated panel: N={N} (n={n} provinces × T={T} years)")
print(f"Y: mean={Y.mean():.3f}, std={Y.std():.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Estimation pipeline
# ─────────────────────────────────────────────────────────────────────────────

# -- First stage --
eps_hat, pi_hat = first_stage(h, Z_W)
print(f"\nFirst stage π̂: {np.round(pi_hat, 3)}")

bar_eps_hat = aggregate_eps_hat(eps_hat, n, T, mode="L_eq_T")

# -- CF-2SLS --
print("\n" + "─" * 50)
res_2sls = cf_2sls_fit(Y, X, W, bar_eps_hat)
cf_2sls_summary(res_2sls)

# -- CF-QMLE --
res_qmle = cf_qmle(Y, X, W, bar_eps_hat, M=M, W_S=W_S)
print(f"CF-QMLE  δ̂={res_qmle['delta_hat']:.4f}  "
      f"β̂={res_qmle['alpha_hat'][0]:.4f}  "
      f"θ̂={res_qmle['alpha_hat'][1]:.4f}  "
      f"δ̂_c={res_qmle['alpha_hat'][2]:.4f}")

# -- CF-GMM --
res_gmm = cf_gmm_fit(Y, X, W, bar_eps_hat, max_iter=2)

summarise_results(cf2sls=res_2sls, cfqmle=res_qmle, cfgmm=res_gmm)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Cross-period spillover effects with FDR control
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

print_ie_table(
    {**ie_res, **mt},
    period_labels=[f"Y{t+1}" for t in range(T)],
)

print(f"\nDone. True δ₀={delta0}, estimated δ̂={res_2sls['kappa'][0]:.4f}")
