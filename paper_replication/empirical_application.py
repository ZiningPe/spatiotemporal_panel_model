"""
paper_replication/empirical_application.py
------------------------------------------
Template for the empirical application section of the paper.

Replace the simulated data below with your actual dataset.
The expected data format is described in the docstring of each section.

Data format expected
--------------------
Y  : (N,) = (n*T,) panel outcome (time-major stacking)
X  : (N, k) regressors (time-major)
coords : (n, 2) geographic coordinates of provinces
gdp_data : (n, T) GDP (or growth rates) for building TWM

Instrument for the entry equation
-----------------------------------
The first-stage instrument Z_W must be an excluded variable that predicts
the temporal weight matrix M (e.g. a national policy index, lagged trade
openness, or weather shocks) but is excluded from the structural equation.
"""

import numpy as np
from spatiotemporal_cf import (
    inverse_distance_weights, compute_morans_i,
    build_twm_from_stats, build_stwm, stwm_summary,
    first_stage, first_stage_stats, aggregate_eps_hat,
    cf_2sls_fit, cf_2sls_summary,
    cf_qmle,
    cf_gmm_fit,
    cf_2sls_avar_corrected,
    multiplier_matrix, ie_inference,
    bh_correction, multiple_testing_summary, print_ie_table,
    summarise_results, ned_test,
)


def load_data():
    """
    Replace this stub with your actual data loading code.

    Returns
    -------
    Y       : (N,) panel outcome, time-major stacking
    X       : (N, k) regressors
    coords  : (n, 2) geographic coordinates
    gdp_ts  : (T,) national aggregate GDP growth rates (for TWM)
    Z_W     : (T, k_Z) first-stage instruments
    n, T    : dimensions
    """
    # ── PLACEHOLDER: generate fake data ──────────────────────────────────
    rng  = np.random.default_rng(0)
    n, T = 30, 8
    N    = n * T

    coords  = rng.uniform(0, 10, (n, 2))
    gdp_ts  = rng.normal(0.07, 0.02, T)
    Z_W     = rng.standard_normal((T, 2))

    X = rng.standard_normal((N, 1))
    Y = rng.standard_normal(N)
    # ─────────────────────────────────────────────────────────────────────
    return Y, X, coords, gdp_ts, Z_W, n, T


def run_empirical():
    Y, X, coords, gdp_ts, Z_W, n, T = load_data()
    N = n * T

    print("=" * 60)
    print("  Empirical Application")
    print(f"  n={n} units, T={T} periods, N={N} observations")
    print("=" * 60)

    # ── 1. Build weight matrices ──────────────────────────────────────────
    W_S = inverse_distance_weights(coords, power=1.0)
    M   = build_twm_from_stats(gdp_ts, method="raw")
    W   = build_stwm(M, W_S)
    info = stwm_summary(W, n, T)
    print(f"\nSTWM: spectral_radius={info['spectral_radius']:.4f}, "
          f"sparsity={info['sparsity']:.3f}")

    # ── 2. First stage ────────────────────────────────────────────────────
    # Entry equation: h_t = Moran's I in period t (or other statistic)
    # Here we use the GDP growth rate as a proxy for the entry equation
    h    = gdp_ts  # replace with your Moran's I or entry statistic
    fs   = first_stage_stats(h, Z_W)
    print(f"\nFirst stage: F={fs['F_stat']:.2f}, R²={fs['R2']:.4f}")
    if fs["F_stat"] < 10:
        print("  WARNING: weak first stage (F < 10)")

    eps_hat     = fs["eps_hat"]
    bar_eps_hat = aggregate_eps_hat(eps_hat, n, T, mode="L_eq_T")

    # ── 3. CF-2SLS ───────────────────────────────────────────────────────
    res_2sls = cf_2sls_fit(Y, X, W, bar_eps_hat)
    cf_2sls_summary(res_2sls)

    # Corrected SE (Ω_A for L=T)
    avar_corr = cf_2sls_avar_corrected(Y, X, W, bar_eps_hat,
                                        eps_hat, Z_W, res_2sls["kappa"],
                                        n, T)
    from spatiotemporal_cf import cf_2sls_se
    se_corr = cf_2sls_se(avar_corr, N)
    print(f"Corrected SE (Ω_A): {np.round(se_corr, 4)}")

    # ── 4. CF-QMLE ───────────────────────────────────────────────────────
    res_qmle = cf_qmle(Y, X, W, bar_eps_hat, M=M, W_S=W_S)
    print(f"\nCF-QMLE: δ̂={res_qmle['delta_hat']:.4f}, "
          f"σ̂²={res_qmle['sigma2_hat']:.4f}")

    # ── 5. CF-GMM ────────────────────────────────────────────────────────
    res_gmm = cf_gmm_fit(Y, X, W, bar_eps_hat, max_iter=2)

    summarise_results(cf2sls=res_2sls, cfqmle=res_qmle, cfgmm=res_gmm)

    # ── 6. Cross-period effects ───────────────────────────────────────────
    T_mat  = multiplier_matrix(res_2sls["kappa"][0], W)
    ie_res = ie_inference(T_mat, res_2sls["kappa"], res_2sls["avar"],
                          W, n, T, N)
    mt     = multiple_testing_summary(ie_res["IE"], ie_res["SE"],
                                       ie_res["p_value"])
    print_ie_table({**ie_res, **mt})

    # ── 7. Residual diagnostics ───────────────────────────────────────────
    kappa = res_2sls["kappa"]
    xi_hat = Y - (
        (W @ Y) * kappa[0]
        + X.ravel() * kappa[1]
        + (W @ X).ravel() * kappa[2]
        + bar_eps_hat * kappa[3]
    )
    ned = ned_test(xi_hat, n, T, W_S)
    print(f"\nNED check: {ned['summary']}")

    return res_2sls, res_qmle, res_gmm, ie_res


if __name__ == "__main__":
    run_empirical()
