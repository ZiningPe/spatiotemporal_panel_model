"""
paper_replication/monte_carlo.py
---------------------------------
Monte Carlo replication of Table 1–3 from the paper.

Experiment 1 : CF-QMLE consistency  (δ̂ bias & RMSE as n → ∞)
Experiment 2 : Asymptotic normality of sqrt(N)(δ̂ − δ₀)
Experiment 3 : CF-2SLS, CF-GMM vs unadjusted 2SLS (endogenous W)

Usage
-----
    python monte_carlo.py --exp all --reps 500 --seed 42

This module replicates the simulations using the gstw_pdm package
instead of the standalone script, so the package API is exercised.
"""

import argparse
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

from gstw_pdm import (
    rook_weights, build_twm_parametric, build_stwm,
    first_stage, aggregate_eps_hat,
    cf_2sls, cf_2sls_avar, cf_2sls_se,
    cf_qmle, qmle_static,
    cf_gmm,
)


# ─────────────────────────────────────────────────────────────────────────────
# DGP
# ─────────────────────────────────────────────────────────────────────────────

def simulate_dgp(n_side, T, delta0, beta0, theta0, sigma_xi2,
                 delta_c=0.0, rho_M=0.6, seed=None):
    rng    = np.random.default_rng(seed)
    n      = n_side ** 2
    N      = n * T

    W_S = rook_weights(n_side)
    M   = build_twm_parametric(T, rho=rho_M)
    W   = build_stwm(M, W_S)

    X    = rng.standard_normal((N, 1))
    Z_W  = rng.standard_normal((T, 1))
    eps  = rng.standard_normal(T)
    h    = Z_W.ravel() + eps

    bar_eps_true = np.repeat(eps, n)
    xi  = rng.standard_normal(N) * np.sqrt(sigma_xi2)
    V   = delta_c * bar_eps_true + xi

    S   = np.eye(N) - delta0 * W
    rhs = X.ravel() * beta0 + (W @ X).ravel() * theta0 + V
    Y   = np.linalg.solve(S, rhs)

    return Y, X, W, W_S, M, h, Z_W, eps, xi


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1: QMLE consistency
# ─────────────────────────────────────────────────────────────────────────────

def experiment_1(n_sides=(5, 10, 15, 20), T=5, reps=500,
                 delta0=0.3, beta0=1.0, theta0=0.5, sigma_xi2=1.0,
                 rho_M=0.6, seed=0):
    print("\n" + "=" * 70)
    print("Experiment 1: CF-QMLE consistency vs static-W QMLE (exogenous W)")
    print(f"  δ₀={delta0}, T={T}, reps={reps}")
    print("=" * 70)
    print(f"{'n':>6} {'N':>6} | {'CF-QMLE Bias':>14} {'RMSE':>8}"
          f" | {'Static Bias':>12} {'RMSE':>8}")
    print("-" * 65)

    rng_m = np.random.default_rng(seed)
    results = {}

    for n_side in n_sides:
        n  = n_side ** 2
        N  = n * T
        seeds = rng_m.integers(0, 2**31, size=reps)

        W_S    = rook_weights(n_side)
        M      = build_twm_parametric(T, rho=rho_M)
        W      = build_stwm(M, W_S)
        W_st   = np.kron(np.eye(T), W_S)
        eigW_st = np.linalg.eigvals(W_st).real

        d_cf  = np.full(reps, np.nan)
        d_sta = np.full(reps, np.nan)

        for r in range(reps):
            try:
                Y, X, _, _, _, h, Z_W, eps, _ = simulate_dgp(
                    n_side, T, delta0, beta0, theta0, sigma_xi2,
                    delta_c=0.0, rho_M=rho_M, seed=int(seeds[r])
                )
                eps_hat, _    = first_stage(h, Z_W)
                bar_eps_hat   = aggregate_eps_hat(eps_hat, n, T)
                res           = cf_qmle(Y, X, W, bar_eps_hat, M=M, W_S=W_S)
                d_cf[r]       = res["delta_hat"]
                d_sta[r]      = qmle_static(Y, X, W_st, eigW_st)
            except Exception:
                pass

        bias_cf  = float(np.nanmean(d_cf - delta0))
        rmse_cf  = float(np.sqrt(np.nanmean((d_cf - delta0) ** 2)))
        bias_sta = float(np.nanmean(d_sta - delta0))
        rmse_sta = float(np.sqrt(np.nanmean((d_sta - delta0) ** 2)))

        results[n] = dict(bias_cf=bias_cf, rmse_cf=rmse_cf,
                          bias_sta=bias_sta, rmse_sta=rmse_sta)
        print(f"{n:>6} {N:>6} | {bias_cf:>14.4f} {rmse_cf:>8.4f}"
              f" | {bias_sta:>12.4f} {rmse_sta:>8.4f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2: Asymptotic normality
# ─────────────────────────────────────────────────────────────────────────────

def experiment_2(n_side=20, T=5, reps=1000,
                 delta0=0.3, beta0=1.0, theta0=0.5, sigma_xi2=1.0,
                 rho_M=0.6, seed=1):
    n = n_side ** 2
    N = n * T
    print("\n" + "=" * 70)
    print(f"Experiment 2: Asymptotic normality  n={n}, N={N}")
    print("=" * 70)

    W_S  = rook_weights(n_side)
    M    = build_twm_parametric(T, rho=rho_M)
    W    = build_stwm(M, W_S)
    rng_m = np.random.default_rng(seed)
    seeds = rng_m.integers(0, 2**31, size=reps)

    d_hat = np.full(reps, np.nan)
    for r in range(reps):
        try:
            Y, X, _, _, _, h, Z_W, eps, _ = simulate_dgp(
                n_side, T, delta0, beta0, theta0, sigma_xi2,
                delta_c=0.0, rho_M=rho_M, seed=int(seeds[r])
            )
            eps_hat, _  = first_stage(h, Z_W)
            bar_eps_hat = aggregate_eps_hat(eps_hat, n, T)
            res         = cf_qmle(Y, X, W, bar_eps_hat, M=M, W_S=W_S)
            d_hat[r]    = res["delta_hat"]
        except Exception:
            pass
        if (r + 1) % 200 == 0:
            print(f"  {r+1}/{reps} ...")

    std    = float(np.nanstd(d_hat))
    z_stat = (d_hat - delta0) / std
    emp_sz = float(np.nanmean(np.abs(z_stat) > 1.96))
    print(f"\n  Bias  = {np.nanmean(d_hat - delta0):.4f}")
    print(f"  Std   = {std:.4f}")
    print(f"  Empirical size (5% nominal) = {emp_sz:.3f}")
    return dict(delta_hat=d_hat, z_stats=z_stat, emp_size=emp_sz)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 3: Endogenous W — CF-2SLS, CF-GMM vs unadjusted 2SLS
# ─────────────────────────────────────────────────────────────────────────────

def experiment_3(n_sides=(5, 10, 15, 20), T=5, reps=500,
                 delta0=0.3, beta0=1.0, theta0=0.5, sigma_xi2=1.0,
                 delta_c=0.5, rho_M=0.6, seed=2):
    print("\n" + "=" * 80)
    print(f"Experiment 3: Endogenous W  δ_c={delta_c}, δ₀={delta0}")
    print("=" * 80)
    hdr = (f"{'n':>6} {'N':>6} | {'Unadj Bias':>11} {'RMSE':>7}"
           f" | {'CF-2SLS Bias':>12} {'RMSE':>7}"
           f" | {'CF-GMM Bias':>11} {'RMSE':>7}"
           f" | {'2SLS Cov':>8} {'GMM Cov':>8}")
    print(hdr); print("-" * len(hdr))

    rng_m   = np.random.default_rng(seed)
    results = {}

    for n_side in n_sides:
        n  = n_side ** 2
        N  = n * T
        seeds = rng_m.integers(0, 2**31, size=reps)

        W_S = rook_weights(n_side)
        M   = build_twm_parametric(T, rho=rho_M)
        W   = build_stwm(M, W_S)

        d_un   = np.full(reps, np.nan)
        d_2sls = np.full(reps, np.nan)
        d_gmm  = np.full(reps, np.nan)
        cov_2sls = np.full(reps, False)
        cov_gmm  = np.full(reps, False)

        for r in range(reps):
            try:
                Y, X, _, _, _, h, Z_W, eps, _ = simulate_dgp(
                    n_side, T, delta0, beta0, theta0, sigma_xi2,
                    delta_c=delta_c, rho_M=rho_M, seed=int(seeds[r])
                )

                # Unadjusted 2SLS (no CF)
                from scipy import linalg
                WY  = W @ Y; WX = W @ X; W2X = W @ WX
                U_un = np.column_stack([WY[:, None], X, WX])
                Q_un = np.column_stack([X, WX, W2X])
                QtQ_inv = linalg.pinv(Q_un.T @ Q_un)
                PQU = Q_un @ (QtQ_inv @ (Q_un.T @ U_un))
                k_un = linalg.solve(PQU.T @ U_un, PQU.T @ Y)
                d_un[r] = k_un[0]

                # CF-2SLS
                eps_hat, _  = first_stage(h, Z_W)
                beh = aggregate_eps_hat(eps_hat, n, T)
                k2  = cf_2sls(Y, X, W, beh)
                d_2sls[r] = k2[0]
                avar2 = cf_2sls_avar(Y, X, W, beh, k2)
                se2   = cf_2sls_se(avar2, N)[0]
                cov_2sls[r] = abs(k2[0] - delta0) <= 1.96 * se2

                # CF-GMM
                kg, _, _ = cf_gmm(Y, X, W, beh, max_iter=2)
                d_gmm[r] = kg[0]
                cov_gmm[r] = abs(kg[0] - delta0) <= 1.96 * se2   # proxy SE

            except Exception:
                pass

        r_dict = dict(
            bias_un   = float(np.nanmean(d_un - delta0)),
            rmse_un   = float(np.sqrt(np.nanmean((d_un - delta0) ** 2))),
            bias_2sls = float(np.nanmean(d_2sls - delta0)),
            rmse_2sls = float(np.sqrt(np.nanmean((d_2sls - delta0) ** 2))),
            bias_gmm  = float(np.nanmean(d_gmm - delta0)),
            rmse_gmm  = float(np.sqrt(np.nanmean((d_gmm - delta0) ** 2))),
            cov_2sls  = float(np.nanmean(cov_2sls)),
            cov_gmm   = float(np.nanmean(cov_gmm)),
        )
        results[n] = r_dict
        print(f"{n:>6} {N:>6} | {r_dict['bias_un']:>11.4f} {r_dict['rmse_un']:>7.4f}"
              f" | {r_dict['bias_2sls']:>12.4f} {r_dict['rmse_2sls']:>7.4f}"
              f" | {r_dict['bias_gmm']:>11.4f} {r_dict['rmse_gmm']:>7.4f}"
              f" | {r_dict['cov_2sls']:>8.3f} {r_dict['cov_gmm']:>8.3f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo replication")
    parser.add_argument("--exp",  default="all", choices=["1", "2", "3", "all"])
    parser.add_argument("--reps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.exp in ("1", "all"):
        experiment_1(reps=args.reps, seed=args.seed)
    if args.exp in ("2", "all"):
        experiment_2(reps=args.reps, seed=args.seed + 1)
    if args.exp in ("3", "all"):
        experiment_3(reps=args.reps, seed=args.seed + 2)
