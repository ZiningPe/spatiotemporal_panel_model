# spatiotemporal_panel_model

**Control-Function estimators for Spatial Durbin Models with endogenous Spatio-Temporal Weight Matrices (STWM)**

Implements and extends the methodology of *Wang (2025): "Endogenous Spatio-Temporal Weight Matrices and the Control-Function Approach"*, integrating with the [ZiningPe/STWM](https://github.com/ZiningPe/STWM) package for weight-matrix construction.

---

## Core features

| Feature | Module |
|---------|--------|
| `W = M ⊗ W_S` Kronecker weight matrix | `weight_construction` |
| Custom non-Kronecker W (user-supplied) | all estimators |
| First-stage OLS, F-stat, R² | `first_stage` |
| Aggregation A: L=T or L=N | `control_function` |
| CF-2SLS + sandwich SE | `cf_2sls` |
| CF-QMLE, fast Kronecker log-det | `cf_qmle` |
| CF-GMM, optimal weighting | `cf_gmm` |
| Cross-period effects IE_{t←s}, Delta method SE | `inference` |
| Benjamini-Hochberg FDR control | `inference` |
| Ω_A variance correction (L=T) | `variance_correction` |
| NED diagnostic, panel reshape, condition checks | `utils` |

---

## Installation

```bash
pip install git+https://github.com/YOUR_USERNAME/spatiotemporal_panel_model.git
```

Or install locally:

```bash
cd spatiotemporal_panel_model
pip install -e .
```

**Requirements:** Python ≥ 3.9, NumPy ≥ 1.24, SciPy ≥ 1.10

---

## Quick start

```python
import numpy as np
from spatiotemporal_panel_model import (
    rook_weights, compute_morans_i,
    build_twm_from_stats, build_stwm,
    first_stage, aggregate_eps_hat,
    cf_2sls_fit, cf_2sls_summary,
    cf_qmle, cf_gmm_fit,
    multiplier_matrix, ie_inference,
    bh_correction, print_ie_table,
)

# 1. Build weight matrices
W_S    = rook_weights(n_side=5)          # 25×25 spatial W
morans = [compute_morans_i(y_t, W_S) for y_t in annual_data]
M      = build_twm_from_stats(morans)    # 6×6 temporal M from Moran's I
W      = build_stwm(M, W_S)             # 150×150 full STWM (Kronecker)

# 2. First stage  h = Z_W π + ε
eps_hat, pi_hat = first_stage(h, Z_W)
bar_eps_hat     = aggregate_eps_hat(eps_hat, n=25, T=6, mode="L_eq_T")

# 3. Estimate
res_2sls = cf_2sls_fit(Y, X, W, bar_eps_hat)
cf_2sls_summary(res_2sls)

res_qmle = cf_qmle(Y, X, W, bar_eps_hat, M=M, W_S=W_S)
print(f"CF-QMLE δ̂ = {res_qmle['delta_hat']:.4f}")

res_gmm  = cf_gmm_fit(Y, X, W, bar_eps_hat)

# 4. Cross-period effects with FDR correction
T_mat  = multiplier_matrix(res_2sls["kappa"][0], W)
ie_res = ie_inference(T_mat, res_2sls["kappa"], res_2sls["avar"], W, n=25, TT=6, N=150)
_, p_bh = bh_correction(ie_res["p_value"], alpha=0.05)
print_ie_table({**ie_res, "p_bh": p_bh, "reject_bh": p_bh < 0.05})
```

---

## Model

Structural equation:

```
Y = δ·WY + X·β + WX·θ + ε̄̂·δ_c + ξ
```

where `W = M ⊗ W_S` and `ε̄̂ = A·ε̂` is the aggregated control function
from the first-stage residuals.

### Aggregation matrix A

| L | A | ε̄̂_p |
|---|---|------|
| L = T | I_T ⊗ ι_n | ε̂_{t(p)} — unit p inherits period t residual |
| L = N | I_N | ε̂_p — unit-level first stage |

---

## Estimators

### CF-2SLS

Instruments: `Q = [X, WX, W²X, ε̄̂]`

```python
from spatiotemporal_panel_model import cf_2sls_fit, cf_2sls_summary
res = cf_2sls_fit(Y, X, W, bar_eps_hat)
cf_2sls_summary(res)
```

With Ω_A variance correction (L = T):

```python
from spatiotemporal_panel_model import cf_2sls_avar_corrected
avar_corr = cf_2sls_avar_corrected(Y, X, W, bar_eps_hat, eps_hat, Z_W,
                                    res["kappa"], n, T)
```

### CF-QMLE

Kronecker log-det: `ln|I − δW| = Σ_j Σ_k ln|1 − δ·μ_j·λ_k|`

```python
from spatiotemporal_panel_model import cf_qmle
res = cf_qmle(Y, X, W, bar_eps_hat, M=M, W_S=W_S)   # Kronecker (fast)
res = cf_qmle(Y, X, W, bar_eps_hat)                  # full eigenvalue (custom W)
```

### CF-GMM

```python
from spatiotemporal_panel_model import cf_gmm_fit
res = cf_gmm_fit(Y, X, W, bar_eps_hat, max_iter=2)
```

---

## Temporal weight matrix options

```python
from spatiotemporal_panel_model import (
    build_twm_from_stats,   # from Moran's I / Geary's C / Getis-Ord / Gini
    build_twm_parametric,   # exponential / power / linear decay
)

# Data-driven
M = build_twm_from_stats(morans, method="moran")

# Parametric (e.g. exponential decay with ρ=0.6)
M = build_twm_parametric(T, rho=0.6, form="exponential")
```

---

## Cross-period effects & FDR

```python
from spatiotemporal_panel_model import (
    multiplier_matrix, ie_inference,
    bh_correction, multiple_testing_summary, print_ie_table,
)

T_mat = multiplier_matrix(delta_hat, W)
ie    = ie_inference(T_mat, kappa, avar, W, n, T, N)
mt    = multiple_testing_summary(ie["IE"], ie["SE"], ie["p_value"])
print_ie_table({**ie, **mt})
```

---

## Package structure

```
spatiotemporal_panel_model/
├── spatiotemporal_panel_model/
│   ├── weight_construction.py   # W = M ⊗ W_S, Moran/parametric TWM
│   ├── first_stage.py           # OLS h = Z_W π + ε → ε̂
│   ├── control_function.py      # ε̄̂ = A ε̂ (L=T or L=N)
│   ├── cf_2sls.py               # CF-2SLS estimator + sandwich SE
│   ├── cf_qmle.py               # CF-QMLE + Kronecker log-det
│   ├── cf_gmm.py                # CF-GMM + optimal weighting
│   ├── inference.py             # IE_{t←s}, Delta method, BH-FDR
│   ├── variance_correction.py   # Ω_A correction (L=T)
│   └── utils.py                 # NED test, panel reshape, diagnostics
├── examples/
│   ├── moran_example.py         # Moran's I TWM end-to-end
│   └── gdp_distance_example.py  # GDP-distance TWM (provincial panel)
├── tests/
│   └── test_estimators.py       # 25+ unit tests (pytest)
└── paper_replication/
    ├── monte_carlo.py           # Table 1–3 replication
    └── empirical_application.py # Empirical template
```

---

## Citation

```bibtex
@software{spatiotemporal_panel_model2025,
  author  = {Wang, Zining},
  title   = {spatiotemporal\_cf: Control-Function Estimators for
             Spatial Durbin Models with Endogenous STWM},
  year    = {2025},
  url     = {https://github.com/YOUR_USERNAME/spatiotemporal_panel_model}
}
```

Also cite the underlying STWM package:

```bibtex
@software{stwm2025,
  author = {Zining, Pe},
  title  = {{STWM}: Spatial-Temporal Weight Matrix for Panel Econometrics},
  year   = {2025},
  url    = {https://github.com/ZiningPe/STWM}
}
```
