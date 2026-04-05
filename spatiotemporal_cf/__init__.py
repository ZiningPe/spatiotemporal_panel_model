"""
spatiotemporal_cf
=================
Control-Function estimators for Spatial Durbin Models with endogenous
Spatio-Temporal Weight Matrices (STWM).

Integrates and extends the ZiningPe/STWM package with:
  - CF-2SLS, CF-QMLE, CF-GMM estimators
  - Kronecker-structure weight matrix utilities
  - Cross-period indirect effects with Delta-method SE
  - Benjamini-Hochberg FDR correction
  - Ω_A variance correction for generated regressors (L = T case)

Quick start
-----------
>>> import numpy as np
>>> from spatiotemporal_cf import (
...     rook_weights, build_twm_parametric, build_stwm,
...     first_stage, aggregate_eps_hat,
...     cf_2sls_fit, cf_qmle, cf_gmm_fit,
...     cross_period_effects_matrix, bh_correction,
... )

References
----------
Wang, Z. (2025). Endogenous Spatio-Temporal Weight Matrices and the
Control-Function Approach. Working paper.
"""

from .weight_construction import (
    # Spatial weight matrices
    rook_weights,
    queen_weights,
    inverse_distance_weights,
    knn_weights,
    # Temporal statistics
    compute_morans_i,
    compute_gearys_c,
    compute_getis_ord_g,
    compute_spatial_gini,
    # Temporal weight matrix builders
    build_twm_from_stats,
    build_twm_parametric,
    # Full STWM assembler
    build_stwm,
    stwm_summary,
    # Kronecker eigenvalue utilities
    eigvals_kronecker,
    logdet_kronecker,
    delta_admissible_range,
)

from .first_stage import (
    first_stage,
    first_stage_stats,
    projection_matrix,
)

from .control_function import (
    aggregate_eps_hat,
    aggregation_matrix,
    detect_mode,
)

from .cf_2sls import (
    cf_2sls,
    cf_2sls_avar,
    cf_2sls_se,
    cf_2sls_fit,
    cf_2sls_summary,
)

from .cf_qmle import (
    logdet_kronecker as qmle_logdet_kronecker,
    logdet_full,
    profile_loglik,
    cf_qmle,
    cf_qmle_avar,
    qmle_static,
)

from .cf_gmm import (
    moment_vector,
    cf_gmm,
    cf_gmm_avar,
    cf_gmm_fit,
)

from .inference import (
    multiplier_matrix,
    multiplier_matrix_approx,
    cross_period_effect,
    cross_period_effects_matrix,
    delta_method_se,
    ie_inference,
    bh_correction,
    multiple_testing_summary,
    print_ie_table,
)

from .variance_correction import (
    omega_A,
    omega_A_simple,
    compute_sigma2_eps,
    cf_2sls_avar_corrected,
)

from .utils import (
    ned_test,
    eigs_sorted,
    spectral_radius,
    check_admissibility,
    to_time_major,
    to_unit_major,
    panel_demean,
    matrix_condition,
    check_instrument_strength,
    summarise_results,
)

__version__ = "0.1.0"
__author__  = "Wang Zining"
__all__ = [
    # weight construction
    "rook_weights", "queen_weights", "inverse_distance_weights", "knn_weights",
    "compute_morans_i", "compute_gearys_c", "compute_getis_ord_g",
    "compute_spatial_gini", "build_twm_from_stats", "build_twm_parametric",
    "build_stwm", "stwm_summary", "eigvals_kronecker", "logdet_kronecker",
    "delta_admissible_range",
    # first stage
    "first_stage", "first_stage_stats", "projection_matrix",
    # control function
    "aggregate_eps_hat", "aggregation_matrix", "detect_mode",
    # CF-2SLS
    "cf_2sls", "cf_2sls_avar", "cf_2sls_se", "cf_2sls_fit", "cf_2sls_summary",
    # CF-QMLE
    "logdet_full", "profile_loglik", "cf_qmle", "cf_qmle_avar", "qmle_static",
    # CF-GMM
    "moment_vector", "cf_gmm", "cf_gmm_avar", "cf_gmm_fit",
    # inference
    "multiplier_matrix", "multiplier_matrix_approx",
    "cross_period_effect", "cross_period_effects_matrix",
    "delta_method_se", "ie_inference", "bh_correction",
    "multiple_testing_summary", "print_ie_table",
    # variance correction
    "omega_A", "omega_A_simple", "compute_sigma2_eps", "cf_2sls_avar_corrected",
    # utils
    "ned_test", "eigs_sorted", "spectral_radius", "check_admissibility",
    "to_time_major", "to_unit_major", "panel_demean",
    "matrix_condition", "check_instrument_strength", "summarise_results",
]
