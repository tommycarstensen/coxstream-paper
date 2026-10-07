# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython AOT Cox PH Efron forward pass -- float32 data variant.

Input arrays t, X, beta, xb, exp_xb are float32: halves peak RSS for the
data and working vectors (saves ~512 MB at n=32M, p=2) vs the float64 pass.

Running accumulators S0, S1, S2 and all Efron correction temporaries are
float64: at n=32M the float32 ULP at ~16M equals 2, so contributions
of exp_xb~1 would be silently dropped, corrupting the Hessian.  Keeping
accumulators in float64 retains full numerical stability at no RAM cost
(S0/S1/S2 are O(p^2) scalars, negligible vs the n-row arrays).

Build from src/coxph/:
    python setup_cython.py build_ext --inplace
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport expf as c_expf, log as c_log

cnp.import_array()


def efron_pass(
    cnp.ndarray[cnp.float32_t, ndim=1] t not None,
    cnp.ndarray[cnp.int32_t,   ndim=1] e not None,
    cnp.ndarray[cnp.float32_t, ndim=2] X not None,
    cnp.ndarray[cnp.float32_t, ndim=1] beta not None,
):
    """Efron log-likelihood, score and neg-Hessian via single descending pass.

    Float32 variant: t, X, beta are float32; ll is float64; score/neg_H
    are returned as float64 for compatibility with the NR step.

    Parameters
    ----------
    t    : (n,) float32  event times sorted DESCENDING
    e    : (n,) int32    event indicator (1=event, 0=censored)
    X    : (n, p) float32 covariates sorted DESCENDING by t
    beta : (p,) float32

    Returns
    -------
    ll    : float (float64)
    score : (p,) float64
    neg_H : (p, p) float64  positive semi-definite
    """
    cdef Py_ssize_t n = t.shape[0]
    cdef Py_ssize_t p = X.shape[1]
    cdef Py_ssize_t i, j, k, l, d
    cdef int        n_ev, n_events
    # float32 temps: only for reading float32 inputs (t, X, beta, xb, exp_xb)
    cdef float      t_cur, ex, xk, frac, s_val_f
    cdef float      eta_max
    # double temps: running accumulators and per-event Efron corrections.
    # S0/S1/S2 must be float64: at n=32M the float32 ULP at ~16M is 2, so
    # additions of exp_xb~1 get rounded away and S1 stops growing correctly.
    cdef double     s0_d, inv_s0, s2kl
    cdef double     S0 = 0.0, tS0 = 0.0
    cdef double     ll_val

    cdef cnp.ndarray[cnp.float32_t, ndim=1] xb     = np.empty(n, dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] exp_xb = np.empty(n, dtype=np.float32)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] score_ = np.zeros(p, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] neg_H_ = np.zeros((p, p), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] S1     = np.zeros(p, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] S2     = np.zeros((p, p), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] tS1    = np.zeros(p, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] tS2    = np.zeros((p, p), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] s1_d   = np.empty(p, dtype=np.float64)

    ll_val = 0.0
    n_events = 0

    # log-sum-exp stabilization: subtract global max before exp to prevent overflow
    for i in range(n):
        s_val_f = 0.0
        for k in range(p):
            s_val_f += X[i, k] * beta[k]
        xb[i] = s_val_f

    eta_max = xb[0]
    for i in range(1, n):
        if xb[i] > eta_max:
            eta_max = xb[i]

    for i in range(n):
        exp_xb[i] = c_expf(xb[i] - eta_max)

    i = 0
    while i < n:
        t_cur = t[i]
        n_ev = 0
        tS0 = 0.0
        for k in range(p):
            tS1[k] = 0.0
            for l in range(p):
                tS2[k, l] = 0.0

        while i < n and t[i] == t_cur:
            ex = exp_xb[i]
            S0 += ex
            for k in range(p):
                xk = X[i, k]
                S1[k] += xk * ex
                for l in range(p):
                    S2[k, l] += xk * X[i, l] * ex
            if e[i] == 1:
                n_ev += 1
                tS0 += ex
                for k in range(p):
                    xk = X[i, k]
                    tS1[k] += xk * ex
                    for l in range(p):
                        tS2[k, l] += xk * X[i, l] * ex
                ll_val += xb[i]
                for k in range(p):
                    score_[k] += X[i, k]
            i += 1

        if n_ev > 0:
            n_events += n_ev
            for d in range(n_ev):
                frac = <float>d / <float>n_ev
                s0_d = S0 - frac * tS0
                if s0_d <= 0.0:
                    s0_d = 1e-300
                inv_s0 = 1.0 / s0_d
                for k in range(p):
                    s1_d[k] = S1[k] - frac * tS1[k]
                    score_[k] -= s1_d[k] * inv_s0
                ll_val -= c_log(s0_d) - eta_max
                for k in range(p):
                    for l in range(p):
                        s2kl = S2[k, l] - frac * tS2[k, l]
                        neg_H_[k, l] += (
                            s2kl * inv_s0
                            - s1_d[k] * s1_d[l] * inv_s0 * inv_s0
                        )

    ll_val -= n_events * eta_max
    return ll_val, score_, neg_H_
