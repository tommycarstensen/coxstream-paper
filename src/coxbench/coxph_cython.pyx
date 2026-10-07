# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython AOT Cox PH Efron forward pass.

Same algorithm as numba_coxph._efron_pass: single descending-order pass,
O(p^2) working memory.  Compiled ahead-of-time; no JIT warm-up cost.

Build from src/coxph/:
    python setup_cython.py build_ext --inplace
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport exp as c_exp, log as c_log

cnp.import_array()


def efron_pass(
    cnp.ndarray[cnp.float64_t, ndim=1] t not None,
    cnp.ndarray[cnp.int32_t,   ndim=1] e not None,
    cnp.ndarray[cnp.float64_t, ndim=2] X not None,
    cnp.ndarray[cnp.float64_t, ndim=1] beta not None,
):
    """Efron log-likelihood, score and neg-Hessian via single descending pass.

    Parameters
    ----------
    t    : (n,) float64  event times sorted DESCENDING
    e    : (n,) int32    event indicator (1=event, 0=censored)
    X    : (n, p) float64 covariates sorted DESCENDING by t
    beta : (p,) float64

    Returns
    -------
    ll    : float
    score : (p,) float64
    neg_H : (p, p) float64  positive semi-definite
    """
    cdef Py_ssize_t n = t.shape[0]
    cdef Py_ssize_t p = X.shape[1]
    cdef Py_ssize_t i, j, k, l, d
    cdef int        n_ev, n_events
    cdef double     t_cur, ex, xk, frac, s0_d, inv_s0, s_val, s2kl, ll_val
    cdef double     S0 = 0.0, tS0 = 0.0, eta_max

    cdef cnp.ndarray[cnp.float64_t, ndim=1] xb     = np.empty(n, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] exp_xb = np.empty(n, dtype=np.float64)
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
        s_val = 0.0
        for k in range(p):
            s_val += X[i, k] * beta[k]
        xb[i] = s_val

    eta_max = xb[0]
    for i in range(1, n):
        if xb[i] > eta_max:
            eta_max = xb[i]

    for i in range(n):
        exp_xb[i] = c_exp(xb[i] - eta_max)

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
                frac = <double>d / <double>n_ev
                s0_d = S0 - frac * tS0
                if s0_d <= 0.0:
                    s0_d = 1e-300
                inv_s0 = 1.0 / s0_d
                for k in range(p):
                    s1_d[k] = S1[k] - frac * tS1[k]
                    score_[k] -= s1_d[k] * inv_s0
                ll_val -= c_log(s0_d)
                for k in range(p):
                    for l in range(p):
                        s2kl = S2[k, l] - frac * tS2[k, l]
                        neg_H_[k, l] += (
                            s2kl * inv_s0
                            - s1_d[k] * s1_d[l] * inv_s0 * inv_s0
                        )

    # restore true log-likelihood (stabilized denominators absorbed n_events * eta_max)
    ll_val -= n_events * eta_max
    return ll_val, score_, neg_H_


def efron_stream_batch(
    cnp.ndarray[cnp.int32_t,   ndim=1] e_b   not None,
    cnp.ndarray[cnp.float64_t, ndim=2] X_b   not None,
    cnp.ndarray[cnp.float64_t, ndim=1] beta  not None,
    cnp.ndarray[cnp.float64_t, ndim=1] S0_arr not None,
    cnp.ndarray[cnp.float64_t, ndim=1] S1    not None,
    cnp.ndarray[cnp.float64_t, ndim=2] S2    not None,
    cnp.ndarray[cnp.float64_t, ndim=1] ll_arr not None,
    cnp.ndarray[cnp.float64_t, ndim=1] score not None,
    cnp.ndarray[cnp.float64_t, ndim=2] neg_H not None,
):
    """Accumulate one batch into streaming Efron carry (in-place, no-tie mode).

    No log-sum-exp stabilization: safe when max(|X @ beta|) << 709.
    Modifies S0_arr[0], S1, S2, ll_arr[0], score, neg_H in-place.
    Assumes no ties: each event is its own tie group of size 1.
    """
    cdef Py_ssize_t n = X_b.shape[0]
    cdef Py_ssize_t p = X_b.shape[1]
    cdef Py_ssize_t i, k, q
    cdef double eta_i, ex, s0, inv_s0, xk

    for i in range(n):
        eta_i = 0.0
        for k in range(p):
            eta_i += X_b[i, k] * beta[k]
        ex = c_exp(eta_i)

        S0_arr[0] += ex
        for k in range(p):
            xk = X_b[i, k]
            S1[k] += xk * ex
            for q in range(p):
                S2[k, q] += xk * X_b[i, q] * ex

        if e_b[i] == 1:
            s0 = S0_arr[0]
            if s0 <= 0.0:
                s0 = 1e-300
            inv_s0 = 1.0 / s0
            ll_arr[0] += eta_i - c_log(s0)
            for k in range(p):
                score[k] += X_b[i, k] - S1[k] * inv_s0
                for q in range(p):
                    neg_H[k, q] += (
                        S2[k, q] * inv_s0
                        - S1[k] * S1[q] * inv_s0 * inv_s0
                    )
