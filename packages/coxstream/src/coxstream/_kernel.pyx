# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Vendored Cython kernel for coxstream: exact Efron streaming accumulation.

Two entry points accumulate the Efron partial-likelihood score and observed
information in a single descending-time pass, with exact tie handling:

  efron_stream_batch_ties     accumulate one chunk; carries the open tie group
                              across chunk boundaries.
  efron_stream_finalize_ties  close the final open tie group after the last
                              chunk.

The p x p accumulators (S2, neg_H, D2) and the (n, p) design matrix are typed as
C-contiguous memoryviews (``double[:, ::1]``) rather than NumPy buffer objects.
The contiguous-memoryview form lets the C compiler prove unit inner stride and
no aliasing, so it auto-vectorises (SIMD) the O(p^2) inner loops that dominate
at large p.

Built automatically by ``pip install coxstream`` (see setup.py).
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport exp as c_exp, log as c_log

cnp.import_array()


cdef inline void _finalize_group(
    double S0,
    double[::1]    S1,
    double[::1, :] S2,
    double[::1]    ll,
    double[::1]    score,
    double[::1, :] neg_H,
    double         D0,
    double[::1]    D1,
    double[::1, :] D2,
    double         Deta,
    double[::1]    Dx,
    double[::1]    s1_d,
    Py_ssize_t     d,
    Py_ssize_t     p,
):
    """Apply one tie group's exact Efron contribution to the carry (in place).

    Mirrors the inner d-loop of ``efron_pass``: S0/S1/S2 are the risk-set sums
    (everyone with time >= this group's time, the group's own rows included),
    D0/D1/D2 the tied-EVENT-only sums, Deta/Dx the summed event eta and x.
    ``s1_d`` is a length-p scratch buffer.  Does nothing for a group with no
    events (d == 0).
    """
    cdef Py_ssize_t l, k, q
    cdef double frac, s0_d, inv, s1q
    if d <= 0:
        return
    ll[0] += Deta
    for k in range(p):
        score[k] += Dx[k]
    for l in range(d):
        frac = <double>l / <double>d
        s0_d = S0 - frac * D0
        if s0_d <= 0.0:
            s0_d = 1e-300
        inv = 1.0 / s0_d
        ll[0] -= c_log(s0_d)
        for k in range(p):
            s1_d[k] = S1[k] - frac * D1[k]
            score[k] -= s1_d[k] * inv
        for q in range(p):
            s1q = s1_d[q] * inv * inv
            for k in range(p):
                neg_H[k, q] += (S2[k, q] - frac * D2[k, q]) * inv - s1_d[k] * s1q


def efron_stream_batch_ties(
    double[::1]    t_b   not None,
    cnp.ndarray[cnp.int32_t, ndim=1] e_b not None,
    double[:, ::1] X_b   not None,
    double[::1]    beta  not None,
    double[::1]    S0_arr not None,
    double[::1]    S1    not None,
    double[::1, :] S2    not None,
    double[::1]    ll_arr not None,
    double[::1]    score not None,
    double[::1, :] neg_H not None,
    double[::1]    D0_arr not None,
    double[::1]    D1    not None,
    double[::1, :] D2    not None,
    double[::1]    Deta_arr not None,
    double[::1]    Dx    not None,
    double[::1]    dcount_arr not None,
    double[::1]    tau_arr not None,
    cnp.ndarray[cnp.int32_t, ndim=1] gopen_arr not None,
    double[::1]    s1_d  not None,
):
    """Accumulate one batch into streaming Efron carry, EXACT tie correction.

    Rows arrive sorted DESCENDING by time (``t_b``).  The risk-set sums
    (S0/S1/S2) and the currently-open tie group's accumulators (D0/D1/D2,
    Deta, Dx, dcount, tau, gopen) all persist across calls, so a tie group
    that straddles a batch boundary is finalized correctly on the next call:
    the group is closed lazily only when a smaller event time is seen.  Call
    ``efron_stream_finalize_ties`` once after the last batch to close the
    final open group.

    No log-sum-exp stabilization (safe when max(|X @ beta|) << 709).  ``s1_d``
    is a length-p scratch buffer.  All carry arrays are modified in place.
    S2 / D2 / neg_H arrive Fortran-contiguous (``[::1, :]``) so the O(p^2)
    inner loops walk the contiguous (k) axis -- the SIMD path shared with the
    no-tie kernel.
    """
    cdef Py_ssize_t n = X_b.shape[0]
    cdef Py_ssize_t p = X_b.shape[1]
    cdef Py_ssize_t i, k, q, d
    cdef double eta_i, ex, xq_ex, t_i

    for i in range(n):
        eta_i = 0.0
        for k in range(p):
            eta_i += X_b[i, k] * beta[k]
        t_i = t_b[i]

        # Time decreased -> the open group is complete (all its rows are
        # already in the risk-set carry).  Finalize it, then reset.
        if gopen_arr[0] == 1 and t_i != tau_arr[0]:
            d = <Py_ssize_t>dcount_arr[0]
            _finalize_group(
                S0_arr[0], S1, S2, ll_arr, score, neg_H,
                D0_arr[0], D1, D2, Deta_arr[0], Dx, s1_d, d, p,
            )
            D0_arr[0] = 0.0
            Deta_arr[0] = 0.0
            dcount_arr[0] = 0.0
            for k in range(p):
                D1[k] = 0.0
                Dx[k] = 0.0
            for q in range(p):
                for k in range(p):
                    D2[k, q] = 0.0
            gopen_arr[0] = 0

        ex = c_exp(eta_i)
        S0_arr[0] += ex
        for k in range(p):
            S1[k] += X_b[i, k] * ex
        for q in range(p):
            xq_ex = X_b[i, q] * ex
            for k in range(p):
                S2[k, q] += X_b[i, k] * xq_ex

        if e_b[i] == 1:
            D0_arr[0] += ex
            for k in range(p):
                D1[k] += X_b[i, k] * ex
                Dx[k] += X_b[i, k]
            for q in range(p):
                xq_ex = X_b[i, q] * ex
                for k in range(p):
                    D2[k, q] += X_b[i, k] * xq_ex
            Deta_arr[0] += eta_i
            dcount_arr[0] += 1.0

        tau_arr[0] = t_i
        gopen_arr[0] = 1


def efron_stream_finalize_ties(
    double[::1]    S0_arr not None,
    double[::1]    S1    not None,
    double[::1, :] S2    not None,
    double[::1]    ll_arr not None,
    double[::1]    score not None,
    double[::1, :] neg_H not None,
    double[::1]    D0_arr not None,
    double[::1]    D1    not None,
    double[::1, :] D2    not None,
    double[::1]    Deta_arr not None,
    double[::1]    Dx    not None,
    double[::1]    dcount_arr not None,
    cnp.ndarray[cnp.int32_t, ndim=1] gopen_arr not None,
    double[::1]    s1_d  not None,
):
    """Close the final open tie group after the last batch (in place)."""
    cdef Py_ssize_t p = S1.shape[0]
    cdef Py_ssize_t d
    if gopen_arr[0] == 1:
        d = <Py_ssize_t>dcount_arr[0]
        _finalize_group(
            S0_arr[0], S1, S2, ll_arr, score, neg_H,
            D0_arr[0], D1, D2, Deta_arr[0], Dx, s1_d, d, p,
        )
        gopen_arr[0] = 0
