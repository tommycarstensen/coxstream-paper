"""Streaming Efron CoxPH accumulation: NumPy, Numba, and JAX backends.

All functions implement the streaming Efron partial log-likelihood accumulation
for one parquet batch.  No-tie assumption applies: synthetic Weibull data has
continuous times so each event is its own tie group of size 1.

No log-sum-exp stabilization: safe when max(|X @ beta|) << 709 (float64
overflow threshold).  For our synthetic data beta stays small at all iterates.

Interface (all backends)
------------------------
batch_fn(e_b, X_b, beta, S0_arr, S1, S2, ll_arr, score, neg_H)

  e_b    : (batch_size,) int32     event indicator (1=event, 0=censored)
  X_b    : (batch_size, p) float64  covariates
  beta   : (p,) float64            current NR iterate
  S0_arr : (1,) float64            S0 carry        [MODIFIED IN PLACE]
  S1     : (p,) float64            S1 carry        [MODIFIED IN PLACE]
  S2     : (p, p) float64          S2 carry        [MODIFIED IN PLACE]
  ll_arr : (1,) float64            ll carry        [MODIFIED IN PLACE]
  score  : (p,) float64            score carry     [MODIFIED IN PLACE]
  neg_H  : (p, p) float64          neg-H carry     [MODIFIED IN PLACE]
"""

import math

import numba
import numpy as np


# ---------------------------------------------------------------------------
# NumPy vectorized (cumulative sums; O(batch_size * p^2) peak RAM per batch)
# ---------------------------------------------------------------------------

def efron_stream_batch_numpy(e_b, X_b, beta, S0_arr, S1, S2, ll_arr, score, neg_H):
    """NumPy vectorized streaming Efron batch accumulation (no-tie mode).

    Uses cumulative sums over the batch to avoid a Python per-row loop.
    Peak RAM per batch: O(batch_size * p^2).  For batch_size=200K, p=10
    this is ~160 MB for the (n, p, p) S2 cumsum array.
    """
    eta = X_b @ beta                                        # (n,)
    w = np.exp(eta)                                         # (n,)
    wX  = w[:, np.newaxis] * X_b                            # (n, p)
    wXX = wX[:, :, np.newaxis] * X_b[:, np.newaxis, :]     # (n, p, p)

    cum_w   = np.cumsum(w)                                  # (n,)
    cum_wX  = np.cumsum(wX,  axis=0)                        # (n, p)
    cum_wXX = np.cumsum(wXX, axis=0)                        # (n, p, p)

    S0_at = S0_arr[0] + cum_w                               # (n,)
    S1_at = S1[np.newaxis, :] + cum_wX                      # (n, p)
    S2_at = S2[np.newaxis, :, :] + cum_wXX                  # (n, p, p)

    ev = e_b == 1
    if ev.any():
        s0_ev  = S0_at[ev]                                  # (n_ev,)
        s0_safe = np.maximum(s0_ev, 1e-300)
        inv_s0  = 1.0 / s0_safe                             # (n_ev,)
        s1_ev   = S1_at[ev]                                 # (n_ev, p)
        s2_ev   = S2_at[ev]                                 # (n_ev, p, p)
        eta_ev  = eta[ev]                                   # (n_ev,)
        X_ev    = X_b[ev]                                   # (n_ev, p)
        s1_norm = s1_ev * inv_s0[:, np.newaxis]             # (n_ev, p)

        ll_arr[0] += float(np.sum(eta_ev) - np.sum(np.log(s0_safe)))
        score += (
            np.sum(X_ev, axis=0)
            - np.sum(s1_norm, axis=0)
        )
        neg_H += (
            np.einsum('ijk,i->jk', s2_ev, inv_s0)
            - np.einsum('ij,ik->jk', s1_norm, s1_norm)
        )

    S0_arr[0] += float(cum_w[-1])
    S1 += cum_wX[-1]
    S2 += cum_wXX[-1]


# ---------------------------------------------------------------------------
# Numba NJIT (sequential per-row loop; O(p^2) peak RAM independent of batch)
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def _efron_stream_batch_njit(
    e_b, X_b, beta, S0_arr, S1, S2, ll_arr, score, neg_H
):
    n, p = X_b.shape
    for i in range(n):
        eta_i = 0.0
        for k in range(p):
            eta_i += X_b[i, k] * beta[k]
        ex = math.exp(eta_i)

        S0_arr[0] += ex
        for k in range(p):
            xk = X_b[i, k]
            S1[k] += xk * ex
            for q in range(p):
                S2[k, q] += xk * X_b[i, q] * ex

        if e_b[i] == 1:
            s0 = S0_arr[0]
            if s0 <= 0.0:
                s0 = 1.0e-300
            inv_s0 = 1.0 / s0
            ll_arr[0] += eta_i - math.log(s0)
            for k in range(p):
                score[k] += X_b[i, k] - S1[k] * inv_s0
                for q in range(p):
                    neg_H[k, q] += (
                        S2[k, q] * inv_s0
                        - S1[k] * S1[q] * inv_s0 * inv_s0
                    )


def efron_stream_batch_numba(e_b, X_b, beta, S0_arr, S1, S2, ll_arr, score, neg_H):
    """Numba NJIT streaming Efron batch accumulation (no-tie mode)."""
    _efron_stream_batch_njit(e_b, X_b, beta, S0_arr, S1, S2, ll_arr, score, neg_H)


def warmup_numba(p):
    """Trigger Numba JIT compilation before timing."""
    _efron_stream_batch_njit(
        np.zeros(4, dtype=np.int32),
        np.zeros((4, p), dtype=np.float64),
        np.zeros(p, dtype=np.float64),
        np.zeros(1, dtype=np.float64),
        np.zeros(p, dtype=np.float64),
        np.zeros((p, p), dtype=np.float64),
        np.zeros(1, dtype=np.float64),
        np.zeros(p, dtype=np.float64),
        np.zeros((p, p), dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# JAX JIT-compiled (XLA; same cumsum approach as NumPy, GPU-capable)
# ---------------------------------------------------------------------------

_JAX_BATCH_FN_CACHE: dict = {}


def _get_jax_batch_fn(p: int):
    """Return a JAX-JIT compiled per-batch function for covariate dimension p."""
    if p in _JAX_BATCH_FN_CACHE:
        return _JAX_BATCH_FN_CACHE[p]

    try:
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            "JAX not installed. Run: pip install 'jax[cpu]'"
        ) from exc

    @jax.jit
    def _batch_fn(e_b, X_b, beta, S0_prev, S1_prev, S2_prev,
                  ll_prev, score_prev, neg_H_prev):
        eta = X_b @ beta
        w   = jnp.exp(eta)
        wX  = w[:, None] * X_b
        wXX = wX[:, :, None] * X_b[:, None, :]

        cum_w   = jnp.cumsum(w)
        cum_wX  = jnp.cumsum(wX,  axis=0)
        cum_wXX = jnp.cumsum(wXX, axis=0)

        S0_at = S0_prev + cum_w
        S1_at = S1_prev[None, :] + cum_wX
        S2_at = S2_prev[None, :, :] + cum_wXX

        ev = (e_b == 1).astype(jnp.float64)
        s0_safe = jnp.where(S0_at > 1e-300, S0_at, jnp.ones_like(S0_at) * 1e-300)
        inv_s0  = 1.0 / s0_safe

        ll_new    = ll_prev + jnp.sum(ev * (eta - jnp.log(s0_safe)))
        s1_norm   = S1_at * inv_s0[:, None]
        score_new = score_prev + jnp.sum(ev[:, None] * (X_b - s1_norm), axis=0)
        neg_H_new = (
            neg_H_prev
            + jnp.einsum('ijk,i->jk', S2_at, ev * inv_s0)
            - jnp.einsum('ij,ik,i->jk', s1_norm, s1_norm, ev)
        )

        S0_new = S0_prev + jnp.sum(w)
        S1_new = S1_prev + jnp.sum(wX, axis=0)
        S2_new = S2_prev + jnp.sum(wXX, axis=0)

        return S0_new, S1_new, S2_new, ll_new, score_new, neg_H_new

    _JAX_BATCH_FN_CACHE[p] = _batch_fn
    return _batch_fn


def efron_stream_batch_jax(e_b, X_b, beta, S0_arr, S1, S2, ll_arr, score, neg_H):
    """JAX JIT-compiled streaming Efron batch accumulation (no-tie mode)."""
    try:
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            "JAX not installed. Run: pip install 'jax[cpu]'"
        ) from exc

    p = X_b.shape[1]
    fn = _get_jax_batch_fn(p)

    s0_n, s1_n, s2_n, ll_n, sc_n, nh_n = fn(
        jnp.array(e_b, dtype=jnp.int32),
        jnp.array(X_b, dtype=jnp.float64),
        jnp.array(beta, dtype=jnp.float64),
        jnp.float64(S0_arr[0]),
        jnp.array(S1, dtype=jnp.float64),
        jnp.array(S2, dtype=jnp.float64),
        jnp.float64(ll_arr[0]),
        jnp.array(score, dtype=jnp.float64),
        jnp.array(neg_H, dtype=jnp.float64),
    )
    S0_arr[0] = float(s0_n)
    S1[:] = np.asarray(s1_n)
    S2[:] = np.asarray(s2_n)
    ll_arr[0] = float(ll_n)
    score[:] = np.asarray(sc_n)
    neg_H[:] = np.asarray(nh_n)


def warmup_jax(p):
    """Trigger JAX JIT compilation before timing."""
    import jax.numpy as jnp
    fn = _get_jax_batch_fn(p)
    fn(
        jnp.zeros(4, dtype=jnp.int32),
        jnp.zeros((4, p), dtype=jnp.float64),
        jnp.zeros(p, dtype=jnp.float64),
        jnp.float64(0.0),
        jnp.zeros(p, dtype=jnp.float64),
        jnp.zeros((p, p), dtype=jnp.float64),
        jnp.float64(0.0),
        jnp.zeros(p, dtype=jnp.float64),
        jnp.zeros((p, p), dtype=jnp.float64),
    )


# ---------------------------------------------------------------------------
# JAX x32 (jax_enable_x64=False; float32 internally; separate cache so x64
# config is never set in a subprocess that only uses this variant)
# ---------------------------------------------------------------------------

_JAX_BATCH_FN_CACHE_X32: dict = {}


def _get_jax_batch_fn_x32(p: int):
    """Return a JAX-JIT compiled per-batch function in float32 for dimension p."""
    if p in _JAX_BATCH_FN_CACHE_X32:
        return _JAX_BATCH_FN_CACHE_X32[p]

    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            "JAX not installed. Run: pip install 'jax[cpu]'"
        ) from exc

    @jax.jit
    def _batch_fn_x32(e_b, X_b, beta, S0_prev, S1_prev, S2_prev,
                      ll_prev, score_prev, neg_H_prev):
        eta = X_b @ beta
        w   = jnp.exp(eta)
        wX  = w[:, None] * X_b
        wXX = wX[:, :, None] * X_b[:, None, :]

        cum_w   = jnp.cumsum(w)
        cum_wX  = jnp.cumsum(wX,  axis=0)
        cum_wXX = jnp.cumsum(wXX, axis=0)

        S0_at = S0_prev + cum_w
        S1_at = S1_prev[None, :] + cum_wX
        S2_at = S2_prev[None, :, :] + cum_wXX

        ev = (e_b == 1).astype(jnp.float32)
        s0_safe = jnp.where(S0_at > jnp.float32(1e-30), S0_at,
                            jnp.ones_like(S0_at) * jnp.float32(1e-30))
        inv_s0  = jnp.float32(1.0) / s0_safe

        ll_new    = ll_prev + jnp.sum(ev * (eta - jnp.log(s0_safe)))
        s1_norm   = S1_at * inv_s0[:, None]
        score_new = score_prev + jnp.sum(ev[:, None] * (X_b - s1_norm), axis=0)
        neg_H_new = (
            neg_H_prev
            + jnp.einsum('ijk,i->jk', S2_at, ev * inv_s0)
            - jnp.einsum('ij,ik,i->jk', s1_norm, s1_norm, ev)
        )

        S0_new = S0_prev + jnp.sum(w)
        S1_new = S1_prev + jnp.sum(wX, axis=0)
        S2_new = S2_prev + jnp.sum(wXX, axis=0)

        return S0_new, S1_new, S2_new, ll_new, score_new, neg_H_new

    _JAX_BATCH_FN_CACHE_X32[p] = _batch_fn_x32
    return _batch_fn_x32


def efron_stream_batch_jax_x32(
    e_b, X_b, beta, S0_arr, S1, S2, ll_arr, score, neg_H
):
    """JAX JIT-compiled streaming Efron batch accumulation in float32 (no x64)."""
    try:
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            "JAX not installed. Run: pip install 'jax[cpu]'"
        ) from exc

    p = X_b.shape[1]
    fn = _get_jax_batch_fn_x32(p)

    s0_n, s1_n, s2_n, ll_n, sc_n, nh_n = fn(
        jnp.array(e_b, dtype=jnp.int32),
        jnp.array(X_b, dtype=jnp.float32),
        jnp.array(beta, dtype=jnp.float32),
        jnp.float32(S0_arr[0]),
        jnp.array(S1, dtype=jnp.float32),
        jnp.array(S2, dtype=jnp.float32),
        jnp.float32(ll_arr[0]),
        jnp.array(score, dtype=jnp.float32),
        jnp.array(neg_H, dtype=jnp.float32),
    )
    S0_arr[0] = float(s0_n)
    S1[:] = np.asarray(s1_n, dtype=np.float64)
    S2[:] = np.asarray(s2_n, dtype=np.float64)
    ll_arr[0] = float(ll_n)
    score[:] = np.asarray(sc_n, dtype=np.float64)
    neg_H[:] = np.asarray(nh_n, dtype=np.float64)


def warmup_jax_x32(p):
    """Trigger float32 JAX JIT compilation before timing."""
    import jax.numpy as jnp
    fn = _get_jax_batch_fn_x32(p)
    fn(
        jnp.zeros(4, dtype=jnp.int32),
        jnp.zeros((4, p), dtype=jnp.float32),
        jnp.zeros(p, dtype=jnp.float32),
        jnp.float32(0.0),
        jnp.zeros(p, dtype=jnp.float32),
        jnp.zeros((p, p), dtype=jnp.float32),
        jnp.float32(0.0),
        jnp.zeros(p, dtype=jnp.float32),
        jnp.zeros((p, p), dtype=jnp.float32),
    )
