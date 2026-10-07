"""JAX JIT-compiled Efron CoxPH forward pass.

Ports coxph_cython.pyx exactly: single descending-order pass, O(p^2) working
memory, Efron tie correction.

Usage
-----
from coxbench.coxph_jax import efron_pass, warmup

warmup(p=2)          # compile before timing
ll, score, neg_H = efron_pass(t, e, X, beta)
"""

from __future__ import annotations

from typing import Callable

import numpy as np

_EFRON_PASS_CACHE: dict[int, Callable] = {}


def _make_efron_pass(p: int) -> Callable:
    """Build and JIT-compile efron_pass for covariate dimension p."""
    try:
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            "JAX not installed. Run: pip install 'jax[cpu]'"
        ) from exc

    def _efron_pass_jax(t, e, X, beta):
        n = t.shape[0]

        xb = X @ beta
        eta_max = jnp.max(xb)
        exp_xb = jnp.exp(xb - eta_max)

        # carry = (i, S0, S1, S2, ll, score, neg_H, n_events)
        # Each scan step processes one row; tie-group flush happens when
        # t[i+1] < t[i] (i.e. the time is about to change) or at the last row.
        # We keep a running "pending tie group" in tS0/tS1/tS2/n_ev in the
        # carry as well.
        #
        # Full carry layout (all jax arrays, must be static-shape):
        #   S0      : scalar
        #   S1      : (p,)
        #   S2      : (p, p)
        #   tS0     : scalar  -- running tie-group risk sums
        #   tS1     : (p,)
        #   tS2     : (p, p)
        #   n_ev    : scalar int  -- events in current tie group
        #   ll      : scalar
        #   score   : (p,)
        #   neg_H   : (p, p)
        #   n_events: scalar int  -- total events (for eta_max correction)

        def flush_group(carry_inner):
            """Apply Efron correction for the current tie group."""
            S0, S1, S2, tS0, tS1, tS2, n_ev, ll, score, neg_H, n_events = (
                carry_inner
            )

            def efron_step(d, acc):
                ll_a, score_a, neg_H_a = acc
                frac = d / n_ev
                s0_d = jnp.maximum(S0 - frac * tS0, 1e-300)
                inv_s0 = 1.0 / s0_d
                s1_d = S1 - frac * tS1
                s2_d = S2 - frac * tS2
                ll_a = ll_a - jnp.log(s0_d)
                score_a = score_a - s1_d * inv_s0
                neg_H_a = (
                    neg_H_a
                    + s2_d * inv_s0
                    - jnp.outer(s1_d, s1_d) * (inv_s0 * inv_s0)
                )
                return ll_a, score_a, neg_H_a

            ll, score, neg_H = jax.lax.fori_loop(
                0, n_ev, efron_step, (ll, score, neg_H)
            )
            n_events = n_events + n_ev
            tS0_new = jnp.zeros(())
            tS1_new = jnp.zeros((p,))
            tS2_new = jnp.zeros((p, p))
            n_ev_new = jnp.zeros((), dtype=jnp.int32)
            return (
                S0, S1, S2,
                tS0_new, tS1_new, tS2_new, n_ev_new,
                ll, score, neg_H, n_events,
            )

        def no_flush(carry_inner):
            return carry_inner

        def scan_step(carry, i):
            S0, S1, S2, tS0, tS1, tS2, n_ev, ll, score, neg_H, n_events = (
                carry
            )

            xi = X[i]
            ex = exp_xb[i]
            ei = e[i]
            xbi = xb[i]

            # Accumulate into running risk-set sums (always).
            S0 = S0 + ex
            S1 = S1 + xi * ex
            S2 = S2 + jnp.outer(xi, xi) * ex

            # Accumulate into tie-group sums for event rows only.
            is_event = ei == 1
            tS0 = tS0 + jnp.where(is_event, ex, 0.0)
            tS1 = tS1 + jnp.where(is_event, xi * ex, jnp.zeros((p,)))
            tS2 = tS2 + jnp.where(
                is_event, jnp.outer(xi, xi) * ex, jnp.zeros((p, p))
            )
            n_ev = n_ev + jnp.where(is_event, 1, 0).astype(jnp.int32)
            ll = ll + jnp.where(is_event, xbi, 0.0)
            score = score + jnp.where(is_event, xi, jnp.zeros((p,)))

            # Flush when the next row has a strictly smaller time value
            # (descending sort: next time < current time means group ends),
            # OR when we are at the last row.
            is_last = i == n - 1
            next_t = jnp.where(is_last, -jnp.inf, t[jnp.minimum(i + 1, n - 1)])
            time_changes = next_t < t[i]
            should_flush = (time_changes | is_last) & (n_ev > 0)

            carry_inner = (
                S0, S1, S2, tS0, tS1, tS2, n_ev, ll, score, neg_H, n_events
            )
            carry_inner = jax.lax.cond(
                should_flush, flush_group, no_flush, carry_inner
            )
            return carry_inner, None

        init_carry = (
            jnp.zeros(()),           # S0
            jnp.zeros((p,)),         # S1
            jnp.zeros((p, p)),       # S2
            jnp.zeros(()),           # tS0
            jnp.zeros((p,)),         # tS1
            jnp.zeros((p, p)),       # tS2
            jnp.zeros((), dtype=jnp.int32),  # n_ev
            jnp.zeros(()),           # ll
            jnp.zeros((p,)),         # score
            jnp.zeros((p, p)),       # neg_H
            jnp.zeros((), dtype=jnp.int32),  # n_events
        )

        final_carry, _ = jax.lax.scan(
            scan_step, init_carry, jnp.arange(n)
        )

        (
            _S0, _S1, _S2, _tS0, _tS1, _tS2, _n_ev,
            ll, score, neg_H, n_events,
        ) = final_carry

        ll = ll - n_events.astype(jnp.float64) * eta_max
        return ll, score, neg_H

    return jax.jit(_efron_pass_jax)


def efron_pass(
    t: np.ndarray,
    e: np.ndarray,
    X: np.ndarray,
    beta: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
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
    try:
        import jax
        import jax.numpy as jnp
        jax.config.update("jax_enable_x64", True)
    except ImportError as exc:
        raise ImportError(
            "JAX not installed. Run: pip install 'jax[cpu]'"
        ) from exc

    t = jnp.asarray(np.ascontiguousarray(t, dtype=np.float64))  # pyright: ignore[reportAssignmentType]
    e = jnp.asarray(np.ascontiguousarray(e, dtype=np.int32))  # pyright: ignore[reportAssignmentType]
    X = jnp.asarray(np.ascontiguousarray(X, dtype=np.float64))  # pyright: ignore[reportAssignmentType]
    beta = jnp.asarray(np.ascontiguousarray(beta, dtype=np.float64))  # pyright: ignore[reportAssignmentType]

    p = X.shape[1]
    if p not in _EFRON_PASS_CACHE:
        _EFRON_PASS_CACHE[p] = _make_efron_pass(p)

    ll, score, neg_H = _EFRON_PASS_CACHE[p](t, e, X, beta)

    return (
        float(np.asarray(ll)),
        np.asarray(score, dtype=np.float64),
        np.asarray(neg_H, dtype=np.float64),
    )


def warmup(p: int = 2) -> None:
    """Compile and cache the JIT for dimension p on tiny synthetic data.

    Call this before the benchmark starts the timer so JIT compilation
    cost is excluded from the measured wall time.
    """
    n = 4
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0.0, 1.0, n))[::-1].astype(np.float64)
    e = np.array([1, 0, 1, 0], dtype=np.int32)
    X = rng.standard_normal((n, p)).astype(np.float64)
    beta = np.zeros(p, dtype=np.float64)
    efron_pass(t, e, X, beta)
