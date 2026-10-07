"""JAX JIT-compiled Efron CoxPH forward pass -- float32 (jax_enable_x64=False).

Identical algorithm to coxph_jax.py but intentionally does NOT set
jax_enable_x64=True.  JAX operates in float32 mode.  Use this module
ONLY from a subprocess that has never imported coxph_jax, so the x64
config is never activated in that process.

Usage
-----
from coxbench.coxph_jax_x32 import efron_pass_x32, warmup_x32

warmup_x32(p=2)
ll, score, neg_H = efron_pass_x32(t, e, X, beta)
"""

from __future__ import annotations

from typing import Callable

import numpy as np

_EFRON_PASS_CACHE_X32: dict[int, Callable] = {}


def _make_efron_pass_x32(p: int) -> Callable:
    """Build and JIT-compile efron_pass for float32 (no jax_enable_x64)."""
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            "JAX not installed. Run: pip install 'jax[cpu]'"
        ) from exc

    def _efron_pass_f32(t, e, X, beta):
        n = t.shape[0]

        xb = X @ beta
        eta_max = jnp.max(xb)
        exp_xb = jnp.exp(xb - eta_max)

        def flush_group(carry_inner):
            S0, S1, S2, tS0, tS1, tS2, n_ev, ll, score, neg_H, n_events = (
                carry_inner
            )

            def efron_step(d, acc):
                ll_a, score_a, neg_H_a = acc
                frac = d / n_ev
                s0_d = jnp.maximum(S0 - frac * tS0, jnp.float32(1e-30))
                inv_s0 = jnp.float32(1.0) / s0_d
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
            return (
                S0, S1, S2,
                jnp.zeros(()),
                jnp.zeros((p,)),
                jnp.zeros((p, p)),
                jnp.zeros((), dtype=jnp.int32),
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

            S0 = S0 + ex
            S1 = S1 + xi * ex
            S2 = S2 + jnp.outer(xi, xi) * ex

            is_event = ei == 1
            tS0 = tS0 + jnp.where(is_event, ex, jnp.float32(0.0))
            tS1 = tS1 + jnp.where(is_event, xi * ex, jnp.zeros((p,)))
            tS2 = tS2 + jnp.where(
                is_event, jnp.outer(xi, xi) * ex, jnp.zeros((p, p))
            )
            n_ev = n_ev + jnp.where(is_event, 1, 0).astype(jnp.int32)
            ll = ll + jnp.where(is_event, xbi, jnp.float32(0.0))
            score = score + jnp.where(is_event, xi, jnp.zeros((p,)))

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
            jnp.zeros(()),
            jnp.zeros((p,)),
            jnp.zeros((p, p)),
            jnp.zeros(()),
            jnp.zeros((p,)),
            jnp.zeros((p, p)),
            jnp.zeros((), dtype=jnp.int32),
            jnp.zeros(()),
            jnp.zeros((p,)),
            jnp.zeros((p, p)),
            jnp.zeros((), dtype=jnp.int32),
        )

        final_carry, _ = jax.lax.scan(
            scan_step, init_carry, jnp.arange(n)
        )

        (
            _S0, _S1, _S2, _tS0, _tS1, _tS2, _n_ev,
            ll, score, neg_H, n_events,
        ) = final_carry

        ll = ll - n_events.astype(jnp.float32) * eta_max
        return ll, score, neg_H

    return jax.jit(_efron_pass_f32)


def efron_pass_x32(
    t: np.ndarray,
    e: np.ndarray,
    X: np.ndarray,
    beta: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Efron pass in float32 JAX (jax_enable_x64=False).

    Accepts float64 inputs; JAX downcasts to float32 internally.
    Returns float64 outputs cast from float32 results.
    """
    try:
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            "JAX not installed. Run: pip install 'jax[cpu]'"
        ) from exc

    t = jnp.asarray(np.ascontiguousarray(t, dtype=np.float32))  # pyright: ignore[reportAssignmentType]
    e = jnp.asarray(np.ascontiguousarray(e, dtype=np.int32))  # pyright: ignore[reportAssignmentType]
    X = jnp.asarray(np.ascontiguousarray(X, dtype=np.float32))  # pyright: ignore[reportAssignmentType]
    beta = jnp.asarray(np.ascontiguousarray(beta, dtype=np.float32))  # pyright: ignore[reportAssignmentType]

    p = X.shape[1]
    if p not in _EFRON_PASS_CACHE_X32:
        _EFRON_PASS_CACHE_X32[p] = _make_efron_pass_x32(p)

    ll, score, neg_H = _EFRON_PASS_CACHE_X32[p](t, e, X, beta)

    return (
        float(np.asarray(ll)),
        np.asarray(score, dtype=np.float64),
        np.asarray(neg_H, dtype=np.float64),
    )


def warmup_x32(p: int = 2) -> None:
    """Compile and cache the float32 JIT for dimension p."""
    n = 4
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0.0, 1.0, n))[::-1].astype(np.float32)
    e = np.array([1, 0, 1, 0], dtype=np.int32)
    X = rng.standard_normal((n, p)).astype(np.float32)
    beta = np.zeros(p, dtype=np.float32)
    efron_pass_x32(t, e, X, beta)
