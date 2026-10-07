"""CoxStream: exact out-of-core Cox proportional hazards via streaming NR.

The estimator computes the exact Efron partial-likelihood score and observed
information in a single descending-time pass over the data per Newton-Raphson
iteration, holding only ``O(p^2)`` carry state. The information from the
final pass gives model-based standard errors at no extra cost. Working memory
is therefore independent of the number of observations ``n``: the model fits
on a workstation even when the cohort is far larger than RAM.

Two entry points:

* :meth:`CoxStream.fit` -- in-memory arrays (sorted then streamed in batches).
* :meth:`CoxStream.fit_parquet` -- true out-of-core: stream a Parquet file that
  is already sorted by descending event time, chunk by chunk, without
  materialising the cohort. Sorting is left to your pipeline, so this adds only
  pyarrow. Requires the ``parquet`` extra (``pip install coxstream[parquet]``).
"""
from __future__ import annotations

import warnings

import numpy as np

from . import _kernel

__all__ = ["CoxStream", "check_sorted"]

_BATCH = 100_000
_LL_RTOL = 1e-12  # line-search tolerance, relative to |log-likelihood|


def _assert_finite(t, X) -> None:
    """Reject NaN or infinite durations or covariates (missing values)."""
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(X))):
        raise ValueError(
            "durations and covariates must be finite: drop or impute "
            "missing values (NaN) before fitting")


def _assert_desc_sorted(pf, duration_col: str) -> None:
    """Reject a Parquet file that is provably not descending-time sorted.

    Uses only the footer row-group statistics (per-column min/max), so no data
    pages are read: the cost is a few kilobytes regardless of cohort size, paid
    once before fitting. For a file in descending ``duration_col`` order, each
    row group's minimum must be at least the next group's maximum.

    This is a guard against the common mistake -- a file left unsorted, sorted
    ascending, or ordered by another column -- not a proof of order: min/max
    cannot see the row order *within* a single row group. A genuine out-of-core
    sort always writes globally ordered groups, so in practice this catches the
    errors that occur while reading only the footer. Pass
    ``assume_sorted=True`` to skip it.
    """
    md = pf.metadata
    col = pf.schema_arrow.get_field_index(duration_col)
    if col < 0:
        raise ValueError(
            f"{duration_col!r} not in Parquet schema {pf.schema_arrow.names}")
    prev_min = None
    for rg in range(md.num_row_groups):
        st = md.row_group(rg).column(col).statistics
        if st is None or not st.has_min_max:
            raise ValueError(
                f"cannot verify order: row group {rg} has no statistics for "
                f"{duration_col!r}. Re-write the Parquet with statistics "
                "enabled, or pass assume_sorted=True.")
        if st.null_count:
            raise ValueError(
                f"{duration_col!r} has nulls in row group {rg}; descending "
                "order is undefined with null durations.")
        if prev_min is not None and st.max > prev_min:
            raise ValueError(
                f"Parquet is not sorted by descending {duration_col!r}: row "
                f"group {rg} reaches {st.max}, above the previous group's "
                f"minimum {prev_min}. Sort the file by {duration_col!r} "
                "descending once up front (see fit_parquet docstring), or "
                "pass assume_sorted=True.")
        prev_min = st.min


def check_sorted(path, duration_col: str) -> None:
    """Validate that a Parquet file is descending-time sorted, without fitting.

    A dry run for the :meth:`CoxStream.fit_parquet` precondition: it runs
    exactly the check ``fit_parquet`` performs by default, reading only the
    Parquet footer statistics (no data pages, no full pass). Returns ``None``
    if the file is not provably out of order; raises ``ValueError`` with an
    actionable message otherwise. Use it as a pipeline or CI gate -- e.g.
    right after you sort a cohort and before a long fit -- so an ordering
    mistake fails fast instead of yielding a silently wrong estimate.

    Note that footer statistics can prove a file *unsorted* but not *sorted*:
    min/max are blind to row order within a single row group. A genuine
    out-of-core sort always writes globally ordered groups, so this catches the
    realistic mistakes (unsorted, ascending, or ordered by another column).

    Requires the ``parquet`` extra.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "check_sorted needs the 'parquet' extra: "
            "pip install coxstream[parquet]"
        ) from exc
    _assert_desc_sorted(pq.ParquetFile(path), duration_col)


class CoxStream:
    """Exact streaming Efron Cox proportional hazards estimator.

    Parameters
    ----------
    tol : float
        Convergence tolerance on the L2 norm of the coefficient update.
    max_iter : int
        Maximum Newton-Raphson iterations.
    batch_size : int
        Rows per streamed chunk. Bounds peak resident memory together with
        ``p``; does not affect the estimate.

    Attributes
    ----------
    coef_ : numpy.ndarray
        Fitted coefficients, in covariate order.
    n_iter_ : int
        Newton-Raphson iterations performed.
    n_passes_ : int
        Full passes over the data: one per iteration plus the initial pass,
        plus one per line-search backtrack.
    standard_errors_ : numpy.ndarray
        Model-based standard errors, ``sqrt(diag(variance_matrix_))``.
    variance_matrix_ : numpy.ndarray
        Inverse of the observed information at ``coef_``, accumulated in the
        final pass, so it costs no extra pass over the data.
    converged_ : bool
        Whether the coefficient update fell below ``tol`` within
        ``max_iter`` iterations (a ``RuntimeWarning`` is issued if not).
    n_obs_ : int
        Number of observations fitted.
    feature_names_ : list[str] | None
        Covariate names if supplied.
    """

    def __init__(self, tol: float = 1e-8, max_iter: int = 50,
                 batch_size: int = _BATCH) -> None:
        self.tol = tol
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.coef_: np.ndarray | None = None
        self.n_iter_: int | None = None
        self.n_passes_: int | None = None
        self.standard_errors_: np.ndarray | None = None
        self.variance_matrix_: np.ndarray | None = None
        self.converged_: bool | None = None
        self.n_obs_: int | None = None
        self.feature_names_: list[str] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, durations=None, events=None, X=None, feature_names=None,
            *, start=None, stop=None) -> CoxStream:
        """Fit from in-memory arrays.

        Provide the follow-up time either directly as ``durations`` or as a
        ``start``/``stop`` pair, in which case the duration is
        ``stop - start``. This is right-censored follow-up only -- it is *not*
        the left-truncation / counting-process model; entry times are not
        part of the risk set. The arrays are sorted by descending duration
        internally, so the input order is free.

        Parameters
        ----------
        durations : array-like, shape (n,), optional
            Follow-up time. Provide this, or ``start`` and ``stop``.
        events : array-like, shape (n,)   (1 = event, 0 = censored)
        X : array-like, shape (n, p)
        feature_names : sequence[str], optional
        start, stop : array-like, shape (n,), optional
            Interval endpoints; the duration is ``stop - start``. Mutually
            exclusive with ``durations``.
        """
        if durations is None:
            if start is None or stop is None:
                raise ValueError(
                    "provide either durations, or both start and stop")
            start = np.ascontiguousarray(start, dtype=np.float64).ravel()
            stop = np.ascontiguousarray(stop, dtype=np.float64).ravel()
            if len(start) != len(stop):
                raise ValueError("start and stop must share length")
            durations = stop - start
            if np.any(durations < 0.0):
                raise ValueError(
                    "stop must be >= start; negative durations found")
        elif start is not None or stop is not None:
            raise ValueError("pass either durations or start/stop, not both")
        if events is None or X is None:
            raise ValueError("events and X are required")
        t = np.ascontiguousarray(durations, dtype=np.float64).ravel()
        e = np.ascontiguousarray(events, dtype=np.int32).ravel()
        X = np.ascontiguousarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be 2-D (n, p)")
        if not (len(t) == len(e) == X.shape[0]):
            raise ValueError("durations, events, X must share the first axis")
        _assert_finite(t, X)

        order = np.argsort(t, kind="stable")[::-1]  # descending time
        t_d = np.ascontiguousarray(t[order])
        e_d = np.ascontiguousarray(e[order])
        X_d = np.ascontiguousarray(X[order])
        # Cox estimates are invariant to shifting a covariate by a constant;
        # centring keeps exp(x'beta) and S2/S0 - m m' well conditioned.
        X_d -= X_d.mean(axis=0)
        p = X.shape[1]

        def _batches():
            for s in range(0, len(t_d), self.batch_size):
                sl = slice(s, s + self.batch_size)
                yield (np.ascontiguousarray(t_d[sl]),
                       np.ascontiguousarray(e_d[sl]),
                       np.ascontiguousarray(X_d[sl]))

        self._set_fit(*self._newton(_batches, p))
        self.n_obs_ = len(t_d)
        self.feature_names_ = (
            list(feature_names) if feature_names is not None else None
        )
        return self

    def fit_parquet(self, path, duration_col, event_col,
                    covariate_cols, assume_sorted=False) -> CoxStream:
        """Fit out-of-core from a Parquet file pre-sorted by descending time.

        ``path`` must be a single Parquet file already sorted by
        ``duration_col`` in DESCENDING order. The streaming Efron pass consumes
        the time suffix as the risk set, so the on-disk row order *is* the
        algorithm's order.

        Order is verified once, before fitting, from the Parquet footer
        statistics alone (no data pages are read); a file that is provably out
        of order is rejected with a clear message. Pass
        ``assume_sorted=True`` to skip the check.

        Sorting is intentionally left to your pipeline so coxstream's only read
        dependency stays pyarrow. Produce the descending-time file once with an
        out-of-core sorter -- a benchmark of sort engines found these the
        fastest and both spill to disk, so they handle a cohort larger than RAM
        (here the duration column is ``duration``):

            duckdb:  COPY (SELECT * FROM 'src.parquet' ORDER BY duration DESC)
                         TO 'dst.parquet' (FORMAT PARQUET)
            polars:  (pl.scan_parquet('src.parquet')
                        .sort('duration', descending=True)
                        .sink_parquet('dst.parquet'))
            R:       duckdb's R client runs the same COPY ... ORDER BY DESC.

        If the cohort already fits in RAM, skip the file entirely and call
        :meth:`fit`, which sorts the arrays internally. Requires the
        ``parquet`` extra.

        The file is streamed one row group at a time per Newton-Raphson
        iteration; the cohort is never held in full. Peak memory is
        O(row_group * p), so write the parquet with modest row groups.

        Parameters
        ----------
        assume_sorted : bool
            Skip the descending-order pre-flight check. Set only when the file
            is known to be sorted; an unsorted file yields a silently wrong
            fit.
        """
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "fit_parquet needs the 'parquet' extra: "
                "pip install coxstream[parquet]"
            ) from exc

        cov = list(covariate_cols)
        p = len(cov)
        pf = pq.ParquetFile(path)
        if not assume_sorted:
            _assert_desc_sorted(pf, duration_col)

        want = [duration_col, event_col, *cov]
        # Centring shift (estimates are shift-invariant): the first row
        # group's covariate means, read once.
        rg0 = pf.read_row_group(0, columns=cov)
        shift = np.array([
            np.mean(rg0[c].to_numpy(zero_copy_only=False), dtype=np.float64)
            for c in cov])
        del rg0
        unchecked = [True]  # validate finiteness on the first pass only

        def _batches_t():
            # Stream one row group at a time via read_row_group, NOT
            # iter_batches: iter_batches read-aheads and buffers proportional
            # to the file size, so peak RSS grows with n and defeats
            # out-of-core fitting. Reading row groups individually keeps peak
            # at O(row_group * p), independent of n (verified flat to 32M
            # rows).
            # Peak is therefore set by the input's largest row group -- write
            # the sorted parquet with modest row groups.
            for rg in range(pf.metadata.num_row_groups):
                batch = pf.read_row_group(rg, columns=want)
                t_b = np.ascontiguousarray(
                    batch[duration_col].to_numpy(zero_copy_only=False)
                    .astype(np.float64))
                e_b = batch[event_col].to_numpy(
                    zero_copy_only=False).astype(np.int32)
                X_b = np.ascontiguousarray(np.column_stack([
                    batch[c].to_numpy(zero_copy_only=False) for c in cov
                ]).astype(np.float64))
                if unchecked[0]:
                    _assert_finite(t_b, X_b)
                X_b -= shift
                yield t_b, e_b, X_b
            unchecked[0] = False

        self._set_fit(*self._newton(_batches_t, p))
        self.feature_names_ = cov
        self.n_obs_ = pf.metadata.num_rows
        return self

    # ------------------------------------------------------------------
    # Core: Newton-Raphson over the streaming Efron tie-aware kernel
    # ------------------------------------------------------------------
    def _newton(self, batches, p):
        s1_d = np.empty(p, dtype=np.float64)

        def _one_pass(beta):
            S0 = np.zeros(1)
            S1 = np.zeros(p)
            S2 = np.zeros((p, p), order="F")
            ll = np.zeros(1)
            score = np.zeros(p)
            neg_H = np.zeros((p, p), order="F")
            # Open-tie-group carry, persists across batches within one pass.
            D0 = np.zeros(1)
            D1 = np.zeros(p)
            D2 = np.zeros((p, p), order="F")
            Deta = np.zeros(1)
            Dx = np.zeros(p)
            dcount = np.zeros(1)
            tau = np.zeros(1)
            gopen = np.zeros(1, dtype=np.int32)
            for t_b, e_b, X_b in batches():
                _kernel.efron_stream_batch_ties(
                    t_b, e_b, X_b, beta, S0, S1, S2, ll, score, neg_H,
                    D0, D1, D2, Deta, Dx, dcount, tau, gopen, s1_d,
                )
            _kernel.efron_stream_finalize_ties(
                S0, S1, S2, ll, score, neg_H,
                D0, D1, D2, Deta, Dx, dcount, gopen, s1_d,
            )
            return ll[0], score, neg_H

        beta = np.zeros(p, dtype=np.float64)
        ll, score, neg_H = _one_pass(beta)
        n_passes = 1
        n_iter = 0
        converged = False
        for it in range(self.max_iter):
            # neg_H is the observed information (positive definite); a general
            # solve is ample for the small p x p system -- no scipy needed.
            step = np.linalg.solve(neg_H, score)
            # Accept a step unless it lowers ll by more than the summation
            # rounding of ll itself; an absolute bar sits below that noise at
            # large n and makes the search backtrack spuriously at convergence.
            ll_floor = ll - _LL_RTOL * max(1.0, abs(ll))
            alpha = 1.0
            exhausted = False
            for _ in range(15):  # backtracking line search on ll
                trial = _one_pass(beta + alpha * step)
                n_passes += 1
                if trial[0] >= ll_floor:
                    break
                alpha *= 0.5
            else:  # search exhausted: take the smallest step, evaluated there
                trial = _one_pass(beta + alpha * step)
                n_passes += 1
                exhausted = True
            beta_new = beta + alpha * step
            # The accepted trial pass already holds ll, score and information
            # at beta_new, so the next iteration reuses it instead of
            # re-reading the data: one pass per iteration when the full step
            # is accepted.
            ll, score, neg_H = trial
            n_iter = it + 1
            # A step forced by an exhausted search is short because ll would
            # not rise, not because the estimate has settled.
            converged = (not exhausted
                         and bool(np.linalg.norm(beta_new - beta) < self.tol))
            beta = beta_new
            if converged:
                break
        if not (np.all(np.isfinite(beta)) and np.all(np.isfinite(neg_H))):
            raise FloatingPointError(
                "Newton-Raphson diverged to a non-finite estimate; check for "
                "separation (a covariate that perfectly predicts events) or "
                "extreme covariate scales")
        if not converged:
            warnings.warn(
                f"CoxStream did not converge in {self.max_iter} iterations "
                f"(tol={self.tol}); the estimate may be inaccurate",
                RuntimeWarning, stacklevel=3)
        return beta, n_iter, n_passes, neg_H, converged

    def _set_fit(self, beta, n_iter, n_passes, neg_H, converged) -> None:
        """Store the estimate and its inference from the final pass."""
        self.converged_ = converged
        self.coef_ = beta
        self.n_iter_ = n_iter
        self.n_passes_ = n_passes
        # neg_H was accumulated at the returned estimate, so the model-based
        # covariance costs no extra pass over the data.
        self.variance_matrix_ = np.linalg.inv(neg_H)
        self.standard_errors_ = np.sqrt(np.diag(self.variance_matrix_))
