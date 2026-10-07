"""Tests for coxstream.

The core tests need only the ``test`` extra (pytest); the Parquet tests also
need the ``parquet`` extra. Exactness is checked against independent
plain-numpy Newton-Raphson references (distinct times, and Efron ties), so no
third-party survival package is needed:

    pip install -e '.[test,parquet]'
    pytest
"""
import numpy as np
import pytest

from coxstream import CoxStream


def _coef(model: CoxStream) -> np.ndarray:
    """Fitted coefficients; fails the test if the model was never fitted."""
    assert model.coef_ is not None
    return model.coef_


def _se(model: CoxStream) -> np.ndarray:
    """Standard errors; fails the test if the model was never fitted."""
    assert model.standard_errors_ is not None
    return model.standard_errors_


def _simulate(n=20_000, p=4, seed=0):
    """Weibull proportional-hazards data with known coefficients."""
    rng = np.random.default_rng(seed)
    beta = np.array([0.5, -0.4, 0.3, -0.2])[:p]
    X = rng.standard_normal((n, p))
    eta = X @ beta
    u = rng.uniform(size=n)
    t_event = (-np.log(u) / np.exp(eta)) ** (1.0 / 1.5)
    t_cens = rng.exponential(scale=t_event.mean() * 1.5, size=n)
    t = np.minimum(t_event, t_cens)
    e = (t_event <= t_cens).astype(int)
    return t, e, X, beta


def test_recovers_known_coefficients():
    t, e, X, beta = _simulate()
    model = CoxStream().fit(t, e, X)
    coef = _coef(model)
    assert coef.shape == (X.shape[1],)
    assert model.n_iter_ is not None and model.n_iter_ >= 1
    # Sampling error at n=20k, p=4 is well under 0.1.
    assert np.max(np.abs(coef - beta)) < 0.1


def test_batch_size_invariant():
    """The estimate must not depend on the streaming batch size."""
    t, e, X, _ = _simulate()
    a = _coef(CoxStream(batch_size=512).fit(t, e, X))
    b = _coef(CoxStream(batch_size=50_000).fit(t, e, X))
    np.testing.assert_allclose(a, b, atol=1e-10)


def test_handles_ties():
    """Discretised (heavily tied) times still fit and recover coefficients."""
    t, e, X, beta = _simulate()
    t_tied = np.ceil(t * 4) / 4  # quarter-unit grid -> many ties
    model = CoxStream().fit(t_tied, e, X)
    assert np.max(np.abs(_coef(model) - beta)) < 0.15


def test_fit_start_stop_matches_durations():
    """Passing start/stop must equal passing the precomputed durations."""
    t, e, X, _ = _simulate(n=5_000, p=3)
    start = np.full_like(t, 3.0)        # arbitrary common entry; duration = t
    stop = start + t
    ref = _coef(CoxStream().fit(t, e, X))
    coef = _coef(CoxStream().fit(start=start, stop=stop, events=e, X=X))
    np.testing.assert_allclose(coef, ref, atol=1e-12)


@pytest.mark.parametrize("bad", ["both", "neither", "mismatch", "negative"])
def test_fit_start_stop_validation(bad):
    t, e, X, _ = _simulate(n=100, p=2)
    with pytest.raises(ValueError):
        if bad == "both":
            CoxStream().fit(durations=t, start=t, stop=t, events=e, X=X)
        elif bad == "neither":
            CoxStream().fit(events=e, X=X)
        elif bad == "mismatch":
            CoxStream().fit(start=t, stop=t[:-1], events=e, X=X)
        else:
            CoxStream().fit(start=t, stop=t - 1.0, events=e, X=X)


@pytest.mark.parametrize("bad", ["1d_X", "mismatched"])
def test_input_validation(bad):
    t, e, X, _ = _simulate(n=100)
    with pytest.raises(ValueError):
        if bad == "1d_X":
            CoxStream().fit(t, e, X[:, 0])
        else:
            CoxStream().fit(t[:50], e, X)


def test_fit_parquet_matches_fit(tmp_path):
    """The out-of-core path (sorted Parquet) must equal the in-memory fit."""
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq

    t, e, X, _ = _simulate(n=8_000, p=3)
    t = np.ceil(t * 4) / 4  # induce ties so the tie path runs through disk
    cols = [f"x{i}" for i in range(X.shape[1])]

    # fit_parquet requires the file pre-sorted by descending event time.
    order = np.argsort(t, kind="stable")[::-1]
    table = pa.table({
        "duration": t[order],
        "event": e[order],
        **{c: X[order, i] for i, c in enumerate(cols)},
    })
    path = tmp_path / "cohort_desc.parquet"
    pq.write_table(table, path)

    ref = _coef(CoxStream().fit(t, e, X))
    model = CoxStream().fit_parquet(str(path), "duration", "event", cols)
    np.testing.assert_allclose(_coef(model), ref, atol=1e-8)
    assert model.n_obs_ == len(t)
    assert model.feature_names_ == cols


def test_fit_parquet_rejects_unsorted(tmp_path):
    """An ascending (not DESC) Parquet is rejected from footer stats alone,
    and assume_sorted=True bypasses the check."""
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq

    t, e, X, _ = _simulate(n=8_000, p=3)
    cols = [f"x{i}" for i in range(X.shape[1])]
    order = np.argsort(t, kind="stable")            # ASCENDING -> wrong order
    table = pa.table({
        "duration": t[order],
        "event": e[order],
        **{c: X[order, i] for i, c in enumerate(cols)},
    })
    path = tmp_path / "cohort_asc.parquet"
    # Several row groups, so the cross-group footer check has pairs to compare.
    pq.write_table(table, path, row_group_size=2_000)

    with pytest.raises(ValueError, match="not sorted by descending"):
        CoxStream().fit_parquet(str(path), "duration", "event", cols)
    # Opt-out runs without raising (the result is meaningless; guard is off).
    CoxStream().fit_parquet(str(path), "duration", "event", cols,
                            assume_sorted=True)


def test_check_sorted_dry_run(tmp_path):
    """The public dry-run validator agrees with fit_parquet's own guard."""
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq

    from coxstream import check_sorted

    t, e, _, _ = _simulate(n=8_000, p=2)
    order = np.argsort(t, kind="stable")

    asc = tmp_path / "asc.parquet"
    pq.write_table(pa.table({"duration": t[order], "event": e[order]}),
                   asc, row_group_size=2_000)
    with pytest.raises(ValueError, match="not sorted by descending"):
        check_sorted(str(asc), "duration")

    desc = tmp_path / "desc.parquet"
    rev = order[::-1]
    pq.write_table(pa.table({"duration": t[rev], "event": e[rev]}),
                   desc, row_group_size=2_000)
    assert check_sorted(str(desc), "duration") is None  # passes, returns None


def _cox_nr_reference(t, e, X, max_iter=50, tol=1e-10):
    """Independent plain-numpy exact Cox partial-likelihood Newton-Raphson.

    Risk-set moments via reverse cumulative sums over descending-time order.
    Assumes distinct event times (Efron then equals the exact partial
    likelihood), so it is a clean oracle for continuous simulated data without
    any third-party dependency.
    """
    order = np.argsort(t)[::-1]              # descending time
    e = e[order].astype(bool)
    X = X[order]
    _, p = X.shape
    beta = np.zeros(p)
    for _ in range(max_iter):
        w = np.exp(X @ beta)
        S0 = np.cumsum(w)  # risk-set suffix sum
        S1 = np.cumsum(w[:, None] * X, axis=0)
        outer = X[:, :, None] * X[:, None, :]
        S2 = np.cumsum(w[:, None, None] * outer, axis=0)
        m = S1[e] / S0[e, None]
        score = (X[e] - m).sum(0)
        hess = (S2[e] / S0[e, None, None]
                - m[:, :, None] * m[:, None, :]).sum(0)
        step = np.linalg.solve(hess, score)
        beta = beta + step
        if np.linalg.norm(step) < tol:
            break
    return beta


def test_matches_numpy_reference():
    """CoxStream reproduces an independent numpy MLE (distinct times)."""
    t, e, X, _ = _simulate(n=5_000, p=3)  # continuous Weibull: distinct times
    ref = _cox_nr_reference(t, e, X)
    coef = _coef(CoxStream().fit(t, e, X))
    np.testing.assert_allclose(coef, ref, atol=1e-6)


def _efron_score_info(t, e, X, beta):
    """Independent plain-numpy Efron score and observed information at beta.

    Written from the textbook formulas, one distinct event time at a time,
    with the whole risk set in memory; it shares no code with the streaming
    kernel. For a tie group D of d events at time u, term l = 0..d-1 uses the
    risk-set sums minus the fraction l/d of the tied events' sums.
    """
    e = e.astype(bool)
    p = X.shape[1]
    w = np.exp(X @ beta)
    score = np.zeros(p)
    info = np.zeros((p, p))
    for u in np.unique(t[e]):
        risk = t >= u
        tied = (t == u) & e
        d = int(tied.sum())
        f = np.arange(d) / d
        w_r, x_r = w[risk], X[risk]
        w_d, x_d = w[tied], X[tied]
        s2_risk = (x_r * w_r[:, None]).T @ x_r
        s2_tied = (x_d * w_d[:, None]).T @ x_d
        s0 = w_r.sum() - f * w_d.sum()
        s1 = (w_r @ x_r)[None, :] - f[:, None] * (w_d @ x_d)[None, :]
        m = s1 / s0[:, None]
        score += x_d.sum(axis=0) - m.sum(axis=0)
        info += ((1.0 / s0).sum() * s2_risk - (f / s0).sum() * s2_tied
                 - m.T @ m)
    return score, info


def _efron_nr_reference(t, e, X, max_iter=50, tol=1e-12):
    """Independent plain-numpy Efron Newton-Raphson for tied event times."""
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        score, info = _efron_score_info(t, e, X, beta)
        step = np.linalg.solve(info, score)
        beta = beta + step
        if np.linalg.norm(step) < tol:
            break
    return beta


def _reference_standard_errors(t, e, X, beta):
    """sqrt(diag(I(beta)^-1)) from the independent Efron information."""
    _, info = _efron_score_info(t, e, X, beta)
    return np.sqrt(np.diag(np.linalg.inv(info)))


_TINY_CHUNK = 37  # rows per chunk; every large tie group spans many chunks


def _tied_data():
    """Quarter-unit time grid: tie groups of hundreds of rows each."""
    t, e, X, _ = _simulate(n=4_000, p=3)
    t = np.ceil(t * 4) / 4
    _, group_sizes = np.unique(t[e == 1], return_counts=True)
    assert group_sizes.max() > 10 * _TINY_CHUNK  # groups must span chunks
    return t, e, X


def test_ties_across_chunks_match_efron_reference():
    """Tie groups straddling chunk boundaries give the in-memory Efron MLE."""
    t, e, X = _tied_data()
    ref = _efron_nr_reference(t, e, X)
    coef = _coef(CoxStream(batch_size=_TINY_CHUNK).fit(t, e, X))
    np.testing.assert_allclose(coef, ref, rtol=0, atol=1e-8)


def test_parquet_ties_across_row_groups_match_efron_reference(tmp_path):
    """Out-of-core: tie groups straddling row groups give the Efron MLE."""
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq

    t, e, X = _tied_data()
    cols = [f"x{i}" for i in range(X.shape[1])]
    order = np.argsort(t, kind="stable")[::-1]
    table = pa.table({
        "duration": t[order],
        "event": e[order],
        **{c: X[order, i] for i, c in enumerate(cols)},
    })
    path = tmp_path / "tied_desc.parquet"
    pq.write_table(table, path, row_group_size=_TINY_CHUNK)
    assert pq.ParquetFile(path).num_row_groups > 100

    ref = _efron_nr_reference(t, e, X)
    model = CoxStream().fit_parquet(str(path), "duration", "event", cols)
    np.testing.assert_allclose(_coef(model), ref, rtol=0, atol=1e-8)
    se_ref = _reference_standard_errors(t, e, X, _coef(model))
    np.testing.assert_allclose(_se(model), se_ref, rtol=1e-10)


def test_one_pass_per_iteration():
    """Each accepted Newton step costs one pass; only the first is extra."""
    t, e, X, _ = _simulate()
    model = CoxStream().fit(t, e, X)
    assert model.n_iter_ is not None
    assert model.n_passes_ == model.n_iter_ + 1


@pytest.mark.parametrize("tied", [False, True])
def test_standard_errors_match_efron_reference(tied):
    """SEs from the final streamed pass equal sqrt(diag(I^-1)) at coef_."""
    if tied:
        t, e, X = _tied_data()
        model = CoxStream(batch_size=_TINY_CHUNK).fit(t, e, X)
    else:
        t, e, X, _ = _simulate(n=5_000, p=3)
        model = CoxStream().fit(t, e, X)
    se_ref = _reference_standard_errors(t, e, X, _coef(model))
    np.testing.assert_allclose(_se(model), se_ref, rtol=1e-10)


_OFFSET = np.array([1e4, -3e3, 2e3])  # far from zero, like calendar years


def _write_desc_parquet(path, t, e, X, row_group_size):
    """Write (t, e, X) sorted by descending time, as fit_parquet requires."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    order = np.argsort(t, kind="stable")[::-1]
    cols = [f"x{i}" for i in range(X.shape[1])]
    pq.write_table(pa.table({
        "duration": t[order], "event": e[order],
        **{c: X[order, i] for i, c in enumerate(cols)},
    }), path, row_group_size=row_group_size)
    return cols


def test_uncentred_covariates_match_centred():
    """Cox fits are shift-invariant, so a large offset changes nothing."""
    t, e, X, _ = _simulate(n=5_000, p=3)
    centred = CoxStream().fit(t, e, X)
    shifted = CoxStream().fit(t, e, X + _OFFSET)
    np.testing.assert_allclose(_coef(shifted), _coef(centred), rtol=0,
                               atol=1e-10)
    np.testing.assert_allclose(_se(shifted), _se(centred), rtol=1e-8)


def test_parquet_uncentred_matches_in_memory(tmp_path):
    """The out-of-core path centres too (shift from the first row group)."""
    pytest.importorskip("pyarrow")
    t, e, X, _ = _simulate(n=5_000, p=3)
    path = tmp_path / "offset_desc.parquet"
    cols = _write_desc_parquet(path, t, e, X + _OFFSET, row_group_size=500)
    ref = CoxStream().fit(t, e, X)
    model = CoxStream().fit_parquet(str(path), "duration", "event", cols)
    np.testing.assert_allclose(_coef(model), _coef(ref), rtol=0, atol=1e-10)
    np.testing.assert_allclose(_se(model), _se(ref), rtol=1e-8)


@pytest.mark.parametrize("where", ["X", "durations"])
def test_rejects_non_finite(where):
    t, e, X, _ = _simulate(n=200, p=2)
    if where == "X":
        X[5, 1] = np.nan
    else:
        t[3] = np.inf
    with pytest.raises(ValueError, match="finite"):
        CoxStream().fit(t, e, X)


def test_parquet_rejects_non_finite(tmp_path):
    pytest.importorskip("pyarrow")
    t, e, X, _ = _simulate(n=2_000, p=2)
    X[1_500, 0] = np.nan  # lands in a later row group, not the first
    path = tmp_path / "nan_desc.parquet"
    cols = _write_desc_parquet(path, t, e, X, row_group_size=300)
    with pytest.raises(ValueError, match="finite"):
        CoxStream().fit_parquet(str(path), "duration", "event", cols)


def test_converged_flag_and_warning():
    t, e, X, _ = _simulate(n=2_000, p=3)
    assert CoxStream().fit(t, e, X).converged_ is True
    with pytest.warns(RuntimeWarning, match="did not converge"):
        model = CoxStream(max_iter=1).fit(t, e, X)
    assert model.converged_ is False


def test_exhausted_line_search_is_not_convergence(monkeypatch):
    # After 15 failed halvings the fit takes the 2**-15 step anyway. That step
    # is short because the search failed, not because the estimate converged,
    # so it must not set converged_. A short descent direction makes every
    # trial lower the log-likelihood and the forced step fall below tol.
    t, e, X, _ = _simulate(n=2_000, p=3)
    solve = np.linalg.solve

    def short_descent(a, b):
        step = solve(a, b)
        return -1e-4 * step / np.linalg.norm(step)

    monkeypatch.setattr(np.linalg, "solve", short_descent)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        model = CoxStream(max_iter=2).fit(t, e, X)
    assert model.converged_ is False
