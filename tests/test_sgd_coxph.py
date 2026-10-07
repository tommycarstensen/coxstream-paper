"""The BigSurvSGD reimplementation's stratum gradient uses the risk set.

Run: python3 -m pytest tests -q
"""
import numpy as np

from coxbench.sgd_coxph import BigSurvSGDCoxPH


def _cohort(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 2))
    beta = np.array([0.5, -0.3])
    t_event = (-np.log(rng.random(n)) / np.exp(x @ beta)) ** (1 / 1.5) * 8
    t_cens = np.minimum(rng.exponential(25.0, n), 10.0)
    return (np.minimum(t_event, t_cens), (t_event <= t_cens).astype(int), x,
            beta)


def test_stratum_gradient_matches_explicit_risk_sets():
    t, e, x, _ = _cohort(300)
    order = np.argsort(t, kind="stable")
    t, e, x = t[order], e[order], x[order]
    b = np.array([0.2, 0.1])
    w = np.exp(x @ b)
    expected = np.zeros(2)
    for i in np.flatnonzero(e):
        r = t >= t[i]  # the risk set of event i
        expected += x[i] - (w[r, None] * x[r]).sum(0) / w[r].sum()
    got = BigSurvSGDCoxPH._stratum_gradient(t, e, x, b)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-10)


def test_fit_recovers_the_coefficients():
    t, e, x, beta = _cohort(20_000)
    model = BigSurvSGDCoxPH(strata_size=50, n_epochs=5).partial_fit(t, e, x)
    model.fit()
    assert np.abs(model.coef_ - beta).max() < 0.03
