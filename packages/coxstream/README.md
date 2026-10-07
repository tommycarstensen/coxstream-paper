# coxstream

**Exact out-of-core Cox proportional hazards regression via streaming
Newton-Raphson.**

[![PyPI](https://img.shields.io/pypi/v/coxstream.svg)](https://pypi.org/project/coxstream/)
[![DOI](https://zenodo.org/badge/1269530910.svg)](https://doi.org/10.5281/zenodo.20693783)

Standard CoxPH solvers (`lifelines`, `scikit-survival`, R `survival`) load the
full cohort into memory before fitting, so on registry-scale data they exhaust
RAM long before the computation is hard. `coxstream` computes the **exact** Efron
partial-likelihood estimate by streaming a single time-sorted pass over the data
per Newton-Raphson iteration, holding only `O(p^2)` state for `p` covariates.
Working memory is therefore **independent of the number of observations `n`**:
the model fits on a workstation even when the cohort is far larger than RAM.

The streamed estimate *is* the in-memory maximum-likelihood estimate, and the
Efron tie correction is carried across chunk boundaries, so heavily tied data
are handled exactly.

![coxstream holds peak RAM flat as the cohort grows, while in-memory solvers (lifelines, R survival::coxph) scale with n; coefficients agree to machine precision.](https://raw.githubusercontent.com/tommycarstensen/coxstream/main/docs/benchmark.png)

*Memory vs. speed against `lifelines` and R `survival::coxph`: coxstream's peak
RAM stays flat in the number of rows while in-memory solvers grow with the
cohort, at matching coefficients. See the accompanying paper for the full
methodology.*

## Install

```bash
pip install coxstream             # core (numpy only)
pip install coxstream[parquet]    # + out-of-core fit_parquet (pyarrow)
```

Prebuilt wheels are published for current CPython versions on Linux (x86_64,
aarch64), macOS (universal2), and Windows, so most installs need no toolchain.
A C compiler is only needed when building from the sdist (e.g. on a platform
without a wheel), since the package compiles a small Cython kernel.

## Usage

In memory:

```python
import numpy as np
from coxstream import CoxStream

model = CoxStream().fit(durations, events, X, feature_names=names)
print(model.coef_, model.standard_errors_, model.n_iter_)
```

`standard_errors_` and `variance_matrix_` come from the observed information accumulated in the final pass, so they cost no extra pass over the data; `n_passes_` reports how many full passes the fit made (one per iteration plus one).

If your cohort records interval endpoints rather than a precomputed follow-up
time, pass `start`/`stop` and the duration is `stop - start` (right-censored
follow-up; entry times are not modelled):

```python
model = CoxStream().fit(start=entry, stop=exit, events=events, X=X)
```

Out of core, from a Parquet file **pre-sorted by descending event time** (never
materialises the cohort):

```python
from coxstream import CoxStream

# The file must already be sorted by duration DESC. `fit_parquet` verifies this
# from the Parquet footer statistics alone (no full pass) and rejects a file
# that is out of order; pass assume_sorted=True to skip the check.
#
# Sort it once with an out-of-core sorter -- both spill to disk, so they handle
# a cohort larger than RAM (a sort-engine benchmark found these the fastest):
#   duckdb:  COPY (SELECT * FROM 'cohort.parquet' ORDER BY duration DESC)
#            TO 'cohort_desc.parquet' (FORMAT PARQUET);
#   polars:  (pl.scan_parquet("cohort.parquet")
#              .sort("duration", descending=True)
#              .sink_parquet("cohort_desc.parquet"))
#   R:       duckdb via its R client runs the same COPY ... ORDER BY DESC.
# If the cohort fits in RAM, skip the file and call .fit, which sorts for you.

model = CoxStream().fit_parquet(
    "cohort_desc.parquet",
    duration_col="duration",
    event_col="event",
    covariate_cols=["age_std", "sex", "treatment"],
)
print(model.coef_)
```

To validate a file's order ahead of time -- a dry run, e.g. a CI or pipeline
gate right after you sort and before a long fit -- call `check_sorted`, which
runs the same footer-only check without fitting and raises on a file that is
provably out of order:

```python
from coxstream import check_sorted

check_sorted("cohort_desc.parquet", duration_col="duration")  # raises if unsorted
```

It doubles as a shell gate -- it exits non-zero on an out-of-order file, so a
pipeline step can fail fast without a bespoke CLI:

```bash
python -c "import coxstream; coxstream.check_sorted('cohort_desc.parquet', 'duration')"
```

## Validation

`coxstream` is verified against `lifelines` and R `survival::coxph`:

- It reproduces the in-memory maximum-likelihood estimate to **machine
  precision** on synthetic data.
- On heavily tied synthetic data (monthly-discretised event times) the exact
  Efron pass matches `lifelines` where the Breslow approximation would diverge.
- Peak resident memory is flat in `n` while in-memory solvers grow with the
  cohort and eventually exhaust RAM.

The package's own test suite is dependency-free: it checks exactness against a
self-contained plain-numpy Cox Newton-Raphson reference. The cross-checks
against `lifelines` and R `survival::coxph` above live in the accompanying
benchmark and paper.

The methodology and full results are in the accompanying paper (see
[Citation](#citation)).

## Scope

`coxstream` implements the exact Efron partial likelihood for large-`n`,
modest-`p` tabular survival data. It is a focused estimator, not a full survival
suite: it does not provide baseline-hazard estimation, time-varying covariates,
or proportional-hazards diagnostics.

## Testing

```bash
pip install -e '.[test]'           # core suite (numpy only)
pip install -e '.[test,parquet]'   # + the out-of-core fit_parquet test
pytest
```

## Citation

If you use `coxstream`, please cite it via the metadata in
[`CITATION.cff`](CITATION.cff).

## License

MIT. See [LICENSE](LICENSE).
