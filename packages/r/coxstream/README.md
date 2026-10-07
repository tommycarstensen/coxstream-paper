# coxstream

[![DOI](https://zenodo.org/badge/1269532103.svg)](https://doi.org/10.5281/zenodo.20694001)

Memory-efficient Cox proportional hazards regression via streaming
Newton-Raphson. Peak RAM is O(p^2) in the number of covariates and flat in the
number of rows n, so models fit on datasets that do not fit in memory.
Coefficients are identical to `survival::coxph()` with Efron tie correction.

![coxstream (Python and R) holds peak RAM flat as the cohort grows, while in-memory solvers (lifelines, survival::coxph) scale with n; coefficients agree to machine precision.](https://raw.githubusercontent.com/tommycarstensen/coxstream-r/main/docs/benchmark.png)

## Installation

From CRAN:

    install.packages("coxstream")

Development version from GitHub:

    # install.packages("remotes")
    remotes::install_github("tommycarstensen/coxstream-r")

## Usage

In-memory fit, with the same formula interface as `coxph()`:

    library(survival)
    library(coxstream)

    fit <- coxstream(Surv(time, status) ~ age + sex, data = lung)
    coef(fit)
    fit

Out-of-core fit, streaming a time-DESCENDING-sorted parquet file one row group
at a time (requires the optional `arrow` package):

    fit <- coxstream_arrow(
        "events_sorted.parquet",
        x_cols    = c("age", "sex"),
        time_col  = "duration",
        event_col = "event"
    )
    coef(fit)

The reader loads one row-group chunk at a time and frees it before the next, so
peak RAM stays at O(batch_size * p), flat in n. Efron tie groups that span chunk
boundaries are carried in running state, giving coefficients bit-identical to
the in-memory fit.

## How it works

Each Newton-Raphson iteration makes a single descending-time pass to accumulate
the Cox partial-likelihood score and Hessian. Only running sums of size O(p) and
O(p^2) are held, never the full risk set, so memory does not grow with n. The
accumulation kernel is implemented in C++ via Rcpp.

## Dependencies

- Imports: Rcpp (compiled kernel), survival (Surv response; ships with R)
- Suggests: arrow (only for `coxstream_arrow()`), testthat (tests)

## License

MIT, except `src/arrow_c_abi.h`, which is vendored from Apache Arrow under
Apache-2.0; see `inst/COPYRIGHTS`.
