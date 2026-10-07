library(coxstream)
library(survival)

test_that("coefficients match survival::coxph on lung dataset", {
    ref <- coxph(Surv(time, status) ~ age + sex, data = lung, ties = "efron")
    fit <- coxstream(Surv(time, status) ~ age + sex, data = lung)
    expect_equal(coef(fit), coef(ref), tolerance = 1e-6)
})

test_that("coefficients match survival::coxph on veteran dataset", {
    ref <- coxph(Surv(time, status) ~ trt + karno + age, data = veteran, ties = "efron")
    fit <- coxstream(Surv(time, status) ~ trt + karno + age, data = veteran)
    expect_equal(coef(fit), coef(ref), tolerance = 1e-6)
})

test_that("coef() and vcov() accessors work", {
    fit <- coxstream(Surv(time, status) ~ age + sex, data = lung)
    expect_named(coef(fit), c("age", "sex"))
    expect_equal(dim(vcov(fit)), c(2L, 2L))
})

test_that("coxstream_arrow matches survival::coxph on continuous-time data", {
    skip_if_not_installed("arrow")

    set.seed(7)
    n <- 800L
    x1 <- rnorm(n); x2 <- rnorm(n)
    lp <- 0.5 * x1 - 0.3 * x2
    time_raw <- rweibull(n, shape = 2, scale = exp(-lp / 2))
    df <- data.frame(
        time  = time_raw,
        event = as.integer(runif(n) > 0.3),
        x1    = x1, x2 = x2
    )
    df <- df[order(df$time, decreasing = TRUE), ]
    pq <- tempfile(fileext = ".parquet")
    arrow::write_parquet(df, pq)

    ref <- coxph(Surv(time, event) ~ x1 + x2, data = df, ties = "efron")
    fit <- coxstream_arrow(pq, x_cols = c("x1", "x2"),
                           time_col = "time", event_col = "event",
                           verbose = FALSE)
    expect_equal(coef(fit), coef(ref), tolerance = 1e-5)
    unlink(pq)
})

test_that("coxstream_arrow exact Efron on tied data (lung)", {
    skip_if_not_installed("arrow")

    # lung has integer-day times -- many ties. This tests that tie groups
    # spanning row-group boundaries are handled correctly.
    df <- lung[complete.cases(lung[, c("time", "status", "age", "sex")]), ]
    df$event <- as.integer(df$status == 2)
    df <- df[order(df$time, decreasing = TRUE), ]
    pq <- tempfile(fileext = ".parquet")
    arrow::write_parquet(df, pq)

    ref <- coxph(Surv(time, event) ~ age + sex, data = df, ties = "efron")
    fit <- coxstream_arrow(pq, x_cols = c("age", "sex"),
                           time_col = "time", event_col = "event",
                           verbose = FALSE)
    expect_equal(coef(fit), coef(ref), tolerance = 1e-6)
    unlink(pq)
})

test_that("coxstream_arrow handles tie groups spanning many batches", {
    skip_if_not_installed("arrow")

    # Force a tiny batch_size so heavily-tied lung times are split across many
    # Arrow record batches. Exact Efron only matches survival::coxph if (a) the
    # C-stream batches arrive in on-disk DESC order and (b) the local tie carry
    # correctly closes groups that straddle batch boundaries.
    df <- lung[complete.cases(lung[, c("time", "status", "age", "sex")]), ]
    df$event <- as.integer(df$status == 2)
    df <- df[order(df$time, decreasing = TRUE), ]
    pq <- tempfile(fileext = ".parquet")
    arrow::write_parquet(df, pq)

    ref <- coxph(Surv(time, event) ~ age + sex, data = df, ties = "efron")
    fit <- coxstream_arrow(pq, x_cols = c("age", "sex"),
                           time_col = "time", event_col = "event",
                           batch_size = 17L, verbose = FALSE)
    expect_equal(coef(fit), coef(ref), tolerance = 1e-6)
    unlink(pq)
})
