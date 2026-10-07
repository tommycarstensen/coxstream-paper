#!/usr/bin/env Rscript
# r_competing_coxph.R -- fit one R CoxPH method on a pre-sorted parquet.
#
# Called by bench_competing_coxph.py _run_r_competing():
#   Rscript src/coxbench/r_competing_coxph.R <method> <parquet> <n_rows> <out.json> [<p>]
# <parquet> is the DESC-sorted parquet (both methods accept any row order).
#
# Methods
#   r_survival_coxph   survival::coxph Efron (gold-standard R reference)
#   xue_schifano_cuee  Xue & Schifano CUEE (batch-wise inverse-variance blending)
#
# Output JSON: { method, n, coef, coef_names, n_iter, wall_s }

suppressPackageStartupMessages({
    library(survival)
    library(arrow)
    library(jsonlite)
})

argv   <- commandArgs(trailingOnly = TRUE)
if (length(argv) < 4L) {
    stop("Usage: Rscript r_competing_coxph.R <method> <sorted.parquet> <n_rows> <out.json> [<p>]")
}
method       <- argv[1L]
sorted_path  <- argv[2L]
n_rows       <- as.integer(argv[3L])
out          <- argv[4L]
p_value      <- if (length(argv) >= 5L) as.integer(argv[5L]) else 10L

# Covariate names come from the parquet schema (in canonical write order),
# not a hardcoded list: the p-sweep dataset (--ncovariates 32) carries
# x11..x32 beyond the 10 built-in covariates.  open_dataset is lazy and reads
# only metadata.
ALL_X_COLS <- setdiff(names(arrow::open_dataset(sorted_path)),
                      c("duration", "event"))
X_COLS <- ALL_X_COLS[seq_len(p_value)]

BATCH_ROWS <- 200000L

form <- as.formula(paste(
    "Surv(duration, event) ~",
    paste(X_COLS, collapse = " + ")
))

# ---------------------------------------------------------------------------
# r_survival_coxph: exact in-memory Efron MLE via survival::coxph
# ---------------------------------------------------------------------------
if (method == "r_survival_coxph") {
    cat(sprintf("[%s]  loading %s rows from %s ...\n",
                method, format(n_rows, big.mark = ","), sorted_path))
    flush.console()

    df <- as.data.frame(read_parquet(
        sorted_path,
        col_select = c("duration", "event", X_COLS)
    ))
    if (nrow(df) > n_rows) df <- df[seq_len(n_rows), ]

    cat(sprintf("[%s]  fitting on %s rows ...\n",
                method, format(nrow(df), big.mark = ",")))
    flush.console()

    t0  <- proc.time()
    fit <- coxph(form, data = df, ties = "efron")
    el  <- proc.time() - t0

    coef_vec <- coef(fit)
    n_iter   <- as.integer(fit$iter)

# ---------------------------------------------------------------------------
# xue_schifano_cuee: streaming CUEE (batch-wise inverse-variance blending)
#
# Algorithm (Xue & Schifano 2020 / 2021):
#   For each batch k:
#     1. Fit local coxph -> beta_k, I_k = solve(vcov(fit_k))
#     2. Accumulate: cum_I   += I_k
#                    cum_score += I_k %*% beta_k
#   Final: beta_CUEE = solve(cum_I) %*% cum_score
#
# Approximation: uses local risk sets per batch (not global risk set).
# Memory: O(BATCH_ROWS * p) -- streaming, does not load full dataset.
# ---------------------------------------------------------------------------
} else if (method == "xue_schifano_cuee") {
    p           <- length(X_COLS)
    cum_I       <- matrix(0.0, p, p)
    cum_score   <- numeric(p)
    n_batches   <- 0L
    n_total     <- 0L

    cat(sprintf("[%s]  streaming %s rows by row group ...\n",
                method, format(n_rows, big.mark = ",")))
    flush.console()

    # arrow R uses ParquetFileReader (not ParquetFile as in Python pyarrow)
    reader    <- arrow::ParquetFileReader$create(sorted_path)
    n_rg      <- reader$num_row_groups
    wanted    <- c("duration", "event", X_COLS)
    rows_left <- n_rows

    t0 <- proc.time()
    for (rg_idx in seq_len(n_rg)) {
        if (rows_left <= 0L) break

        tab   <- reader$ReadRowGroup(rg_idx - 1L)
        batch <- as.data.frame(tab)[, wanted, drop = FALSE]
        rm(tab)

        if (nrow(batch) > rows_left) {
            batch <- batch[seq_len(rows_left), , drop = FALSE]
        }
        rows_left  <- rows_left - nrow(batch)
        n_total    <- n_total + nrow(batch)
        n_batches  <- n_batches + 1L

        cat(sprintf("[%s]  batch %d  (%s rows)\r",
                    method, n_batches, format(nrow(batch), big.mark = ",")))
        flush.console()

        fit_b <- tryCatch(
            coxph(form, data = batch, ties = "efron"),
            error = function(e) NULL
        )
        if (is.null(fit_b) || any(is.na(coef(fit_b)))) next

        beta_b <- as.numeric(coef(fit_b))
        I_b    <- tryCatch(solve(vcov(fit_b)), error = function(e) NULL)
        if (is.null(I_b)) next

        cum_I     <- cum_I + I_b
        cum_score <- cum_score + as.numeric(I_b %*% beta_b)
    }
    cat("\n")
    el <- proc.time() - t0

    coef_vec       <- as.numeric(solve(cum_I) %*% cum_score)
    names(coef_vec) <- X_COLS
    n_iter         <- n_batches

# ---------------------------------------------------------------------------
# cyclops: unregularised Cox MLE via Cyclops (cyclic coordinate descent).
# The established large-n / high-dimensional Cox tool (Mittal et al. 2014).
# In-memory: loads the cohort and builds a column-major structure; the "none"
# prior fits the unpenalised MLE, matching survival::coxph on untied data.
# Needs install.packages("Cyclops").
# ---------------------------------------------------------------------------
} else if (method == "cyclops") {
    library(Cyclops)
    cat(sprintf("[%s]  loading %s rows from %s ...\n",
                method, format(n_rows, big.mark = ","), sorted_path))
    flush.console()

    df <- as.data.frame(read_parquet(
        sorted_path,
        col_select = c("duration", "event", X_COLS)
    ))
    if (nrow(df) > n_rows) df <- df[seq_len(n_rows), ]

    cat(sprintf("[%s]  fitting unregularised Cox MLE on %s rows ...\n",
                method, format(nrow(df), big.mark = ",")))
    flush.console()

    t0  <- proc.time()
    cd  <- createCyclopsData(form, data = df, modelType = "cox")
    fit <- fitCyclopsModel(cd, prior = createPrior("none"))
    el  <- proc.time() - t0

    coef_vec <- coef(fit)
    if (all(X_COLS %in% names(coef_vec))) coef_vec <- coef_vec[X_COLS]
    n_iter   <- -1L  # Cyclops does not expose an iteration count

# ---------------------------------------------------------------------------
# r_coxstream: exact streaming Efron MLE via the CRAN coxstream package.
# Same algorithm as the Python coxstream_cython method -- coxstream_arrow()
# streams the DESC-sorted parquet row group by row group, so peak RAM is
# O(row_group * p) + O(p^2) carry state, independent of n. Bit-identical to
# survival::coxph on untied data. Included to confirm the R package is as fast
# and memory-lean as the Python one. Needs install.packages("coxstream").
# tol matches the Python _nr_loop (1e-8 on the coefficient step).
# ---------------------------------------------------------------------------
} else if (method == "r_coxstream") {
    library(coxstream)
    cat(sprintf("[%s]  streaming %s rows from %s (%d covariates) ...\n",
                method, format(n_rows, big.mark = ","), sorted_path,
                length(X_COLS)))
    flush.console()

    t0  <- proc.time()
    fit <- coxstream_arrow(
        sorted_path,
        x_cols    = X_COLS,
        time_col  = "duration",
        event_col = "event",
        tol       = 1e-8,
        verbose   = FALSE
    )
    el  <- proc.time() - t0

    coef_vec        <- coef(fit)
    names(coef_vec) <- X_COLS
    n_iter          <- as.integer(fit$n_iter)
    n_total         <- as.integer(fit$n)  # result block reads n_total if set

} else {
    stop(paste("Unknown method:", method))
}

result <- list(
    method     = method,
    n          = if (exists("n_total")) n_total else nrow(df),
    coef       = as.numeric(unname(coef_vec)),
    coef_names = as.character(names(coef_vec)),
    n_iter     = as.integer(n_iter),
    wall_s     = as.numeric(el["elapsed"])
)
writeLines(toJSON(result, auto_unbox = TRUE, digits = 15), out)
cat(sprintf("[%s]  done. wall=%.1fs\n", method, el["elapsed"]))
