#!/usr/bin/env Rscript
# In-memory reference fits for bench_coxph_ties.py: R survival::coxph on one
# Parquet file with the requested tie methods, written as JSON at full
# precision. Called by bench_coxph_ties.py; not a standalone benchmark.
#
#   Rscript r_ties_reference.R <in.parquet> <out.json> <ties, comma-separated>
#
# The control is tighter than coxph's defaults so the reference is at least as
# precise as the streamed fit it is compared with: eps is a relative change in
# the log-likelihood (default 1e-9), toler.chol (singularity detection) is
# set below eps as coxph.control asks, and timefix = FALSE stops coxph from
# merging continuous times closer than ~1.5e-8 into ties the streamed fit
# (exact double equality) does not see.

suppressPackageStartupMessages({
    library(survival)
    library(arrow)
    library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
    stop("usage: r_ties_reference.R <in.parquet> <out.json> <ties,...>")
}
in_path <- args[1]
out_path <- args[2]
ties <- strsplit(args[3], ",", fixed = TRUE)[[1]]

df <- as.data.frame(arrow::read_parquet(in_path))
x_cols <- setdiff(names(df), c("duration", "event"))
df$event <- as.numeric(df$event)
fml <- as.formula(paste("Surv(duration, event) ~",
                        paste(x_cols, collapse = " + ")))
ctl <- coxph.control(eps = 1e-12, toler.chol = 1e-13, iter.max = 50,
                     timefix = FALSE)

fit_one <- function(tie) {
    warns <- character()
    t0 <- proc.time()
    fit <- withCallingHandlers(
        coxph(fml, data = df, ties = tie, control = ctl,
              x = FALSE, y = FALSE, model = FALSE),
        warning = function(w) {
            warns <<- c(warns, conditionMessage(w))
            invokeRestart("muffleWarning")
        })
    elapsed <- (proc.time() - t0)[["elapsed"]]
    cat(sprintf("  R coxph ties=%s: %d iterations, %.1f s\n",
                tie, fit$iter, elapsed))
    list(ties = tie,
         coef_names = names(coef(fit)),
         coef = unname(coef(fit)),
         se = unname(sqrt(diag(vcov(fit)))),
         loglik = fit$loglik[2],
         loglik_null = fit$loglik[1],
         iter = fit$iter,
         elapsed_s = elapsed,
         warnings = I(warns))
}

out <- list(n_rows = nrow(df),
            survival_version = as.character(packageVersion("survival")),
            r_version = R.version.string,
            control = list(eps = ctl$eps, iter_max = ctl$iter.max,
                           toler_chol = ctl$toler.chol,
                           timefix = ctl$timefix),
            fits = lapply(ties, fit_one))

tmp_path <- paste0(out_path, ".tmp")
writeLines(jsonlite::toJSON(out, digits = NA, auto_unbox = TRUE,
                            na = "null"), tmp_path)
if (!file.rename(tmp_path, out_path)) stop("could not write ", out_path)
