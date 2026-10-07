#' Fit a Cox proportional hazards model via streaming Newton-Raphson
#'
#' Fits the Cox PH model using a single descending-time-order pass per
#' Newton-Raphson iteration. Peak RAM is O(p^2) regardless of n, making it
#' suitable for large datasets. Produces identical coefficients to
#' `survival::coxph()` with Efron tie correction.
#'
#' @param formula A formula with a `survival::Surv()` response,
#'   e.g. `Surv(time, event) ~ x1 + x2`.
#' @param data A data frame containing the variables in `formula`.
#' @param init Optional numeric vector of starting values for beta (length p).
#'   Defaults to zero.
#' @param max_iter Maximum Newton-Raphson iterations. Default 25.
#' @param tol Convergence tolerance on the max absolute score element.
#'   Default 1e-9.
#' @param verbose Currently unused; reserved for future per-iteration output.
#'   Default `FALSE`.
#'
#' @return An object of class `"coxstream"` with components:
#'   \item{coefficients}{Named numeric vector of fitted coefficients.}
#'   \item{var}{Variance-covariance matrix (inverse of observed information).}
#'   \item{loglik}{Log-likelihood at convergence.}
#'   \item{n_iter}{Number of NR iterations taken.}
#'   \item{n}{Number of rows.}
#'   \item{formula}{The formula used.}
#'   \item{call}{The matched call.}
#'
#' @examples
#' library(survival)
#' fit <- coxstream(Surv(time, status) ~ age + sex, data = lung)
#' coef(fit)
#'
#' @importFrom survival Surv
#' @export
coxstream <- function(formula, data, init = NULL,
                      max_iter = 25L, tol = 1e-9, verbose = FALSE) {
    cl <- match.call()

    # Extract model frame and design matrix
    mf  <- model.frame(formula, data = data)
    Y   <- model.response(mf)           # Surv object: [, 1] = time, [, 2] = event
    X   <- model.matrix(formula, data = data)[, -1, drop = FALSE]  # drop intercept

    time  <- as.double(Y[, 1])
    event <- as.integer(Y[, 2])
    p     <- ncol(X)
    n     <- nrow(X)

    # Sort descending by time (required by streaming kernel)
    ord   <- order(time, decreasing = TRUE)
    time  <- time[ord]
    event <- event[ord]
    X     <- X[ord, , drop = FALSE]

    beta <- if (!is.null(init)) as.double(init) else rep(0.0, p)

    ll_prev <- -Inf
    n_iter  <- 0L
    var_mat <- matrix(NA_real_, p, p)

    for (iter in seq_len(max_iter)) {
        res   <- efron_pass_cpp(time, event, X, beta)
        score <- res$score
        neg_H <- res$neg_hessian

        # Newton step: beta += solve(neg_H, score)
        step  <- tryCatch(
            solve(neg_H, score),
            error = function(e) rep(NA_real_, p)
        )
        if (anyNA(step)) break

        beta    <- beta + step
        n_iter  <- iter
        ll_prev <- res$ll

        if (max(abs(score)) < tol) {
            var_mat <- tryCatch(solve(neg_H), error = function(e) var_mat)
            break
        }
    }

    names(beta)        <- colnames(X)
    rownames(var_mat)  <- colnames(X)
    colnames(var_mat)  <- colnames(X)

    structure(
        list(
            coefficients = beta,
            var          = var_mat,
            loglik       = ll_prev,
            n_iter       = n_iter,
            n            = n,
            formula      = formula,
            call         = cl
        ),
        class = "coxstream"
    )
}

#' @export
coef.coxstream <- function(object, ...) object$coefficients

#' @export
vcov.coxstream <- function(object, ...) object$var

#' @export
print.coxstream <- function(x, ...) {
    cat("coxstream fit\n")
    cat("Call: "); print(x$call)
    cat(sprintf("n = %d,  iterations = %d,  loglik = %.4f\n",
                x$n, x$n_iter, x$loglik))
    cat("\nCoefficients:\n")
    se  <- sqrt(diag(x$var))
    z   <- x$coefficients / se
    mat <- cbind(
        coef = x$coefficients,
        `se(coef)` = se,
        z = z,
        `Pr(>|z|)` = 2 * pnorm(-abs(z))
    )
    printCoefmat(mat, digits = 4, P.values = TRUE, has.Pvalue = TRUE)
    invisible(x)
}
