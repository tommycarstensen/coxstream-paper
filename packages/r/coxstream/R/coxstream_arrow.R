#' Fit a Cox PH model by streaming a DESC-sorted parquet file
#'
#' Like `coxstream()` but reads data row-group by row-group from parquet.
#' Peak RAM is O(batch_size * p) for the active chunk plus O(p^2) for the
#' carry state, independent of total n. Uses exact Efron tie correction: tie
#' groups that span row-group boundaries are handled via local carry state,
#' giving bit-identical coefficients to `coxstream()` on any data.
#'
#' Each NR iteration reads one row-group chunk at a time with `mmap = FALSE`
#' (pread into heap buffers freed after each chunk -- a memory-mapped reader
#' would instead leave every touched file page resident for the mapping's
#' lifetime, making RSS grow O(n)). Each chunk is exported to a C
#' `ArrowArrayStream` and consumed zero-copy in C++ by
#' `efron_stream_chunk_inplace()`, with the Efron tie-state carried across
#' chunks in R -- no R-level column materialisation (`as.vector` / `cbind` /
#' `concat_tables`), which is what previously left a ~1.5x gap behind the Python
#' streaming path.
#'
#' @param parquet_path Path to a parquet file sorted by time DESCENDING.
#' @param x_cols     Character vector of covariate column names.
#' @param time_col   Column name for event/censoring time. Default `"duration"`.
#' @param event_col  Column name for event indicator (1 = event). Default `"event"`.
#' @param init       Optional starting values for beta (length p). Default zero.
#' @param max_iter   Maximum NR iterations. Default 25.
#' @param tol        Convergence tolerance on ||NR step|| (L2 norm of beta
#'                   update). Default 1e-8. Same criterion as the Python
#'                   coxstream implementations.
#' @param batch_size Target rows per read call. Consecutive row groups are
#'                   merged until the total reaches this size, then freed (with a
#'                   gc()) before the next is read, so peak RAM is O(batch_size *
#'                   p), flat in n. The default 250 000 keeps RAM genuinely flat;
#'                   larger chunks are slightly faster but let the allocator's
#'                   high-water ratchet up, so RAM regains a mild upward drift.
#' @param verbose    Print per-iteration progress. Default TRUE.
#'
#' @return A `"coxstream"` object (same class as `coxstream()`).
#' @export
coxstream_arrow <- function(parquet_path, x_cols,
                             time_col   = "duration",
                             event_col  = "event",
                             init       = NULL,
                             max_iter   = 25L,
                             tol        = 1e-8,
                             batch_size = 250000L,
                             verbose    = TRUE) {
    if (!requireNamespace("arrow", quietly = TRUE))
        stop("arrow package required; install with install.packages('arrow')")

    parquet_path <- as.character(parquet_path)
    p      <- length(x_cols)
    beta   <- if (!is.null(init)) as.double(init) else rep(0.0, p)
    wanted <- c(time_col, event_col, x_cols)

    # Arrow C Stream Interface plumbing, using only exported arrow API:
    # cox_alloc_arrow_array_stream() (this package's C++) returns an external
    # pointer owning a zeroed C `ArrowArrayStream`; a chunk's exported
    # `RecordBatchReader$export_to_c()` fills it (arrow accepts any external
    # pointer to an ArrowArrayStream); efron_stream_chunk_inplace() then reads
    # the struct via that same pointer. No unexported arrow function and no
    # build-time dependency on the Arrow C++ libraries are involved.

    # mmap = FALSE: read row groups via pread into heap buffers that are freed
    # after each chunk, so peak RAM stays at O(batch_size * p) regardless of n.
    reader <- arrow::ParquetFileReader$create(parquet_path, mmap = FALSE)
    n_rg   <- reader$num_row_groups
    schema_names <- reader$GetSchema()$names
    col_idx <- match(wanted, schema_names) - 1L   # 0-based indices into schema
    if (anyNA(col_idx))
        stop("columns not found in parquet: ",
             paste(wanted[is.na(col_idx)], collapse = ", "))

    # Merge consecutive row groups into ~batch_size-row chunks to cut per-chunk
    # dispatch.  Probe the first row group's size to choose the chunk width.
    rg0_nrows <- reader$ReadRowGroups(0L, col_idx)$num_rows
    rg_chunk  <- max(1L, as.integer(ceiling(batch_size / rg0_nrows)))

    one_pass <- function(beta) {
        # Global carry: accumulated over the full dataset this NR iteration.
        S0_v  <- c(0.0); S1 <- rep(0.0, p); S2 <- matrix(0.0, p, p)
        score <- rep(0.0, p); neg_H <- matrix(0.0, p, p)

        # Local carry: the currently-open tie group (may span chunk boundaries).
        t_open_v   <- c(Inf)            # sentinel: no group open yet
        n_pend_v   <- integer(1L)
        tS0_pend_v <- c(0.0); tS1_pend <- rep(0.0, p); tS2_pend <- matrix(0.0, p, p)
        ll_raw_v   <- c(0.0); sc_raw   <- rep(0.0, p)

        ll_acc    <- 0.0
        rows_seen <- 0L
        for (rg_start in seq(1L, n_rg, by = rg_chunk)) {
            rg_end <- min(rg_start + rg_chunk - 1L, n_rg)
            rg_idx <- seq.int(rg_start, rg_end) - 1L   # 0-based row-group ids
            # Read only the wanted columns; subset by name to fix the
            # (time, event, x...) order the C++ kernel expects.
            tab    <- reader$ReadRowGroups(rg_idx, col_idx)[wanted]
            rows_seen <- rows_seen + tab$num_rows
            rbr    <- arrow::as_record_batch_reader(tab)
            stream <- cox_alloc_arrow_array_stream()
            rbr$export_to_c(stream)
            ll_acc <- ll_acc + efron_stream_chunk_inplace(
                stream, p, beta,
                S0_v, S1, S2, score, neg_H,
                t_open_v, n_pend_v, tS0_pend_v, tS1_pend, tS2_pend,
                ll_raw_v, sc_raw
            )
            # Free the chunk's Arrow objects NOW, before reading the next one.
            # rm() alone is not enough: Arrow data lives behind externalptrs
            # whose memory R reclaims only at garbage collection, so without a
            # per-chunk gc the freed chunks pile up until the end of the pass and
            # peak RAM grows O(n). full = FALSE runs only a young-generation
            # collection: the chunk objects were just allocated so they are
            # always young and get reclaimed (with their Arrow finalizers), but
            # it skips the full-heap scan -- same flat O(batch * p) RAM at a
            # fraction of the wall-time cost of a full gc().
            rm(tab, rbr, stream)
            gc(verbose = FALSE, full = FALSE)
        }

        # Close the final pending tie group.
        ll_acc <- ll_acc + efron_flush_exact_inplace(
            S0_v, S1, S2, score, neg_H,
            n_pend_v, tS0_pend_v, tS1_pend, tS2_pend, ll_raw_v, sc_raw
        )
        list(ll = ll_acc, score = score, neg_hessian = neg_H, n = rows_seen)
    }

    n_total <- 0L
    n_iter  <- 0L
    var_mat <- matrix(NA_real_, p, p)
    ll_val  <- 0.0

    for (iter in seq_len(max_iter)) {
        res   <- one_pass(beta)
        score <- res$score
        neg_H <- res$neg_hessian
        if (iter == 1L) n_total <- as.integer(res$n)

        step <- tryCatch(
            solve(neg_H, score),
            error = function(e) rep(NA_real_, p)
        )
        if (anyNA(step)) break

        beta      <- beta + step
        n_iter    <- iter
        ll_val    <- res$ll
        norm_step <- sqrt(sum(step^2))

        if (verbose) {
            cat(sprintf(
                "  [coxstream_arrow]  iter %d  ll=%.6f  ||step||=%.3e\n",
                iter, ll_val, norm_step))
            flush(stdout())
        }

        if (norm_step < tol) {
            var_mat <- tryCatch(solve(neg_H), error = function(e) var_mat)
            break
        }
    }

    names(beta)       <- x_cols
    rownames(var_mat) <- x_cols
    colnames(var_mat) <- x_cols

    structure(
        list(
            coefficients = beta,
            var          = var_mat,
            loglik       = ll_val,
            n_iter       = n_iter,
            n            = n_total,
            formula      = NULL,
            call         = match.call()
        ),
        class = "coxstream"
    )
}
