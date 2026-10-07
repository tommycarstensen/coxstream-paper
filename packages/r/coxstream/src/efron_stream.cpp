// [[Rcpp::depends(Rcpp)]]
#include <Rcpp.h>
#include <cmath>
#include <cstdint>
#include <vector>
#include "arrow_c_abi.h"
using namespace Rcpp;

// ---------------------------------------------------------------------------
// Zero-copy Arrow-stream exact-Efron batch kernel (in-place carry)
// ---------------------------------------------------------------------------
//
// efron_stream_chunk_inplace() consumes one row-group chunk -- handed to it as
// an Arrow C *stream* (an ArrowArrayStream exported on the R side from a
// RecordBatchReader over a single ParquetFileReader$ReadRowGroups() chunk) --
// and folds every row into the running Efron risk sets. All per-row work
// (linear predictor, exp, S0/S1/S2 accumulation, tie-group closing) happens in
// C++, reading the column buffers zero-copy; there is no R-level column
// materialisation (no as.vector / cbind / concat_tables).
//
// Carry state lives in R objects and persists across successive calls, so a tie
// group may span row-group-chunk boundaries. For speed the carry is copied into
// C++ locals on entry, updated by the per-row inner loops, and written back to
// the R objects on exit -- going through Rcpp NumericVector/NumericMatrix
// proxies on every access inside a loop that runs millions of times is
// materially slower than raw std::vector. The caller reads chunks one at a time
// with mmap = FALSE and frees each (with a gc()) before reading the next, which
// keeps peak RAM at O(batch_size * p), independent of n.
//
// Projection order from the R side is fixed: child 0 = time, child 1 = event,
// children 2 .. p+1 = covariates in x_cols order. No log-sum-exp stabilisation
// (safe when max|X @ beta| << 709, which holds for the benchmark data).

// Read element `idx` of a primitive Arrow buffer as a double, dispatching on
// the one-character Arrow format code. Covers every primitive type the parquet
// columns use (float64 "g" covariates, int8 "c" event / sex / treatment) plus
// the other fixed-width integer/float widths and bit-packed booleans.
static inline double read_f64(const void* buf, char fmt, int64_t idx) {
    switch (fmt) {
        case 'g': return static_cast<const double*>(buf)[idx];
        case 'f': return static_cast<const float*>(buf)[idx];
        case 'c': return static_cast<const int8_t*>(buf)[idx];
        case 'C': return static_cast<const uint8_t*>(buf)[idx];
        case 's': return static_cast<const int16_t*>(buf)[idx];
        case 'S': return static_cast<const uint16_t*>(buf)[idx];
        case 'i': return static_cast<const int32_t*>(buf)[idx];
        case 'I': return static_cast<const uint32_t*>(buf)[idx];
        case 'l': return static_cast<double>(static_cast<const int64_t*>(buf)[idx]);
        case 'L': return static_cast<double>(static_cast<const uint64_t*>(buf)[idx]);
        case 'b': {
            const uint8_t byte = static_cast<const uint8_t*>(buf)[idx >> 3];
            return ((byte >> (idx & 7)) & 1) ? 1.0 : 0.0;
        }
        default:
            Rcpp::stop("efron_stream: unsupported Arrow column format '%c'", fmt);
    }
    return 0.0;  // unreachable
}

// Close the currently-open tie group: fold its Efron contribution into the
// global score / negative Hessian and return its log-likelihood delta.
// R-typed-carry version, used by efron_flush_exact_inplace() (called once per
// pass, so proxy overhead is irrelevant there).
static double close_group(
    int n_ev, int p,
    double S0, const NumericVector& S1, const NumericMatrix& S2,
    double tS0, const NumericVector& tS1, const NumericMatrix& tS2,
    double ll_raw, const NumericVector& sc_raw,
    NumericVector& score, NumericMatrix& neg_H
) {
    if (n_ev == 0) return 0.0;

    double ll_delta = ll_raw;
    for (int k = 0; k < p; ++k) score[k] += sc_raw[k];

    for (int d = 0; d < n_ev; ++d) {
        const double frac = static_cast<double>(d) / static_cast<double>(n_ev);
        double s0_d = S0 - frac * tS0;
        if (s0_d <= 0.0) s0_d = 1e-300;
        const double inv_s0 = 1.0 / s0_d;
        ll_delta -= std::log(s0_d);
        for (int k = 0; k < p; ++k) {
            const double s1k = S1[k] - frac * tS1[k];
            score[k] -= s1k * inv_s0;
            for (int l = 0; l < p; ++l) {
                const double s2kl = S2(k, l) - frac * tS2(k, l);
                const double s1l  = S1[l] - frac * tS1[l];
                neg_H(k, l) += s2kl * inv_s0 - s1k * s1l * inv_s0 * inv_s0;
            }
        }
    }
    return ll_delta;
}

// std::vector-carry version of close_group() for the hot per-chunk loop.
// Matrices are flat row-major (index k*p + l).
static double close_group_local(
    int n_ev, int p,
    double S0, const std::vector<double>& S1, const std::vector<double>& S2,
    double tS0, const std::vector<double>& tS1, const std::vector<double>& tS2,
    double ll_raw, const std::vector<double>& sc_raw,
    std::vector<double>& score, std::vector<double>& neg_H
) {
    if (n_ev == 0) return 0.0;

    double ll_delta = ll_raw;
    for (int k = 0; k < p; ++k) score[k] += sc_raw[k];

    for (int d = 0; d < n_ev; ++d) {
        const double frac = static_cast<double>(d) / static_cast<double>(n_ev);
        double s0_d = S0 - frac * tS0;
        if (s0_d <= 0.0) s0_d = 1e-300;
        const double inv_s0 = 1.0 / s0_d;
        ll_delta -= std::log(s0_d);
        for (int k = 0; k < p; ++k) {
            const double s1k = S1[k] - frac * tS1[k];
            score[k] -= s1k * inv_s0;
            const double* S2k  = &S2[static_cast<size_t>(k) * p];
            const double* tS2k = &tS2[static_cast<size_t>(k) * p];
            double*       nHk  = &neg_H[static_cast<size_t>(k) * p];
            for (int l = 0; l < p; ++l) {
                const double s2kl = S2k[l] - frac * tS2k[l];
                const double s1l  = S1[l] - frac * tS1[l];
                nHk[l] += s2kl * inv_s0 - s1k * s1l * inv_s0 * inv_s0;
            }
        }
    }
    return ll_delta;
}

// Finalizer for a coxstream-owned ArrowArrayStream external pointer: release
// any still-held stream resources (a well-behaved release callback nulls
// itself per the Arrow C stream contract, so this is a no-op once
// efron_stream_chunk_inplace() has already released it), then free the struct.
static void cox_arrow_stream_finalizer(SEXP xp) {
    ArrowArrayStream* s = static_cast<ArrowArrayStream*>(R_ExternalPtrAddr(xp));
    if (s == nullptr) return;
    if (s->release != nullptr) s->release(s);
    delete s;
    R_ClearExternalPtr(xp);
}

// Allocate a zeroed ArrowArrayStream and return it as an external pointer that
// owns the struct. The caller fills it via the exported
// RecordBatchReader$export_to_c() (arrow accepts any external pointer whose
// address points to an ArrowArrayStream), then hands it to
// efron_stream_chunk_inplace(). This replaces arrow's unexported
// allocate_arrow_array_stream(): we own the struct layout via the vendored
// arrow_c_abi.h, so no internal arrow function is needed.
// [[Rcpp::export]]
SEXP cox_alloc_arrow_array_stream() {
    ArrowArrayStream* s = new ArrowArrayStream();
    s->release = nullptr;  // empty until export_to_c() populates it
    SEXP xp = PROTECT(R_MakeExternalPtr(s, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(xp, cox_arrow_stream_finalizer, TRUE);
    UNPROTECT(1);
    return xp;
}

// Consume one row-group chunk's Arrow stream, updating carry in place.
// Returns the ll delta for all tie groups closed within this chunk. The final
// pending group is closed once, after the last chunk, by
// efron_flush_exact_inplace().
// [[Rcpp::export]]
double efron_stream_chunk_inplace(
    SEXP           stream_xptr,  // external ptr to an exported ArrowArrayStream
    int            p,
    NumericVector  beta,         // (p,) current coefficients (read-only)
    NumericVector  S0_v,         // (1,)   global carry: risk-set denominator
    NumericVector  S1,           // (p,)   global carry: risk-set first moment
    NumericMatrix  S2,           // (p,p)  global carry: risk-set second moment
    NumericVector  score,        // (p,)   global carry: score
    NumericMatrix  neg_H,        // (p,p)  global carry: negative Hessian
    NumericVector  t_open_v,     // (1,)   local carry: open group time; init +Inf
    IntegerVector  n_pend_v,     // (1,)   local carry: pending event count; init 0L
    NumericVector  tS0_pend_v,   // (1,)   local carry: event risk-set denominator
    NumericVector  tS1_pend,     // (p,)   local carry: event risk-set first moment
    NumericMatrix  tS2_pend,     // (p,p)  local carry: event risk-set second moment
    NumericVector  ll_raw_v,     // (1,)   local carry: sum raw xb for pending events
    NumericVector  sc_raw        // (p,)   local carry: sum raw X for pending events
) {
    ArrowArrayStream* stream =
        static_cast<ArrowArrayStream*>(R_ExternalPtrAddr(stream_xptr));
    if (stream == nullptr || stream->release == nullptr)
        Rcpp::stop("efron_stream: null or already-released Arrow stream");

    // Schema: record one format char per column, validate child count.
    ArrowSchema schema;
    schema.release = nullptr;
    if (stream->get_schema(stream, &schema) != 0) {
        const char* msg = stream->get_last_error ? stream->get_last_error(stream) : "?";
        Rcpp::stop("efron_stream: get_schema failed: %s", msg ? msg : "?");
    }
    const int n_cols = p + 2;  // time, event, p covariates
    if (schema.n_children != n_cols) {
        if (schema.release) schema.release(&schema);
        Rcpp::stop("efron_stream: schema has %lld columns, expected %d",
                   static_cast<long long>(schema.n_children), n_cols);
    }
    std::vector<char> fmt(n_cols);
    for (int j = 0; j < n_cols; ++j)
        fmt[j] = schema.children[j]->format[0];

    // Copy carry into C++ locals (flat row-major matrices) for the hot loop.
    const size_t pp = static_cast<size_t>(p) * p;
    double S0 = S0_v[0];
    std::vector<double> betal(beta.begin(), beta.end());
    std::vector<double> S1l(S1.begin(), S1.end());
    std::vector<double> scorel(score.begin(), score.end());
    std::vector<double> tS1l(tS1_pend.begin(), tS1_pend.end());
    std::vector<double> sc_rawl(sc_raw.begin(), sc_raw.end());
    std::vector<double> S2l(pp), negHl(pp), tS2l(pp);
    for (int k = 0; k < p; ++k)
        for (int l = 0; l < p; ++l) {
            S2l[static_cast<size_t>(k) * p + l]   = S2(k, l);
            negHl[static_cast<size_t>(k) * p + l] = neg_H(k, l);
            tS2l[static_cast<size_t>(k) * p + l]  = tS2_pend(k, l);
        }
    double t_open = t_open_v[0];
    int    n_pend = n_pend_v[0];
    double tS0    = tS0_pend_v[0];
    double ll_raw = ll_raw_v[0];

    double ll_delta = 0.0;
    std::vector<double> xrow(p);

    for (;;) {
        ArrowArray batch;
        batch.release = nullptr;
        if (stream->get_next(stream, &batch) != 0) {
            const char* msg = stream->get_last_error ? stream->get_last_error(stream) : "?";
            if (schema.release) schema.release(&schema);
            stream->release(stream);
            Rcpp::stop("efron_stream: get_next failed: %s", msg ? msg : "?");
        }
        if (batch.release == nullptr) break;  // end of chunk stream

        const int64_t nrow = batch.length;
        const int64_t base = batch.offset;

        std::vector<const void*> data(n_cols);
        std::vector<int64_t>     coff(n_cols);
        for (int j = 0; j < n_cols; ++j) {
            const ArrowArray* col = batch.children[j];
            data[j] = col->buffers[1];  // values buffer (buffers[0] is validity)
            coff[j] = col->offset;
        }

        for (int64_t i = 0; i < nrow; ++i) {
            const int64_t logical = base + i;
            const double  t_i = read_f64(data[0], fmt[0], coff[0] + logical);
            const double  e_i = read_f64(data[1], fmt[1], coff[1] + logical);

            if (t_i != t_open) {
                // Time changed: close the previous group, open a new one.
                ll_delta += close_group_local(
                    n_pend, p, S0, S1l, S2l, tS0, tS1l, tS2l,
                    ll_raw, sc_rawl, scorel, negHl);
                t_open = t_i;
                n_pend = 0;
                tS0    = 0.0;
                ll_raw = 0.0;
                std::fill(tS1l.begin(), tS1l.end(), 0.0);
                std::fill(tS2l.begin(), tS2l.end(), 0.0);
                std::fill(sc_rawl.begin(), sc_rawl.end(), 0.0);
            }

            double xb_i = 0.0;
            for (int k = 0; k < p; ++k) {
                const double xk = read_f64(data[k + 2], fmt[k + 2], coff[k + 2] + logical);
                xrow[k] = xk;
                xb_i += xk * betal[k];
            }
            const double ex = std::exp(xb_i);

            // Global risk set: all rows.
            S0 += ex;
            for (int k = 0; k < p; ++k) {
                const double xk = xrow[k];
                S1l[k] += xk * ex;
                double* S2k = &S2l[static_cast<size_t>(k) * p];
                for (int l = 0; l < p; ++l)
                    S2k[l] += xk * xrow[l] * ex;
            }

            // Local tie-group state: events only.
            if (e_i != 0.0) {
                n_pend++;
                tS0    += ex;
                ll_raw += xb_i;
                for (int k = 0; k < p; ++k) {
                    const double xk = xrow[k];
                    tS1l[k]    += xk * ex;
                    sc_rawl[k] += xk;
                    double* tS2k = &tS2l[static_cast<size_t>(k) * p];
                    for (int l = 0; l < p; ++l)
                        tS2k[l] += xk * xrow[l] * ex;
                }
            }
        }

        batch.release(&batch);
    }

    if (schema.release) schema.release(&schema);
    stream->release(stream);

    // Write carry back to the R objects so it persists into the next chunk.
    S0_v[0] = S0;
    for (int k = 0; k < p; ++k) {
        S1[k]       = S1l[k];
        score[k]    = scorel[k];
        tS1_pend[k] = tS1l[k];
        sc_raw[k]   = sc_rawl[k];
        for (int l = 0; l < p; ++l) {
            S2(k, l)       = S2l[static_cast<size_t>(k) * p + l];
            neg_H(k, l)    = negHl[static_cast<size_t>(k) * p + l];
            tS2_pend(k, l) = tS2l[static_cast<size_t>(k) * p + l];
        }
    }
    t_open_v[0]   = t_open;
    n_pend_v[0]   = n_pend;
    tS0_pend_v[0] = tS0;
    ll_raw_v[0]   = ll_raw;
    return ll_delta;
}

// Close the final pending tie group after the last row-group chunk. Called once
// per NR pass, after efron_stream_chunk_inplace() has consumed every chunk.
// Returns the ll delta for the last group (same scalar-return contract as
// efron_stream_chunk_inplace; the caller accumulates on the R side).
// [[Rcpp::export]]
double efron_flush_exact_inplace(
    NumericVector  S0_v,
    NumericVector  S1,
    NumericMatrix  S2,
    NumericVector  score,
    NumericMatrix  neg_H,
    IntegerVector  n_pend_v,
    NumericVector  tS0_pend_v,
    NumericVector  tS1_pend,
    NumericMatrix  tS2_pend,
    NumericVector  ll_raw_v,
    NumericVector  sc_raw
) {
    const int p = S1.size();
    return close_group(
        n_pend_v[0], p,
        S0_v[0], S1, S2,
        tS0_pend_v[0], tS1_pend, tS2_pend,
        ll_raw_v[0], sc_raw,
        score, neg_H
    );
}
