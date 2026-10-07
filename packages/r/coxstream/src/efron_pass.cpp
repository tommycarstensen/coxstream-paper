// [[Rcpp::depends(Rcpp)]]
#include <Rcpp.h>
#include <cmath>
#include <algorithm>
using namespace Rcpp;

// Single descending-order Efron pass over an in-memory design matrix.
// t, e, X must be sorted by t DESCENDING (ties allowed).
// Returns list(ll, score, neg_hessian). Used by coxstream() (in-memory fit);
// the out-of-core coxstream_arrow() path lives in efron_stream.cpp.
// [[Rcpp::export]]
List efron_pass_cpp(
    NumericVector t,      // (n,)   event times, descending
    IntegerVector e,      // (n,)   event indicator (1/0)
    NumericMatrix X,      // (n, p) covariates
    NumericVector beta    // (p,)   current coefficients
) {
    const int n = t.size();
    const int p = beta.size();

    // xb = X %*% beta, log-sum-exp stabilised
    NumericVector xb(n, 0.0);
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < p; ++k)
            xb[i] += X(i, k) * beta[k];

    double eta_max = *std::max_element(xb.begin(), xb.end());
    NumericVector exp_xb(n);
    for (int i = 0; i < n; ++i)
        exp_xb[i] = std::exp(xb[i] - eta_max);

    double S0 = 0.0, tS0 = 0.0, ll_val = 0.0;
    int n_events = 0;
    NumericVector S1(p, 0.0), tS1(p, 0.0), s1_d(p, 0.0), score_(p, 0.0);
    NumericMatrix S2(p, p), tS2(p, p), neg_H_(p, p);
    std::fill(S2.begin(),     S2.end(),     0.0);
    std::fill(tS2.begin(),    tS2.end(),    0.0);
    std::fill(neg_H_.begin(), neg_H_.end(), 0.0);

    int i = 0;
    while (i < n) {
        double t_cur = t[i];
        int n_ev = 0;
        tS0 = 0.0;
        std::fill(tS1.begin(), tS1.end(), 0.0);
        std::fill(tS2.begin(), tS2.end(), 0.0);

        // Accumulate all rows at t_cur into S0/S1/S2;
        // events also accumulate tS0/tS1/tS2 and contribute to ll/score.
        while (i < n && t[i] == t_cur) {
            double ex = exp_xb[i];
            S0 += ex;
            for (int k = 0; k < p; ++k) {
                double xk = X(i, k);
                S1[k] += xk * ex;
                for (int l = 0; l < p; ++l)
                    S2(k, l) += xk * X(i, l) * ex;
            }
            if (e[i] == 1) {
                ++n_ev;
                tS0 += ex;
                for (int k = 0; k < p; ++k) {
                    double xk = X(i, k);
                    tS1[k] += xk * ex;
                    for (int l = 0; l < p; ++l)
                        tS2(k, l) += xk * X(i, l) * ex;
                    score_[k] += xk;        // raw covariate sum for events
                }
                ll_val += xb[i];            // stabilised linear predictor
            }
            ++i;
        }

        if (n_ev > 0) {
            n_events += n_ev;
            for (int d = 0; d < n_ev; ++d) {
                double frac   = (double)d / (double)n_ev;
                double s0_d   = S0 - frac * tS0;
                if (s0_d <= 0.0) s0_d = 1e-300;
                double inv_s0 = 1.0 / s0_d;
                for (int k = 0; k < p; ++k)
                    s1_d[k] = S1[k] - frac * tS1[k];
                for (int k = 0; k < p; ++k)
                    score_[k] -= s1_d[k] * inv_s0;
                ll_val -= std::log(s0_d);
                for (int k = 0; k < p; ++k)
                    for (int l = 0; l < p; ++l) {
                        double s2kl = S2(k, l) - frac * tS2(k, l);
                        neg_H_(k, l) += s2kl * inv_s0
                                      - s1_d[k] * s1_d[l] * inv_s0 * inv_s0;
                    }
            }
        }
    }

    // Restore true log-likelihood (stabilised denominators absorbed n_events * eta_max)
    ll_val -= (double)n_events * eta_max;

    return List::create(
        Named("ll")          = ll_val,
        Named("score")       = score_,
        Named("neg_hessian") = neg_H_
    );
}
