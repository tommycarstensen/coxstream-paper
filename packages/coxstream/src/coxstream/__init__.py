"""coxstream: exact out-of-core Cox proportional hazards via streaming NR.

Public API
----------
CoxStream
    Exact Efron Cox proportional hazards estimator. Computes the score and
    observed information in a single descending-time pass per Newton-Raphson
    iteration, with O(p^2) working memory independent of the cohort size.
    ``fit`` takes in-memory arrays; ``fit_parquet`` streams out-of-core.
check_sorted
    Dry run for the ``fit_parquet`` precondition: validate that a Parquet file
    is descending-time sorted, from footer statistics alone (no full pass), so a
    sort mistake fails fast instead of yielding a silently wrong fit.
"""
from coxstream.coxstream import CoxStream, check_sorted

__all__ = ["CoxStream", "check_sorted"]
__version__ = "0.3.0"
