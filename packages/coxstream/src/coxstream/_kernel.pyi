"""Type stub for the compiled Cython Efron streaming kernel.

The kernel accumulates the exact Efron partial-likelihood score and observed
information in place. All arrays are float64 except event indicators (int32)
and the open-group flag (int32); ``S2`` / ``neg_H`` / ``D2`` are Fortran-ordered.
"""
import numpy as np

def efron_stream_batch_ties(
    t_b: np.ndarray,
    e_b: np.ndarray,
    X_b: np.ndarray,
    beta: np.ndarray,
    S0_arr: np.ndarray,
    S1: np.ndarray,
    S2: np.ndarray,
    ll_arr: np.ndarray,
    score: np.ndarray,
    neg_H: np.ndarray,
    D0_arr: np.ndarray,
    D1: np.ndarray,
    D2: np.ndarray,
    Deta_arr: np.ndarray,
    Dx: np.ndarray,
    dcount_arr: np.ndarray,
    tau_arr: np.ndarray,
    gopen_arr: np.ndarray,
    s1_d: np.ndarray,
) -> None: ...
def efron_stream_finalize_ties(
    S0_arr: np.ndarray,
    S1: np.ndarray,
    S2: np.ndarray,
    ll_arr: np.ndarray,
    score: np.ndarray,
    neg_H: np.ndarray,
    D0_arr: np.ndarray,
    D1: np.ndarray,
    D2: np.ndarray,
    Deta_arr: np.ndarray,
    Dx: np.ndarray,
    dcount_arr: np.ndarray,
    gopen_arr: np.ndarray,
    s1_d: np.ndarray,
) -> None: ...
