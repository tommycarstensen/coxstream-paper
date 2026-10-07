"""
Benchmark: competing CoxPH implementations on synthetic survival data.

Uses data/survival_parts/ (10 covariates, Weibull PH, known true coef).
Two pre-sorted parquets are created once outside timing via DuckDB ORDER BY.
True coefficients are loaded from data/survival_parts/true_coef.json.

Methods
-------
Streaming NR  (O(batch_size * p^2) peak RAM; exact Efron MLE):
  coxstream_numpy   NumPy vectorized per batch (cumsum; ~160 MB/batch)
  coxstream_numba   Numba NJIT sequential per-row loop (~800 B carry)
  coxstream_jax     JAX JIT-compiled vectorized (XLA; same as NumPy RAM)
  coxstream_cython  Cython AOT sequential per-row loop (~800 B carry);
                    distinct-times kernel, full Newton steps
  coxstream_package the released coxstream package (fit_parquet): tie-aware
                    Efron, step-halving line search, records SEs + n_passes
  coxstream_cpp     C++ pybind11 sequential per-row loop (~800 B carry)
  coxstream_rs      Rust PyO3 sequential per-row loop (~800 B carry)

In-memory compiled  (all data in RAM; O(n) peak):
  renew_coxph_cython         RenewCoxPH Cython AOT efron_pass (pre-sorted)
  renew_coxph_cython_f32     RenewCoxPH Cython f32 efron_pass (pre-sorted)
  renew_coxph_cython_uns     RenewCoxPH Cython (unsorted; argsort in timing)
  renew_coxph_cpp            RenewCoxPH C++ efron_pass (pre-sorted)
  renew_coxph_cpp_uns        RenewCoxPH C++ (unsorted; argsort in timing)
  renew_coxph_rs             RenewCoxPH Rust efron_pass (pre-sorted)
  renew_coxph_rs_uns         RenewCoxPH Rust (unsorted; argsort in timing)
  renew_coxph_jax            RenewCoxPH JAX efron_pass (pre-sorted)
  renew_coxph_jax_uns        RenewCoxPH JAX (unsorted; argsort in timing)

In-memory Python / JVM  (all data in RAM):
  bigsurvsgd                 BigSurvSGD APPROXIMATE mini-batch SGD
  h2o_coxph                  H2O CoxPH (Java JVM; wall includes ~10s JVM init)
  lifelines                  lifelines.CoxPHFitter (MLE reference)
  sksurv                     scikit-survival CoxPHSurvivalAnalysis
  statsmodels_phreg          statsmodels.PHReg (Efron)

R methods  (via Rscript subprocess):
  r_survival_coxph           R survival::coxph (Efron, exact)
  xue_schifano_cuee          Xue & Schifano CUEE (streaming batch-wise, approx)

Sorted vs unsorted pairs isolate the sort overhead:
  renew_coxph_cython vs renew_coxph_cython_uns  -- NumPy argsort cost
  (likewise for cpp, rs, jax)

Streaming NR algorithm
----------------------
Reads a DESC-sorted parquet in batches (BATCH_SIZE rows each).  For each
NR iteration, one sequential pass accumulates carry state (S0, S1, S2,
ll, score, neg_H) across batches without storing the full dataset.  Peak
RAM = O(batch_size * p^2) for NumPy/JAX vectorized variants, O(p^2) for
compiled sequential variants (Numba, Cython, C++, Rust).
No log-sum-exp stabilization (safe for our data: max|eta| << 709).

Two input files (created once outside timing):
  data/competing_coxph_<nrows>_sorted.parquet       -- DuckDB ORDER BY ASC
  data/competing_coxph_<nrows>_sorted_desc.parquet  -- DuckDB ORDER BY DESC
  data/survival_parts/part_*.parquet                -- original unsorted parts

Generate the data first (not from VSCode -- CPU/disk intensive):
    python3 -m coxbench.generate_survival_data --nrows 30_000_000 --nparts 30

Run from the project root:
    python3.13 src/coxbench/bench_competing_coxph.py
    python3.13 src/coxbench/bench_competing_coxph.py --mrows 10
    python3 -m coxbench.bench_competing_coxph --methods lifelines
    python3.13 src/coxbench/bench_competing_coxph.py --reps 10

Each method is run --reps times (default 5), with the order shuffled per
pass.  Per-rep arrays plus mean/median/sd/se/cv are stored per row.  The
cache key is (method, n_rows, p, cache); cached reps count toward the --reps
target, so only the shortfall is run.  Results are written after every rep
(crash-safe).  Default cache state is cold (OS page cache purged before each
subprocess); pass --cache warm to skip purge.

Results written to results/competing_coxph.json.
"""

import argparse
import json
import math
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import polars as pl
import psutil
import pyarrow.parquet as pq
from scipy.linalg import solve

from coxbench.bench_utils import _purge_page_cache, check_sudo_purge
from coxbench.paths import data_path, results_path

# coxstream_source_hash is re-exported: cloud/run_coxph_nsweep.sh imports it
# from this module.
from coxbench.provenance import (
    coxstream_source_hash,
    host_label,
    scrub_paths,
)
from coxbench.sgd_coxph import BigSurvSGDCoxPH

_R_DISPATCHER = Path(__file__).parent / "r_competing_coxph.R"

SURVIVAL_DIR = data_path("survival_parts")
RESULTS_PATH = results_path("competing_coxph.json")
BATCH_SIZE = 200_000
# Interval at which the monitor thread samples a run's resident set size;
# gen_arxiv_macros quotes it in the paper.
RSS_POLL_S = 0.05

# Stamp every measured rep with the host so cross-machine rows (e.g. Mac vs
# a GCloud VM, whose RSS semantics differ) are never silently mixed.
_HOST = host_label()

_ALL_X_COLS = [
    "age_std", "sex", "treatment", "smoking_pack_years",
    "bmi_std", "systolic_bp_std", "hemoglobin_std",
    "log_creatinine", "albumin_std", "charlson_std",
]
X_COLS: list[str] = list(_ALL_X_COLS)
TIME_COL = "duration"
EVENT_COL = "event"


def _load_x_cols(parts_dir):
    """Ordered X-column names from the dataset's true_coef.json sidecar.

    The p-sweep dataset (--ncovariates 32) carries x11..x32 beyond the 10
    built-in covariates; the sidecar is the single source of truth for both
    the column order and how many exist.  Both the parent and the --run-one
    child resolve _ALL_X_COLS through this so a p>10 sweep reads the right
    columns instead of silently truncating to the 10 built-ins.
    """
    coef_path = Path(parts_dir) / "true_coef.json"
    with open(coef_path) as fh:
        return list(json.load(fh)["coef"].keys())


def _parquet_has_cols(path, cols):
    """True if the parquet at *path* contains every name in *cols*."""
    schema_names = set(pq.ParquetFile(str(path)).schema.names)
    return all(c in schema_names for c in cols)


# ---------------------------------------------------------------------------
# Pre-sorted parquet
# ---------------------------------------------------------------------------

def _ensure_sorted_parquet(parts, sorted_path):
    """Create a single time-sorted parquet from parts (DuckDB ORDER BY).

    Skipped if a cached copy already holds every X column; a stale copy
    built for fewer covariates (e.g. an older p=10 run sharing the same n) is
    rebuilt.
    """
    if sorted_path.exists() and _parquet_has_cols(sorted_path, _ALL_X_COLS):
        print(f"  Sorted parquet cached: {sorted_path}", flush=True)
        return
    print(
        f"  Building sorted parquet from {len(parts)} parts -> "
        f"{sorted_path} ...",
        flush=True,
    )
    import duckdb

    x_sql = ", ".join(_ALL_X_COLS)
    # Build an explicit list of part paths so we don't pick up stale files.
    part_list = ", ".join(f"'{p}'" for p in parts)
    con = duckdb.connect()
    con.execute(
        f"COPY ("
        f"  SELECT {TIME_COL}, {EVENT_COL}, {x_sql}"
        f"  FROM read_parquet([{part_list}])"
        f"  ORDER BY {TIME_COL}"
        f") TO '{sorted_path}' (FORMAT PARQUET, COMPRESSION SNAPPY, "
        f"ROW_GROUP_SIZE {BATCH_SIZE})"
    )
    con.close()
    size_mb = sorted_path.stat().st_size / 1024 ** 2
    print(f"  -> {sorted_path} ({size_mb:.0f} MB)", flush=True)


def _ensure_sorted_desc_parquet(parts, desc_path):
    """Create one DESC-time-sorted parquet from parts (DuckDB ORDER BY DESC).

    Required by streaming NR methods.  Skipped if a cached copy already holds
    every X column; a stale copy built for fewer covariates is rebuilt.
    """
    if desc_path.exists() and _parquet_has_cols(desc_path, _ALL_X_COLS):
        print(f"  DESC-sorted parquet cached: {desc_path}", flush=True)
        return
    print(
        f"  Building DESC-sorted parquet from {len(parts)} parts -> "
        f"{desc_path} ...",
        flush=True,
    )
    import duckdb

    x_sql = ", ".join(_ALL_X_COLS)
    part_list = ", ".join(f"'{p}'" for p in parts)
    con = duckdb.connect()
    con.execute(
        f"COPY ("
        f"  SELECT {TIME_COL}, {EVENT_COL}, {x_sql}"
        f"  FROM read_parquet([{part_list}])"
        f"  ORDER BY {TIME_COL} DESC"
        f") TO '{desc_path}' (FORMAT PARQUET, COMPRESSION SNAPPY, "
        f"ROW_GROUP_SIZE {BATCH_SIZE})"
    )
    con.close()
    size_mb = desc_path.stat().st_size / 1024 ** 2
    print(f"  -> {desc_path} ({size_mb:.0f} MB)", flush=True)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_sorted_numpy(sorted_path, n_rows):
    """Load first n_rows rows from sorted parquet; return (t, e, X) float64."""
    cols = [TIME_COL, EVENT_COL, *X_COLS]
    n_batches = math.ceil(n_rows / BATCH_SIZE)
    pf = pq.ParquetFile(str(sorted_path), memory_map=True)
    ts, es, Xs = [], [], []
    rows_left = n_rows
    for b, batch in enumerate(pf.iter_batches(BATCH_SIZE, columns=cols), 1):
        print(f"    loading batch {b}/{n_batches}", end="\r", flush=True)
        chunk = min(len(batch), rows_left)
        if chunk < len(batch):
            batch = batch.slice(0, chunk)
        ts.append(batch[TIME_COL].to_numpy(zero_copy_only=False)
                  .astype(np.float64))
        es.append(
            batch[EVENT_COL].to_numpy(zero_copy_only=False).astype(np.int32)
        )
        Xs.append(
            np.column_stack([
                batch[c].to_numpy(zero_copy_only=False)
                for c in X_COLS
            ]).astype(np.float64)
        )
        rows_left -= chunk
        if rows_left <= 0:
            break
    print(flush=True)
    t = np.concatenate(ts)
    e = np.concatenate(es)
    X = np.vstack(Xs)
    return t, e, X


def _load_sorted_numpy_f32(sorted_path, n_rows):
    """Like _load_sorted_numpy but returns float32 (t and X)."""
    cols = [TIME_COL, EVENT_COL, *X_COLS]
    n_batches = math.ceil(n_rows / BATCH_SIZE)
    pf = pq.ParquetFile(str(sorted_path), memory_map=True)
    ts, es, Xs = [], [], []
    rows_left = n_rows
    for b, batch in enumerate(pf.iter_batches(BATCH_SIZE, columns=cols), 1):
        print(f"    loading batch {b}/{n_batches} (f32)", end="\r", flush=True)
        chunk = min(len(batch), rows_left)
        if chunk < len(batch):
            batch = batch.slice(0, chunk)
        ts.append(batch[TIME_COL].to_numpy(zero_copy_only=False)
                  .astype(np.float32))
        es.append(
            batch[EVENT_COL].to_numpy(zero_copy_only=False).astype(np.int32)
        )
        Xs.append(
            np.column_stack([
                batch[c].to_numpy(zero_copy_only=False)
                for c in X_COLS
            ]).astype(np.float32)
        )
        rows_left -= chunk
        if rows_left <= 0:
            break
    print(flush=True)
    t = np.concatenate(ts)
    e = np.concatenate(es)
    X = np.vstack(Xs)
    return t, e, X


def _iter_sorted_batches_arrow(sorted_path, n_rows):
    """Yield (t, e, X) batches from sorted parquet via pyarrow."""
    cols = [TIME_COL, EVENT_COL, *X_COLS]
    n_batches = math.ceil(n_rows / BATCH_SIZE)
    pf = pq.ParquetFile(str(sorted_path), memory_map=False)
    rows_left = n_rows
    for b, batch in enumerate(pf.iter_batches(BATCH_SIZE, columns=cols), 1):
        print(f"    loading batch {b}/{n_batches}", end="\r", flush=True)
        chunk = min(len(batch), rows_left)
        if chunk < len(batch):
            batch = batch.slice(0, chunk)
        t = batch[TIME_COL].to_numpy(zero_copy_only=False).astype(np.float64)
        e = batch[EVENT_COL].to_numpy(zero_copy_only=False).astype(np.int32)
        X = np.column_stack([
            batch[c].to_numpy(zero_copy_only=False) for c in X_COLS
        ]).astype(np.float64)
        rows_left -= chunk
        yield t, e, X
        if rows_left <= 0:
            break
    print(flush=True)


def _iter_sorted_batches_polars(sorted_path, n_rows):
    """Yield (t, e, X) batches from sorted parquet via polars scan+slice."""
    cols = [TIME_COL, EVENT_COL, *X_COLS]
    n_batches = math.ceil(n_rows / BATCH_SIZE)
    offset = 0
    b = 0
    while offset < n_rows:
        b += 1
        length = min(BATCH_SIZE, n_rows - offset)
        print(f"    loading batch {b}/{n_batches} (polars)", end="\r",
              flush=True)
        df = (
            pl.scan_parquet(str(sorted_path))
            .select(cols)
            .slice(offset, length)
            .collect()
        )
        t = df[TIME_COL].to_numpy().astype(np.float64)
        e = df[EVENT_COL].cast(pl.Int32).to_numpy()
        X = np.column_stack(
            [df[c].to_numpy() for c in X_COLS]).astype(np.float64)
        offset += length
        yield t, e, X
    print(flush=True)


def _iter_desc_batches_arrow(pf, n_rows):
    """Yield (e, X) float64 batches from an open DESC-sorted ParquetFile, one
    row group at a time.

    Used by streaming NR methods: rows arrive in descending-time order so the
    running risk-set accumulators (S0, S1, S2) grow correctly batch by batch.

    Reads via ``read_row_group``, NOT ``iter_batches``: iter_batches
    read-aheads and buffers proportional to the file size, so peak RSS grew
    with n (160 -> 588 MB over 1M -> 8M, on either the mimalloc or system Arrow
    allocator). Reading one row group at a time is flat (95 -> 118 MB) and
    matches the R C++ streaming reader. Row groups hold BATCH_SIZE rows (set
    via ROW_GROUP_SIZE at write time), so the peak footprint is
    O(BATCH_SIZE * p), independent of n. The time column is not yielded
    (the ordering is already correct).
    """
    cols = [EVENT_COL, *X_COLS]
    n_rg = pf.metadata.num_row_groups
    rows_left = n_rows
    for rg in range(n_rg):
        if rows_left <= 0:
            break
        print(f"    stream row group {rg + 1}/{n_rg}", end="\r", flush=True)
        batch = pf.read_row_group(rg, columns=cols)
        chunk = min(len(batch), rows_left)
        if chunk < len(batch):
            batch = batch.slice(0, chunk)
        e = batch[EVENT_COL].to_numpy(zero_copy_only=False).astype(np.int32)
        X = np.column_stack([
            batch[c].to_numpy(zero_copy_only=False) for c in X_COLS
        ]).astype(np.float64)
        rows_left -= chunk
        yield e, X
    print(flush=True)


def _load_unsorted_numpy(parts, n_rows):
    """Load n_rows rows from unsorted parts; return (t, e, X) float64.

    Rows arrive in file order (unsorted by time).  Callers that need sorted
    data must call np.argsort(t) themselves -- and that sort is then inside
    the method's timing, showing the true sort overhead.
    """
    cols = [TIME_COL, EVENT_COL, *X_COLS]
    ts, es, Xs = [], [], []
    rows_left = n_rows
    n_parts = math.ceil(n_rows / 1_000_000)
    for i, part in enumerate(parts[:n_parts], 1):
        print(f"    loading part {i}/{n_parts} (unsorted)", end="\r",
              flush=True)
        pf = pq.ParquetFile(str(part), memory_map=True)
        for batch in pf.iter_batches(BATCH_SIZE, columns=cols):
            chunk = min(len(batch), rows_left)
            if chunk < len(batch):
                batch = batch.slice(0, chunk)
            ts.append(
                batch[TIME_COL].to_numpy(zero_copy_only=False)
                .astype(np.float64)
            )
            es.append(
                batch[EVENT_COL].to_numpy(zero_copy_only=False)
                .astype(np.int32)
            )
            Xs.append(
                np.column_stack([
                    batch[c].to_numpy(zero_copy_only=False) for c in X_COLS
                ]).astype(np.float64)
            )
            rows_left -= chunk
            if rows_left <= 0:
                break
        if rows_left <= 0:
            break
    print(flush=True)
    return np.concatenate(ts), np.concatenate(es), np.vstack(Xs)


def _iter_unsorted_batches_arrow(parts, n_rows):
    """Yield (t, e, X) batches from unsorted parts via pyarrow."""
    cols = [TIME_COL, EVENT_COL, *X_COLS]
    n_parts = math.ceil(n_rows / 1_000_000)
    rows_left = n_rows
    total_batches = math.ceil(n_rows / BATCH_SIZE)
    b = 0
    for part in parts[:n_parts]:
        if rows_left <= 0:
            break
        pf = pq.ParquetFile(str(part), memory_map=True)
        for batch in pf.iter_batches(BATCH_SIZE, columns=cols):
            b += 1
            print(f"    loading batch {b}/{total_batches} (unsorted)",
                  end="\r", flush=True)
            chunk = min(len(batch), rows_left)
            if chunk < len(batch):
                batch = batch.slice(0, chunk)
            t = (batch[TIME_COL].to_numpy(zero_copy_only=False)
                 .astype(np.float64))
            e = (batch[EVENT_COL].to_numpy(zero_copy_only=False)
                 .astype(np.int32))
            X = np.column_stack([
                batch[c].to_numpy(zero_copy_only=False) for c in X_COLS
            ]).astype(np.float64)
            rows_left -= chunk
            yield t, e, X
            if rows_left <= 0:
                break
    print(flush=True)


# ---------------------------------------------------------------------------
# NR loop (for compiled efron_pass backends)
# ---------------------------------------------------------------------------

def _nr_loop(efron_pass_fn, t_d, e_d, X_d, tag, tol=1e-8, max_iter=50):
    """Newton-Raphson over a compiled efron_pass -> (ll, score, neg_H).

    t_d / e_d / X_d must be sorted DESCENDING (descending time = standard
    sorted-pass convention for the Efron correction).
    """
    beta = np.zeros(X_d.shape[1])
    n_iter = 0
    for it in range(max_iter):
        ll, score, neg_H = efron_pass_fn(t_d, e_d, X_d, beta)
        step = solve(neg_H, score, assume_a="pos")
        alpha = 1.0
        for _ in range(15):
            ll_new, _, _ = efron_pass_fn(t_d, e_d, X_d, beta + alpha * step)
            if ll_new >= ll - 1e-10:
                break
            alpha *= 0.5
        beta_new = beta + alpha * step
        n_iter = it + 1
        print(f"  [{tag}]  iter {n_iter}", flush=True)
        if np.linalg.norm(beta_new - beta) < tol:
            beta = beta_new
            break
        beta = beta_new
    return beta, n_iter


def _nr_loop_f32(efron_pass_fn, t_d, e_d, X_d, tag, max_iter=50):
    """Like _nr_loop but for float32 efron_pass (coxph_cython_f32).

    Convergence is checked on float32-rounded betas to avoid infinite loops
    when the f64 step norm stays above tol due to float32 ULP quantisation.
    """
    beta = np.zeros(X_d.shape[1])
    n_iter = 0
    for it in range(max_iter):
        beta_f32 = beta.astype(np.float32)
        ll, score, neg_H = efron_pass_fn(t_d, e_d, X_d, beta_f32)
        step = solve(neg_H, score, assume_a="pos")
        alpha = 1.0
        for _ in range(15):
            b_f32 = (beta + alpha * step).astype(np.float32)
            ll_new, _, _ = efron_pass_fn(t_d, e_d, X_d, b_f32)
            if ll_new >= ll - 1e-10:
                break
            alpha *= 0.5
        beta_new = beta + alpha * step
        n_iter = it + 1
        print(f"  [{tag}]  iter {n_iter}", flush=True)
        if np.linalg.norm(
            beta_new.astype(np.float32) - beta.astype(np.float32)
        ) < 1e-6:
            beta = beta_new
            break
        beta = beta_new
    return beta, n_iter


# ---------------------------------------------------------------------------
# Streaming NR loop (for streaming efron_stream_batch backends)
# ---------------------------------------------------------------------------

def _stream_nr_loop(desc_path, n_rows, batch_fn, p, tag, warmup_fn=None,
                    tol=1e-8, max_iter=50):
    """Newton-Raphson over a streaming efron_stream_batch backend.

    Each NR iteration makes one sequential pass over the DESC-sorted parquet,
    accumulating carry state (S0, S1, S2, ll, score, neg_H) batch by batch.
    Peak RAM: O(batch_size * p^2) for vectorized variants, O(p^2) for compiled.

    No step-halving: an extra backward pass per backtrack would double I/O.
    Well-conditioned CoxPH data (our synthetic Weibull) converges without it.

    Parameters
    ----------
    desc_path : Path    DESC-sorted parquet (time DESCENDING)
    n_rows    : int
    batch_fn  : callable  efron_stream_batch(e, X, beta, S0, S1, S2, ll, sc,
                          nH)
    p         : int       number of covariates
    tag       : str       label for progress prints
    warmup_fn : callable  called once before the first timed iteration
    """
    if warmup_fn is not None:
        warmup_fn()
    # Open the reader ONCE and reuse it across every NR pass. Re-opening a
    # fresh ParquetFile per pass leaks ~O(file) via the mimalloc pool retaining
    # freed read buffers (see _iter_desc_batches_arrow), which is what inflated
    # peak RSS at large n and masked the O(p^2) footprint.
    pf = pq.ParquetFile(str(desc_path), memory_map=False)
    beta = np.zeros(p, dtype=np.float64)
    n_iter = 0
    for it in range(max_iter):
        S0 = np.zeros(1, dtype=np.float64)
        S1 = np.zeros(p, dtype=np.float64)
        # F-order for S2/neg_H: required by C++ Eigen::Ref<MatrixXd> binding;
        # harmless for NumPy/Numba/Cython/Rust (element access ignores layout).
        S2 = np.zeros((p, p), dtype=np.float64, order='F')
        ll = np.zeros(1, dtype=np.float64)
        score = np.zeros(p, dtype=np.float64)
        neg_H = np.zeros((p, p), dtype=np.float64, order='F')
        for e_b, X_b in _iter_desc_batches_arrow(pf, n_rows):
            batch_fn(e_b, X_b, beta, S0, S1, S2, ll, score, neg_H)
        step = solve(neg_H, score, assume_a="pos")
        beta_new = beta + step
        n_iter = it + 1
        print(f"  [{tag}]  iter {n_iter}  ll={ll[0]:.6f}", flush=True)
        if np.linalg.norm(beta_new - beta) < tol:
            beta = beta_new
            break
        beta = beta_new
    return beta, n_iter


# ---------------------------------------------------------------------------
# Subprocess polling
# ---------------------------------------------------------------------------

def _run_subprocess_with_polling(cmd, out_path, method_name):
    """Spawn cmd, poll its RSS every RSS_POLL_S via psutil, wait, parse JSON.

    Returns dict with coef, n_iter, elapsed_s, rss_peak_mb, rss_delta_mb.
    """
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    ps = psutil.Process(proc.pid)
    try:
        before = ps.memory_info().rss / 1024 ** 2
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        before = 0.0
    peak = [before]
    stop = threading.Event()

    def _poll():
        while not stop.is_set():
            try:
                now = ps.memory_info().rss / 1024 ** 2
                peak[0] = max(peak[0], now)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            stop.wait(RSS_POLL_S)

    mon = threading.Thread(target=_poll, daemon=True)
    t0 = time.perf_counter()
    mon.start()
    stdout, stderr = proc.communicate()
    elapsed = time.perf_counter() - t0
    stop.set()
    mon.join(timeout=1)

    if proc.returncode != 0:
        raise RuntimeError(
            f"subprocess {method_name} failed "
            f"(returncode={proc.returncode}):\n"
            f"--- stdout ---\n{stdout.decode(errors='replace')}\n"
            f"--- stderr ---\n{stderr.decode(errors='replace')}"
        )
    if not Path(out_path).exists():
        raise RuntimeError(
            f"subprocess {method_name} exited 0 but did not write {out_path}"
        )
    result = json.loads(Path(out_path).read_text())
    coef = np.asarray(result["coef"], dtype=np.float64)
    return {
        "coef": coef,
        "n_iter": int(result.get("n_iter", -1)),
        "se": result.get("se"),
        "n_passes": result.get("n_passes"),
        "pkg_src": result.get("pkg_src"),
        "elapsed_s": elapsed,
        "rss_peak_mb": peak[0],
        "rss_delta_mb": peak[0] - before,
    }


def _run_method_subproc(method_key, sorted_path, desc_path, parts, n_rows,
                        args, p: int = 10, cold: bool = False):
    """Spawn one CoxPH method fit in a fresh subprocess; return its dict."""
    if cold:
        _purge_page_cache()
    _, is_r = METHODS[method_key]
    # Use the actual covariate count after _ALL_X_COLS[:p] clamping.  If the
    # requested p exceeds the dataset's covariate count, X_COLS is shorter than
    # p and both the Python subprocess and R must use the effective count so
    # their returned coef vectors match true_coef.
    effective_p = len(X_COLS)
    if is_r:
        return _run_r_competing(method_key, desc_path, n_rows, effective_p)
    parquet_path = (sorted_path if method_key in _NEEDS_ASC_SORTED
                    else desc_path)
    tmpdir = Path(tempfile.mkdtemp(prefix=f"coxph_{method_key}_"))
    out = tmpdir / "out.json"
    try:
        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--run-one", method_key,
            "--sorted-path", str(parquet_path),
            "--n-rows", str(n_rows),
            "--sgd-epochs", str(args.sgd_epochs),
            "--sgd-strata", str(args.sgd_strata),
            "--p-value", str(effective_p),
            "--parts-dir", str(args.parts_dir),
            "--out", str(out),
        ]
        if method_key in _UNSORTED_METHODS:
            # Pass part file paths so child can read unsorted data.
            n_parts = math.ceil(n_rows / 1_000_000)
            cmd += ["--parts"] + [str(part) for part in parts[:n_parts]]
        return _run_subprocess_with_polling(cmd, out, method_key)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _run_r_competing(method_name, sorted_path, n_rows, p: int = 10):
    """Spawn r_competing_coxph.R for one R method; return measurement dict."""
    tmpdir = Path(tempfile.mkdtemp(prefix=f"coxph_{method_name}_"))
    out = tmpdir / "out.json"
    try:
        cmd = [
            "Rscript", str(_R_DISPATCHER),
            method_name, str(sorted_path), str(n_rows), str(out), str(p),
        ]
        return _run_subprocess_with_polling(cmd, out, method_name)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Rep accounting helpers
# ---------------------------------------------------------------------------

_SUMMARY_KEYS = ("elapsed_s", "rss_peak_mb", "rss_delta_mb")


def _migrate_row(row):
    """Promote scalar fields from pre-reps rows to per-rep arrays."""
    if "elapsed_s_reps" not in row:
        es = row.get("elapsed_s")
        row["elapsed_s_reps"] = [es] if isinstance(es, (int, float)) else []
    if "n_iter_reps" not in row:
        ni = row.get("n_iter")
        row["n_iter_reps"] = [ni] if isinstance(ni, (int, float)) else []
    if "rss_peak_mb_reps" not in row:
        row["rss_peak_mb_reps"] = []
    if "rss_delta_mb_reps" not in row:
        row["rss_delta_mb_reps"] = []
    if "p" not in row:
        row["p"] = len(row.get("coef_names", _ALL_X_COLS))
    if "cache" not in row:
        row["cache"] = "warm"
    return row


def _update_summary(row):
    """Recompute mean/median/sd/se/cv from rep arrays."""
    for key in _SUMMARY_KEYS:
        reps = row[f"{key}_reps"]
        if not reps:
            continue
        arr = np.asarray(reps, dtype=float)
        mean = float(arr.mean())
        med = float(np.median(arr))
        n = arr.size
        if n > 1:
            sd = float(arr.std(ddof=1))
            se = sd / np.sqrt(n)
            cv = sd / mean if mean != 0 else None
        else:
            sd = se = cv = None
        row[key] = round(mean, 3)
        row[f"mean_{key}"] = round(mean, 3)
        row[f"median_{key}"] = round(med, 3)
        row[f"sd_{key}"] = round(sd, 4) if sd is not None else None
        row[f"se_{key}"] = round(se, 4) if se is not None else None
        row[f"cv_{key}"] = round(cv, 4) if cv is not None else None
    row["n_reps"] = len(row["elapsed_s_reps"])


def _reps_done(row):
    if row is None:
        return 0
    return min(
        len(row["elapsed_s_reps"]),
        len(row["rss_peak_mb_reps"]),
    )


# ---------------------------------------------------------------------------
# Fit functions  (each returns (coef: np.ndarray, n_iter: int))
# ---------------------------------------------------------------------------

def _optional_backend(name):
    """Import an optional compiled or JVM backend by name, at call time.

    coxph_cpp / coxph_rs are built from packages/coxph-cpp and
    packages/coxph-rs, and h2o needs a JVM; most hosts have none of them, so
    they are resolved only when their method runs (as coxph_cython is).
    """
    import importlib
    return importlib.import_module(name)


def _desc_path_for(sorted_path):
    """Derive the DESC-sorted parquet path from the ASC sorted path."""
    return Path(str(sorted_path).replace("_sorted.parquet",
                                         "_sorted_desc.parquet"))


def fit_coxstream_numpy(sorted_path, n_rows, **_kw):
    """Streaming NR (NumPy vectorized cumsum): exact MLE, O(batch*p^2) RAM."""
    from coxbench.coxph_stream import efron_stream_batch_numpy
    desc_path = _desc_path_for(sorted_path)
    p = len(X_COLS)
    print(f"    streaming NR (numpy, {n_rows:,} rows) ...", flush=True)
    return _stream_nr_loop(
        desc_path, n_rows, efron_stream_batch_numpy, p, tag="numpy"
    )


def fit_coxstream_numba(sorted_path, n_rows, **_kw):
    """Streaming NR (Numba NJIT sequential): exact MLE, O(p^2) carry RAM."""
    from coxbench.coxph_stream import (
        efron_stream_batch_numba,
        warmup_numba,
    )
    desc_path = _desc_path_for(sorted_path)
    p = len(X_COLS)
    print(f"    JIT warmup (numba, p={p}) ...", flush=True)
    return _stream_nr_loop(
        desc_path, n_rows, efron_stream_batch_numba, p, tag="numba",
        warmup_fn=lambda: warmup_numba(p),
    )


def fit_coxstream_jax(sorted_path, n_rows, **_kw):
    """Streaming NR (JAX JIT-compiled XLA): exact MLE, O(batch*p^2) RAM."""
    from coxbench.coxph_stream import (
        efron_stream_batch_jax,
        warmup_jax,
    )
    desc_path = _desc_path_for(sorted_path)
    p = len(X_COLS)
    print(f"    JIT warmup (jax, p={p}) ...", flush=True)
    return _stream_nr_loop(
        desc_path, n_rows, efron_stream_batch_jax, p, tag="jax",
        warmup_fn=lambda: warmup_jax(p),
    )


def fit_coxstream_cython(sorted_path, n_rows, **_kw):
    """Streaming NR (Cython AOT, distinct-times kernel): O(p^2) carry RAM."""
    import importlib
    mod = importlib.import_module("coxph_cython")
    desc_path = _desc_path_for(sorted_path)
    p = len(X_COLS)
    print(f"    streaming NR (cython, {n_rows:,} rows) ...", flush=True)
    return _stream_nr_loop(
        desc_path, n_rows, mod.efron_stream_batch, p, tag="cython_stream"
    )


def fit_coxstream_package(sorted_path, n_rows, **_kw):
    """The released coxstream package (PyPI): ``CoxStream.fit_parquet``.

    What a reader installs: the tie-aware Efron kernel, the step-halving line
    search, one pass per iteration and model-based SEs from the final pass.
    Streams the DESC parquet one row group (``BATCH_SIZE`` rows) at a time.
    """
    import coxstream
    import pyarrow.parquet as pq
    from coxstream import CoxStream
    desc_path = _desc_path_for(sorted_path)
    n_file = pq.ParquetFile(str(desc_path)).metadata.num_rows
    if n_file != n_rows:
        raise RuntimeError(
            f"{desc_path} holds {n_file:,} rows, not n_rows={n_rows:,}; "
            "fit_parquet fits the whole file"
        )
    print(f"    coxstream package fit_parquet ({n_rows:,} rows) ...",
          flush=True)
    model = CoxStream().fit_parquet(
        str(desc_path), TIME_COL, EVENT_COL, list(X_COLS)
    )
    assert model.standard_errors_ is not None
    return model.coef_, model.n_iter_, {
        "se": model.standard_errors_.tolist(),
        "n_passes": model.n_passes_,
        "pkg_src": coxstream_source_hash(Path(coxstream.__file__).parent),
    }


def fit_coxstream_cpp(sorted_path, n_rows, **_kw):
    """Streaming NR (C++ pybind11 sequential): exact MLE, O(p^2) carry RAM."""
    _stream_cpp = _optional_backend("coxph_cpp").efron_stream_batch
    desc_path = _desc_path_for(sorted_path)
    p = len(X_COLS)
    print(f"    streaming NR (cpp, {n_rows:,} rows) ...", flush=True)
    return _stream_nr_loop(
        desc_path, n_rows, _stream_cpp, p, tag="cpp_stream"
    )


def fit_coxstream_rs(sorted_path, n_rows, **_kw):
    """Streaming NR (Rust PyO3 sequential): exact MLE, O(p^2) carry RAM."""
    _stream_rs = _optional_backend("coxph_rs").efron_stream_batch
    desc_path = _desc_path_for(sorted_path)
    p = len(X_COLS)
    print(f"    streaming NR (rust, {n_rows:,} rows) ...", flush=True)
    return _stream_nr_loop(
        desc_path, n_rows, _stream_rs, p, tag="rs_stream"
    )


def fit_bigsurvsgd(sorted_path, n_rows, strata_size=50, n_epochs=10, **_kw):
    """BigSurvSGD APPROXIMATE streaming mini-batch SGD."""
    model = BigSurvSGDCoxPH(
        strata_size=strata_size,
        batch_size=1,
        n_epochs=n_epochs,
        opt_method="AMSGrad",
        beta_type="averaged",
        standardize=True,
    )
    for t, e, X in _iter_sorted_batches_arrow(sorted_path, n_rows):
        model.partial_fit(t, e, X)
    print("    fitting (SGD) ...", flush=True)
    model.fit()
    return model.coef_, model.n_iter_


def fit_renew_coxph_cython(sorted_path, n_rows, **_kw):
    """RenewCoxPH Cython AOT efron_pass, in-memory, float64."""
    import importlib
    mod = importlib.import_module("coxph_cython")
    t_asc, e_asc, X_asc = _load_sorted_numpy(sorted_path, n_rows)
    t_d = np.ascontiguousarray(t_asc[::-1], dtype=np.float64)
    e_d = np.ascontiguousarray(e_asc[::-1], dtype=np.int32)
    X_d = np.ascontiguousarray(X_asc[::-1], dtype=np.float64)
    del t_asc, e_asc, X_asc
    print(f"    fitting cython ({n_rows:,} rows) ...", flush=True)
    return _nr_loop(mod.efron_pass, t_d, e_d, X_d, "cython")


def fit_renew_coxph_cython_f32(sorted_path, n_rows, **_kw):
    """RenewCoxPH Cython f32 efron_pass, in-memory, float32."""
    import importlib
    mod = importlib.import_module("coxph_cython_f32")
    t_asc, e_asc, X_asc = _load_sorted_numpy_f32(sorted_path, n_rows)
    t_d = np.ascontiguousarray(t_asc[::-1], dtype=np.float32)
    e_d = np.ascontiguousarray(e_asc[::-1], dtype=np.int32)
    X_d = np.ascontiguousarray(X_asc[::-1], dtype=np.float32)
    del t_asc, e_asc, X_asc
    print(f"    fitting cython_f32 ({n_rows:,} rows) ...", flush=True)
    return _nr_loop_f32(mod.efron_pass, t_d, e_d, X_d, "cython_f32")


def fit_renew_coxph_cpp(sorted_path, n_rows, **_kw):
    """RenewCoxPH C++ efron_pass, in-memory, float64."""
    _efron_cpp = _optional_backend("coxph_cpp").efron_pass
    t_asc, e_asc, X_asc = _load_sorted_numpy(sorted_path, n_rows)
    t_d = np.ascontiguousarray(t_asc[::-1], dtype=np.float64)
    e_d = np.ascontiguousarray(e_asc[::-1], dtype=np.int32)
    X_d = np.ascontiguousarray(X_asc[::-1], dtype=np.float64)
    del t_asc, e_asc, X_asc
    print(f"    fitting cpp ({n_rows:,} rows) ...", flush=True)
    return _nr_loop(_efron_cpp, t_d, e_d, X_d, "cpp")


def fit_renew_coxph_rs(sorted_path, n_rows, **_kw):
    """RenewCoxPH Rust efron_pass, in-memory, float64."""
    _efron_rs = _optional_backend("coxph_rs").efron_pass
    t_asc, e_asc, X_asc = _load_sorted_numpy(sorted_path, n_rows)
    t_d = np.ascontiguousarray(t_asc[::-1], dtype=np.float64)
    e_d = np.ascontiguousarray(e_asc[::-1], dtype=np.int32)
    X_d = np.ascontiguousarray(X_asc[::-1], dtype=np.float64)
    del t_asc, e_asc, X_asc
    print(f"    fitting rust ({n_rows:,} rows) ...", flush=True)
    return _nr_loop(_efron_rs, t_d, e_d, X_d, "rust")


def fit_renew_coxph_jax(sorted_path, n_rows, **_kw):
    """RenewCoxPH JAX efron_pass, in-memory, float64 (jax_enable_x64=True)."""
    from coxbench.coxph_jax import efron_pass as _efron_jax
    from coxbench.coxph_jax import warmup as _warmup_jax
    print(f"    JIT warmup (p={len(X_COLS)}) ...", flush=True)
    _warmup_jax(p=len(X_COLS))
    t_asc, e_asc, X_asc = _load_sorted_numpy(sorted_path, n_rows)
    t_d = np.ascontiguousarray(t_asc[::-1], dtype=np.float64)
    e_d = np.ascontiguousarray(e_asc[::-1], dtype=np.int32)
    X_d = np.ascontiguousarray(X_asc[::-1], dtype=np.float64)
    del t_asc, e_asc, X_asc
    print(f"    fitting jax ({n_rows:,} rows) ...", flush=True)
    return _nr_loop(_efron_jax, t_d, e_d, X_d, "jax")


def fit_renew_coxph_jax_x32(sorted_path, n_rows, **_kw):
    """RenewCoxPH JAX efron_pass, float32 (jax_enable_x64=False)."""
    from coxbench.coxph_jax_x32 import (
        efron_pass_x32 as _efron_jax_x32,
    )
    from coxbench.coxph_jax_x32 import warmup_x32 as _warmup_x32
    print(f"    JIT warmup (jax x32, p={len(X_COLS)}) ...", flush=True)
    _warmup_x32(p=len(X_COLS))
    t_asc, e_asc, X_asc = _load_sorted_numpy(sorted_path, n_rows)
    t_d = np.ascontiguousarray(t_asc[::-1], dtype=np.float64)
    e_d = np.ascontiguousarray(e_asc[::-1], dtype=np.int32)
    X_d = np.ascontiguousarray(X_asc[::-1], dtype=np.float64)
    del t_asc, e_asc, X_asc
    print(f"    fitting jax_x32 ({n_rows:,} rows) ...", flush=True)
    return _nr_loop(_efron_jax_x32, t_d, e_d, X_d, "jax_x32")


def fit_renew_coxph_jax_uns_x32(sorted_path, n_rows, parts=None, **_kw):
    """RenewCoxPH JAX float32: unsorted parts; np.argsort in timing."""
    from coxbench.coxph_jax_x32 import (
        efron_pass_x32 as _efron_jax_x32,
    )
    from coxbench.coxph_jax_x32 import warmup_x32 as _warmup_x32
    print(f"    JIT warmup (jax x32 uns, p={len(X_COLS)}) ...", flush=True)
    _warmup_x32(p=len(X_COLS))
    t, e, X = _load_unsorted_numpy(parts, n_rows)
    print(f"    argsort ({n_rows:,} rows) ...", flush=True)
    order = np.argsort(t, kind="stable")[::-1]
    t_d = np.ascontiguousarray(t[order], dtype=np.float64)
    e_d = np.ascontiguousarray(e[order], dtype=np.int32)
    X_d = np.ascontiguousarray(X[order], dtype=np.float64)
    del t, e, X
    print(f"    fitting jax_x32_uns ({n_rows:,} rows) ...", flush=True)
    return _nr_loop(_efron_jax_x32, t_d, e_d, X_d, "jax_x32_uns")


def fit_coxstream_jax_x32(sorted_path, n_rows, **_kw):
    """Streaming NR JAX float32 (jax_enable_x64=False): exact Efron MLE."""
    from coxbench.coxph_stream import (
        efron_stream_batch_jax_x32,
        warmup_jax_x32,
    )
    desc_path = _desc_path_for(sorted_path)
    p = len(X_COLS)
    print(f"    JIT warmup (jax x32 stream, p={p}) ...", flush=True)
    return _stream_nr_loop(
        desc_path, n_rows, efron_stream_batch_jax_x32, p, tag="jax_x32_stream",
        warmup_fn=lambda: warmup_jax_x32(p),
    )


def fit_renew_coxph_cython_uns(sorted_path, n_rows, parts=None, **_kw):
    """RenewCoxPH Cython AOT: unsorted parts; np.argsort in timing."""
    import importlib
    mod = importlib.import_module("coxph_cython")
    t, e, X = _load_unsorted_numpy(parts, n_rows)
    print(f"    argsort ({n_rows:,} rows) ...", flush=True)
    order = np.argsort(t, kind="stable")[::-1]
    t_d = np.ascontiguousarray(t[order], dtype=np.float64)
    e_d = np.ascontiguousarray(e[order], dtype=np.int32)
    X_d = np.ascontiguousarray(X[order], dtype=np.float64)
    del t, e, X
    print(f"    fitting cython_uns ({n_rows:,} rows) ...", flush=True)
    return _nr_loop(mod.efron_pass, t_d, e_d, X_d, "cython_uns")


def fit_renew_coxph_cpp_uns(sorted_path, n_rows, parts=None, **_kw):
    """RenewCoxPH C++: unsorted parts; np.argsort in timing."""
    _efron_cpp = _optional_backend("coxph_cpp").efron_pass
    t, e, X = _load_unsorted_numpy(parts, n_rows)
    print(f"    argsort ({n_rows:,} rows) ...", flush=True)
    order = np.argsort(t, kind="stable")[::-1]
    t_d = np.ascontiguousarray(t[order], dtype=np.float64)
    e_d = np.ascontiguousarray(e[order], dtype=np.int32)
    X_d = np.ascontiguousarray(X[order], dtype=np.float64)
    del t, e, X
    print(f"    fitting cpp_uns ({n_rows:,} rows) ...", flush=True)
    return _nr_loop(_efron_cpp, t_d, e_d, X_d, "cpp_uns")


def fit_renew_coxph_rs_uns(sorted_path, n_rows, parts=None, **_kw):
    """RenewCoxPH Rust: unsorted parts; np.argsort in timing."""
    _efron_rs = _optional_backend("coxph_rs").efron_pass
    t, e, X = _load_unsorted_numpy(parts, n_rows)
    print(f"    argsort ({n_rows:,} rows) ...", flush=True)
    order = np.argsort(t, kind="stable")[::-1]
    t_d = np.ascontiguousarray(t[order], dtype=np.float64)
    e_d = np.ascontiguousarray(e[order], dtype=np.int32)
    X_d = np.ascontiguousarray(X[order], dtype=np.float64)
    del t, e, X
    print(f"    fitting rs_uns ({n_rows:,} rows) ...", flush=True)
    return _nr_loop(_efron_rs, t_d, e_d, X_d, "rs_uns")


def fit_renew_coxph_jax_uns(sorted_path, n_rows, parts=None, **_kw):
    """RenewCoxPH JAX: unsorted parts; np.argsort in timing."""
    from coxbench.coxph_jax import efron_pass as _efron_jax
    from coxbench.coxph_jax import warmup as _warmup_jax
    print(f"    JIT warmup (p={len(X_COLS)}) ...", flush=True)
    _warmup_jax(p=len(X_COLS))
    t, e, X = _load_unsorted_numpy(parts, n_rows)
    print(f"    argsort ({n_rows:,} rows) ...", flush=True)
    order = np.argsort(t, kind="stable")[::-1]
    t_d = np.ascontiguousarray(t[order], dtype=np.float64)
    e_d = np.ascontiguousarray(e[order], dtype=np.int32)
    X_d = np.ascontiguousarray(X[order], dtype=np.float64)
    del t, e, X
    print(f"    fitting jax_uns ({n_rows:,} rows) ...", flush=True)
    return _nr_loop(_efron_jax, t_d, e_d, X_d, "jax_uns")


def fit_h2o_coxph(sorted_path, n_rows, **_kw):
    """H2O CoxPH: Java-backed distributed Efron MLE.

    Wall time includes JVM initialisation (~10-15 s on first call) and
    H2OFrame conversion.  RAM reflects the Python process only; the JVM
    heap is tracked by a separate OS process.
    """
    import pandas as pd
    h2o = _optional_backend("h2o")
    H2OCoxProportionalHazardsEstimator = _optional_backend(
        "h2o.estimators").H2OCoxProportionalHazardsEstimator

    h2o.init(nthreads=-1, max_mem_size="8g", verbose=False)
    h2o.no_progress()

    try:
        t_asc, e_asc, X_asc = _load_sorted_numpy(sorted_path, n_rows)
        df_pd = pd.DataFrame(
            np.column_stack([t_asc, X_asc]),
            columns=[TIME_COL, *X_COLS],  # pyright: ignore[reportArgumentType]
        )
        df_pd[EVENT_COL] = e_asc  # keep as int32, not float
        del t_asc, e_asc, X_asc
        print(f"    converting to H2OFrame ({n_rows:,} rows) ...", flush=True)
        hf = h2o.H2OFrame(df_pd)
        del df_pd
        print(f"    fitting h2o CoxPH ({n_rows:,} rows) ...", flush=True)
        model = H2OCoxProportionalHazardsEstimator(
            stop_column=TIME_COL,
            ties="efron",
        )
        model.train(x=X_COLS, y=EVENT_COL, training_frame=hf)
        h2o.remove(hf)
        coef_dict = model.coef()
        coef = np.array([coef_dict[c] for c in X_COLS])
        try:
            n_iter = int(model.iter)
        except (AttributeError, TypeError, ValueError):
            n_iter = -1
        return coef, n_iter
    finally:
        cluster = h2o.cluster()
        if cluster is not None:
            cluster.shutdown(prompt=False)


def fit_lifelines(sorted_path, n_rows, **_kw):
    """lifelines.CoxPHFitter: in-memory exact MLE (Efron)."""
    import pandas as pd
    from lifelines import CoxPHFitter
    t_asc, e_asc, X_asc = _load_sorted_numpy(sorted_path, n_rows)
    df = pd.DataFrame(
        np.column_stack([t_asc, e_asc.astype(float), X_asc]),
        # pandas' stubs type `columns` as an Axes, which a list[str] is at
        # runtime; pyright does not accept the list literal.
        columns=[  # pyright: ignore[reportArgumentType]
            TIME_COL, EVENT_COL, *X_COLS],
    )
    del t_asc, e_asc, X_asc
    print(f"    fitting lifelines ({n_rows:,} rows) ...", flush=True)
    cph = CoxPHFitter()
    cph.fit(df, duration_col=TIME_COL, event_col=EVENT_COL,
            show_progress=False)
    return cph.params_.values, -1, {
        "se": cph.standard_errors_.values.tolist(),
    }


def fit_sksurv(sorted_path, n_rows, **_kw):
    """scikit-survival CoxPHSurvivalAnalysis: in-memory exact Efron MLE."""
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    t_asc, e_asc, X_asc = _load_sorted_numpy(sorted_path, n_rows)
    y = np.array(
        [(bool(ev), ti) for ev, ti in zip(e_asc, t_asc)],
        dtype=[("event", bool), ("time", float)],
    )
    del t_asc, e_asc
    print(f"    fitting sksurv ({n_rows:,} rows) ...", flush=True)
    model = CoxPHSurvivalAnalysis(ties="efron")
    model.fit(X_asc, y)
    return model.coef_, -1


def fit_statsmodels_phreg(sorted_path, n_rows, **_kw):
    """statsmodels PHReg (Efron): in-memory exact MLE."""
    from statsmodels.duration.hazard_regression import PHReg
    t_asc, e_asc, X_asc = _load_sorted_numpy(sorted_path, n_rows)
    print(f"    fitting statsmodels PHReg ({n_rows:,} rows) ...", flush=True)
    model = PHReg(t_asc, X_asc, status=e_asc.astype(np.float64), ties="efron")
    result = model.fit(disp=False)
    return (
        np.asarray(result.params),
        int(getattr(result, "mle_retvals", {}).get("iters", -1) or -1),
    )


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

# is_r=True -> dispatched via _run_r_competing (Rscript subprocess)
# is_r=False, fn=None -> bug; all non-R methods must have a fn
#
# _uns variants read from original unsorted parts; the argsort / DuckDB sort
# is inside their timing so wall time includes sort overhead.
#
#  key                             fn                               is_r
METHODS = {
    # Streaming NR (exact Efron MLE, O(batch * p^2) or O(p^2) carry RAM)
    "coxstream_numpy":    (fit_coxstream_numpy,   False),
    "coxstream_numba":    (fit_coxstream_numba,   False),
    "coxstream_jax":      (fit_coxstream_jax,     False),
    "coxstream_cython":            (fit_coxstream_cython,           False),
    "coxstream_package":           (fit_coxstream_package,          False),
    "coxstream_cpp":      (fit_coxstream_cpp,     False),
    "coxstream_rs":       (fit_coxstream_rs,      False),
    # In-memory compiled (pre-sorted; argsort outside timing)
    "renew_coxph_cython":          (fit_renew_coxph_cython,         False),
    "renew_coxph_cython_f32":      (fit_renew_coxph_cython_f32,     False),
    "renew_coxph_cython_uns":      (fit_renew_coxph_cython_uns,     False),
    "renew_coxph_cpp":             (fit_renew_coxph_cpp,            False),
    "renew_coxph_cpp_uns":         (fit_renew_coxph_cpp_uns,        False),
    "renew_coxph_rs":              (fit_renew_coxph_rs,             False),
    "renew_coxph_rs_uns":          (fit_renew_coxph_rs_uns,         False),
    "renew_coxph_jax":             (fit_renew_coxph_jax,            False),
    "renew_coxph_jax_uns":         (fit_renew_coxph_jax_uns,        False),
    "renew_coxph_jax_x32":         (fit_renew_coxph_jax_x32,        False),
    "renew_coxph_jax_uns_x32":     (fit_renew_coxph_jax_uns_x32,    False),
    "coxstream_jax_x32":  (fit_coxstream_jax_x32, False),
    # In-memory Python (+ Java via H2O)
    "bigsurvsgd":                  (fit_bigsurvsgd,                 False),
    "h2o_coxph":                   (fit_h2o_coxph,                  False),
    "lifelines":                   (fit_lifelines,                  False),
    "sksurv":                      (fit_sksurv,                     False),
    "statsmodels_phreg":           (fit_statsmodels_phreg,          False),
    # R methods (Rscript subprocess)
    "r_survival_coxph":            (None,                           True),
    "xue_schifano_cuee":           (None,                           True),
    "cyclops":                     (None,                           True),
    # R streaming coxstream (CRAN package): same algorithm as coxstream_cython,
    # streamed via coxstream_arrow(). Not in-memory -> not capped.
    "r_coxstream":                 (None,                           True),
}

# Methods that require the ASC-sorted parquet (time-ascending pre-sort).
_NEEDS_ASC_SORTED = frozenset(
    m for m in METHODS
    if m.startswith("renew_coxph_") and "_stream_" not in m
)

# In-memory methods load the full dataset; cap n to avoid OOM on a 16 GB
# laptop.
# renew_coxph_cpp is the in-memory exact C++ reference -- kept in defaults to
# quantify streaming overhead at small n, capped like the other in-mem methods.
# bigsurvsgd holds the data in R too (peak RSS 6.4 GB at 16M, 12.5 GB at 32M on
# the VM); left out, --in-mem-cap did not stop it and the VM sweep ran its
# 100-epoch fit at 16M and 32M, about an hour per rep at 32M.
_IN_MEM_METHODS = frozenset(
    {"lifelines", "sksurv", "r_survival_coxph", "renew_coxph_cpp", "cyclops",
     "bigsurvsgd"}
)
_IN_MEM_CAP_N = 8_000_000  # overridden by --in-mem-cap at runtime

# Methods that need original unsorted part files (not the sorted parquet).
_UNSORTED_METHODS = frozenset({
    "renew_coxph_cython_uns",
    "renew_coxph_cpp_uns",
    "renew_coxph_rs_uns",
    "renew_coxph_jax_uns",
    "renew_coxph_jax_uns_x32",
})


# ---------------------------------------------------------------------------
# --run-one: child subprocess mode
# ---------------------------------------------------------------------------

def _run_one_command(args):
    """Run ONE method fit and write {coef, n_iter} JSON to args.out."""
    global _ALL_X_COLS, X_COLS
    _ALL_X_COLS = _load_x_cols(args.parts_dir)
    X_COLS = _ALL_X_COLS[:args.p_value]
    sorted_path = Path(args.sorted_path)
    n_rows = args.n_rows
    method = args.run_one

    fn, _ = METHODS.get(method, (None, False))
    if fn is None:
        sys.exit(f"--run-one: unknown or R-only method '{method}'")

    parts = [Path(p) for p in args.parts] if args.parts else []

    result = fn(
        sorted_path, n_rows,
        parts=parts,
        strata_size=args.sgd_strata,
        n_epochs=args.sgd_epochs,
    )
    # (coef, n_iter) or (coef, n_iter, extra): extra carries optional per-fit
    # fields such as model-based SEs ("se") and data passes ("n_passes").
    coef, n_iter = result[0], result[1]
    extra = result[2] if len(result) > 2 else {}

    Path(args.out).write_text(json.dumps({
        "coef": np.asarray(coef).tolist(),
        "n_iter": int(n_iter) if n_iter is not None else -1,
        **extra,
    }))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _log_rep(display_name, m):
    print(f"    {display_name:<26}  {m['elapsed_s']:>8.2f}s  "
          f"peak={m['rss_peak_mb']:>6.0f}MB  d={m['rss_delta_mb']:>+6.0f}MB  "
          f"iters={m['n_iter']}")


def _load_existing(path):
    if path.exists():
        with open(path) as fh:
            return json.load(fh).get("results", [])
    return []


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mrows", type=int, default=5,
        help="millions of rows to benchmark (default 5 = 5M rows).",
    )
    parser.add_argument(
        "--parts-dir", default=str(SURVIVAL_DIR), dest="parts_dir",
        help=f"directory of part_*.parquet files (default: {SURVIVAL_DIR}). "
             "Use a separate dir with --ncovariates 32 for the p-sweep.",
    )
    _default_methods = [
        m for m in METHODS
        if m not in {
            "coxstream_jax_x32", "coxstream_jax",
            "coxstream_numpy",
            "statsmodels_phreg", "h2o_coxph",
            "coxstream_cpp", "coxstream_rs",
            "coxstream_numba",
            "bigsurvsgd", "xue_schifano_cuee",
        }
        and not (
            m.startswith("renew_coxph_")
            and "_stream_" not in m
        )
    ]
    parser.add_argument(
        "--methods", nargs="+", default=_default_methods,
        choices=list(METHODS),
        help="restrict to these methods (default: all except "
             "coxstream_jax_x32). "
             "Cached reps for unselected methods are preserved.",
    )
    parser.add_argument(
        "--skip", nargs="+", default=[], choices=list(METHODS),
        help="skip these methods even if selected by --methods. "
             "Cached reps for skipped methods are preserved.",
    )
    parser.add_argument(
        "--sgd-epochs", type=int, default=10, dest="sgd_epochs",
        help="BigSurvSGD epochs (default 10; paper uses 100 for convergence)",
    )
    parser.add_argument(
        "--sgd-strata", type=int, default=50, dest="sgd_strata",
        help="BigSurvSGD strata size (default 50; paper uses 20-50)",
    )
    parser.add_argument(
        "--reps", type=int, default=5,
        help="target reps per method (default 5); cached reps count toward "
             "this target so only the shortfall is run",
    )
    parser.add_argument(
        "--in-mem-cap", type=int, default=_IN_MEM_CAP_N, dest="in_mem_cap",
        help="skip in-memory methods for n > this (default 8M; "
             "raise on VMs with more RAM, e.g. --in-mem-cap 32000000)",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="seed for per-pass shuffle order (default 0)",
    )
    parser.add_argument(
        "--cache", choices=["warm", "cold", "both"], default="cold",
        help="cache state: cold (default; purge OS page cache before each "
             "subprocess), warm, or both. Cold requires passwordless sudo "
             "purge.",
    )
    parser.add_argument(
        "-p", nargs="+", type=int, default=[10], dest="p_values",
        metavar="P",
        help="covariate counts to sweep (default: 10). "
             "Each value benchmarks the first P columns of the data.",
    )
    # Subprocess child mode (hidden from --help):
    parser.add_argument("--run-one", metavar="METHOD", dest="run_one",
                        help=argparse.SUPPRESS)
    parser.add_argument("--sorted-path", metavar="PATH", dest="sorted_path",
                        help=argparse.SUPPRESS)
    parser.add_argument("--n-rows", type=int, dest="n_rows",
                        help=argparse.SUPPRESS)
    parser.add_argument("--parts", nargs="+", metavar="PATH",
                        help=argparse.SUPPRESS)
    parser.add_argument("--out", metavar="PATH", help=argparse.SUPPRESS)
    parser.add_argument("--p-value", type=int, default=10, dest="p_value",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.run_one is not None:
        _run_one_command(args)
        return

    cache_states = (
        ["warm", "cold"] if args.cache == "both" else [args.cache]
    )
    if "cold" in cache_states:
        check_sudo_purge()

    parts_dir = Path(args.parts_dir)
    all_parts = sorted(parts_dir.glob("part_*.parquet"))
    if not all_parts:
        sys.exit(
            f"No parquet files found in {parts_dir}/\n"
            "Generate the survival dataset first:\n"
            "  python3.13 src/coxbench/generate_survival_data.py\n"
            "Or for the p-sweep (p=4..32, n=1M):\n"
            "  python3.13 src/coxbench/generate_survival_data.py "
            "--ncovariates 32 --outdir data/survival_parts_p32"
        )

    global _ALL_X_COLS
    _ALL_X_COLS = _load_x_cols(parts_dir)

    # Fail loudly rather than silently clamp p down to the covariate count and
    # store a row mislabeled with the requested p (which would duplicate the
    # widest fit-able p across the sweep).  Validate before any parquet build.
    n_avail = len(_ALL_X_COLS)
    over_p = sorted(p for p in args.p_values if p > n_avail)
    if over_p:
        sys.exit(
            f"Requested p={over_p} exceeds the {n_avail} covariates in "
            f"{parts_dir}/true_coef.json.\n"
            "Point --parts-dir at a dataset generated with enough "
            "covariates, e.g. the existing p-sweep set:\n"
            "  python3 -m coxbench.bench_competing_coxph "
            "--parts-dir data/survival_parts_p32 "
            f"-p {' '.join(map(str, args.p_values))}\n"
            "Or generate a wider one:\n"
            "  python3.13 src/coxbench/generate_survival_data.py "
            f"--ncovariates {max(over_p)} "
            f"--outdir data/survival_parts_p{max(over_p)}"
        )

    parts = all_parts[: args.mrows]
    if len(parts) < args.mrows:
        print(
            f"Warning: only {len(parts)} parts available "
            f"(requested {args.mrows}); using {len(parts)}.",
            flush=True,
        )
    n_rows = len(parts) * 1_000_000

    sorted_path = Path("data") / f"competing_coxph_{n_rows}_sorted.parquet"
    desc_path = Path("data") / f"competing_coxph_{n_rows}_sorted_desc.parquet"

    skip_set = set(args.skip)
    selected_methods = [m for m in args.methods if m not in skip_set]

    _ensure_sorted_desc_parquet(parts, desc_path)
    if any(m in _NEEDS_ASC_SORTED for m in selected_methods):
        _ensure_sorted_parquet(parts, sorted_path)

    with open(parts_dir / "true_coef.json") as fh:
        meta = json.load(fh)
    all_coef_names = list(meta["coef"].keys())
    all_true_coef = np.array(list(meta["coef"].values()))

    existing = [_migrate_row(r) for r in _load_existing(RESULTS_PATH)]
    rows_by_key = {
        (r["method"], r["n_rows"],
         r.get("p", len(r.get("coef_names", _ALL_X_COLS))),
         r.get("cache", "warm")): r
        for r in existing
    }

    def _persist():
        RESULTS_PATH.parent.mkdir(exist_ok=True)
        with open(RESULTS_PATH, "w") as fh:
            json.dump({"results": list(rows_by_key.values())}, fh, indent=2)
        RESULTS_PATH.with_suffix(".bak.json").write_text(
            RESULTS_PATH.read_text()
        )

    rng = random.Random(args.seed)

    for cache_state in cache_states:
        cold = cache_state == "cold"

        for p in args.p_values:
            global X_COLS
            X_COLS = _ALL_X_COLS[:p]
            coef_names = all_coef_names[:p]
            true_coef = all_true_coef[:p]

            print(f"\n{'=' * 68}")
            print("Competing CoxPH benchmark")
            print(f"  Cache state     : {cache_state}")
            print(f"  Data            : {desc_path.name}")
            print(f"  Rows            : {n_rows:,}  ({len(parts)} parts)")
            print(f"  BigSurvSGD      : strata={args.sgd_strata}  "
                  f"epochs={args.sgd_epochs}")
            if skip_set:
                print(f"  Skip            : {', '.join(sorted(skip_set))}")
            print(f"  Methods         : {', '.join(selected_methods)}")
            in_mem_cap = args.in_mem_cap
            skipped_slow = [m for m in selected_methods
                            if m in _IN_MEM_METHODS and n_rows > in_mem_cap]
            if skipped_slow:
                print(f"  Skipped (slow)  : {', '.join(skipped_slow)} "
                      f"(n>{in_mem_cap:,})")
            print(f"  Reps            : {args.reps} (shuffled order per pass)")
            print(f"  Covariates      : {p} ({', '.join(coef_names)})")
            print(f"{'=' * 68}\n")

            combos = []
            for method_key in selected_methods:
                if method_key in _IN_MEM_METHODS and n_rows > in_mem_cap:
                    continue
                sgd_label = f"bigsurvsgd_s{args.sgd_strata}_e{args.sgd_epochs}"
                label = sgd_label if method_key == "bigsurvsgd" else method_key
                combos.append((label, method_key, n_rows, p))

            for rep in range(args.reps):
                pending = [
                    combo for combo in combos
                    if _reps_done(
                        rows_by_key.get(
                            (combo[0], combo[2], combo[3], cache_state)
                        )
                    ) < args.reps
                ]
                if not pending:
                    print(f"\nAll combos at {args.reps} reps -- "
                          "nothing to do.")
                    break
                rng.shuffle(pending)
                print(f"\n=== Pass {rep + 1}/{args.reps} "
                      f"(cache={cache_state}, p={p}) -- "
                      f"{len(pending)} combo(s) need more reps ===")
                for (display_name, method_key, n_rows_method,
                     p_combo) in pending:
                    key = (display_name, n_rows_method, p_combo, cache_state)
                    row = rows_by_key.get(key)
                    if row is None:
                        row = {
                            "method": display_name,
                            "n_rows": n_rows_method,
                            "p": p_combo,
                            "cache": cache_state,
                            "coef_names": coef_names,
                            "true_coef": true_coef.tolist(),
                            "elapsed_s_reps": [],
                            "rss_peak_mb_reps": [],
                            "rss_delta_mb_reps": [],
                            "n_iter_reps": [],
                        }
                        rows_by_key[key] = row
                    if _reps_done(row) >= args.reps:
                        continue
                    print(f"  -> {display_name} ...", flush=True)
                    try:
                        m = _run_method_subproc(
                            method_key, sorted_path, desc_path, parts,
                            n_rows_method, args, p=p_combo, cold=cold,
                        )
                    except (RuntimeError, OSError, ValueError, KeyError,
                            psutil.Error, subprocess.SubprocessError) as exc:
                        # Every runner funnels through
                        # _run_subprocess_with_polling: record the failed
                        # rep and move on; anything else is a harness bug.
                        # The message carries the child's traceback, and
                        # the row goes into a public results file.
                        msg = scrub_paths(
                            f"pass {rep + 1}: {type(exc).__name__}: {exc}")
                        row.setdefault("errors", []).append(msg)
                        preview = (msg if len(msg) <= 200
                                   else msg[:200] + "...[truncated]")
                        print(f"    {display_name:<26}  FAILED  {preview}",
                              flush=True)
                        _persist()
                        continue
                    _log_rep(display_name, m)
                    row["elapsed_s_reps"].append(round(m["elapsed_s"], 3))
                    row["rss_peak_mb_reps"].append(round(m["rss_peak_mb"], 1))
                    row["rss_delta_mb_reps"].append(
                        round(m["rss_delta_mb"], 1))
                    row["n_iter_reps"].append(m["n_iter"])
                    row.setdefault("host_reps", []).append(_HOST)
                    row["host"] = _HOST
                    if len(set(row["host_reps"])) > 1:
                        print(f"    WARNING: {display_name} n={n_rows_method} "
                              f"has reps from multiple hosts "
                              f"{sorted(set(row['host_reps']))}; delete "
                              "the row and re-run on one machine.",
                              flush=True)
                    n_c = min(len(m["coef"]), len(true_coef))
                    row["max_err_true"] = float(
                        np.max(np.abs(m["coef"][:n_c] - true_coef[:n_c]))
                    )
                    row["coef"] = m["coef"].tolist()
                    if m.get("se") is not None:
                        row["se"] = m["se"]
                    if m.get("n_passes") is not None:
                        row.setdefault("n_passes_reps", []).append(
                            m["n_passes"])
                    if m.get("pkg_src") is not None:
                        if row.get("pkg_src") not in (None, m["pkg_src"]):
                            print(f"    WARNING: {display_name} "
                                  f"n={n_rows_method} has reps from two "
                                  "package sources; delete the row and "
                                  "re-run.", flush=True)
                        row["pkg_src"] = m["pkg_src"]
                    _update_summary(row)
                    _persist()

    print(f"\nResults written to {RESULTS_PATH} "
          f"({len(rows_by_key)} combo rows)")


if __name__ == "__main__":
    main()
