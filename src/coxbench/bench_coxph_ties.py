"""Tie experiment for the arXiv paper: streamed CoxStream against Efron.

Question: does the released coxstream package, streaming a descending-time
Parquet file one row group (chunk) at a time, reproduce the in-memory Efron
fit when tie groups straddle row-group boundaries?

Design (the ten-role advice of 2026-10-07):
- One synthetic cohort, generated here from --seed with
  generate_survival_data.make_chunk (no dependency on data/). Its times are
  put on three grids: continuous (no tied events; a sanity check), monthly,
  ceil(12 t) (the headline) and yearly, ceil(t) (large tie groups, where
  Efron and Breslow differ clearly). Events and censorings are rounded by the
  same rule, so the administrative censoring at 10 years shares the last
  tied time with that interval's events.
- Rows are sorted by time descending, in a seeded random order within each
  tied time, so a tie group's events fall on both sides of row-group
  boundaries. How many tied event groups do so is computed from the sorted
  data and recorded; the run refuses to write results if they do not.
- In-memory references: R survival::coxph with ties = "efron" and
  "breslow" (r_ties_reference.R, tight control) and lifelines, a second,
  independent Efron implementation.
- Streamed fits: coxstream CoxStream.fit_parquet at several row-group sizes
  per grid (a prime among them, and a single chunk), plus a small subsample
  with row groups of 1, 7, 37 and 1009 rows, checked against R and against
  a plain-numpy Efron written from the textbook formulas.

Accuracy only: nothing is timed for the paper (elapsed_s is informational)
and memory is not measured. results/coxph_ties.json keeps the labels and
fields that readers of the earlier file rely on (covariate_names; efron_/
breslow_/lifelines_ continuous and discrete rows with err_vs_true and
n_rows), now produced by R survival and lifelines, and adds the rest.

Run it in a terminal, not from an IDE session (about 3 to 4 minutes and up
to about 4 GB of RAM on a laptop):
    cd ~/coxstream && python3 -m coxbench.bench_coxph_ties
It logs to tmp/coxph_ties_<UTC>.log (path on the first line, exit status on
the last) and writes results/coxph_ties.json.
"""
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import TextIO

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from coxbench import generate_survival_data as gen
from coxbench.paths import REPO_ROOT, results_path
from coxbench.provenance import coxstream_source_hash, host_label

SCHEMA_VERSION = 2
R_SCRIPT = Path(__file__).with_name("r_ties_reference.R")
COVARIATES = list(gen.TRUE_COEF)
TRUE_COEF = np.array([gen.TRUE_COEF[c] for c in COVARIATES])
GRID_RULE = {"continuous": "t", "monthly": "ceil(12 t)", "yearly": "ceil(t)"}
# Label suffix of the in-memory rows: "discrete" is the monthly grid, the
# name readers of the earlier results file look up.
LABEL_SUFFIX = {"continuous": "continuous", "monthly": "discrete",
                "yearly": "yearly"}
STRADDLE_MIN_FRACTION = 0.99  # of tied event groups, at the small row groups
LIFELINES_PRECISION = 1e-10   # lifelines' step-norm stop (default 1e-7)


# --- logging ----------------------------------------------------------------
_T0 = time.perf_counter()
_LOG: list[TextIO] = []  # the open log file, once main() has opened it


def echo(text):
    """Print to the terminal and append to the log."""
    print(text, flush=True)
    for log in _LOG:
        log.write(text + "\n")
        log.flush()


def say(msg):
    echo(f"[{time.perf_counter() - _T0:7.1f} s] {msg}")


# --- data -------------------------------------------------------------------
def cohort(n, seed):
    """(t, event, X) from the paper's generator; X in COVARIATES order."""
    table = gen.make_chunk(n, np.random.default_rng(seed))
    t = table["duration"].to_numpy().astype(np.float64)
    e = table["event"].to_numpy().astype(np.int8)
    x = np.column_stack([table[c].to_numpy().astype(np.float64)
                         for c in COVARIATES])
    return t, e, x


def on_grid(t, grid):
    """Times on `grid`; the same rule for events and censorings."""
    if grid == "continuous":
        return t.copy()
    per_year = {"monthly": 12, "yearly": 1}[grid]
    return np.ceil(t * per_year).astype(np.int64).astype(np.float64)


def descending(t, e, x, seed):
    """Sort by time descending, in a seeded random order within a tied time."""
    order = np.lexsort((np.random.default_rng(seed).random(t.size), -t))
    return t[order], e[order], x[order]


def tie_structure(t, e, rows_per_group):
    """Tie groups of a descending-sorted file and how they meet row groups.

    A tied event group is a distinct time with at least two events; it
    straddles when its events lie in more than one row group.
    """
    n = t.size
    starts = np.flatnonzero(np.r_[True, t[1:] != t[:-1]])
    ends = np.r_[starts[1:], n]
    ev_idx = np.flatnonzero(e)
    if ev_idx.size == 0:
        raise RuntimeError("the cohort has no events")
    d = (np.searchsorted(ev_idx, ends) - np.searchsorted(ev_idx, starts))
    first = ev_idx[np.minimum(np.searchsorted(ev_idx, starts),
                              ev_idx.size - 1)]
    last = ev_idx[np.maximum(np.searchsorted(ev_idx, ends) - 1, 0)]
    tied = d >= 2
    straddle = tied & (first // rows_per_group != last // rows_per_group)
    rows = ends - starts
    return {
        "row_group_rows": int(rows_per_group),
        "n_row_groups": int(-(-n // rows_per_group)),
        "n_times": int(starts.size),
        "n_tied_event_groups": int(tied.sum()),
        # rows (events and censorings) of the smallest tied event group
        "min_tied_group_rows": int(rows[tied].min()) if tied.any() else 0,
        "n_tied_event_groups_straddling": int(straddle.sum()),
        "events_in_tied_groups": int(d[tied].sum()),
        "events_in_straddling_groups": int(d[straddle].sum()),
        "max_group_rows": int(rows.max()),
        "max_group_events": int(d.max()),
        # the first group of a descending file is the latest time
        "latest_time_rows": int(rows[0]),
        "latest_time_events": int(d[0]),
    }


def write_parquet(path, t, e, x, rows_per_group):
    table = pa.table({"duration": t, "event": e,
                      **{c: x[:, j] for j, c in enumerate(COVARIATES)}})
    pq.write_table(table, path, row_group_size=rows_per_group)


# --- fits -------------------------------------------------------------------
def fit_r(path, out_json):
    """R survival Efron and Breslow on one file: {ties: fit}, versions."""
    cmd = ["Rscript", str(R_SCRIPT), str(path), str(out_json),
           "efron,breslow"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    for line in (proc.stdout + proc.stderr).splitlines():
        echo(f"    {line}")
    if proc.returncode:
        raise RuntimeError(f"R reference failed (exit {proc.returncode})")
    out = json.loads(Path(out_json).read_text())
    if [c.replace("`", "") for c in out["fits"][0]["coef_names"]] \
            != COVARIATES:
        raise RuntimeError(f"R returned coefficients in another order: "
                           f"{out['fits'][0]['coef_names']}")
    return {f["ties"]: f for f in out["fits"]}, out


def fit_lifelines(t, e, x):
    import pandas as pd
    from lifelines import CoxPHFitter

    df = pd.DataFrame(x, columns=COVARIATES)
    df["duration"] = t
    df["event"] = e
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = CoxPHFitter().fit(df, "duration", "event",
                              fit_options={"precision": LIFELINES_PRECISION})
    return {
        "coef": np.asarray(m.params_.loc[COVARIATES], float).tolist(),
        "se": np.asarray(m.standard_errors_.loc[COVARIATES], float).tolist(),
        "loglik": float(m.log_likelihood_),
        "warnings": [str(w.message) for w in caught],
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }


def fit_coxstream(path):
    from coxstream import CoxStream

    t0 = time.perf_counter()
    m = CoxStream().fit_parquet(str(path), "duration", "event", COVARIATES)
    if m.n_iter_ is None or m.n_passes_ is None:
        raise RuntimeError(f"coxstream returned no fit for {path}")
    return {
        "coef": np.asarray(m.coef_, float).tolist(),
        "se": np.asarray(m.standard_errors_, float).tolist(),
        "n_iter": m.n_iter_,
        "n_passes": m.n_passes_,
        "converged": bool(m.converged_),
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }


def numpy_efron(t, e, x, max_iter=50, tol=1e-12):
    """Plain-numpy Efron Newton-Raphson from the textbook formulas: the whole
    risk set in memory, one distinct event time at a time. Shares no code with
    the streaming kernel; for the subsample only (memory n x distinct times).
    """
    ev = e.astype(bool)
    p = x.shape[1]
    beta = np.zeros(p)
    info = np.eye(p)
    for _ in range(max_iter):
        w = np.exp(x @ beta)
        score, info = np.zeros(p), np.zeros((p, p))
        for u in np.unique(t[ev]):
            risk, tied = t >= u, (t == u) & ev
            f = np.arange(int(tied.sum())) / tied.sum()
            w_r, x_r, w_d, x_d = w[risk], x[risk], w[tied], x[tied]
            s0 = w_r.sum() - f * w_d.sum()
            s1 = (w_r @ x_r)[None, :] - f[:, None] * (w_d @ x_d)[None, :]
            m = s1 / s0[:, None]
            score += x_d.sum(axis=0) - m.sum(axis=0)
            info += ((1.0 / s0).sum() * ((x_r * w_r[:, None]).T @ x_r)
                     - (f / s0).sum() * ((x_d * w_d[:, None]).T @ x_d)
                     - m.T @ m)
        step = np.linalg.solve(info, score)
        beta = beta + step
        if np.linalg.norm(step) < tol:
            break
    return {"coef": beta.tolist(),
            "se": np.sqrt(np.diag(np.linalg.inv(info))).tolist()}


# --- rows and summaries -----------------------------------------------------
def row(label, method, grid, ties, n, fit, **extra):
    coef = np.asarray(fit["coef"])
    return {"label": label, "method": method, "data": grid, "ties": ties,
            "n_rows": int(n), **fit,
            "err_vs_true": (coef - TRUE_COEF).tolist(), **extra}


def gap(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def rel_gap(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b) / np.abs(b)))


# --- provenance -------------------------------------------------------------
def _git(*args):
    out = subprocess.run(["git", "-C", str(REPO_ROOT), *args],
                         capture_output=True, text=True, check=False)
    return out.stdout.strip()


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def preflight(args):
    if shutil.which("Rscript") is None:
        raise RuntimeError("Rscript not found on PATH")
    r_probe = ('for (p in c("survival", "arrow", "jsonlite")) '
               'if (!requireNamespace(p, quietly = TRUE)) '
               'stop("missing R package ", p)')
    probe = subprocess.run(["Rscript", "-e", r_probe], capture_output=True,
                           text=True, check=False)
    if probe.returncode:
        raise RuntimeError(probe.stderr.strip())
    import coxstream
    pkg_dir = Path(coxstream.__file__).resolve().parent
    # fail before the long run, not in it
    if not args.no_lifelines and importlib.util.find_spec("lifelines") is None:
        raise RuntimeError("lifelines is not installed; install it or pass "
                           "--no-lifelines")
    tracked = [Path(__file__).relative_to(REPO_ROOT),
               R_SCRIPT.relative_to(REPO_ROOT),
               Path("src/coxbench/generate_survival_data.py"),
               Path("packages/coxstream")]
    dirty = _git("status", "--porcelain", "--", *map(str, tracked))
    if dirty and not args.allow_dirty:
        raise RuntimeError("uncommitted changes in the code that produces "
                           f"the results:\n{dirty}\ncommit them, or pass "
                           "--allow-dirty (recorded in the JSON)")
    return {
        "coxstream_version": coxstream.__version__,
        # repository-relative when editable, so no home directory is recorded
        "coxstream_dir": (str(pkg_dir.relative_to(REPO_ROOT))
                          if pkg_dir.is_relative_to(REPO_ROOT)
                          else pkg_dir.name),
        "coxstream_editable": pkg_dir.is_relative_to(REPO_ROOT),
        "pkg_src": coxstream_source_hash(pkg_dir),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(dirty),
    }


# --- the experiment ---------------------------------------------------------
def run_grid(grid, t0, e0, x0, args, workdir, results, structure, r_info):
    t, e, x = descending(on_grid(t0, grid), e0, x0, args.seed + 1)
    n = t.size
    sizes = {"continuous": [200_000], "monthly": args.row_groups,
             "yearly": [7_919, 200_000]}[grid]
    sizes = sorted({min(s if s > 0 else n, n) for s in sizes})
    suffix = LABEL_SUFFIX[grid]
    say(f"{grid} ({GRID_RULE[grid]}): {np.unique(t).size} distinct times, "
        f"row groups {sizes}")

    ref_path = workdir / f"{grid}_reference.parquet"
    write_parquet(ref_path, t, e, x, 200_000)
    say(f"  R survival coxph (Efron, Breslow) on {n:,} rows ...")
    r_fits, r_meta = fit_r(ref_path, workdir / f"{grid}_r.json")
    r_info.update({k: r_meta[k] for k in ("survival_version", "r_version",
                                          "control")})
    for ties in ("efron", "breslow"):
        f = r_fits[ties]
        results.append(row(f"{ties}_{suffix}", "r_survival_coxph", grid,
                           ties, n, {
                               "coef": f["coef"], "se": f["se"],
                               "loglik": f["loglik"], "n_iter": f["iter"],
                               "converged": not f["warnings"],
                               "warnings": f["warnings"],
                               "elapsed_s": f["elapsed_s"]}))
    ref_path.unlink()

    if not args.no_lifelines:
        say(f"  lifelines (Efron) on {n:,} rows ...")
        fit = fit_lifelines(t, e, x)
        results.append(row(f"lifelines_{suffix}", "lifelines", grid, "efron",
                           n, fit))

    structure[grid] = {}
    for size in sizes:
        path = workdir / f"{grid}_rg{size}.parquet"
        write_parquet(path, t, e, x, size)
        ts = tie_structure(t, e, size)
        structure[grid][str(size)] = ts
        fit = fit_coxstream(path)
        say(f"  coxstream, row groups of {size:,}: {ts['n_row_groups']:,} "
            f"row groups, {ts['n_tied_event_groups_straddling']}/"
            f"{ts['n_tied_event_groups']} tied event groups straddle; "
            f"{fit['n_iter']} iterations, gap to R Efron "
            f"{gap(fit['coef'], r_fits['efron']['coef']):.2e}")
        results.append(row(f"coxstream_{grid}_rg{size}", "coxstream_package",
                           grid, "efron", n, fit, row_group_rows=size,
                           tie_structure=ts))
        if not args.keep:
            path.unlink()


def run_subsample(args, workdir, results, structure):
    t0, e0, x0 = cohort(args.sub_nrows, args.seed + 2)
    t, e, x = descending(on_grid(t0, "monthly"), e0, x0, args.seed + 3)
    n = t.size
    say(f"subsample: {n:,} rows, monthly grid, row groups "
        f"{args.sub_row_groups}")
    ref_path = workdir / "subsample_reference.parquet"
    write_parquet(ref_path, t, e, x, n)
    r_fits, _ = fit_r(ref_path, workdir / "subsample_r.json")
    ref_path.unlink()
    r_eff = r_fits["efron"]
    results.append(row("sub_efron_r", "r_survival_coxph", "monthly_subsample",
                       "efron", n, {"coef": r_eff["coef"],
                                    "se": r_eff["se"],
                                    "loglik": r_eff["loglik"],
                                    "n_iter": r_eff["iter"],
                                    "converged": not r_eff["warnings"],
                                    "warnings": r_eff["warnings"]}))
    say("  plain-numpy Efron ...")
    results.append(row("sub_efron_numpy", "numpy_textbook_efron",
                       "monthly_subsample", "efron", n,
                       numpy_efron(t, e, x)))
    if not args.no_lifelines:
        results.append(row("sub_lifelines", "lifelines", "monthly_subsample",
                           "efron", n, fit_lifelines(t, e, x)))
    structure["monthly_subsample"] = {}
    for size in args.sub_row_groups:
        path = workdir / f"subsample_rg{size}.parquet"
        write_parquet(path, t, e, x, size)
        ts = tie_structure(t, e, size)
        structure["monthly_subsample"][str(size)] = ts
        fit = fit_coxstream(path)
        say(f"  coxstream, row groups of {size}: "
            f"{ts['n_tied_event_groups_straddling']}/"
            f"{ts['n_tied_event_groups']} tied event groups straddle; gap to "
            f"R Efron {gap(fit['coef'], r_eff['coef']):.2e}")
        results.append(row(f"sub_coxstream_rg{size}", "coxstream_package",
                           "monthly_subsample", "efron", n, fit,
                           row_group_rows=size, tie_structure=ts))
        if not args.keep:
            path.unlink()


def summarise(results, structure):
    """Gaps per grid, and the design check (refuses a run that tests
    nothing). Returns the summary dict for the JSON."""
    by = {r["label"]: r for r in results}
    summary, problems = {}, []
    for grid, suffix in [*LABEL_SUFFIX.items(),
                         ("monthly_subsample", None)]:
        ref = by["sub_efron_r" if suffix is None else f"efron_{suffix}"]
        streams = [r for r in results if r["data"] == grid
                   and r["method"] == "coxstream_package"]
        s = {
            "coxstream_vs_r_efron_coef": max(gap(r["coef"], ref["coef"])
                                             for r in streams),
            "coxstream_vs_r_efron_se_rel": max(rel_gap(r["se"], ref["se"])
                                               for r in streams),
        }
        if suffix is not None:
            s["r_breslow_vs_r_efron_coef"] = gap(by[f"breslow_{suffix}"]
                                                 ["coef"], ref["coef"])
        life = by.get("sub_lifelines" if suffix is None
                      else f"lifelines_{suffix}")
        if life is not None:
            s["lifelines_vs_r_efron_coef"] = gap(life["coef"], ref["coef"])
            s["coxstream_vs_lifelines_coef"] = max(
                gap(r["coef"], life["coef"]) for r in streams)
        if suffix is None:
            s["coxstream_vs_numpy_efron_coef"] = max(
                gap(r["coef"], by["sub_efron_numpy"]["coef"])
                for r in streams)
        summary[grid] = s
        for r in results:
            if r["data"] == grid and r.get("converged") is False:
                problems.append(f"{r['label']} did not converge "
                                f"{r.get('warnings', '')}")
    # Design check. A row group under half the smallest tied group must cut
    # nearly every tied group (else the within-time shuffle failed), and each
    # tied grid needs at least one such row-group size (else the run tests
    # nothing). Row groups larger than the groups are controls and may not.
    for grid in ("monthly", "yearly", "monthly_subsample"):
        best = 0.0
        for size, ts in structure[grid].items():
            frac = (ts["n_tied_event_groups_straddling"]
                    / max(ts["n_tied_event_groups"], 1))
            best = max(best, frac)
            if (int(size) <= ts["min_tied_group_rows"] // 2
                    and frac < STRADDLE_MIN_FRACTION):
                raise RuntimeError(
                    f"design failure: on the {grid} grid, row groups of "
                    f"{size} rows are under half the smallest tied group "
                    f"({ts['min_tied_group_rows']} rows), yet only "
                    f"{frac:.1%} of tied event groups straddle a boundary")
        if best < STRADDLE_MIN_FRACTION:
            raise RuntimeError(
                f"design failure: on the {grid} grid no row-group size makes "
                f"{STRADDLE_MIN_FRACTION:.0%} of tied event groups straddle a "
                "boundary; the run would test nothing")
    return summary, problems


def run(args):
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    say(f"tie experiment: n = {args.nrows:,}, seed {args.seed}, grids "
        f"{list(GRID_RULE)}, monthly row groups {args.row_groups}, "
        f"subsample {args.sub_nrows:,} rows with row groups "
        f"{args.sub_row_groups}; lifelines "
        f"{'off' if args.no_lifelines else 'on'}; output {out}")
    prov = preflight(args)
    say(f"coxstream {prov['coxstream_version']} (source {prov['pkg_src']}, "
        f"{'editable' if prov['coxstream_editable'] else 'installed'}), "
        f"git {prov['git_commit'][:7]}"
        f"{' with uncommitted changes' if prov['git_dirty'] else ''}")

    say(f"generating the cohort ({args.nrows:,} rows) ...")
    t0, e0, x0 = cohort(args.nrows, args.seed)
    say(f"  {int(e0.sum()):,} events; "
        f"{int((t0 == gen.MAX_FOLLOW_UP).sum()):,} censored at the "
        f"{gen.MAX_FOLLOW_UP:g}-year follow-up cap")

    results, structure, r_info = [], {}, {}
    for grid in GRID_RULE:
        run_grid(grid, t0, e0, x0, args, workdir, results, structure, r_info)
    run_subsample(args, workdir, results, structure)
    results.append({"label": "true", "method": None, "data": None,
                    "ties": None, "n_rows": None,
                    "coef": TRUE_COEF.tolist(),
                    "err_vs_true": [0.0] * len(COVARIATES)})

    summary, problems = summarise(results, structure)
    say("summary (max absolute coefficient gap; SE gap relative):")
    for grid, s in summary.items():
        say(f"  {grid}: " + ", ".join(f"{k} {v:.2e}" for k, v in s.items()))
    for p in problems:
        say(f"WARNING: {p}")

    lib_versions = {"python": platform.python_version(),
                    "numpy": np.__version__, "pyarrow": pa.__version__}
    if not args.no_lifelines:
        import lifelines
        lib_versions["lifelines"] = lifelines.__version__
    doc = {
        "covariate_names": COVARIATES,
        "results": results,
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "script": Path(__file__).name,
            "script_sha256": _sha256(__file__),
            "r_script_sha256": _sha256(R_SCRIPT),
            "created_utc": dt.datetime.now(dt.UTC).isoformat(
                timespec="seconds"),
            "command_line": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
            "host": host_label(),
            **prov,
            "versions": {**lib_versions, **r_info},
            "lifelines_precision": LIFELINES_PRECISION,
            "seed": args.seed,
            "n_rows": args.nrows,
            "sub_n_rows": args.sub_nrows,
            "grids": GRID_RULE,
            "within_time_order": "seeded random (seed + 1; subsample "
                                 "seed + 3)",
            "generator": {"function": "generate_survival_data.make_chunk",
                          "weibull_shape": gen.WEIBULL_SHAPE,
                          "cens_rate": gen.CENS_RATE,
                          "max_follow_up": gen.MAX_FOLLOW_UP,
                          "true_coef": TRUE_COEF.tolist()},
            "tie_structure": structure,
            "summary": summary,
            "problems": problems,
        },
    }
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    os.replace(tmp, out)
    say(f"wrote {len(results)} rows -> {out}")


def _ints(text):
    return [int(v) for v in text.split(",") if v]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nrows", type=int, default=3_000_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--row-groups", type=_ints,
                    default=[1_000, 7_919, 200_000, 0],
                    help="monthly-grid row-group sizes; 0 = one chunk")
    ap.add_argument("--sub-nrows", type=int, default=20_000)
    ap.add_argument("--sub-row-groups", type=_ints,
                    default=[1, 7, 37, 1_009])
    ap.add_argument("--no-lifelines", action="store_true",
                    help="skip the lifelines Efron cross-check")
    ap.add_argument("--workdir", default=str(REPO_ROOT / "tmp" / "ties"))
    ap.add_argument("--out", default=str(results_path("coxph_ties.json")))
    ap.add_argument("--keep", action="store_true",
                    help="keep the streamed Parquet files in --workdir")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="run with uncommitted changes (recorded)")
    args = ap.parse_args()

    log_path = (REPO_ROOT / "tmp" / "coxph_ties_"
                f"{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%SZ}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    status = 1
    with open(log_path, "a") as log:
        _LOG.append(log)
        echo(f"log: {log_path}")
        try:
            run(args)
            status = 0
        except RuntimeError as exc:
            echo(f"ERROR: {exc}")
        except Exception:
            echo(traceback.format_exc())  # into the log, then re-raised
            raise
        finally:
            echo(f"exit status: {status}")
            _LOG.clear()
    sys.exit(status)


if __name__ == "__main__":
    main()
