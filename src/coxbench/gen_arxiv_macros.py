"""Generate papers/arxiv/paper_macros.tex from the benchmark results.

Every empirical number in the arXiv CoxStream paper is emitted here, so the
manuscript tracks results/*.json instead of hand-transcription (which drifts:
the old hand-set \\CoxStreamSERefMaxRelDev was 2.22e-16 while the SE JSON had
moved to 1.65e-13).

Sources
-------
- results/competing_coxph.json      speed, RAM, recovery error, coef and SE
                                     gaps, repeat counts
- results/coxph_ties.json            the tie experiment (bench_coxph_ties.py):
                                     streamed coxstream vs R survival Efron,
                                     lifelines, numpy Efron and R Breslow
- results/vm_environment.txt         the Environment paragraph: machine, CPU,
                                     RAM, disk, OS, Python/R and versions
- bench_competing_coxph.py           BATCH_SIZE (the streamed chunk size) and
                                     RSS_POLL_S (the memory monitor's interval)
- generate_survival_data.py          the data-generating process: Weibull
                                     shape, censoring rate, follow-up cap
- packages/coxstream/.../coxstream.py  the fitter's tolerance, iteration cap,
                                     step-halving limit and line-search slack
- packages/r/coxstream/R/coxstream_arrow.R  the R package's iteration cap

The data-dependent sentences of the text (who is fastest, which footprint is
larger, that no fit backtracked, ...) are checked
in check_claims(): a regeneration that would contradict the prose fails and
names the sentence, instead of silently printing a number that disagrees.

Two kinds of number:
- Hardware-dependent (speed, RAM ratios): recomputed from whichever host wrote
  competing_coxph.json -- rerun this after the GCloud sweep to refresh them.
- Platform-independent (recovery error, coef gaps, tie error):
  bit-level agreement figures that do not change with the machine.

The cells the paper reports at (COMPARISON_N, the p-sweep endpoints) are
editorial choices, kept as named constants below; the values *at* those cells
are all data-driven.

Usage
-----
    python3 -m coxbench.gen_arxiv_macros          # write it
    python3 -m coxbench.gen_arxiv_macros --check  # diff only
    python3 -m coxbench.gen_arxiv_macros --results <path>
"""

import argparse
import json
import re
from itertools import pairwise
from math import floor, log10
from pathlib import Path

from coxbench.provenance import coxstream_source_hash

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
MACROS_OUT = REPO_ROOT / "papers" / "arxiv" / "paper_macros.tex"
ENV_TXT = RESULTS_DIR / "vm_environment.txt"
PKG_SRC = (REPO_ROOT / "packages" / "coxstream" / "src" / "coxstream"
           / "coxstream.py")
R_PKG_SRC = (REPO_ROOT / "packages" / "r" / "coxstream" / "R"
             / "coxstream_arrow.R")

# --- editorial choices (which cells the paper reports at) -------------------
N_SWEEP_P = 10             # model width of the n-sweep
COMPARISON_N = 4_000_000   # head-to-head cell: coef gaps, speedups, RAM ratios
PSWEEP_N = 1_000_000       # cell for the p-sweep recovery-error endpoints
P_LO, P_HI = 8, 128        # p-sweep endpoints the paper cites
RAM_FLAT_TOL = 1.5         # "near-constant" = within this factor of min-n RSS
RAM_ORDER_OF_MAG = 10      # "more than an order of magnitude more" RAM
EXTRAP_N = 100_000_000     # registry scale the lifelines footprint is
#                            extrapolated to (the intro's 10^7 to 10^8 rows)
LINEAR_FIT_TOL = 0.1       # "grows linearly": max relative residual of the
#                            straight-line fit of RAM added on n

# Method keys in competing_coxph.json. The timed CoxStream is the released
# package (tie-aware kernel, step-halving, one pass per iteration), the
# method the text describes.
M_STREAM = "coxstream_package"
M_LIFE = "lifelines"
M_CYCLOPS = "cyclops"
M_RSURV = "r_survival_coxph"
BENCH_SRC = Path(__file__).with_name("bench_competing_coxph.py")

# The tie experiment (bench_coxph_ties.py). Every row must come from one of
# these methods; each streamed row is compared with R survival's Efron fit of
# the same data.
TIES_SCHEMA = 2
TIES_METHODS = {"r_survival_coxph", "lifelines", "coxstream_package",
                "numpy_textbook_efron", None}  # None: the "true" row
TIES_EFRON_LABEL = {"continuous": "efron_continuous",
                    "monthly": "efron_discrete", "yearly": "efron_yearly",
                    "monthly_subsample": "sub_efron_r"}
TIES_LIFELINES_LABEL = {"continuous": "lifelines_continuous",
                        "monthly": "lifelines_discrete",
                        "yearly": "lifelines_yearly",
                        "monthly_subsample": "sub_lifelines"}
TIES_TOL = 1e-8    # "to rounding error": streamed vs R Efron, coefficients
TIES_TEETH = 1000  # Breslow-vs-Efron gap must exceed this x the streamed gap
GEN_SRC = Path(__file__).with_name("generate_survival_data.py")


# --- formatting -------------------------------------------------------------
def fmt_int(n):
    """4000000 -> '4\\,000\\,000' (LaTeX thin-space thousands)."""
    return f"{round(n):,}".replace(",", "\\,")


def fmt_sci(x, sig=3):
    """0.003394 -> '3.39 \\times 10^{-3}' (sig significant figures)."""
    if x == 0:
        return "0"
    e = floor(log10(abs(x)))
    m = round(x / 10 ** e, sig - 1)
    if abs(m) >= 10:  # rounding pushed the mantissa to 10.x
        m /= 10
        e += 1
    return f"{m:.{sig - 1}f} \\times 10^{{{e}}}"


def fmt_num(x, dp=1):
    """6.34 -> '6.3' (fixed decimals; ratios and speedups)."""
    return f"{x:.{dp}f}"


def fmt_pow10(x):
    """1e-12 -> '10^{-12}'; refuses a value that is not a power of ten."""
    e = round(log10(x))
    if abs(x - 10.0 ** e) > 1e-9 * 10.0 ** e:
        raise SystemExit(f"{x} is not a power of ten")
    return f"10^{{{e}}}"


# --- data access ------------------------------------------------------------
def _load(path):
    with open(path) as fh:
        return json.load(fh)


def one_host(rows, what):
    """Refuse numbers that mix machines: speed and RAM are host-specific."""
    hosts = {r.get("host") for r in rows}
    if len(hosts) != 1:
        raise SystemExit(f"{what} come from {len(hosts)} hosts "
                         f"{sorted(map(str, hosts))}: re-measure them on one "
                         "machine (cloud/run_coxph_nsweep.sh)")


def _source_value(path, pattern, what):
    """Group 1 of `pattern` in a source file. A renamed constant fails here
    instead of leaving a stale number in the paper."""
    m = re.search(pattern, path.read_text(), re.MULTILINE)
    if m is None:
        raise SystemExit(f"{what} not found in {path}")
    return m.group(1)


def _bench_chunk_rows():
    """BATCH_SIZE in the benchmark source: the DESC parquet's row groups."""
    return int(_source_value(BENCH_SRC, r"^BATCH_SIZE = ([\d_]+)$",
                             "BATCH_SIZE").replace("_", ""))


def _bench_rss_poll_ms():
    """RSS_POLL_S in the benchmark source, in milliseconds."""
    return 1000 * float(_source_value(BENCH_SRC, r"^RSS_POLL_S = ([\d.]+)$",
                                      "RSS_POLL_S"))


def dgp_macros():
    """The data-generating process the Data paragraph states, read from the
    generator source (a changed constant fails here or changes the text)."""
    def num(name):
        return _source_value(GEN_SRC, rf"^{name} = ([\d.]+)", name)

    return {
        "DGPShape": f"{float(num('WEIBULL_SHAPE')):g}",
        "DGPCensRate": f"{float(num('CENS_RATE')):g}",
        "DGPFollowUp": f"{float(num('MAX_FOLLOW_UP')):g}",
    }


def package_macros():
    """The fitter's defaults the Method section quotes, read from the package
    source (not imported, so the paper builds without the compiled kernel)."""
    def num(pattern, what):
        return _source_value(PKG_SRC, pattern, what)

    return {
        "CoxStreamTol": fmt_pow10(float(num(
            r"tol: float = ([\d.eE+-]+)", "tol default"))),
        "CoxStreamMaxIter": num(r"max_iter: int = (\d+)", "max_iter default"),
        "CoxStreamMaxHalvings": num(
            r"for _ in range\((\d+)\):  # backtracking line search",
            "step-halving limit"),
        "CoxStreamLLRtol": fmt_pow10(float(num(
            r"^_LL_RTOL = ([\d.eE+-]+)", "_LL_RTOL"))),
        # the R package's out-of-core entry point differs: the text says so
        "CoxStreamRMaxIter": _source_value(
            R_PKG_SRC, r"max_iter\s+= (\d+)L", "R max_iter default"),
    }


def _env(text, pattern, what):
    m = re.search(pattern, text, re.MULTILINE)
    if m is None:
        raise SystemExit(f"{what} not found in {ENV_TXT}")
    return m.group(1).strip()


def env_macros(text, host):
    """The Environment paragraph, from the VM's vm_environment.txt. `host` is
    the host stamp of the speed rows: the file must describe that machine."""
    node = _env(text, r"^Linux (\S+) ", "kernel host name")
    if not host.startswith(f"{node} "):
        raise SystemExit(f"{ENV_TXT.name} describes {node}, but the speed "
                         f"rows come from {host}")
    machine = _env(text, r"machine-type: \S*/machineTypes/(\S+)",
                   "machine type")
    vcpus = int(_env(text, r"^CPU\(s\):\s+(\d+)", "vCPU count"))
    # GCE n2-standard: 4 GB per vCPU. Other families need their own rule.
    m = re.fullmatch(r"n2-standard-(\d+)", machine)
    if m is None or int(m.group(1)) != vcpus:
        raise SystemExit(f"machine {machine} with {vcpus} vCPUs: only "
                         "n2-standard memory is known here")
    ram_gb = 4 * vcpus
    mem_kb = int(_env(text, r"^MemTotal:\s+(\d+) kB", "MemTotal"))
    if not 0.9 * ram_gb * 2 ** 20 < mem_kb <= ram_gb * 2 ** 20:
        raise SystemExit(f"MemTotal {mem_kb} kB does not fit {ram_gb} GB")
    cores = (int(_env(text, r"^Core\(s\) per socket:\s+(\d+)", "cores"))
             * int(_env(text, r"^Socket\(s\):\s+(\d+)", "sockets")))
    model = _env(text, r"^Model name:\s+(.+?) CPU @", "CPU model")
    model = model.replace("(R)", "")
    ghz = float(_env(text, r"^Model name:.*@ ([\d.]+)GHz", "CPU clock"))
    platform = _env(text, r"cpu-platform: (.+)$", "CPU platform")
    platform = platform.removeprefix(model.split()[0] + " ")
    # The text says "no thread limits were set".
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        if f"env {var} = (unset)" not in text:
            raise SystemExit(f"{var} was set on the VM; the text says no "
                             "thread limits were set")
    if '"openblas configuration": "OpenBLAS' not in text:
        raise SystemExit("NumPy is not built with OpenBLAS; fix the text")
    r_blas = _env(text, r"R BLAS: (\S+)", "R BLAS")
    if "openblas" in r_blas:
        r_blas = "OpenBLAS"
    elif re.search(r"/blas/libblas\.so", r_blas):
        r_blas = "the reference BLAS"
    else:
        raise SystemExit(f"unrecognised R BLAS {r_blas}")

    def ver(name):
        return _env(text, rf"^\s+{name}: (\S+)", f"{name} version")

    return {
        "EnvMachine": machine,
        "EnvVCPUs": str(vcpus),
        "EnvCores": str(cores),
        "EnvCPU": f"{model} ({platform}) at {ghz:g}\\,GHz",
        "EnvRAMGB": str(ram_gb),
        "EnvDiskGB": _env(text, r"^\S+\s+(\d+)G\s+\d\s+PersistentDisk",
                          "persistent disk"),
        "EnvUbuntu": _env(text, r"^Linux .*~(\d+\.\d+)\.\d+-Ubuntu",
                          "Ubuntu release"),
        "EnvPython": _env(text, r"^python: (\S+)", "Python version"),
        "EnvNumPy": ver("numpy"),
        "EnvSciPy": ver("scipy"),
        "EnvPyarrow": ver("pyarrow"),
        "EnvLifelines": ver("lifelines"),
        "EnvR": _env(text, r"^R version (\S+)", "R version"),
        "EnvSurvival": ver("survival"),
        "EnvCyclops": ver("Cyclops"),
        "EnvRBLAS": r_blas,
    }


def _row(rows, method, n_rows, p=N_SWEEP_P):
    """The single result row for (method, n_rows, p); error if absent."""
    hits = [
        r for r in rows
        if r["method"] == method and r["n_rows"] == n_rows and r.get("p") == p
    ]
    if not hits:
        raise SystemExit(
            f"missing row: method={method} n_rows={n_rows} p={p} "
            f"(has the sweep produced this cell yet?)"
        )
    row = hits[0]
    if row.get("coef") is None or row.get("errors"):
        raise SystemExit(
            f"failed cell: method={method} n_rows={n_rows} p={p} "
            f"(coef is null / errors present -- re-run this cell before "
            f"regenerating macros)"
        )
    return row


def _wall(row):
    return row["median_elapsed_s"]


def _rss(row):
    return row["median_rss_delta_mb"]


def _coef(row):
    return row["coef"]


def _max_abs_gap(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def _recovery_err(row, ref_true=None):
    """max |coef - true_coef|. Computed here rather than read from the
    'max_err_true' field, which only some methods (not cyclops) carry. Uses the
    row's own true_coef when present, else a reference truth passed in (all
    methods at one cell share the same data-generating coefficients)."""
    true = row.get("true_coef", ref_true)
    if true is None:
        raise SystemExit(
            f"no true_coef for {row['method']} @ n={row['n_rows']}")
    return max(abs(c - t) for c, t in zip(row["coef"], true))


# --- claims in the prose ----------------------------------------------------
def _sweep(rows, method, key):
    """{n_rows: median value} over a method's successful n-sweep cells."""
    return {r["n_rows"]: r[f"median_{key}"] for r in rows
            if r["method"] == method and r.get("p") == N_SWEEP_P
            and r.get("coef")}


def _line_fit(pts):
    """Least-squares (slope, intercept) of y on x over [(x, y)]."""
    xs, ys = zip(*pts)
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    slope = (sum((x - mx) * (y - my) for x, y in pts)
             / sum((x - mx) ** 2 for x in xs))
    return slope, my - slope * mx


def _claim(ok, sentence):
    if not ok:
        raise SystemExit("the data no longer support this sentence of "
                         f"paper.tex; rewrite it, then regenerate: {sentence}")


def check_claims(rows, ram_ratio_n):
    """The sentences about the sweep as a whole, checked at every n measured
    (the comparison-cell ones are checked where build_macros computes them)."""
    wall = {m: _sweep(rows, m, "elapsed_s") for m in
            (M_STREAM, M_LIFE, M_CYCLOPS, M_RSURV)}
    ram = {m: _sweep(rows, m, "rss_delta_mb") for m in wall}
    others = [m for m in wall if m != M_STREAM]
    _claim(all(wall[M_STREAM][n] < wall[m][n]
               for m in others for n in wall[m] if n in wall[M_STREAM]),
           "CoxStream is fastest at every n (abstract, Figure 3 caption, "
           "conclusion)")
    _claim(all(ram[M_STREAM][n] < ram[m][n]
               for m in others for n in ram[m] if n in ram[M_STREAM]),
           "every baseline adds more RAM than CoxStream at every n "
           "(Figure 4 caption: 'low-memory, low-time corner')")
    _claim(all(wall[M_CYCLOPS][n] < wall[m][n]
               for m in (M_RSURV, M_LIFE) for n in wall[M_CYCLOPS]
               if n in wall[m]),
           "Cyclops is faster than R survival::coxph and lifelines at every n "
           "(Figure 3 caption)")
    _claim(ram[M_LIFE][ram_ratio_n]
           > RAM_ORDER_OF_MAG * ram[M_STREAM][ram_ratio_n],
           "by n = CoxPHRAMRatioN lifelines uses more than an order of "
           "magnitude more RAM than CoxStream (abstract; Scaling)")


# --- the tie experiment (results/coxph_ties.json, bench_coxph_ties.py) ------
def ties_macros(ties):
    """Macros and claim checks for the streamed-vs-in-memory tie experiment.

    Rows are selected by label and checked against the method that must have
    produced them, so a row from other code cannot slip into the paper.
    """
    meta = ties.get("meta", {})
    if meta.get("schema_version") != TIES_SCHEMA:
        raise SystemExit("results/coxph_ties.json is not schema "
                         f"{TIES_SCHEMA}: run python3 -m "
                         "coxbench.bench_coxph_ties")
    if meta.get("pkg_src") != coxstream_source_hash():
        raise SystemExit("the tie experiment ran on coxstream source "
                         f"{meta.get('pkg_src')}, not the current "
                         f"{coxstream_source_hash()}: re-run "
                         "python3 -m coxbench.bench_coxph_ties")
    if meta.get("git_dirty") or meta.get("problems"):
        raise SystemExit("the tie experiment ran with uncommitted code or "
                         f"reported problems: {meta.get('problems')}")
    results = ties["results"]
    unknown = {r["method"] for r in results} - TIES_METHODS
    if unknown:
        raise SystemExit(f"coxph_ties.json has rows from {sorted(unknown)}")
    by_label = {r["label"]: r for r in results}
    if len(by_label) != len(results):
        raise SystemExit("coxph_ties.json has duplicate labels")

    def pick(label, method):
        r = by_label.get(label)
        if r is None or r["method"] != method:
            raise SystemExit(f"coxph_ties.json: no {method} row {label!r}")
        return r

    def r_efron(data):
        return pick(TIES_EFRON_LABEL[data], "r_survival_coxph")

    streams = [r for r in results if r["method"] == "coxstream_package"]
    stream_gap = max(_max_abs_gap(r["coef"], r_efron(r["data"])["coef"])
                     for r in streams)
    se_rel_gap = max(abs(s - t) / abs(t) for r in streams
                     for s, t in zip(r["se"], r_efron(r["data"])["se"]))
    numpy_ref = pick("sub_efron_numpy", "numpy_textbook_efron")
    numpy_gap = max(_max_abs_gap(r["coef"], numpy_ref["coef"])
                    for r in streams if r["data"] == "monthly_subsample")
    r_vs_life = max(
        _max_abs_gap(pick(life, "lifelines")["coef"], r_efron(data)["coef"])
        for data, life in TIES_LIFELINES_LABEL.items())
    life_ref = {data: pick(label, "lifelines")["coef"]
                for data, label in TIES_LIFELINES_LABEL.items()}
    stream_vs_life = max(_max_abs_gap(r["coef"], life_ref[r["data"]])
                         for r in streams if r["data"] in life_ref)
    bres_month = _max_abs_gap(pick("breslow_discrete", "r_survival_coxph")
                              ["coef"], r_efron("monthly")["coef"])
    bres_year = _max_abs_gap(pick("breslow_yearly", "r_survival_coxph")
                             ["coef"], r_efron("yearly")["coef"])

    structure = meta["tie_structure"]
    month_sizes = sorted(int(s) for s in structure["monthly"])
    year_sizes = sorted(int(s) for s in structure["yearly"])
    sub_sizes = sorted(int(s) for s in structure["monthly_subsample"])
    small = structure["monthly"][str(month_sizes[0])]
    _claim(small["n_tied_event_groups_straddling"]
           == small["n_tied_event_groups"] > 0,
           "with the smallest chunks every tied event time has its events "
           "split across chunks (Exactness)")
    _claim(month_sizes[-1] == meta["n_rows"] and sub_sizes[0] == 1,
           "the monthly grid's chunk sizes run up to the whole file in one "
           "chunk, and the subsample's down to single rows (Exactness)")
    _claim(len(year_sizes) > 1 and month_sizes[0] < year_sizes[0]
           and year_sizes[-1] < month_sizes[-1],
           "the yearly grid was fitted at chunk sizes in between "
           "(Exactness)")
    _claim(stream_gap < TIES_TOL,
           "the streamed fit matches an in-memory Efron fit at every chunk "
           "size (abstract; Tie correction; Exactness)")
    _claim(10 * stream_vs_life < r_vs_life,
           "the stream agrees with lifelines far more closely than R's fit "
           "does, so the residual against R is on R's side (Exactness)")
    _claim(min(bres_month, bres_year) > TIES_TEETH * stream_gap,
           "Breslow differs from Efron by far more than the streamed fit "
           "does, so the experiment tells the corrections apart "
           "(Exactness)")
    return {
        "TiesN": fmt_int(meta["n_rows"]),
        "TiesSubN": fmt_int(meta["sub_n_rows"]),
        "TiesRowGroupMin": fmt_int(month_sizes[0]),
        "TiesNTieGroups": fmt_int(small["n_tied_event_groups"]),
        "TiesMaxGroupEvents": fmt_int(small["max_group_events"]),
        "TiesStreamVsEfronCoef": fmt_sci(stream_gap),
        "TiesStreamVsEfronSERel": fmt_sci(se_rel_gap),
        "TiesStreamVsNumpyCoef": fmt_sci(numpy_gap),
        "TiesRVsLifelinesCoef": fmt_sci(r_vs_life),
        "TiesStreamVsLifelinesCoef": fmt_sci(stream_vs_life),
        "TiesBresVsEfronMonthly": fmt_sci(bres_month),
        "TiesBresVsEfronYearly": fmt_sci(bres_year),
    }


# --- macro computation ------------------------------------------------------
def build_macros(competing, ties, env_text):
    rows = competing["results"]

    def cell(method, n=COMPARISON_N, p=N_SWEEP_P):
        return _row(rows, method, n, p)

    stream = cell(M_STREAM)
    life = cell(M_LIFE)
    cyc = cell(M_CYCLOPS)
    rsurv = cell(M_RSURV)
    # Every speed and memory macro must come from one machine.
    one_host([stream, life, cyc, rsurv] + [
        r for r in rows
        if r.get("p") == N_SWEEP_P and r["method"] in (M_STREAM, M_LIFE)
        and r.get("coef")], "speed/RAM macros")

    # recovery error (platform-independent): max |coef - true_coef|. All
    # methods at the comparison cell share one data-generating truth, so
    # cyclops (which omits true_coef) borrows the stream row's.
    ref_true = stream["true_coef"]
    cox_err = _recovery_err(stream)
    _claim(all(fmt_sci(_recovery_err(r, ref_true)) == fmt_sci(cox_err)
               for r in (life, cyc, rsurv)),
           "Cyclops's recovery error matches every exact solver (Speed)")
    _claim(_max_abs_gap(_coef(rsurv), _coef(life)) < 1e-6,
           "the same MLE is returned by R survival::coxph (Exactness)")

    # every coefficient within a few model-based SEs of the truth: the
    # recovery error is sampling error, not bias
    err_se = max(abs(c - t) / s for c, t, s
                 in zip(stream["coef"], ref_true, stream["se"]))
    _claim(err_se < 3, "the recovery error is sampling error: every "
           "coefficient within CoxStreamMaxErrSE standard errors (Exactness)")

    # p-sweep recovery error (each p has its own truth). The p = N_SWEEP_P
    # row at PSWEEP_N is the n-sweep's cohort, not the width sweep's.
    def psweep(method):
        return {r["p"]: _recovery_err(r) for r in rows
                if r["method"] == method and r["n_rows"] == PSWEEP_N
                and r.get("p") != N_SWEEP_P and r.get("coef")}

    p_err = psweep(M_STREAM)
    _claim(all(p_err[a] >= p_err[b] for a, b in pairwise(sorted(p_err))),
           "the recovery error falls as p grows towards the full model "
           "(Exactness: Model width)")
    for m in (M_LIFE, M_RSURV):
        other = psweep(m)
        _claim(bool(other) and min(other) == P_LO and max(other) == P_HI
               and all(fmt_sci(e) == fmt_sci(p_err[p])
                       for p, e in other.items()),
               f"{m} returns the same recovery error as CoxStream at every "
               "width from CoxPHPLo (Exactness: Model width)")
    err_lo, err_hi = p_err[P_LO], p_err[P_HI]
    wide_true = _row(rows, M_STREAM, PSWEEP_N, P_HI)["true_coef"]
    extra_coef = set(wide_true[N_SWEEP_P:])
    if wide_true[:N_SWEEP_P] != ref_true or len(extra_coef) != 1:
        raise SystemExit("the width sweep's truth is not the n-sweep's ten "
                         "coefficients plus one shared extra; fix the Data "
                         "paragraph")

    # coxstream RAM range for the "near-constant" claim. RAMHiN is the
    # largest n where the footprint is still within RAM_FLAT_TOL of the
    # smallest-n footprint, so the sentence "growing n to RAMHiN leaves memory
    # near-constant" stays true by construction: if RSS ever jumps (e.g. a
    # macOS-compression artifact, or a genuine leak), that n is excluded.
    stream_rss = _sweep(rows, M_STREAM, "rss_delta_mb")
    stream_ns = sorted(stream_rss)
    base_rss = stream_rss[stream_ns[0]]
    ram_lo_n = stream_ns[0]
    ram_hi_n = max(n for n in stream_ns
                   if stream_rss[n] <= RAM_FLAT_TOL * base_rss)

    # RAM-ratio headline cell: largest n where an in-memory baseline is also
    # measured (most dramatic honest comparison).
    life_rss = _sweep(rows, M_LIFE, "rss_delta_mb")
    ram_ratio_n = max(n for n in stream_ns if n in life_rss)
    check_claims(rows, ram_ratio_n)

    # lifelines' footprint extrapolated to registry scale along its measured
    # straight line; refused if the line does not describe the data
    slope, icept = _line_fit(sorted(life_rss.items()))
    _claim(all(abs(slope * n + icept - v) <= LINEAR_FIT_TOL * v
               for n, v in life_rss.items()),
           "lifelines' RAM added grows linearly in n (Scaling: the "
           "extrapolation to ExtrapN rows)")

    # run-to-run spread over the repeats of every quoted method
    quoted = (M_STREAM, M_LIFE, M_CYCLOPS, M_RSURV)

    def spread_pct(key):
        return 100 * max(max(r[key]) / min(r[key]) - 1 for r in rows
                         if r["method"] in quoted and r.get("p") == N_SWEEP_P
                         and r.get("coef"))

    # pass counts of the measured CoxStream fits: one pass per iteration plus
    # the initial one means no step-halving anywhere
    pk = [r for r in rows if r["method"] == M_STREAM and r.get("coef")]
    if not all(r.get("n_iter_reps") and r.get("n_passes_reps") for r in pk):
        raise SystemExit(f"{M_STREAM} rows lack n_iter_reps/n_passes_reps: "
                         "re-run them with the current benchmark")
    _claim(all(p == i + 1 for r in pk
               for i, p in zip(r["n_iter_reps"], r["n_passes_reps"])),
           "no measured CoxStream fit backtracked: passes = iterations + 1 "
           "(Speed)")
    sweep_iters = {i for r in pk if r.get("p") == N_SWEEP_P
                   for i in r["n_iter_reps"]}
    if len(sweep_iters) != 1:
        raise SystemExit(f"the p = {N_SWEEP_P} fits took "
                         f"{sorted(sweep_iters)} iterations; the text quotes "
                         "one count")
    stream_iters = sweep_iters.pop()

    # repeat counts at the comparison cell; the baselines must agree
    base_reps = {r["n_reps"] for r in (life, cyc, rsurv)}
    if len(base_reps) != 1:
        raise SystemExit(f"baselines differ in repeats at the comparison "
                         f"cell: {sorted(base_reps)}")

    # standard errors: both rows carry the model-based SEs of the same fit
    if not (stream.get("se") and life.get("se")):
        raise SystemExit(f"{M_STREAM} or {M_LIFE} @ n={COMPARISON_N} has no "
                         "'se': re-run the cell with the current benchmark")
    se_rel_gap = max(abs(s - t) / abs(t)
                     for s, t in zip(stream["se"], life["se"]))

    if ram_hi_n % ram_lo_n:
        raise SystemExit(f"RAM range {ram_lo_n}..{ram_hi_n} is not a whole "
                         "multiple; the text says 'a ...x increase'")

    return {
        # cells (editorial)
        "CoxPHNSweepP": str(N_SWEEP_P),
        "CoxPHPSweepN": fmt_int(PSWEEP_N),
        "CompetingCoxPHN": fmt_int(COMPARISON_N),
        "CoxStreamRAMLoN": fmt_int(ram_lo_n),
        "CoxStreamRAMHiN": fmt_int(ram_hi_n),
        "CoxStreamRAMNFactor": str(ram_hi_n // ram_lo_n),
        "CoxPHRAMRatioN": fmt_int(ram_ratio_n),
        "CoxPHPLo": str(P_LO),
        "CoxPHPHi": str(P_HI),
        # recovery error (platform-independent)
        "CoxStreamMaxErr": fmt_sci(cox_err),
        "CoxStreamMaxErrSE": fmt_num(err_se),
        "CyclopsMaxErr": fmt_sci(_recovery_err(cyc, ref_true)),
        "CoxStreamErrLoP": fmt_sci(err_lo),
        "CoxStreamErrHiP": fmt_sci(err_hi),
        # coefficient agreement (platform-independent)
        "CoxStreamVsLifelinesCoef": fmt_sci(
            _max_abs_gap(_coef(stream), _coef(life))),
        "CoxStreamVsLifelinesSERel": fmt_sci(se_rel_gap),
        "CyclopsVsLifelinesCoef": fmt_sci(
            _max_abs_gap(_coef(cyc), _coef(life))),
        # tie experiment (platform-independent)
        **ties_macros(ties),
        # measurement protocol
        "CoxStreamIters": str(stream_iters),
        "CoxStreamPasses": str(stream_iters + 1),
        "CoxStreamReps": str(stream["n_reps"]),
        "BaselineReps": str(base_reps.pop()),
        "WallSpreadPct": fmt_num(spread_pct("elapsed_s_reps")),
        "RAMSpreadPct": fmt_num(spread_pct("rss_delta_mb_reps")),
        "CoxStreamChunkRows": fmt_int(_bench_chunk_rows()),
        "RSSPollMs": f"{_bench_rss_poll_ms():g}",
        # data-generating process (generator source, and the rows' truth)
        **dgp_macros(),
        "DGPCoefMin": f"{min(map(abs, ref_true)):g}",
        "DGPCoefMax": f"{max(map(abs, ref_true)):g}",
        "DGPExtraCoef": f"{extra_coef.pop():g}",
        # speed (hardware-dependent)
        "LifelinesVsCoxStreamSpeedup": fmt_num(_wall(life) / _wall(stream)),
        "CoxStreamVsCyclopsSpeedup": fmt_num(_wall(cyc) / _wall(stream)),
        "CyclopsVsLifelinesSpeedup": fmt_num(_wall(life) / _wall(cyc)),
        # RAM ratios (hardware-dependent)
        "CyclopsRAMRatio": fmt_num(_rss(cyc) / _rss(stream)),
        "RSurvivalRAMRatio": fmt_num(_rss(rsurv) / _rss(stream)),
        # absolute RAM added, GB (hardware-dependent; MB here are MiB)
        "CoxStreamRAMMinGB": fmt_num(min(stream_rss.values()) / 1024, 2),
        "CoxStreamRAMMaxGB": fmt_num(max(stream_rss.values()) / 1024, 2),
        "LifelinesRAMGB": fmt_num(life_rss[ram_ratio_n] / 1024),
        "ExtrapN": fmt_pow10(EXTRAP_N),
        "LifelinesRAMExtrapGB": fmt_num(
            (slope * EXTRAP_N + icept) / 1024, 0),
        # fitter defaults (from the package source)
        **package_macros(),
        # environment (from the VM that measured speed and RAM)
        **env_macros(env_text, stream["host"]),
    }


HEADER = (
    "% GENERATED by src/coxbench/gen_arxiv_macros.py -- do not\n"
    "% edit by hand. Regenerate after a benchmark re-run:\n"
    "%   python3 -m coxbench.gen_arxiv_macros\n"
    "% Speed/RAM macros track the host that wrote competing_coxph.json; the\n"
    "% recovery-error, coef-gap and tie macros are platform-independent.\n"
)


def render(macros):
    lines = [HEADER]
    for name, value in macros.items():
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path,
                    default=RESULTS_DIR / "competing_coxph.json")
    ap.add_argument("--ties", type=Path,
                    default=RESULTS_DIR / "coxph_ties.json")
    ap.add_argument("--env", type=Path, default=ENV_TXT)
    ap.add_argument("--out", type=Path, default=MACROS_OUT)
    ap.add_argument("--check", action="store_true",
                    help="print values and diff against the current file; "
                         "write nothing")
    args = ap.parse_args()

    macros = build_macros(_load(args.results), _load(args.ties),
                          args.env.read_text())
    text = render(macros)

    if args.check:
        import re
        old = {}
        if args.out.exists():
            for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{(.*)\}",
                                 args.out.read_text()):
                old[m.group(1)] = m.group(2)
        print(f"{'macro':<30} {'new':<26} {'old (current file)'}")
        for name, value in macros.items():
            prev = old.get(name, "(absent)")
            flag = "" if prev == value else "  <-- CHANGED"
            print(f"{name:<30} {value:<26} {prev}{flag}")
        return

    args.out.write_text(text)
    print(f"wrote {len(macros)} macros -> {args.out}")


if __name__ == "__main__":
    main()
