#!/usr/bin/env bash
# Every speed and memory number in the arXiv CoxStream paper, measured on ONE
# Linux VM (GCloud n2-standard-8, 32 GB) where RSS is trustworthy:
#   - p=10 n-sweep 1M..32M: coxstream_package (the released package the paper
#     describes) and coxstream_cython (research kernel, for comparison) at 5
#     reps; lifelines, R survival::coxph and Cyclops at 2 reps up to 16M
#     (32M too with BASELINES_32M=1). lifelines and coxstream_package record
#     SEs. BIGSURVSGD=1 also times the harness's BigSurvSGD reimplementation
#     (10 epochs up to 16M, 100 epochs up to 8M, 2 reps), which the paper
#     does not report.
#   - width sweep at n=1M, p=4..128: coxstream_package, 1 rep (accuracy).
# Rows of these methods measured on any OTHER host are dropped first, so no
# ratio mixes machines, and coxstream_package rows measured with a different
# package source (hash of coxstream.py + _kernel.pyx) are dropped too; rows
# already measured here with the current code are kept, so a re-run after an
# interruption, or after updating packages/coxstream, resumes instead of
# starting over.
#
# The laptop commands below take the cloud project, zone and VM name from the
# environment, so nothing here names a project. Set them once per shell:
#   export GCP_PROJECT=<your project id> GCP_ZONE=europe-west1-b VM=coxph-sweep
# Laptop (once): bundle the repo and create the VM. Pin Cascade Lake, the
# platform of the August VM ("Intel Xeon @ 2.80GHz"):
#   cd ~/coxstream && bash cloud/make_bundle.sh && gcloud compute instances create "$VM" --project="$GCP_PROJECT" --zone="$GCP_ZONE" --machine-type=n2-standard-8 --min-cpu-platform="Intel Cascade Lake" --image-family=ubuntu-2404-lts-amd64 --image-project=ubuntu-os-cloud --boot-disk-size=100GB --boot-disk-type=pd-ssd
# Laptop: copy the bundle and log in. A fresh VM refuses SSH ("Connection
# refused") for about a minute while it boots; if so, rerun this line:
#   cd ~/coxstream && gcloud compute scp coxstream-bench-bundle.tar.gz "$VM":~/ --project="$GCP_PROJECT" --zone="$GCP_ZONE" && gcloud compute ssh "$VM" --project="$GCP_PROJECT" --zone="$GCP_ZONE"
# The prompt then shows the VM's name; run the VM lines below only there.
# VM, first time: unpack and set up (interactive; about 15 min):
#   cd ~ && tar xzf coxstream-bench-bundle.tar.gz && cd ~/coxstream && bash cloud/setup_coxph_sweep.sh
# VM: start the sweep detached from the terminal, so closing the window, a
# dropped SSH session or a sleeping laptop does not stop it (the first run on
# 2026-10-06 died at 16M when its session went away). It writes its own log:
#   cd ~/coxstream && setsid -f nohup bash cloud/run_coxph_nsweep.sh > /dev/null 2>&1 < /dev/null
# Or from the laptop without logging in; it returns at once. (The form ending in "&" instead of setsid -f makes this ssh wait until the whole sweep ends, because the backgrounded "cd && setsid" list keeps the session's output open.)
#   gcloud compute ssh "$VM" --project="$GCP_PROJECT" --zone="$GCP_ZONE" --command='cd ~/coxstream && setsid -f nohup bash cloud/run_coxph_nsweep.sh > /dev/null 2>&1 < /dev/null'
# Watch it (Ctrl-C stops only the watching, not the run), on the VM or from the laptop:
#   tail -f "$(ls -t ~/coxstream/tmp/run_coxph_nsweep_*.log | head -1)"
#   gcloud compute ssh "$VM" --project="$GCP_PROJECT" --zone="$GCP_ZONE" --command='tail -f "$(ls -t ~/coxstream/tmp/run_coxph_nsweep_*.log | head -1)"'
# A run that stopped part-way resumes where it left off when started again.
# Laptop (after): fetch the results, then delete the VM to stop billing:
#   cd ~/coxstream && gcloud compute scp "$VM":~/coxstream/results/competing_coxph.json "$VM":~/coxstream/results/vm_environment.txt results/ --project="$GCP_PROJECT" --zone="$GCP_ZONE" && gcloud compute instances delete "$VM" --project="$GCP_PROJECT" --zone="$GCP_ZONE"
#
# Log: tmp/run_coxph_nsweep_<time>.log (path printed first, exit status last).
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p tmp results
LOG="tmp/run_coxph_nsweep_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "log: $LOG"
trap 'echo "exit status: $?"' EXIT
# One run at a time: two runs would write results/competing_coxph.json
# concurrently and lose reps.
exec 9> tmp/run_coxph_nsweep.lock
if ! flock -n 9; then
    echo "another run_coxph_nsweep.sh is already running; not starting"
    exit 1
fi

if [ ! -f .venv/bin/activate ]; then
    echo "no .venv here: run cloud/setup_coxph_sweep.sh first"
    exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate
BENCH="python -m coxbench.bench_competing_coxph"
SIZES="1 2 4 8 16 32"
CAP=16000000
if [ "${BASELINES_32M:-0}" = "1" ]; then
    CAP=32000000
fi
echo "in-memory baselines up to n=${CAP}"

echo "== [1/7] R packages: Cyclops, coxstream =="
# setup_coxph_sweep.sh installs survival/arrow/jsonlite; these two are
# opt-in, so the sweep installs them itself and fails at setup, not hours in.
# sudo: the system library is root-writable; a plain Rscript cannot write it
# non-interactively. Posit Package Manager serves prebuilt Ubuntu 24.04
# binaries (seconds instead of a ~30 min Cyclops compile) given this
# HTTPUserAgent, and falls back to source if a binary is missing.
sudo Rscript -e 'options(HTTPUserAgent = sprintf("R/%s R (%s)", getRversion(), paste(getRversion(), R.version$platform, R.version$arch, R.version$os))); repo <- "https://packagemanager.posit.co/cran/__linux__/noble/latest"; pkgs <- c("Cyclops", "coxstream"); for (p in pkgs) if (!requireNamespace(p, quietly = TRUE)) install.packages(p, repos = repo, Ncpus = max(1L, parallel::detectCores())); miss <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]; if (length(miss)) stop("failed to install: ", paste(miss, collapse = ", "))' || exit 1

echo "== [2/7] coxstream 0.3.0 from packages/coxstream =="
python -m pip install --quiet -e packages/coxstream || exit 1
python -c "import coxstream; v = coxstream.__version__; assert v == '0.3.0', v; print('coxstream', v)" || exit 1

echo "== [3/7] record the VM environment -> results/vm_environment.txt =="
ENV_OUT="results/vm_environment.txt"
{
    echo "# CoxPH sweep VM environment"
    echo
    # The cloud project is kept out of the record (it ends up in a public
    # results file): its id sits in the node name, its number in the
    # machine-type path.
    echo "## OS / kernel"
    uname -a | sed -E 's/\.c\.[a-z0-9-]+\.internal//'
    echo
    echo "## GCE machine type, CPU platform and disks"
    for key in machine-type cpu-platform; do
        printf '  %s: ' "$key"
        { curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/$key" || printf '(metadata unavailable)'; } | sed -E 's#projects/[0-9]+#projects/PROJECT#'
        echo
    done
    lsblk -d -o NAME,SIZE,ROTA,MODEL 2>/dev/null
    echo
    echo "## CPU"
    lscpu | grep -Ei 'model name|^cpu\(s\)|thread|core|socket|bogomips|flags' | head -20
    echo
    echo "## Memory"
    grep -Ei 'memtotal' /proc/meminfo
    echo
    echo "## Python + modules + BLAS"
    python - <<'PY'
import os
import platform

print("python:", platform.python_version(), "(", platform.platform(), ")")
for mod in ("numpy", "scipy", "pyarrow", "cython", "lifelines",
            "coxstream", "sksurv", "pandas", "duckdb"):
    try:
        m = __import__(mod)
        print(f"  {mod}: {getattr(m, '__version__', '?')}")
    except ImportError as e:
        print(f"  {mod}: (not importable: {type(e).__name__})")
print()
print("numpy BLAS/LAPACK build:")
import numpy as np  # noqa: E402

np.show_config()
for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    print(f"  env {var} =", os.environ.get(var, "(unset)"))
PY
    echo
    echo "## R + packages"
    R --version | head -1
    Rscript -e 'for (p in c("survival","Cyclops","arrow","coxstream","jsonlite")) {
        v <- tryCatch(as.character(packageVersion(p)), error=function(e) "(absent)")
        cat(sprintf("  %s: %s\n", p, v)) }'
    Rscript -e 'cat("  R BLAS:", extSoftVersion()[["BLAS"]], "\n");
                cat("  R LAPACK:", La_library(), "\n")'
} 2>&1 | sed -e "s#${PWD}/##g" -e "s#${HOME}#~#g" > "$ENV_OUT"  # a warning's absolute path names the user and the checkout
sed 's/^/    /' "$ENV_OUT"

echo "== [4/7] back up results; drop the swept methods' rows from other hosts =="
cp results/competing_coxph.json "results/competing_coxph.json.bak_$(date +%Y%m%d_%H%M%S)"
python - <<'PY'
import json

from coxbench.provenance import coxstream_source_hash, host_label

host = host_label()  # as the benchmark stamps
src = coxstream_source_hash("packages/coxstream/src/coxstream")
swept = {"coxstream_package", "coxstream_cython", "lifelines",
         "r_survival_coxph", "cyclops",
         "bigsurvsgd_s50_e10", "bigsurvsgd_s50_e100"}
path = "results/competing_coxph.json"
with open(path) as fh:
    data = json.load(fh)


def stale(r):
    if r["method"] not in swept:
        return False
    if r["method"] == "coxstream_package":      # n-sweep and width sweep
        return r.get("host") != host or r.get("pkg_src") != src
    if r.get("p") != 10:
        return False
    if r.get("host") != host:
        return True
    return r["method"] == "lifelines" and "se" not in r   # needs SEs


keep = [r for r in data["results"] if not stale(r)]
print(f"  this host: {host}; coxstream source {src}")
print(f"  dropped {len(data['results']) - len(keep)} rows (other host, "
      f"other package source, or lifelines without SEs); kept {len(keep)}")
data["results"] = keep
with open(path, "w") as fh:
    json.dump(data, fh, indent=2)
PY

echo "== [5/7] data: the seeded cohorts (generated only if missing) =="
nparts=$(find data/survival_parts -maxdepth 1 -name 'part_*.parquet' 2>/dev/null | wc -l | tr -d ' ')
if [ ! -f data/survival_parts/true_coef.json ] || [ "$nparts" -lt 32 ]; then
    python -m coxbench.generate_survival_data --nrows 32000000 --nparts 32 || exit 1
fi
P128=data/survival_parts_p128
if [ ! -f "$P128/true_coef.json" ]; then
    python -m coxbench.generate_survival_data --ncovariates 128 --nrows 1000000 --nparts 1 --outdir "$P128" || exit 1
fi

echo "== [6/7] measurements =="
for n in $SIZES; do
    echo "======== n = ${n}M  ($(date +%H:%M:%S)) ========"
    $BENCH --methods coxstream_package coxstream_cython --mrows "$n" --reps 5 --cache cold
    $BENCH --methods lifelines r_survival_coxph cyclops --mrows "$n" --reps 2 --cache cold --in-mem-cap "$CAP"
    if [ "${BIGSURVSGD:-0}" = "1" ]; then
        $BENCH --methods bigsurvsgd --mrows "$n" --reps 2 --cache cold --sgd-epochs 10 --in-mem-cap 16000000
        $BENCH --methods bigsurvsgd --mrows "$n" --reps 2 --cache cold --sgd-epochs 100 --in-mem-cap 8000000
    fi
done
for p in 4 8 16 32 64 128; do
    echo "======== width sweep: n = 1M, p = ${p}  ($(date +%H:%M:%S)) ========"
    $BENCH --methods coxstream_package --mrows 1 -p "$p" --reps 1 --cache cold --parts-dir "$P128"
done

echo "== [7/7] summary =="
python - <<'PY'
import json

with open("results/competing_coxph.json") as fh:
    rows = json.load(fh)["results"]
# Rows from before the benchmark stamped a host have none; sorting None
# against a string raised TypeError and ended the 2026-10-06 run with status 1.
print("hosts of p=10 rows:", sorted({r.get("host") or "(none)" for r in rows
                                     if r.get("p") == 10}))


def cell(method, n, p=10):
    return next((r for r in rows if r["method"] == method
                 and r["n_rows"] == n and r.get("p") == p), None)


print(f"{'method':<20} {'n':>4} {'median s':>9} {'median MB':>10} "
      f"{'reps':>5} {'passes':>7} errors")
for m in ("coxstream_package", "coxstream_cython", "lifelines", "cyclops",
          "r_survival_coxph", "bigsurvsgd_s50_e10", "bigsurvsgd_s50_e100"):
    for r in sorted((r for r in rows if r["method"] == m and r.get("p") == 10),
                    key=lambda r: r["n_rows"]):
        print(f"{m:<20} {r['n_rows'] // 10**6:>3}M "
              f"{r.get('median_elapsed_s') or float('nan'):>9.2f} "
              f"{r.get('median_rss_delta_mb') or float('nan'):>10.0f} "
              f"{r.get('n_reps', 0):>5} "
              f"{(r.get('n_passes_reps') or ['-'])[-1]:>7} "
              f"{len(r.get('errors', []))}")
life, pkg = cell("lifelines", 4_000_000), cell("coxstream_package", 4_000_000)
if life and pkg and life.get("se") and pkg.get("se"):
    gap = max(abs(x - y) / abs(y) for x, y in zip(pkg["se"], life["se"]))
    print(f"max relative SE gap, package vs lifelines @ 4M: {gap:.2e}")
else:
    print("SE comparison @ 4M: MISSING (a row lacks 'se')")
print("width sweep p present:", [p for p in (4, 8, 16, 32, 64, 128)
                                 if cell("coxstream_package", 1_000_000, p)])
PY
