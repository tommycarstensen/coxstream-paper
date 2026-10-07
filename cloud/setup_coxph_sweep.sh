#!/usr/bin/env bash
# setup_coxph_sweep.sh -- provision a Debian/Ubuntu VM for the competing CoxPH
# benchmark (cloud/run_coxph_nsweep.sh).  Installs Python 3.13, Cython/C++/Rust
# CoxPH backends, R + survival, and wires passwordless sudo for cold-cache purge.
# Idempotent.
#
#   bash cloud/setup_coxph_sweep.sh
#
# Tunables: PYTHON_BIN (default python3.13), VENV_DIR (.venv),
#           SKIP_R=1, SKIP_COLD=1.

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.13}"
VENV_DIR="${VENV_DIR:-.venv}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "=== [1/7] System packages (apt) ==="
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
  build-essential git curl ca-certificates pkg-config cmake libeigen3-dev \
  libcurl4-openssl-dev libssl-dev libxml2-dev software-properties-common

echo ""
echo "=== [2/7] Rust toolchain (for coxph-rs / maturin) ==="
# The C++ and Rust backends are research code the paper does not report. The
# public reproducibility archive leaves them out, and then this step and their
# installs below are skipped.
if [ -d packages/coxph-rs ]; then
  if ! command -v cargo >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  fi
  # shellcheck disable=SC1091
  source "${HOME}/.cargo/env"
  cargo --version
else
  echo "packages/coxph-rs not present: skipping the Rust toolchain"
fi

echo ""
echo "=== [3/7] Python ${PYTHON_BIN} ==="
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt-get update -y
  sudo apt-get install -y python3.13 python3.13-venv python3.13-dev
fi
"${PYTHON_BIN}" --version

echo ""
echo "=== [4/7] Python venv + pip deps ==="
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip wheel
python -m pip install \
  cython setuptools \
  numpy scipy pandas polars pyarrow psutil \
  lifelines scikit-survival \
  duckdb pypdf

echo ""
echo "=== [5/7] CoxPH backends (Cython + C++ + Rust) ==="
python -m pip install -e packages/coxstream  # the released package (coxstream_package)
python -c "import coxstream; print('coxstream', coxstream.__version__)"
if [ -d packages/coxph-cpp ]; then
  python -m pip install ./packages/coxph-cpp   # scikit-build-core + CMake
fi
if [ -d packages/coxph-rs ]; then
  python -m pip install ./packages/coxph-rs    # maturin + Rust
fi
python -m pip install -e .                   # the coxbench harness itself
# Build Cython kernel in-place (coxph_cython + coxph_cython_f32)
( cd src/coxbench && python setup_cython.py build_ext --inplace )
# Smoke-test the backends that were installed. coxph_cython is built in-place
# in src/coxbench, so import from there (the bench's --run-one child is
# launched by file path, which puts that dir on sys.path the same way);
# coxph_cpp / coxph_rs are pip-installed and import from anywhere.
BACKENDS="coxph_cython"
if [ -d packages/coxph-cpp ]; then BACKENDS="${BACKENDS} coxph_cpp"; fi
if [ -d packages/coxph-rs ]; then BACKENDS="${BACKENDS} coxph_rs"; fi
( cd src/coxbench && python -c "import importlib, sys; [importlib.import_module(m) for m in sys.argv[1:]]; print('CoxPH backends OK:', *sys.argv[1:])" ${BACKENDS} )

echo ""
if [ "${SKIP_R:-}" = "1" ]; then
  echo "=== [6/7] R toolchain SKIPPED (SKIP_R=1) ==="
else
  echo "=== [6/7] R toolchain + CRAN packages (survival, arrow, jsonlite) ==="
  if ! command -v R >/dev/null 2>&1; then
    sudo apt-get install -y r-base r-base-dev
  fi
  R --version | head -1
  sudo Rscript -e '
    p <- c("survival","arrow","jsonlite")
    m <- setdiff(p, rownames(installed.packages()))
    if (length(m)) install.packages(m, repos="https://cloud.r-project.org",
                                    Ncpus=max(1L, parallel::detectCores()))
    if (length(setdiff(p, rownames(installed.packages()))))
      stop("R install failed")
    cat("R packages OK\n")
  '
fi

echo ""
if [ "${SKIP_COLD:-}" = "1" ]; then
  echo "=== [7/7] Cold-cache sudoers SKIPPED (SKIP_COLD=1) ==="
else
  echo "=== [7/7] Passwordless sudo for cold-cache purge ==="
  SUDOERS_FILE="/etc/sudoers.d/benchmark-drop-caches"
  echo "${USER} ALL=(ALL) NOPASSWD: /usr/bin/tee /proc/sys/vm/drop_caches" \
    | sudo tee "${SUDOERS_FILE}" >/dev/null
  sudo chmod 0440 "${SUDOERS_FILE}"
  sudo visudo -cf "${SUDOERS_FILE}"
fi

echo ""
echo "=== Setup complete. ==="
echo "  source ${VENV_DIR}/bin/activate"
echo "  bash cloud/run_coxph_nsweep.sh   # its header shows how to start it detached"
