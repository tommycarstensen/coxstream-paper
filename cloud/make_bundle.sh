#!/usr/bin/env bash
# make_bundle.sh -- package the repo (tracked files at HEAD) into a tarball you
# can copy to a GCP VM. Run LOCALLY from the repo root. This does NOT run any
# benchmark; it only archives source. Data are gitignored and are generated on
# the VM by cloud/run_coxph_nsweep.sh.
#
#   bash cloud/make_bundle.sh
#
# Output: coxstream-bench-bundle.tar.gz in the repo root. Uncommitted edits are
# NOT in it: commit first.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

OUT="coxstream-bench-bundle.tar.gz"
PREFIX="coxstream/"

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "WARNING: uncommitted changes to tracked files are NOT in the bundle:"
  git status --short --untracked-files=no
fi

echo "Archiving tracked files at HEAD ($(git rev-parse --short HEAD)) -> ${OUT}"
git archive --format=tar.gz --prefix="${PREFIX}" -o "${OUT}" HEAD

SIZE=$(du -h "${OUT}" | cut -f1)
echo "Wrote ${OUT} (${SIZE})"
echo ""
echo "Copy to the VM and run: see the header of cloud/run_coxph_nsweep.sh."
