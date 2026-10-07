"""Provenance helpers shared by the benchmarks and the paper's macro generator.

Kept free of heavy imports, so a script can stamp or check where a result
came from without loading the benchmark harness.
"""
import hashlib
import platform
import re
from pathlib import Path

from coxbench.paths import REPO_ROOT

# The coxstream package source the paper describes (editable install).
PKG_SRC_DIR = REPO_ROOT / "packages" / "coxstream" / "src" / "coxstream"

# A Google Cloud VM's node name ends in .c.<project>.internal.
_CLOUD_PROJECT_SUFFIX = re.compile(r"\.c\.[a-z0-9-]+\.internal$")


def host_label(node=None, machine=None):
    """'<node> (<arch>)': the host stamp on measured rows.

    The cloud project is dropped from a Google Cloud node name, so result
    files never carry a tenant's identifier; the VM name and zone, which tell
    one measurement machine from another, stay.
    """
    node = platform.node() if node is None else node
    machine = platform.machine() if machine is None else machine
    return f"{_CLOUD_PROJECT_SUFFIX.sub('', node)} ({machine})"


def scrub_paths(text, repo_root=REPO_ROOT, home=None):
    """`text` with the machine's own directories taken out.

    A traceback or a warning kept in a result file names absolute paths, and
    with them the user and whatever the checkout's parent directories are
    called. Paths inside the repository become repository-relative, and the
    home directory becomes ~.
    """
    home = Path.home() if home is None else Path(home)
    text = text.replace(f"{repo_root}/", "")
    return text.replace(str(home), "~")


def coxstream_source_hash(src_dir=PKG_SRC_DIR):
    """Short hash of the coxstream package source a row was measured with.

    cloud/run_coxph_nsweep.sh hashes packages/coxstream/src/coxstream the
    same way and re-measures coxstream_package rows whose hash differs.
    """
    blob = b"".join((Path(src_dir) / f).read_bytes()
                    for f in ("coxstream.py", "_kernel.pyx"))
    return hashlib.sha256(blob).hexdigest()[:12]
