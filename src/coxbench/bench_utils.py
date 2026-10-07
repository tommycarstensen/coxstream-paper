"""Benchmark utilities the CoxPH harness shares across scripts.

The commit hash a result row records, and the page-cache drop for cold runs.
"""
from __future__ import annotations

import platform
import subprocess

# ---------------------------------------------------------------------------
# OS page-cache drop (cold benchmarks)
# ---------------------------------------------------------------------------

# Drop the page cache between cold runs. macOS uses `purge`; Linux writes 3 to
# /proc/sys/vm/drop_caches. Both need passwordless sudo for the one command so
# a cold sweep does not stall on a password prompt mid-run.
_DROP_CACHES = "/proc/sys/vm/drop_caches"
_PURGE_SUDOERS = {
    "Darwin": "    <username> ALL=(ALL) NOPASSWD: /usr/sbin/purge",
    "Linux": f"    <username> ALL=(ALL) NOPASSWD: /usr/bin/tee {_DROP_CACHES}",
}


def _purge_page_cache() -> None:
    """Drop the OS page cache so the next read hits disk, not RAM.

    macOS: `sudo purge`.
    Linux: `sync`, then write 3 to /proc/sys/vm/drop_caches via `sudo tee`.
    """
    system = platform.system()
    if system == "Darwin":
        subprocess.run(["sudo", "purge"], check=True)
    elif system == "Linux":
        subprocess.run(["sync"], check=True)
        subprocess.run(
            ["sudo", "tee", _DROP_CACHES],
            input=b"3", stdout=subprocess.DEVNULL, check=True,
        )
    else:
        raise NotImplementedError(
            f"Page-cache drop is not implemented for {system}."
        )


def check_sudo_purge() -> None:
    """Verify passwordless sudo cache-purge works before a cold run.

    Fails loudly so the user knows to configure passwordless sudo rather than
    hitting a password prompt mid-run. The check itself drops the cache (a
    harmless no-op right before a cold run).
    """
    system = platform.system()
    if system == "Darwin":
        cmd = ["sudo", "-n", "purge"]
        proc_input = None
    elif system == "Linux":
        cmd = ["sudo", "-n", "tee", _DROP_CACHES]
        proc_input = b"3"
    else:
        raise NotImplementedError(
            f"Page-cache drop is not implemented for {system}."
        )
    result = subprocess.run(cmd, input=proc_input, capture_output=True,
                            check=False)
    if result.returncode != 0:
        raise PermissionError(
            "Cold benchmarks require passwordless sudo for the cache purge.\n"
            "Add this line to /etc/sudoers (via `sudo visudo`):\n"
            + _PURGE_SUDOERS[system]
        )


def git_hash() -> str:
    """Return the short HEAD commit hash, or 'unknown' without git."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
