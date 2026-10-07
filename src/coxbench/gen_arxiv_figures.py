"""Generate the arXiv CoxStream paper's figures from competing_coxph.json.

Writes papers/arxiv/figures/{coxph_ram_scaling, coxph_time_scaling,
coxph_efficiency_scatter}.pdf, which paper.tex includes from its own directory
so the arXiv source bundle is self-contained. The figures read only the
committed results/competing_coxph.json and take their method keys from
gen_arxiv_macros, so the plotted rows are the ones the text quotes. Values are
the per-cell medians over repeats, as in the text.

Usage
-----
    python3 -m coxbench.gen_arxiv_figures
"""

import json

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator

from coxbench.colors import BLU, DGR, NAV
from coxbench.gen_arxiv_macros import (
    M_CYCLOPS,
    M_LIFE,
    M_RSURV,
    M_STREAM,
    N_SWEEP_P,
    REPO_ROOT,
    RESULTS_DIR,
    one_host,
)

OUT_DIR = REPO_ROOT / "papers" / "arxiv" / "figures"
P = N_SWEEP_P  # the n-sweep's model width

# (method key, legend label, colour, line style), in the paper's method order
# (Python before R). The line figures' legends follow the curves' vertical
# order instead (scaling_figure); the scatter's follows this list.
SERIES = [
    (M_STREAM, "CoxStream", NAV, "o-"),
    (M_LIFE, "lifelines", DGR, "s--"),
    # Purple, not a second blue: Cyclops and R overlap at 4-8M in the RAM
    # figure and must stay apart in colour and in greyscale.
    (M_RSURV, "R survival::coxph", "#6a3d9a", "v--"),
    (M_CYCLOPS, "Cyclops", BLU, "^-."),
]


def _sweep(rows, method, key):
    """Ascending-n [(n_rows, median value)] of a method's p = 10 n-sweep."""
    pts = sorted(
        (r["n_rows"], r[f"median_{key}"]) for r in rows
        if r.get("method") == method and r.get("p") == P
        and r.get(f"median_{key}") is not None
    )
    if not pts:
        raise SystemExit(f"no p={P} rows for {method} in competing_coxph.json")
    return pts


def _save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.relative_to(REPO_ROOT)}", flush=True)


def scaling_figure(rows, key, ylabel, name, logy=False):
    """Line figure of the median `key` against n for every SERIES method.

    Each line runs to its own largest n: the in-memory baselines were run up
    to the benchmark's in-memory cap and CoxStream further, so lines are not
    clipped to a shared range.
    """
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    entries = []
    for method, label, color, style in SERIES:
        pts = _sweep(rows, method, key)
        line, = ax.plot([n / 1e6 for n, _ in pts], [v for _, v in pts],
                        style, color=color, label=label)
        entries.append((pts, line))
    ax.set_xlabel("$n$ (millions of rows)")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi * 4)  # headroom: the legend sits above the curves
    # Legend top to bottom by each curve's value at the largest n all curves
    # share, so the key matches the visual order where the curves overlap.
    shared_n = min(pts[-1][0] for pts, _ in entries)

    def at_shared(pts):
        below = [v for n, v in pts if n <= shared_n]
        return below[-1] if below else pts[0][1]

    entries.sort(key=lambda e: at_shared(e[0]), reverse=True)
    ax.legend(handles=[line for _, line in entries], loc="upper right")
    fig.tight_layout()
    _save(fig, name)


def efficiency_scatter(rows, name):
    """Median wall time against median RAM, one point per (method, n)."""
    series = []
    for method, label, color, style in SERIES:
        wall = dict(_sweep(rows, method, "elapsed_s"))
        ram = dict(_sweep(rows, method, "rss_delta_mb"))
        pts = [(wall[n], ram[n], n) for n in sorted(wall) if n in ram]
        series.append((pts, style[0], color, label))
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    for pts, marker, color, label in series:
        ax.scatter([t for t, _, _ in pts], [m for _, m, _ in pts],
                   marker=marker, color=color, s=50, label=label, zorder=3)
        for t, m, n in pts:
            # Centred above the point: a label to the right runs into the
            # next point where n doubles in a short time (CoxStream at 1-2M).
            ax.annotate(f"{n // 1_000_000}M", (t, m), xytext=(0, 5),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=10, color=color)
    ax.set_xscale("log")
    ax.set_yscale("log")
    # Explicit limits: tight_layout's pre-render log-tick pass needs positive
    # bounds (matplotlib >= 3.10 otherwise fails with "no positive values").
    ts = [t for pts, _, _, _ in series for t, _, _ in pts]
    ms = [m for pts, _, _, _ in series for _, m, _ in pts]
    ax.set_xlim(min(ts) * 0.7, max(ts) * 1.5)
    ax.set_ylim(min(ms) * 0.7, max(ms) * 1.5)
    # Label 1-3-10 steps, not only the decades: CoxStream's ~400 MB points
    # would otherwise sit below the first labelled tick.
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10, subs=(1.0, 3.0)))
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("Wall time (s)")
    ax.set_ylabel("RAM added (MB)")
    # No corner of this plot stays free of points once the labels are
    # legible, so the legend goes below the axes. It reads row by row in the
    # paper's method order (Python before R); matplotlib fills legend columns
    # first, so hand it the handles transposed.
    ncol = 2
    handles, labels = ax.get_legend_handles_labels()
    order = [i * ncol + j for j in range(ncol)
             for i in range(-(-len(handles) // ncol))
             if i * ncol + j < len(handles)]
    ax.legend([handles[k] for k in order], [labels[k] for k in order],
              loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=ncol,
              frameon=False)
    fig.tight_layout()
    _save(fig, name)


def main():
    mpl.rcParams["font.family"] = "sans-serif"
    # The figures are included at 0.65-0.7 of the text width (about 0.8 of
    # their natural size), so 11 pt here prints at about 9 pt.
    mpl.rcParams["font.size"] = 11
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False
    # Embed TrueType (Type 42), not Type 3: searchable text in the PDF and
    # clean glyphs at any zoom.
    mpl.rcParams["pdf.fonttype"] = 42
    src = RESULTS_DIR / "competing_coxph.json"
    with open(src) as fh:
        rows = json.load(fh)["results"]
    plotted = {m for m, *_ in SERIES}
    one_host([r for r in rows if r.get("p") == P and r["method"] in plotted],
             "figure rows")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"reading {src.relative_to(REPO_ROOT)} ({len(rows)} rows); "
          f"CoxStream series = {M_STREAM}", flush=True)
    scaling_figure(rows, "rss_delta_mb", "RAM added (MB)",
                   "coxph_ram_scaling.pdf")
    scaling_figure(rows, "elapsed_s", "Wall time (s)",
                   "coxph_time_scaling.pdf", logy=True)
    efficiency_scatter(rows, "coxph_efficiency_scatter.pdf")
    print("done: 3 figures", flush=True)


if __name__ == "__main__":
    main()
