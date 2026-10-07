"""Project palette loader.

Loads config/colors.json once and exposes its contents as module-level
constants so consumers can write `from colors import NAV, BLU, DGR`
instead of hardcoding hex strings or re-deriving palettes per file.

Three layers:
  BASE     -- the navy, blue and grey family every chart uses, as a dict.
              Shorthand constants NAV, BLU, DGR, LGR, NEAR_BLACK are also
              exposed at module level for the common cases.
  PAIRED_12 -- ColorBrewer Paired (12, light/dark pairs from matplotlib).
  SET1_9    -- ColorBrewer Set1 (9 qualitative).
  TABLEAU_10 -- matplotlib tab10 (scatter facets).

config/colors.json is the single source of truth; edit it (not this file)
to change the palette.
"""
from __future__ import annotations

import json

from coxbench.paths import REPO_ROOT

_CONFIG = REPO_ROOT / "config" / "colors.json"
with open(_CONFIG) as _fh:
    _DATA = json.load(_fh)

BASE = _DATA["base"]
PAIRED_12 = _DATA["paired_12"]
SET1_9 = _DATA["set1_9"]
TABLEAU_10 = _DATA["tableau_10"]

# Shorthand constants for the base colours the figure generators import.
NAV = BASE["navy"]
BLU = BASE["blue"]
DGR = BASE["dark_grey"]
LGR = BASE["light_grey"]
NEAR_BLACK = BASE["near_black"]

# Canonical library -> colour, used across EVERY benchmark chart so a library
# is the same colour everywhere. One blue/grey/teal family (no red, green, or
# orange); the four Python engines match the headline Results scatter (pandas
# navy, polars blue, duckdb grey, arrow light-blue). Bars/points are labelled,
# so near-shades still read fine.
LIBRARY_COLORS = {
    "pandas":         "#002555",   # navy (headline)
    "data.frame":     "#3a7ca5",   # steel blue (R eager baseline)
    "data.frame (R)": "#3a7ca5",
    "polars":         "#007dbb",   # blue (headline)
    "duckdb":         "#646c6f",   # grey (headline)
    "duckdb (R)":     "#3f474a",   # dark grey
    "arrow":          "#7ab8dd",   # light blue (headline)
    "arrow (R)":      "#2f7f8c",   # teal
    "dask":           "#9aa7b0",   # grey-blue
    "data.table":     "#5a6b7a",   # slate
    "fst":            "#a9c7dc",   # pale blue
    "qs2":            "#8098aa",   # muted blue
}


def lib_color(name, default="#aaaaaa"):
    """Canonical colour for a library label ('(R)' suffix, fallback)."""
    if name in LIBRARY_COLORS:
        return LIBRARY_COLORS[name]
    return LIBRARY_COLORS.get(name.split("(")[0].strip(), default)


__all__ = [
    "BASE",
    "BLU",
    "DGR",
    "LGR",
    "LIBRARY_COLORS",
    "NAV",
    "NEAR_BLACK",
    "PAIRED_12",
    "SET1_9",
    "TABLEAU_10",
    "lib_color",
]
