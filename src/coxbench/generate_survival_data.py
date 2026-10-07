"""
Generate large synthetic survival data for the CoxPH benchmarks.

Covariates mimic a clinical cohort study: age, sex, treatment arm, smoking
pack-years, BMI, systolic blood pressure, haemoglobin, creatinine, albumin,
and Charlson comorbidity index. Survival times follow a Weibull proportional-
hazards model with known true coefficients saved to a sidecar JSON, so a
fit's recovery error can be measured downstream.

Default output: data/survival_parts/ (1 file x 1M rows, ~230 MB Snappy).
Use --nparts / --nrows for larger n-sweep datasets.

Run from the project root (NOT from an IDE session -- disk/CPU intensive):
    python3 -m coxbench.generate_survival_data
    python3 -m coxbench.generate_survival_data --nrows 100_000_000 --nparts 100
    python3 -m coxbench.generate_survival_data --nrows 50_000_000 --nparts 50
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from coxbench.paths import data_path

WEIBULL_SHAPE = 1.5

# Baseline log-hazard (intercept in eta = intercept + X @ beta).
# Chosen so the average subject has ~30% event probability at 10 years.
_INTERCEPT = -4.3

# True PH coefficients -- exact values used to generate data.
# CoxPH does not estimate an intercept (absorbed into baseline hazard), so
# only these 10 beta values are recoverable by fitting the model.
TRUE_COEF = {
    "age_std": 0.50,
    "sex": -0.15,   # female vs male
    "treatment": -0.40,   # active vs control
    "smoking_pack_years": 0.010,  # per pack-year
    "bmi_std": 0.05,
    "systolic_bp_std": 0.10,
    "hemoglobin_std": -0.15,   # lower Hgb -> higher hazard
    "log_creatinine": 0.25,
    "albumin_std": -0.20,   # lower albumin -> higher hazard
    "charlson_std": 0.30,
}

AGE_MEAN, AGE_SD = 60.0, 15.0
SMOKING_PREV = 0.30
CHARLSON_MEAN = 2.0
CHARLSON_SD = np.sqrt(2.0)   # Poisson(2): SD = sqrt(mean)
MAX_FOLLOW_UP = 10.0           # administrative censoring (years)
CENS_RATE = 0.04           # independent exponential censoring rate


def make_chunk(
    n: int, rng: np.random.Generator, ncovariates: int = 10,
) -> pa.Table:
    age_raw = np.clip(rng.normal(AGE_MEAN, AGE_SD, n), 18.0, 100.0)
    age_std = (age_raw - AGE_MEAN) / AGE_SD

    sex = rng.integers(0, 2, n, dtype=np.int8)
    treatment = rng.integers(0, 2, n, dtype=np.int8)

    is_smoker = rng.random(n) < SMOKING_PREV
    smoking_pkyrs = np.where(
        is_smoker, np.clip(rng.gamma(5.0, 8.0, n), 0.0, 120.0), 0.0
    )

    bmi_std = rng.standard_normal(n)
    systolic_bp_std = rng.standard_normal(n)
    hemoglobin_std = rng.standard_normal(n)
    log_creatinine = rng.normal(0.0, 0.3, n)
    albumin_std = rng.standard_normal(n)
    charlson_raw = np.clip(rng.poisson(CHARLSON_MEAN, n), 0, 15)
    charlson_std = (charlson_raw - CHARLSON_MEAN) / CHARLSON_SD

    eta = (
        _INTERCEPT
        + TRUE_COEF["age_std"] * age_std
        + TRUE_COEF["sex"] * sex
        + TRUE_COEF["treatment"] * treatment
        + TRUE_COEF["smoking_pack_years"] * smoking_pkyrs
        + TRUE_COEF["bmi_std"] * bmi_std
        + TRUE_COEF["systolic_bp_std"] * systolic_bp_std
        + TRUE_COEF["hemoglobin_std"] * hemoglobin_std
        + TRUE_COEF["log_creatinine"] * log_creatinine
        + TRUE_COEF["albumin_std"] * albumin_std
        + TRUE_COEF["charlson_std"] * charlson_std
    )

    extra_cols: list[np.ndarray] = []
    for i in range(10, ncovariates):
        col = rng.standard_normal(n)
        extra_cols.append(col)
        eta = eta + 0.1 * col

    # Weibull PH inverse CDF: T = (-log(U) / exp(eta))^(1/k)
    t_event = (-np.log(rng.random(n)) / np.exp(eta)) ** (1.0 / WEIBULL_SHAPE)
    t_cens = np.clip(rng.exponential(1.0 / CENS_RATE, n), 0.0, MAX_FOLLOW_UP)
    duration = np.minimum(t_event, t_cens)
    event = (t_event <= t_cens).astype(np.int8)

    table: dict[str, pa.Array] = {
        "duration": pa.array(duration, type=pa.float64()),
        "event": pa.array(event, type=pa.int8()),
        "age_std": pa.array(age_std, type=pa.float64()),
        "sex": pa.array(sex, type=pa.int8()),
        "treatment": pa.array(treatment, type=pa.int8()),
        "smoking_pack_years": pa.array(smoking_pkyrs, type=pa.float64()),
        "bmi_std": pa.array(bmi_std, type=pa.float64()),
        "systolic_bp_std": pa.array(systolic_bp_std, type=pa.float64()),
        "hemoglobin_std": pa.array(hemoglobin_std, type=pa.float64()),
        "log_creatinine": pa.array(log_creatinine, type=pa.float64()),
        "albumin_std": pa.array(albumin_std, type=pa.float64()),
        "charlson_std": pa.array(charlson_std, type=pa.float64()),
    }
    for i, col in enumerate(extra_cols):
        table[f"x{i + 11}"] = pa.array(col, type=pa.float64())
    return pa.table(table)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--nrows", type=int, default=1_000_000,
                        help="total rows (default 1M)")
    parser.add_argument("--nparts", type=int, default=1,
                        help="number of parquet files (default 1)")
    parser.add_argument("--outdir", type=str,
                        default=str(data_path("survival_parts")),
                        help="output directory (default data/survival_parts)")
    parser.add_argument("--ncovariates", type=int, default=10, metavar="P_MAX",
                        help="total covariates to generate (default 10); "
                             "values >10 add extra synthetic x11, x12, ... "
                             "columns")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.ncovariates < 10:
        parser.error("--ncovariates must be >= 10")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows_per_part = args.nrows // args.nparts
    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()

    n_events = 0
    n_total = 0
    for part in range(args.nparts):
        chunk = make_chunk(rows_per_part, rng, ncovariates=args.ncovariates)
        pq.write_table(
            chunk, outdir / f"part_{part:03d}.parquet", compression="snappy"
        )
        n_events += int(chunk["event"].to_numpy().sum())
        n_total += rows_per_part
        if (part + 1) % 10 == 0 or part == args.nparts - 1:
            print(
                f"  {part + 1:4d}/{args.nparts}  rows={n_total:,}  "
                f"elapsed={time.perf_counter() - t0:.0f}s",
                flush=True,
            )

    all_coef = dict(TRUE_COEF)
    for i in range(10, args.ncovariates):
        all_coef[f"x{i + 1}"] = 0.1

    coef_path = outdir / "true_coef.json"
    with open(coef_path, "w") as fh:
        json.dump(
            {
                "model": "weibull_ph",
                "weibull_shape": WEIBULL_SHAPE,
                "intercept": _INTERCEPT,
                "coef": all_coef,
                "population": {
                    "age_mean": AGE_MEAN,
                    "age_sd": AGE_SD,
                    "smoking_prev": SMOKING_PREV,
                    "charlson_mean": CHARLSON_MEAN,
                    "charlson_sd": CHARLSON_SD,
                    "max_follow_up": MAX_FOLLOW_UP,
                    "censoring_rate": CENS_RATE,
                },
            },
            fh,
            indent=2,
        )

    disk_gb = sum(
        os.path.getsize(p) for p in outdir.glob("part_*.parquet")
    ) / 1024 ** 3
    print(f"\nDone in {time.perf_counter() - t0:.0f}s")
    print(f"  Rows:       {n_total:,}")
    print(f"  Event rate: {n_events / n_total:.1%}")
    print(f"  Disk:       {disk_gb:.1f} GB  ({outdir})")
    print(f"  True coef:  {coef_path}")


if __name__ == "__main__":
    main()
