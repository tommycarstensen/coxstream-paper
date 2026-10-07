# CoxStream paper: reproducibility archive

Everything needed to check and rerun the numbers in "CoxStream: Out-of-Core Cox Proportional Hazards Regression by Streaming Newton-Raphson" (Tommy Carstensen and Clara S. Grønkjær): the benchmark harness, the seeded data generator, the result files, the paper source, and the software exactly as it was measured. It is archived on Zenodo: [doi:10.5281/zenodo.23210002](https://doi.org/10.5281/zenodo.23210002).

The software itself is published separately: the Python package `coxstream` ([PyPI](https://pypi.org/project/coxstream/), [GitHub](https://github.com/tommycarstensen/coxstream), [Zenodo](https://doi.org/10.5281/zenodo.20693783)) and the R package `coxstream` ([CRAN](https://cran.r-project.org/package=coxstream), [GitHub](https://github.com/tommycarstensen/coxstream-r), [Zenodo](https://doi.org/10.5281/zenodo.20694001)). Use those to fit models; use this archive to reproduce the paper.

## What is here

| Path | What |
| --- | --- |
| `results/competing_coxph.json` | Speed, memory and accuracy of every method at every cohort size. The paper uses the rows measured on one Linux VM; the file also holds rows it does not use (laptop runs, scikit-survival, research kernels), each stamped with its host. |
| `results/vm_environment.txt` | That VM: machine type, CPU, RAM, disk, operating system, and the version of Python, R and each package measured. |
| `results/coxph_ties.json` | The tie experiment: streamed fits against in-memory Efron and Breslow references. Its `meta` block records the laptop it ran on, the versions, the seed and the generator's parameters. |
| `papers/arxiv/` | The paper source. `paper_macros.tex` holds every number in the text and is generated, never typed. |
| `src/coxbench/gen_arxiv_macros.py`, `gen_arxiv_figures.py` | Turn the result files into the paper's numbers and figures, and check each data-dependent sentence against the data. |
| `src/coxbench/generate_survival_data.py` | The seeded generator of the synthetic cohorts. No data are downloaded. |
| `src/coxbench/bench_competing_coxph.py`, `r_competing_coxph.R`, `cloud/` | The speed and memory benchmark and the scripts that ran it on the VM. |
| `src/coxbench/bench_coxph_ties.py`, `r_ties_reference.R` | The tie experiment. |
| `src/coxbench/` (the rest), `config/colors.json`, `pyproject.toml`, `tests/` | What those scripts import: paths, provenance stamps and plotting colours; and methods the benchmark can also time but the paper does not report: research kernels (`coxph_stream.py`, `coxph_cython*.pyx`, `coxph_jax*.py`) and a reimplementation of BigSurvSGD (`sgd_coxph.py`, with its test). |
| `packages/coxstream/`, `packages/r/coxstream/` | The Python and R packages as measured (Python 0.3.0, R 0.1.0): source, tests and what is needed to install them. The image their READMEs show is not copied here. |
| `SOURCE_COMMIT` | The commit of the authors' working repository this archive was exported from. |
| `CHANGELOG.md` | What each release of this archive holds, and which one matches which version of the paper. |
| `LICENSE`, `CITATION.cff`, `.zenodo.json` | The licence, and how to cite the archive (the last is the record Zenodo shows). |

## Three levels of reproduction

Commands are run from the root of this archive. Python 3.13 or newer. The first level works from a downloaded copy; the other two need a git clone (`git clone https://github.com/tommycarstensen/coxstream-paper.git`), because the scripts record the commit they ran at.

### 1. Rebuild the paper from the committed results (minutes, any machine)

This regenerates every number and figure from `results/` and compiles the PDF. It needs `matplotlib` and a LaTeX installation with `latexmk` and BibTeX (TeX Live's standard packages are enough).

    python3 -m pip install -e . matplotlib && python3 -m coxbench.gen_arxiv_macros && python3 -m coxbench.gen_arxiv_figures && cd papers/arxiv && latexmk -pdf paper.tex

`gen_arxiv_macros` stops and names the sentence if the data do not support a claim in the text (for example "fastest at every n"). `python3 -m coxbench.gen_arxiv_macros --check` prints every macro next to its current value without writing anything.

### 2. Rerun the tie experiment (about 3 minutes on a laptop, about 4 GB of RAM)

It generates a 3,000,000-row cohort from a seed, rounds its times to months and to years, and compares the streamed fit at several chunk sizes with R `survival::coxph` (Efron and Breslow), `lifelines` and a plain-numpy Efron. It needs a C compiler (to build the package's kernel), R with `survival`, `arrow` and `jsonlite`, and the Python packages below.

    python3 -m pip install -e . -e "packages/coxstream[parquet]" pandas lifelines && python3 -m coxbench.bench_coxph_ties --out tmp/coxph_ties_rerun.json

It writes its results to the file named by `--out` and a log under `tmp/`; without `--out` it overwrites `results/coxph_ties.json`, which is the file the paper is built from. The script refuses to run when the files that produce the results have uncommitted changes (`--allow-dirty` overrides that, and the output records it). The versions the published run used are in the `meta` block of `results/coxph_ties.json`.

### 3. Rerun the speed and memory benchmark (hours, one Linux VM)

Memory was measured on Linux because macOS memory compression under-reports the resident set. The header of `cloud/run_coxph_nsweep.sh` gives every step: creating a Google Cloud `n2-standard-8` VM, setting it up with `cloud/setup_coxph_sweep.sh`, running the sweep, and fetching the results. Its commands assume the clone is at `~/coxstream` on the laptop (clone it under that name, or adjust the `cd`), and take your cloud project, zone and VM name from three environment variables, set once. The setup needs `sudo` on the VM: it installs system packages and one sudoers rule, so that the benchmark can drop the page cache between runs without a password prompt. The data are generated on the VM from the seed. The benchmark can also time two research backends in C++ and Rust that the paper does not report; they are not in this archive, and the setup script skips them.

This level reruns the protocol; it does not rebuild the identical software stack. The setup installs the current versions of Python and R packages, which will drift from the ones measured. Those are listed in `results/vm_environment.txt`; install them explicitly to come closer to the original.

## What to expect from a rerun

- **Numbers in the paper from the committed results (level 1):** identical, on any machine.
- **Accuracy (coefficient gaps, the tie experiment):** the cohorts are drawn with numpy's `default_rng(seed)`. With the numpy version recorded with each result file (2.5.3 for both), the cohorts should be the same, and the gaps should reproduce to about the digits the paper prints; the last digits can differ between platforms, BLAS builds and library versions. numpy does not promise the same random stream across its versions, so with another version expect a statistically equivalent cohort and gaps of the same order, not the same digits.
- **Speed and memory (level 3):** specific to one `n2-standard-8` VM on one date. The ratios between methods are expected, not guaranteed, to hold on comparable hardware. The memory profile (flat for the streamed fit, growing with the cohort for the in-memory solvers) follows from the algorithms and should hold anywhere.

## Data

All data are synthetic. Event times follow a Weibull proportional-hazards model with known coefficients, with exponential censoring capped at an administrative follow-up; the parameters are constants at the top of `src/coxbench/generate_survival_data.py`, and the default seed is 42. The generated files are large (gigabytes at 32 million rows) and are not part of this archive.

## Licence and citation

- The code and the result files (everything outside `papers/`) are under the MIT licence: see `LICENSE`.
- `packages/r/coxstream/src/arrow_c_abi.h` is vendored from the Apache Arrow project under the Apache License 2.0, as `packages/r/coxstream/inst/COPYRIGHTS` states.
- The manuscript source in `papers/` is included so that the paper can be checked and rebuilt. It is copyright of the two authors and is not under the MIT licence; the licence of the paper is the one stated on its preprint.

To cite the archive, use its DOI, [10.5281/zenodo.23210002](https://doi.org/10.5281/zenodo.23210002), which covers every version and resolves to the latest (each version also has a DOI of its own, shown on its Zenodo page; `CITATION.cff` has the rest); to cite the method, cite the paper.
