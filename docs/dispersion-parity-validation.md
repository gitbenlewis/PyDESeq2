# Dispersion parity validation

Date: 2026-09-21. Branch: `fix/deseq2-dispersion-parity`.
Original validation base: `0e56de24ee53321d52f4e85be8f410d99f764ac3`.
The results below describe that original validation; see the rebase note at the end
for checks against the updated PR #451 baseline.

## Scope and implementation

The diagnosis was reproduced before editing production code. This change improves
parity but does **not** establish complete DESeq2 parity.

1. `pydeseq2/dds.py`: initialize the outer trend iteration at `(0.1, 1)`, exclude
   gene-wise estimates at or below `100 * min_disp`, select genes by their ratio to
   the current curve before fitting, and reconsider the complete eligible set
   each iteration. Bound the fit at 11 iterations, retaining the existing
   mean-trend fallback. This also applies to the parametric VST trend.
2. The dispersion-outlier formula is unchanged. It already matches R's comparison
   against twice the standard deviation of unadjusted log residuals. Correcting
   the trend corrects the four named outlier disagreements in the diagnosis.
3. `pydeseq2/ds.py`: apply `dds.min_mu` to count-scale fitted means before computing
   Wald covariance. This respects the configured floor and both normalization
   interfaces. It does not change fitted coefficients.
4. `pydeseq2/utils.py`: preserve prior variance and both regularization flags when
   dispersion optimization falls back to grid search. Previously MAP fallback
   silently optimized an objective without the dispersion prior.
5. `tests/test_dispersion_parity.py`: 13 focused cases cover trend re-entry for DEA
   and VST, a known parametric curve, iteration exhaustion, the outlier boundary,
   Wald covariance with two floors and both normalization layouts, and fallback
   regularization. The existing all-zero-after-replacement test retains its result
   assertions but no longer requires a trend-fit warning: the corrected fit
   converges on that example.

No public signatures, dependencies, parity thresholds, or R reference estimates
were added to production code. The original pytximport checkout and PR #451 were
not modified. Validation was completed locally before publication.

## Reproduction

The suite at `/projects/pydeseq2_vs_R_deseq2_parity_tests` was copied to isolated
working directories, including its input cache. Its current commit was
`3ceac3cd70c59839a1e0e98ea95682438f212ff7` (the diagnosis cites an older commit).
Only each copy's ignored `config/local_env.sh` was redirected to this worktree.
The original suite was not edited. All four datasets stayed enabled.

The original, baseline-copy, and final-copy YAML files have identical SHA-256:
`a459d496ba64e82c28381930db36ffa4542cb97f8f6bbb90f2908e9fb97080ca`.

Commands:

```bash
bash /tmp/dispersion-parity-baseline/scripts/000_run_parity.bash
bash /tmp/dispersion-parity-final/scripts/000_run_parity.bash
```

Python 3.11.15, R 4.5.3, DESeq2 1.50.2, NumPy 2.4.6, pandas 2.3.3, and AnnData
0.12.19 were used. The runner verifies the imported PyDESeq2 module is in the
requested worktree. Each results directory contains full gates, comparisons,
input hashes, and runtime/source provenance.

## Four-dataset outcome

| Dataset | Baseline | Updated |
| --- | --- | --- |
| GEUVADIS Salmon | 24/46 gates fail | 12/46 gates fail |
| SRP254919 tximport | 20/20 pass | 20/20 pass |
| Pasilla | 21/21 pass | 21/21 pass |
| Pickrell | 21/21 pass | 21/21 pass |

Both canonical commands exit nonzero because Salmon still fails. Each invocation
also passes the parity suite's 47 unit tests and the checkout's transcript-length
normalization tests (41 passed, 12 skipped).

## Salmon results

Both independent Python input interfaces agree. Import, rounded counts,
normalization factors, normalized counts, and base means pass before and after.
The artificial three-versus-three contrast is a software validation fixture,
not a biological comparison.

| Primary Python-versus-R metric | Baseline | Updated | Unchanged gate |
| --- | ---: | ---: | ---: |
| Maximum absolute LFC error | 0.0362193 | 0.000834781 | <= 0.001 |
| Maximum absolute lfcSE error | 2.10372 | 0.332443 | <= 0.02 |
| Maximum absolute Wald-statistic error | 2.36606 | 0.210832 | <= 0.05 |
| Maximum absolute p-value error | 0.210403 | 0.142311 | <= 0.005 |
| Maximum absolute adjusted-p-value error | 0.532471 | 0.0301229 | <= 0.01 |
| Adjusted-p-value NA-mask disagreements | 409 | 409 | 0 |
| LFC sign concordance | 0.999906 | 1 | 1 |
| Significant sets at alpha 0.05 and 0.1 | Only 0.05 exact | Both exact | Both exact |

Salmon failures decrease from 24/46 gates to 12/46. The remaining failures are
maximum lfcSE, statistic, and p-value errors, adjusted-p-value rank correlation,
maximum adjusted-p-value error, and adjusted-p-value missingness, each for both
the native Python import and R-imported shared-input comparisons.

## Controlled investigations and remaining differences

On the saved Python gene-wise estimates, reconsidering eligibility each iteration
produced coefficients `(0.07639088, 7.07826594)`, close to R's fitter applied to
those same estimates. Permanently removing genes produced
`(0.08020658, 5.62569311)` in the otherwise matched experiment. The original R
full-data trend is approximately `(0.07570902, 7.12299595)`; remaining gene-wise
estimate differences therefore still matter. Python retains its existing
L-BFGS-B Gamma optimizer; this change does not reproduce R's inner GLM optimizer.

The trend correction makes GSTT2, KAL1, HGF, and OR3A2 non-outliers, agreeing with
R. A fresh production-code stage rerun confirms CDCP1 is the sole remaining
dispersion-outlier flag disagreement. CDCP1 remains near the boundary and is still
shrunk in Python but retained as an outlier in R. Its remaining dispersion difference drives the largest updated
standard-error and p-value discrepancies.

Separately, on the saved shared-input fit with corrected trend, applying the
Wald floor reduced maximum standard-error error from approximately 0.703368 to
0.332434. The floor alone is not a complete parity fix.

A reachable MAP grid fallback for HSP90B1 gave dispersion 0.141149 without the
prior versus 0.107227 with it (R: 0.106835). Forced-failure tests independently
reproduce the missing regularization settings.

For BMS1P20, HSP90B1, RAB20, and GUCY1A3, R's saved gene-wise estimates equal
Python's initial method-of-moments estimates. R's source has a likelihood-gain
acceptance rule that can retain initialization. Matching counts and fitted means
were verified, and a separate scalar minimizer agreed with Python's gene-wise
optima for these examples. This supports investigating R's stopping/acceptance
rules next; it does not justify copying reference dispersions or changing
classification cutoffs. The acceptance rule was not ported in this change.

Independent filtering is unchanged. The original diagnosis's filtering-only
experiment is evidence that upstream p-values can explain mask differences,
not proof of filtering equivalence for every dataset.

## Checks and evidence

The full PyDESeq2 test suite passed: **134 passed, 12 skipped**. The initial
sandboxed run could not download two public example files; the complete rerun
with network access passed. Ruff checks, Ruff formatting, mypy, and
`git diff --check` passed. Nine targeted cases fail against an isolated copy of
the original commit and pass after the changes. All 13 new cases pass.

Commands used the comparison environment's Python with `OPENBLAS_NUM_THREADS=1`,
`OMP_NUM_THREADS=1`, and a writable temporary `MPLCONFIGDIR`:

```bash
python -m pytest -q
python -m pytest -q tests/test_dispersion_parity.py
ruff check pydeseq2 tests/test_dispersion_parity.py tests/test_edge_cases.py
ruff format --check pydeseq2 tests/test_dispersion_parity.py tests/test_edge_cases.py
mypy -p pydeseq2
git diff --check
```

Local evidence (temporary, not committed fixtures):

- Baseline: `/tmp/dispersion-parity-baseline/results/parity/`.
- Final: `/tmp/dispersion-parity-final/results/parity/`.
- Fresh stage rerun: `/tmp/dispersion-final-stages.py`,
  `/tmp/dispersion-final-stages.tsv`, and `/tmp/dispersion-final-stages.log`.
- Full tests: `/tmp/dispersion-pytest-final.log`.
- Regression failures on the original commit: `/tmp/dispersion-regression-before.log`.
- Controlled experiments: `/tmp/dispersion-experiment.py` and its log.

Python 3.12/3.13, the separate SciPy 1.13 compatibility environment, documentation
builds, and package builds were not run. This is focused statistical validation,
not a simulation of every CI matrix job.

## Rebase onto updated PR #451

On 2026-09-21, this change was adapted to PR #451 baseline
`1281923642080b9d44d6dedab8d75c34cd920c92`, which merges upstream main.
The trend and Wald changes now live in `src/pydeseq2/dds.py` and
`src/pydeseq2/ds.py`. The grid-fallback fix moved from the removed `utils.py`
to `src/pydeseq2/dispersions.py`, and its regression test imports that module.
The statistical changes and test assertions were preserved.

Validation used the existing `/tmp/pr451-venv` environment: Python 3.12.3,
NumPy 2.5.1, pandas 3.0.3, AnnData 0.13.2, and SciPy 1.18.0.
With `PYTHONPATH=src`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and
`MPLCONFIGDIR=/tmp/dispersion-rebase-mpl`:

```bash
/tmp/pr451-venv/bin/python -m pytest -q tests/test_dispersion_parity.py
/tmp/pr451-venv/bin/python -m pytest -q
/tmp/pr451-venv/bin/mypy src/pydeseq2 --ignore-missing-imports
```

All 13 dispersion regression cases passed. The full suite passed with
**134 passed, 12 skipped**. The initial sandboxed run failed only the two public
example-data downloads; the network-enabled rerun passed in 34.97 seconds.
Its log is `/tmp/dispersion-rebase-pytest-network.log`.
Ruff 0.15.16 lint and formatting checks passed for the three affected source
files and two affected test files; mypy passed for all 15 source files.
`git diff --check` passed.

The external four-dataset R parity suite, documentation/package builds, and
other CI environments were not rerun. Earlier parity results in this document
remain evidence for the original baseline, not fresh validation of this rebase.

### Follow-up baseline update

Later on 2026-09-21, PR #451 advanced to
`9b42a302fced879b9a721ca855abdd8d9ce91727`, adding normalization validation
tests and CI sparse-compatibility coverage collection. The dispersion commit
rebased cleanly; `git range-diff` confirmed its patch was unchanged before this
validation note was added. No production-code adaptation was needed.

The same Python 3.12 environment and full-suite command above passed with
**137 passed, 12 skipped** in 45.24 seconds, including all 13 dispersion cases.
The log is `/tmp/dispersion-rebase-9b42a30-pytest.log`.
`git diff --check` passed. The external R parity suite and full CI matrix were
not rerun.
