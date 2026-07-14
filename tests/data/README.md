## Test data

This folder contains data for the pytest CI.

The files in `single_factor` and `multi_factor` contain the outputs of DESeq2 (v1.34.0) on the synthetic data provided
in `datasets/synthetic/`, respectively using `~condition` and `~condition + group` as design. More precisely:

- `r_iterative_size_factors.csv` contains DESeq2's `estimateSizeFactorsIterate` output,
- `r_lfc_shrink.csv` contains DESeq2 results after running `lfcShrink`,
- `r_test_dispersions.csv` contains DESeq2 dispersions estimates (post-filtering and refitting),
- `r_test_res.csv` contains DESeq2's `results` output,
- `r_test_size_factors.csv` contains DESeq2's `estimateSizeFactors` output,
- `r_vst.csv` contains DESeq2's `varianceStabilizingTransformation` output with `blind=TRUE` and `fitType="parametric"`,
- `r_vst_with_design.csv` contains DESeq2's `varianceStabilizingTransformation` output with `blind=FALSE` and `fitType="parametric"`.

### Likelihood-ratio test fixtures

`generate_lrt_fixtures.R` generates the reference outputs for PyDESeq2's
classical likelihood-ratio test (LRT). From the repository root, run it in an
R environment containing DESeq2 1.34.0:

```console
Rscript tests/data/generate_lrt_fixtures.R
```

The fixtures in this repository were generated with the Bioconductor 3.14
Docker image; `lrt_session_info.txt` records the complete package environment.

The script deliberately requires **DESeq2 1.34.0**, the version used for the
other reference data in this directory. It reads the existing synthetic counts
and metadata from `datasets/synthetic/`, sets deterministic seeds, and writes
the following files under `tests/data/lrt/`:

- `r_lrt_single_factor.csv`: `~ condition` compared with `~ 1`;
- `r_lrt_multi_factor.csv`: `~ group + condition` compared with `~ group`;
- `r_lrt_multilevel.csv`: a three-level factor compared with an intercept-only
  model, exercising an omnibus test with two degrees of freedom;
- `r_lrt_outlier.csv`: `~ condition` compared with `~ 1` after setting
  `gene1`/`sample1` to 200, which reproducibly triggers Cook's replacement while
  retaining a parametric dispersion trend;
- `r_lrt_outlier_replace_counts.csv`: the count matrix after DESeq2's outlier
  replacement; and
- `r_lrt_manifest.csv` and `lrt_session_info.txt`: model definitions, seeds,
  outlier magnitude, and R package versions needed to audit or regenerate the
  fixtures.

The `log2FoldChange` and `lfcSE` columns in these files correspond to the
explicit contrast passed to `results()`. The `stat`, `pvalue`, and `padj`
columns are omnibus full-versus-reduced LRT results and do not change with that
contrast.
