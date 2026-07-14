"""Tests for classical negative-binomial likelihood-ratio tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from formulaic import model_matrix
from scipy.stats import chi2

from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
from pydeseq2.utils import dispersion_trend
from pydeseq2.utils import load_example_data
from pydeseq2.utils import nb_nll


class CountingInference(DefaultInference):
    """Default inference implementation recording each IRLS invocation."""

    def __init__(self) -> None:
        super().__init__(n_cpus=1)
        self.irls_inputs: list[tuple[np.ndarray, np.ndarray]] = []

    def irls(self, *args: Any, **kwargs: Any):
        """Record counts and design matrix, then run the default IRLS fit."""
        counts = kwargs.get("counts", args[0] if args else None)
        design_matrix = kwargs.get("design_matrix", args[2] if len(args) > 2 else None)
        assert counts is not None
        assert design_matrix is not None
        self.irls_inputs.append(
            (np.asarray(counts).copy(), np.asarray(design_matrix).copy())
        )
        return super().irls(*args, **kwargs)


class NonconvergedFullInference(DefaultInference):
    """Inference backend marking one full-model IRLS result nonconverged."""

    def __init__(self) -> None:
        super().__init__(n_cpus=1)

    def irls(self, *args: Any, **kwargs: Any):
        """Return normal estimates while overriding one full-fit status."""
        result = super().irls(*args, **kwargs)
        design_matrix = kwargs.get("design_matrix", args[2] if len(args) > 2 else None)
        assert design_matrix is not None
        if np.asarray(design_matrix).shape[1] > 1:
            converged = np.asarray(result[3], dtype=bool).copy()
            converged[0] = False
            return result[0], result[1], result[2], converged
        return result


@pytest.fixture(scope="module")
def counts_df() -> pd.DataFrame:
    """Return the bundled synthetic count matrix."""
    return load_example_data(modality="raw_counts", dataset="synthetic", debug=False)


@pytest.fixture(scope="module")
def metadata() -> pd.DataFrame:
    """Return the bundled synthetic sample metadata."""
    return load_example_data(modality="metadata", dataset="synthetic", debug=False)


def _fit_dds(
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
    design: str | pd.DataFrame,
    *,
    inference: DefaultInference | None = None,
    low_memory: bool = False,
    refit_cooks: bool = False,
    test: str = "Wald",
    reduced: str | pd.DataFrame | None = None,
) -> DeseqDataSet:
    """Construct and fit a small, quiet dataset for a requested test."""
    dds = DeseqDataSet(
        counts=counts.copy(),
        metadata=metadata.copy(),
        design=design,
        inference=inference or DefaultInference(n_cpus=1),
        low_memory=low_memory,
        refit_cooks=refit_cooks,
        quiet=True,
    )
    dds.deseq2(test=test, reduced=reduced)
    return dds


def _condition_stats(
    dds: DeseqDataSet,
    *,
    contrast: list[str] | np.ndarray | None = None,
    test: str | None = "LRT",
    reduced: str | pd.DataFrame | None = "~1",
    inference: DefaultInference | None = None,
    cooks_filter: bool = False,
    independent_filter: bool = False,
) -> DeseqStats:
    """Build and summarize condition statistics with deterministic filtering."""
    kwargs: dict[str, Any] = {}
    if inference is not None:
        kwargs["inference"] = inference
    stats = DeseqStats(
        dds,
        contrast=(contrast if contrast is not None else ["condition", "B", "A"]),
        cooks_filter=cooks_filter,
        independent_filter=independent_filter,
        quiet=True,
        test=test,
        reduced=reduced,
        **kwargs,
    )
    stats.summary()
    return stats


def _assert_same_test_results(left: DeseqStats, right: DeseqStats) -> None:
    """Assert equality of the test-specific parts of two results tables."""
    for column in ["stat", "pvalue", "padj"]:
        np.testing.assert_allclose(
            left.results_df[column],
            right.results_df[column],
            rtol=1e-10,
            atol=1e-12,
            equal_nan=True,
        )


def test_dds_lrt_cache_is_reused_by_stats(counts_df, metadata):
    """The dataset API should fit once and the stats API should reuse its cache."""
    inference = CountingInference()
    dds = _fit_dds(
        counts_df,
        metadata,
        "~condition",
        inference=inference,
        test="LRT",
        reduced="~1",
    )

    # One full-model fit and one reduced-model fit are sufficient.
    assert len(inference.irls_inputs) == 2
    calls_after_dds_fit = len(inference.irls_inputs)

    stats = DeseqStats(
        dds,
        contrast=["condition", "B", "A"],
        test="LRT",
        reduced="~1",
        inference=inference,
        cooks_filter=False,
        independent_filter=False,
        quiet=True,
    )
    stats.summary()
    stats.summary()

    assert len(inference.irls_inputs) == calls_after_dds_fit
    np.testing.assert_allclose(stats.statistics, dds.var["_lrt_statistic"])
    np.testing.assert_allclose(stats.p_values, dds.var["_lrt_pvalue"])


def test_stats_lrt_on_demand_is_cached(counts_df, metadata):
    """The stats API should add one reduced fit and cache repeated calls."""
    inference = CountingInference()
    dds = _fit_dds(counts_df, metadata, "~condition", inference=inference)
    assert len(inference.irls_inputs) == 1

    stats = DeseqStats(
        dds,
        contrast=["condition", "B", "A"],
        test="LRT",
        reduced="~1",
        inference=inference,
        cooks_filter=False,
        independent_filter=False,
        quiet=True,
    )
    stats.summary()
    assert len(inference.irls_inputs) == 2

    stats.summary()
    stats.run_lrt_test()
    assert len(inference.irls_inputs) == 2


def test_formula_and_matrix_reduced_models_match(counts_df, metadata):
    """Formula and explicit-matrix reduced designs should give identical LRTs."""
    formula_dds = _fit_dds(counts_df, metadata, "~group + condition")
    formula_stats = DeseqStats(
        formula_dds,
        contrast=["condition", "B", "A"],
        test="LRT",
        reduced="~group",
        cooks_filter=False,
        independent_filter=False,
        quiet=True,
    )
    formula_stats.summary()

    full_matrix = pd.DataFrame(model_matrix("~group + condition", metadata.copy()))
    reduced_matrix = pd.DataFrame(model_matrix("~group", metadata.copy()))
    matrix_dds = _fit_dds(counts_df, metadata, full_matrix)
    contrast = np.zeros(full_matrix.shape[1])
    contrast[full_matrix.columns.get_loc("condition[T.B]")] = 1
    matrix_stats = DeseqStats(
        matrix_dds,
        contrast=contrast,
        test="LRT",
        reduced=reduced_matrix,
        cooks_filter=False,
        independent_filter=False,
        quiet=True,
    )
    matrix_stats.summary()

    _assert_same_test_results(formula_stats, matrix_stats)


def test_explicit_full_and_reduced_matrices(counts_df, metadata):
    """The LRT should support the existing explicit-design-matrix DDS API."""
    full_matrix = pd.DataFrame(model_matrix("~group + condition", metadata.copy()))
    reduced_matrix = pd.DataFrame(model_matrix("~group", metadata.copy()))
    dds = _fit_dds(counts_df, metadata, full_matrix)

    contrast = np.zeros(full_matrix.shape[1])
    contrast[full_matrix.columns.get_loc("condition[T.B]")] = 1
    stats = DeseqStats(
        dds,
        contrast=contrast,
        test="LRT",
        reduced=reduced_matrix,
        cooks_filter=False,
        independent_filter=False,
        quiet=True,
    )
    stats.summary()

    assert np.isfinite(stats.results_df.loc[dds.var["non_zero"], "stat"]).all()
    assert stats.lrt_df == 1


def test_multidf_lrt_and_chi_square_math(counts_df, metadata):
    """An omnibus multi-level factor should use the model-rank difference as df."""
    three_level_metadata = metadata.copy()
    three_level_metadata["time"] = [
        f"T{sample_idx % 3}" for sample_idx in range(len(three_level_metadata))
    ]
    dds = _fit_dds(
        counts_df,
        three_level_metadata,
        "~group + time",
        test="LRT",
        reduced="~group",
    )
    contrast = np.asarray(
        dds.contrast(column="time", baseline="T0", group_to_compare="T2")
    )
    stats = DeseqStats(
        dds,
        contrast=contrast,
        test=None,
        cooks_filter=False,
        independent_filter=False,
        quiet=True,
    )
    stats.summary()

    assert stats.lrt_df == 2
    non_zero = dds.var["non_zero"]
    np.testing.assert_allclose(
        stats.p_values[non_zero],
        chi2.sf(stats.statistics[non_zero], df=2),
        rtol=1e-13,
        atol=0,
    )


def test_lrt_statistic_matches_manual_nb_likelihood(counts_df, metadata):
    """Check the LRT sign, deviance, clipping and chi-square calculation."""
    dds = _fit_dds(
        counts_df,
        metadata,
        "~condition",
        test="LRT",
        reduced="~1",
    )
    non_zero = dds.var["non_zero"].to_numpy()
    counts = np.asarray(dds.X[:, non_zero])
    size_factors = dds.obs["size_factors"].to_numpy()
    dispersions = dds.var.loc[non_zero, "dispersions"].to_numpy()

    full_design = dds.obsm["design_matrix"].to_numpy()
    full_lfc = dds.varm["LFC"].loc[non_zero].to_numpy()
    full_mu = size_factors[:, None] * np.exp(full_design @ full_lfc.T)

    reduced_design = dds.obsm["_lrt_reduced_design_matrix"].to_numpy()
    _, reduced_mu, _, _ = DefaultInference(n_cpus=1).irls(
        counts=counts,
        size_factors=size_factors,
        design_matrix=reduced_design,
        disp=dispersions,
        min_mu=dds.min_mu,
        beta_tol=dds.beta_tol,
    )
    full_nll = np.asarray(nb_nll(counts, full_mu, dispersions))
    reduced_nll = np.asarray(nb_nll(counts, reduced_mu, dispersions))
    expected_statistic = np.maximum(2 * (reduced_nll - full_nll), 0)

    np.testing.assert_allclose(
        dds.var.loc[non_zero, "_lrt_statistic"],
        expected_statistic,
        rtol=1e-8,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        dds.var.loc[non_zero, "_lrt_full_deviance"],
        2 * full_nll,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        dds.var.loc[non_zero, "_lrt_pvalue"],
        chi2.sf(expected_statistic, df=1),
        rtol=1e-13,
        atol=0,
    )
    assert (dds.var.loc[non_zero, "_lrt_statistic"] >= 0).all()


def test_display_contrast_does_not_change_lrt(counts_df, metadata):
    """Changing display contrast must not change an omnibus LRT."""
    dds = _fit_dds(
        counts_df,
        metadata,
        "~condition",
        test="LRT",
        reduced="~1",
    )
    b_vs_a = _condition_stats(dds, contrast=["condition", "B", "A"])
    a_vs_b = _condition_stats(dds, contrast=["condition", "A", "B"])

    _assert_same_test_results(b_vs_a, a_vs_b)
    np.testing.assert_allclose(
        b_vs_a.results_df["log2FoldChange"],
        -a_vs_b.results_df["log2FoldChange"],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        b_vs_a.results_df["lfcSE"],
        a_vs_b.results_df["lfcSE"],
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("low_memory", [False, True])
def test_lrt_with_low_memory(counts_df, metadata, low_memory):
    """LRT summaries should not rely on discarded full-model intermediates."""
    dds = _fit_dds(
        counts_df,
        metadata,
        "~condition",
        low_memory=low_memory,
        test="LRT",
        reduced="~1",
    )
    stats = _condition_stats(dds)

    assert np.isfinite(stats.results_df.loc[dds.var["non_zero"], "pvalue"]).all()
    if low_memory:
        assert "_mu_LFC" not in dds.obsm


def test_all_zero_gene_has_missing_lrt_results(counts_df, metadata):
    """A gene that starts all-zero should have the Wald-compatible NA semantics."""
    counts = counts_df.copy()
    counts["all_zero"] = 0
    dds = _fit_dds(
        counts,
        metadata,
        "~condition",
        test="LRT",
        reduced="~1",
    )
    stats = _condition_stats(dds)

    assert stats.results_df.loc["all_zero", "baseMean"] == 0
    assert (
        stats.results_df.loc[
            "all_zero", ["log2FoldChange", "lfcSE", "stat", "pvalue", "padj"]
        ]
        .isna()
        .all()
    )
    assert pd.isna(dds.var.loc["all_zero", "_lrt_statistic"])
    assert pd.isna(dds.var.loc["all_zero", "_lrt_pvalue"])


def _assert_matches_deseq2(
    stats: DeseqStats,
    filename: str,
    *,
    tolerance: float = 0.02,
    lfc_se_tolerance: float | None = None,
) -> None:
    """Compare a result table with a generated DESeq2 1.34.0 fixture."""
    reference = pd.read_csv(
        Path(__file__).parent / "data" / "lrt" / filename,
        index_col=0,
    )
    assert stats.results_df.index.equals(reference.index)
    for column in ["baseMean", "log2FoldChange", "lfcSE"]:
        assert stats.results_df[column].isna().equals(reference[column].isna())
        absolute_tolerance = 1e-4 if column == "log2FoldChange" else 1e-8
        relative_tolerance = (
            lfc_se_tolerance
            if column == "lfcSE" and lfc_se_tolerance is not None
            else tolerance
        )
        np.testing.assert_allclose(
            stats.results_df[column],
            reference[column],
            rtol=relative_tolerance,
            atol=absolute_tolerance,
            equal_nan=True,
        )

    # At a practically null optimum, tiny likelihood differences are sensitive to
    # optimizer stopping tolerances. Keep the statistic comparison absolute there,
    # while retaining strict relative parity for meaningful statistics and p-values.
    near_null = reference["stat"] < 0.05
    stable = ~near_null
    np.testing.assert_allclose(
        stats.results_df["stat"],
        reference["stat"],
        rtol=tolerance,
        atol=0.01,
        equal_nan=True,
    )
    for column in ["pvalue", "padj"]:
        assert stats.results_df[column].isna().equals(reference[column].isna())
        np.testing.assert_allclose(
            stats.results_df.loc[stable, column],
            reference.loc[stable, column],
            rtol=tolerance,
            atol=1e-8,
            equal_nan=True,
        )
        assert ((stats.results_df[column] < 0.05) == (reference[column] < 0.05)).all()
        assert (stats.results_df.loc[near_null, column].dropna() > 0.8).all()


@pytest.mark.parametrize(
    ("case", "full", "reduced", "contrast", "filename", "expected_df"),
    [
        (
            "single_factor",
            "~condition",
            "~1",
            ["condition", "B", "A"],
            "r_lrt_single_factor.csv",
            1,
        ),
        (
            "multi_factor",
            "~group + condition",
            "~group",
            ["condition", "B", "A"],
            "r_lrt_multi_factor.csv",
            1,
        ),
        (
            "multilevel",
            "~condition3",
            "~1",
            ["condition3", "B", "A"],
            "r_lrt_multilevel.csv",
            2,
        ),
    ],
)
def test_lrt_matches_deseq2_reference(
    counts_df,
    metadata,
    case,
    full,
    reduced,
    contrast,
    filename,
    expected_df,
):
    """Match R DESeq2 for single-factor, adjusted, and multi-df LRTs."""
    case_metadata = metadata.copy()
    if case == "multilevel":
        case_metadata["condition3"] = np.resize(
            np.array(["A", "B", "C"]),
            len(case_metadata),
        )

    dds = _fit_dds(
        counts_df,
        case_metadata,
        full,
        test="LRT",
        reduced=reduced,
    )
    stats = DeseqStats(
        dds,
        contrast=contrast,
        test=None,
        reduced=None,
        cooks_filter=True,
        independent_filter=True,
        quiet=True,
    )
    stats.summary()

    assert stats.lrt_df == expected_df
    _assert_matches_deseq2(stats, filename)


def test_cooks_replacement_is_used_by_both_lrt_models(counts_df, metadata):
    """The reduced LRT fit and R parity must use Cook-replaced effective counts."""
    counts = counts_df.copy()
    counts.loc["sample1", "gene1"] = 200
    inference = CountingInference()
    dds = _fit_dds(
        counts,
        metadata,
        "~condition",
        inference=inference,
        refit_cooks=True,
        test="LRT",
        reduced="~1",
    )
    calls_after_dds_fit = len(inference.irls_inputs)
    stats = _condition_stats(
        dds,
        test=None,
        reduced=None,
        cooks_filter=True,
        independent_filter=True,
    )

    assert len(inference.irls_inputs) == calls_after_dds_fit
    assert bool(dds.var.loc["gene1", "replaced"])
    assert dds.uns["disp_function_type"] == "parametric"
    assert "replace_counts" in dds.layers
    expected_counts = pd.read_csv(
        Path(__file__).parent / "data" / "lrt" / "r_lrt_outlier_replace_counts.csv",
        index_col=0,
    ).T
    expected_counts = expected_counts.loc[dds.obs_names, dds.var_names]
    np.testing.assert_array_equal(dds.layers["replace_counts"], expected_counts)

    effective_counts = np.asarray(dds.layers["replace_counts"])
    fit_mask = (
        (effective_counts != 0).any(axis=0)
        & np.isfinite(dds.var["dispersions"])
        & np.isfinite(dds.varm["LFC"]).all(axis=1)
    )
    reduced_calls = [
        inputs for inputs in inference.irls_inputs if inputs[1].shape[1] == 1
    ]
    assert len(reduced_calls) == 1
    np.testing.assert_array_equal(
        reduced_calls[0][0],
        effective_counts[:, np.asarray(fit_mask)],
    )
    _assert_matches_deseq2(stats, "r_lrt_outlier.csv")


def test_cooks_new_all_zero_gene_has_lrt_zero_semantics(
    counts_df, metadata, monkeypatch
):
    """Cook replacement may create a new all-zero gene, which remains reportable."""
    samples = [f"sample{i}" for i in [*range(1, 11), *range(91, 101)]]
    subset_metadata = metadata.loc[samples].copy()
    counts = counts_df.loc[samples].copy()
    counts["single_outlier"] = 0
    counts.loc["sample100", "single_outlier"] = 100
    dds = DeseqDataSet(
        counts=counts,
        metadata=subset_metadata,
        design="~condition",
        inference=DefaultInference(n_cpus=1),
        refit_cooks=True,
        quiet=True,
    )

    with pytest.warns(UserWarning):
        dds.deseq2(test="LRT", reduced="~1")
    stats = _condition_stats(
        dds,
        test=None,
        reduced=None,
        cooks_filter=True,
        independent_filter=True,
    )

    assert dds.new_all_zeroes_genes.equals(pd.Index(["single_outlier"]))
    assert "replace_counts" in dds.layers
    assert not np.asarray(
        dds.layers["replace_counts"][:, dds.var_names == "single_outlier"]
    ).any()
    assert stats.results_df.loc["single_outlier", "baseMean"] == 0
    assert (
        stats.results_df.loc["single_outlier", ["log2FoldChange", "lfcSE", "stat"]]
        .eq(0)
        .all()
    )
    assert stats.results_df.loc["single_outlier", "pvalue"] == 1
    assert pd.isna(stats.results_df.loc["single_outlier", "padj"])
    fitted = dds.var["non_zero"] & ~dds.var.index.isin(dds.new_all_zeroes_genes)
    assert np.isfinite(stats.statistics.loc[fitted]).all()

    captured: dict[str, np.ndarray] = {}
    original_fit_prior = stats._fit_prior_var

    def record_prior_mask(*args: Any, **kwargs: Any) -> float:
        captured["gene_mask"] = np.asarray(kwargs["gene_mask"], dtype=bool).copy()
        return original_fit_prior(*args, **kwargs)

    monkeypatch.setattr(stats, "_fit_prior_var", record_prior_mask)
    stats.lfc_shrink(coeff="condition[T.B]")

    expected_mask = dds.var["non_zero"].to_numpy(dtype=bool) & (
        dds._effective_counts() != 0
    ).any(axis=0)
    np.testing.assert_array_equal(captured["gene_mask"], expected_mask)
    assert not captured["gene_mask"][dds.var_names.get_loc("single_outlier")]
    assert stats.LFC.loc["single_outlier", "condition[T.B]"] == 0


def test_custom_stats_inference_refits_locally_without_mutating_dds(counts_df, metadata):
    """A distinct Stats inference should fit both models without poisoning DDS cache."""
    dds_inference = CountingInference()
    dds = _fit_dds(
        counts_df,
        metadata,
        "~condition",
        inference=dds_inference,
    )
    dds_calls = len(dds_inference.irls_inputs)
    stats_inference = CountingInference()
    stats = _condition_stats(
        dds,
        inference=stats_inference,
        test="LRT",
        reduced="~1",
    )

    assert len(dds_inference.irls_inputs) == dds_calls
    assert [design.shape[1] for _, design in stats_inference.irls_inputs] == [2, 1]
    assert "_lrt" not in dds.uns
    assert "_lrt_reduced_design_matrix" not in dds.obsm
    assert np.isfinite(stats.results_df.loc[dds.var["non_zero"], "pvalue"]).all()


def test_lfc_shrink_preserves_lrt_test_results(counts_df, metadata):
    """Shrinkage changes the displayed effect and SE, but not omnibus LRT results."""
    dds = _fit_dds(
        counts_df,
        metadata,
        "~condition",
        test="LRT",
        reduced="~1",
    )
    stats = _condition_stats(dds, test=None, reduced=None)
    before = stats.results_df.copy()

    stats.lfc_shrink(coeff="condition[T.B]")

    assert stats.shrunk_LFCs
    assert not np.allclose(
        stats.results_df["log2FoldChange"],
        before["log2FoldChange"],
        equal_nan=True,
    )
    for column in ["stat", "pvalue", "padj"]:
        np.testing.assert_allclose(
            stats.results_df[column],
            before[column],
            rtol=0,
            atol=0,
            equal_nan=True,
        )


def test_wald_and_positional_apis_remain_backward_compatible(counts_df, metadata):
    """New keyword-only LRT arguments must leave existing positional APIs unchanged."""
    dds = _fit_dds(counts_df, metadata, "~condition")
    legacy = DeseqStats(
        dds,
        ["condition", "B", "A"],
        0.1,
        False,
        False,
        quiet=True,
    )
    explicit = DeseqStats(
        dds,
        ["condition", "B", "A"],
        0.1,
        False,
        False,
        quiet=True,
        test="Wald",
    )
    legacy.summary()
    explicit.summary()
    pd.testing.assert_frame_equal(legacy.results_df, explicit.results_df)

    mean_dds = DeseqDataSet(
        counts=counts_df.copy(),
        metadata=metadata.copy(),
        design="~condition",
        inference=DefaultInference(n_cpus=1),
        quiet=True,
    )
    mean_dds.deseq2("mean")
    assert mean_dds.fit_type == "mean"
    assert mean_dds.uns["_deseq2_test"] == "Wald"


def _summarize_lrt(dds: DeseqDataSet, **kwargs: Any) -> None:
    """Construct and summarize stats so validation may occur at either stage."""
    stats = DeseqStats(
        dds,
        contrast=["condition", "B", "A"],
        cooks_filter=False,
        independent_filter=False,
        quiet=True,
        **kwargs,
    )
    stats.summary()


def test_lrt_test_and_wald_option_validation(counts_df, metadata):
    """Reject missing/unknown tests and Wald-only null-hypothesis options."""
    dds = _fit_dds(counts_df, metadata, "~condition")

    with pytest.raises(ValueError, match="reduced"):
        _summarize_lrt(dds, test="LRT", reduced=None)
    with pytest.raises(ValueError, match="test"):
        _summarize_lrt(dds, test="score", reduced=None)
    with pytest.raises(ValueError, match="reduced"):
        _summarize_lrt(dds, test="Wald", reduced="~1")
    with pytest.raises(ValueError, match="lfc_null"):
        _summarize_lrt(dds, test="LRT", reduced="~1", lfc_null=1.0)
    with pytest.raises(ValueError, match="alt_hypothesis"):
        _summarize_lrt(
            dds,
            test="LRT",
            reduced="~1",
            alt_hypothesis="greater",
        )
    with pytest.raises(ValueError, match="prior_LFC_var"):
        _summarize_lrt(
            dds,
            test="LRT",
            reduced="~1",
            prior_LFC_var=np.ones(dds.obsm["design_matrix"].shape[1]),
        )

    for test, reduced, pattern in [
        ("LRT", None, "reduced"),
        ("score", None, "test"),
        ("Wald", "~1", "reduced"),
    ]:
        fresh = DeseqDataSet(
            counts=counts_df.copy(),
            metadata=metadata.copy(),
            design="~condition",
            inference=DefaultInference(n_cpus=1),
            quiet=True,
        )
        with pytest.raises(ValueError, match=pattern):
            fresh.deseq2(test=test, reduced=reduced)


def test_reduced_design_validation(counts_df, metadata):
    """Reduced models must match representation, rank, samples, and nesting."""
    formula_dds = _fit_dds(counts_df, metadata, "~group + condition")
    reduced_matrix = pd.DataFrame(model_matrix("~group", metadata.copy()))
    with pytest.raises(ValueError, match="formula"):
        _summarize_lrt(
            formula_dds,
            test="LRT",
            reduced=reduced_matrix,
        )

    full_matrix = pd.DataFrame(model_matrix("~group + condition", metadata.copy()))
    matrix_dds = _fit_dds(counts_df, metadata, full_matrix)
    contrast = np.zeros(full_matrix.shape[1])
    contrast[full_matrix.columns.get_loc("condition[T.B]")] = 1

    def summarize_matrix(reduced: str | pd.DataFrame) -> None:
        stats = DeseqStats(
            matrix_dds,
            contrast=contrast,
            test="LRT",
            reduced=reduced,
            cooks_filter=False,
            independent_filter=False,
            quiet=True,
        )
        stats.summary()

    with pytest.raises(ValueError, match="DataFrame"):
        summarize_matrix("~group")
    with pytest.raises(ValueError, match="more columns"):
        summarize_matrix(full_matrix)

    wrong_index = reduced_matrix.iloc[::-1]
    with pytest.raises(ValueError, match="sample index"):
        summarize_matrix(wrong_index)

    nonfinite = reduced_matrix.copy()
    nonfinite.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite numeric"):
        summarize_matrix(nonfinite)

    rank_deficient = pd.DataFrame(
        {
            "Intercept": np.ones(len(metadata)),
            "duplicate": np.ones(len(metadata)),
        },
        index=metadata.index,
    )
    with pytest.raises(ValueError, match="full column rank"):
        summarize_matrix(rank_deficient)

    nonnested = pd.DataFrame(
        {
            "Intercept": np.ones(len(metadata)),
            "ramp": np.arange(len(metadata)),
        },
        index=metadata.index,
    )
    with pytest.raises(ValueError, match="not nested"):
        summarize_matrix(nonnested)


def test_direct_lfc_refit_invalidates_cached_lrt(counts_df, metadata):
    """A model mutation must make an existing LRT cache impossible to reuse."""
    dds = _fit_dds(
        counts_df,
        metadata,
        "~condition",
        test="LRT",
        reduced="~1",
    )
    cached_var_keys = {key for key in dds.var.columns if str(key).startswith("_lrt_")}
    assert cached_var_keys
    assert "_lrt" in dds.uns
    assert "_lrt_reduced_design_matrix" in dds.obsm
    assert "_lrt_full_LFC" in dds.varm
    fit_generation = int(dds.uns.get("_fit_generation", 0))

    dds.fit_LFC()

    assert int(dds.uns["_fit_generation"]) == fit_generation + 1
    assert "_deseq2_test" not in dds.uns
    assert "_lrt" not in dds.uns
    assert "_lrt_reduced_design_matrix" not in dds.obsm
    assert "_lrt_full_LFC" not in dds.varm
    assert cached_var_keys.isdisjoint(dds.var.columns)


def test_lrt_state_survives_picklable_h5ad_roundtrip(
    counts_df, metadata, tmp_path: Path
):
    """Serialized LRT state should retain effective counts and model matrices."""
    counts = counts_df.copy()
    counts.loc["sample1", "gene1"] = 200
    dds = _fit_dds(
        counts,
        metadata,
        "~condition",
        refit_cooks=True,
        test="LRT",
        reduced="~1",
    )
    original_trend_coeffs = dds.uns["trend_coeffs"].copy()
    adata = dds.to_picklable_anndata()
    pd.testing.assert_series_equal(dds.uns["trend_coeffs"], original_trend_coeffs)
    assert isinstance(adata.uns["trend_coeffs"], np.ndarray)
    output_path = tmp_path / "lrt_state.h5ad"

    adata.write_h5ad(output_path)
    restored = ad.read_h5ad(output_path)

    trend_inputs = np.array([1.0, 10.0])
    assert isinstance(restored.uns["trend_coeffs"], np.ndarray)
    np.testing.assert_allclose(
        dispersion_trend(trend_inputs, restored.uns["trend_coeffs"]),
        dispersion_trend(trend_inputs, original_trend_coeffs),
        rtol=0,
        atol=0,
    )

    assert "replace_counts" in restored.layers
    np.testing.assert_array_equal(
        restored.layers["replace_counts"],
        dds.layers["replace_counts"],
    )
    assert "_lrt_reduced_design_matrix" in restored.obsm
    np.testing.assert_allclose(
        np.asarray(restored.obsm["_lrt_reduced_design_matrix"]),
        np.asarray(dds.obsm["_lrt_reduced_design_matrix"]),
        rtol=0,
        atol=0,
    )

    lrt_fields = [key for key in dds.var.columns if str(key).startswith("_lrt_")]
    assert lrt_fields
    pd.testing.assert_frame_equal(
        restored.var[lrt_fields],
        dds.var[lrt_fields],
        check_dtype=False,
    )
    assert int(restored.uns["_lrt"]["df"]) == int(dds.uns["_lrt"]["df"])
    assert restored.uns["_lrt"]["reduced_formula"] == "~1"
    assert list(restored.uns["_lrt"]["full_design_columns"]) == list(
        dds.uns["_lrt"]["full_design_columns"]
    )
    assert list(restored.uns["_lrt"]["reduced_design_columns"]) == list(
        dds.uns["_lrt"]["reduced_design_columns"]
    )
    assert bool(restored.uns["_pydeseq2_replace_counts_owned"])
    assert "_lrt_full_LFC" in restored.varm
    np.testing.assert_allclose(
        np.asarray(restored.varm["_lrt_full_LFC"]),
        np.asarray(dds.varm["_lrt_full_LFC"]),
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    assert list(restored.uns["_lrt"]["obs_names"]) == [
        str(name) for name in dds.obs_names
    ]
    np.testing.assert_allclose(
        restored.uns["_lrt"]["full_design_values"],
        np.asarray(dds.obsm["design_matrix"], dtype=float),
        rtol=0,
        atol=0,
    )

    restored_inference = CountingInference()
    restored_dds = DeseqDataSet(
        adata=restored,
        design="~condition",
        inference=restored_inference,
        refit_cooks=True,
        quiet=True,
    )
    canonical_lfcs = restored_dds.varm["LFC"].copy()
    assert restored_dds._lrt_fit_state_matches()
    stats = _condition_stats(restored_dds, test=None, reduced=None)
    assert not restored_inference.irls_inputs
    np.testing.assert_allclose(
        np.asarray(stats.LFC),
        np.asarray(restored_dds.varm["_lrt_full_LFC"]),
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    pd.testing.assert_frame_equal(restored_dds.varm["LFC"], canonical_lfcs)


def test_stale_lrt_cache_rejected_after_balanced_sample_subset(counts_df, metadata):
    """Observation subsetting must not reuse a full-dataset LRT cache."""
    dds = _fit_dds(
        counts_df,
        metadata,
        "~condition",
        test="LRT",
        reduced="~1",
    )
    samples = metadata.groupby("condition", observed=True).head(10).index
    subset_adata = dds.to_picklable_anndata()[samples].copy()
    subset_dds = DeseqDataSet(
        adata=subset_adata,
        design="~condition",
        inference=DefaultInference(n_cpus=1),
        refit_cooks=False,
        quiet=True,
    )

    assert "_lrt" in subset_dds.uns
    assert not subset_dds._lrt_fit_state_matches()
    with pytest.raises(RuntimeError, match="Rerun dds.deseq2"):
        DeseqStats(
            subset_dds,
            contrast=["condition", "B", "A"],
            test=None,
            cooks_filter=False,
            independent_filter=False,
            quiet=True,
        )


def test_stale_lrt_cache_rejected_after_full_design_value_mutation(counts_df, metadata):
    """Changing design values without changing columns must invalidate cache use."""
    dds = _fit_dds(
        counts_df,
        metadata,
        "~condition",
        test="LRT",
        reduced="~1",
    )
    dds.obsm["design_matrix"] = dds.obsm["design_matrix"].copy()
    dds.obsm["design_matrix"].iloc[0, 1] += 1.0

    assert not dds._lrt_fit_state_matches()
    with pytest.raises(RuntimeError, match="Rerun dds.deseq2"):
        DeseqStats(
            dds,
            contrast=["condition", "B", "A"],
            test=None,
            cooks_filter=False,
            independent_filter=False,
            quiet=True,
        )


def test_user_replace_counts_layer_is_preserved_and_ignored(counts_df, metadata):
    """Ordinary Wald fitting must not claim or consume a user-owned layer."""
    dds = DeseqDataSet(
        counts=counts_df.copy(),
        metadata=metadata.copy(),
        design="~condition",
        inference=DefaultInference(n_cpus=1),
        refit_cooks=False,
        quiet=True,
    )
    user_layer = np.asarray(dds.X).copy() + 1000
    dds.layers["replace_counts"] = user_layer.copy()

    dds.deseq2()

    np.testing.assert_array_equal(dds.layers["replace_counts"], user_layer)
    np.testing.assert_array_equal(dds._effective_counts(), np.asarray(dds.X))
    assert "_pydeseq2_replace_counts_owned" not in dds.uns
    assert dds.uns["_deseq2_test"] == "Wald"


def test_cook_replacement_rejects_unowned_layer_collision(counts_df, metadata):
    """Cook replacement must not overwrite a user layer with the reserved name."""
    counts = counts_df.copy()
    counts.loc["sample1", "gene1"] = 200
    dds = DeseqDataSet(
        counts=counts,
        metadata=metadata.copy(),
        design="~condition",
        inference=DefaultInference(n_cpus=1),
        refit_cooks=True,
        quiet=True,
    )
    user_layer = np.asarray(dds.X).copy()
    dds.layers["replace_counts"] = user_layer.copy()

    with pytest.raises(ValueError, match="not owned by PyDESeq2"):
        dds.deseq2()
    np.testing.assert_array_equal(dds.layers["replace_counts"], user_layer)


def test_malformed_replace_counts_owner_does_not_claim_user_layer(counts_df, metadata):
    """A truthy non-boolean marker must not authorize deleting a user layer."""
    dds = DeseqDataSet(
        counts=counts_df.copy(),
        metadata=metadata.copy(),
        design="~condition",
        inference=DefaultInference(n_cpus=1),
        quiet=True,
    )
    user_layer = np.asarray(dds.X).copy()
    dds.layers["replace_counts"] = user_layer.copy()
    dds.uns["_pydeseq2_replace_counts_owned"] = "False"

    assert not dds._owns_replace_counts()
    dds._discard_owned_replace_counts()

    np.testing.assert_array_equal(dds.layers["replace_counts"], user_layer)
    assert dds.uns["_pydeseq2_replace_counts_owned"] == "False"


def test_lrt_inherited_inference_honors_n_cpus(counts_df, metadata):
    """Stats n_cpus must configure an inference inherited from its dataset."""
    inference = DefaultInference(n_cpus=1)
    dds = _fit_dds(
        counts_df,
        metadata,
        "~condition",
        inference=inference,
    )

    stats = DeseqStats(
        dds,
        contrast=["condition", "B", "A"],
        test="LRT",
        reduced="~1",
        n_cpus=2,
        cooks_filter=False,
        independent_filter=False,
        quiet=True,
    )

    assert stats.inference is dds.inference
    assert stats.inference.n_cpus == 2


def test_lrt_cache_digest_rejects_mutated_scientific_inputs(counts_df, metadata):
    """Prepared LRT results must not survive mutations of fitted model inputs."""
    dds = _fit_dds(
        counts_df,
        metadata,
        "~condition",
        test="LRT",
        reduced="~1",
    )
    original_x = np.asarray(dds.X).copy()
    original_size_factors = dds.obs["size_factors"].copy()
    original_dispersions = dds.var["dispersions"].copy()
    original_lfcs = dds.varm["LFC"].copy()
    original_lfc_converged = dds.var["_LFC_converged"].copy()
    original_reduced = dds.obsm["_lrt_reduced_design_matrix"].copy()
    original_min_mu = dds.min_mu
    original_beta_tol = dds.beta_tol

    def assert_stale() -> None:
        assert not dds._lrt_fit_state_matches()
        with pytest.raises(RuntimeError, match="Cached LRT fit state"):
            DeseqStats(
                dds,
                contrast=["condition", "B", "A"],
                test=None,
                cooks_filter=False,
                independent_filter=False,
                quiet=True,
            )

    changed_x = original_x.copy()
    changed_x[0, 0] += 1
    dds.X = changed_x
    assert_stale()
    dds.X = original_x.copy()
    assert dds._lrt_fit_state_matches()

    for invalid_count in [np.nan, -1.0, 0.25, float(2**63)]:
        changed_x = original_x.astype(float)
        changed_x[0, 0] = invalid_count
        dds.X = changed_x
        assert_stale()
        with pytest.raises(
            ValueError,
            match="finite, non-negative integers within the int64 range",
        ):
            dds._compute_lrt(original_reduced)
    dds.X = original_x.copy()
    assert dds._lrt_fit_state_matches()

    dds.obs["size_factors"] = original_size_factors.to_numpy() * 1.01
    assert_stale()
    dds.obs["size_factors"] = original_size_factors
    assert dds._lrt_fit_state_matches()

    dds.var["dispersions"] = original_dispersions.to_numpy() * 1.01
    assert_stale()
    dds.var["dispersions"] = original_dispersions
    assert dds._lrt_fit_state_matches()

    dds.varm["LFC"] = original_lfcs + 0.01
    assert_stale()
    dds.varm["LFC"] = original_lfcs.copy()
    assert dds._lrt_fit_state_matches()

    dds.var["_LFC_converged"] = original_lfc_converged
    dds.var.loc[dds.var_names[0], "_LFC_converged"] = pd.NA
    assert_stale()
    dds.var["_LFC_converged"] = original_lfc_converged
    assert dds._lrt_fit_state_matches()

    changed_reduced = original_reduced.copy()
    changed_reduced.iloc[0, 0] += 1
    dds.obsm["_lrt_reduced_design_matrix"] = changed_reduced
    assert_stale()
    dds.obsm["_lrt_reduced_design_matrix"] = original_reduced
    assert dds._lrt_fit_state_matches()

    dds.min_mu = original_min_mu * 2
    assert_stale()
    dds.min_mu = original_min_mu
    assert dds._lrt_fit_state_matches()

    dds.beta_tol = original_beta_tol * 2
    assert_stale()
    dds.beta_tol = original_beta_tol
    assert dds._lrt_fit_state_matches()


def test_lrt_cache_digest_rejects_mutated_outputs_and_corruption(counts_df, metadata):
    """Private cached outputs and malformed metadata must fail closed."""
    dds = _fit_dds(
        counts_df,
        metadata,
        "~condition",
        test="LRT",
        reduced="~1",
    )
    original_pvalues = dds.var["_lrt_pvalue"].copy()
    original_full_lfcs = dds.varm["_lrt_full_LFC"].copy()
    original_metadata = dict(dds.uns["_lrt"])

    changed_pvalues = original_pvalues.copy()
    changed_pvalues.loc[changed_pvalues.first_valid_index()] += 0.01
    dds.var["_lrt_pvalue"] = changed_pvalues
    assert not dds._lrt_fit_state_matches()
    dds.var["_lrt_pvalue"] = original_pvalues
    assert dds._lrt_fit_state_matches()

    dds.varm["_lrt_full_LFC"] = original_full_lfcs + 0.01
    assert not dds._lrt_fit_state_matches()
    dds.varm["_lrt_full_LFC"] = original_full_lfcs
    assert dds._lrt_fit_state_matches()

    dds.uns["_lrt"] = {"fit_generation": "corrupt"}
    assert not dds._lrt_fit_state_matches()
    dds.uns["_lrt"] = original_metadata
    assert dds._lrt_fit_state_matches()


def test_lrt_rejects_missing_or_mutated_cook_replacement_state(counts_df, metadata):
    """Cook-refit LRTs require the exact internally managed effective counts."""
    counts = counts_df.copy()
    counts.loc["sample1", "gene1"] = 200
    dds = _fit_dds(
        counts,
        metadata,
        "~condition",
        refit_cooks=True,
        test="LRT",
        reduced="~1",
    )
    replacement_counts = np.asarray(dds.layers["replace_counts"]).copy()
    assert dds.var["replaced"].any()
    assert dds._lrt_fit_state_matches()

    dds.layers["replace_counts"] = replacement_counts + 1
    assert not dds._lrt_fit_state_matches()
    dds.layers["replace_counts"] = replacement_counts
    assert dds._lrt_fit_state_matches()

    dds.uns["_pydeseq2_replace_counts_owned"] = False
    assert not dds._lrt_fit_state_matches()
    dds.uns["_pydeseq2_replace_counts_owned"] = True
    assert dds._lrt_fit_state_matches()

    dds._invalidate_lrt_cache()
    del dds.layers["replace_counts"]
    del dds.uns["_pydeseq2_replace_counts_owned"]
    with pytest.raises(RuntimeError, match="effective counts used during Cook"):
        _condition_stats(dds, test="LRT", reduced="~1")


def test_full_model_nonconvergence_warns_but_keeps_lrt_results(counts_df, metadata):
    """A failed full-model convergence flag is visible without discarding the test."""
    dds = _fit_dds(counts_df, metadata, "~condition")
    with pytest.warns(UserWarning, match="full-model fit did not converge"):
        stats = _condition_stats(
            dds,
            test="LRT",
            reduced="~1",
            inference=NonconvergedFullInference(),
        )

    assert not bool(stats.full_converged.iloc[0])
    assert np.isfinite(stats.statistics.iloc[0])
    assert np.isfinite(stats.p_values.iloc[0])


def test_materially_negative_statistic_triggers_targeted_full_refit(counts_df, metadata):
    """A poor cached full fit is recovered gene-wise before reporting an LRT."""
    dds = _fit_dds(counts_df, metadata, "~condition")
    bad_lfcs = dds.varm["LFC"].copy()
    bad_lfcs.loc["gene1", :] = [15.0, -30.0]
    dds.varm["LFC"] = bad_lfcs
    inference = CountingInference()

    results = dds._compute_lrt(
        dds._prepare_lrt_reduced_design("~1"),
        inference=inference,
    )
    statistics, _, _, _, _, _, recovered_lfcs = results

    assert [design.shape[1] for _, design in inference.irls_inputs] == [1, 2]
    assert np.isfinite(statistics.loc["gene1"])
    assert statistics.loc["gene1"] >= 0
    assert not np.allclose(recovered_lfcs.loc["gene1"], bad_lfcs.loc["gene1"])
