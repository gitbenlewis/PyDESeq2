"""Tests for classical negative-binomial likelihood-ratio tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from formulaic import model_matrix
from scipy.sparse import csc_array
from scipy.sparse import csr_matrix
from scipy.sparse import issparse
from scipy.stats import chi2

from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
from pydeseq2.utils import load_example_data
from pydeseq2.utils import nb_nll

_REPLACEMENT_DELTA_LAYER = "_pydeseq2_cook_replacement_delta"


def _dense_counts(values: Any) -> np.ndarray:
    """Return dense count values for small test fixtures."""
    return values.toarray() if issparse(values) else np.asarray(values)


class CountingInference(DefaultInference):
    """Default inference implementation recording each IRLS invocation."""

    def __init__(self) -> None:
        super().__init__(n_cpus=1)
        self.irls_inputs: list[tuple[np.ndarray, np.ndarray]] = []
        self.irls_size_factors: list[np.ndarray] = []

    def irls(self, *args: Any, **kwargs: Any):
        """Record counts, normalization factors, and design before fitting."""
        counts = kwargs.get("counts", args[0] if args else None)
        size_factors = kwargs.get("size_factors", args[1] if len(args) > 1 else None)
        design_matrix = kwargs.get("design_matrix", args[2] if len(args) > 2 else None)
        assert counts is not None
        assert size_factors is not None
        assert design_matrix is not None
        self.irls_inputs.append(
            (_dense_counts(counts).copy(), np.asarray(design_matrix).copy())
        )
        self.irls_size_factors.append(np.asarray(size_factors).copy())
        return super().irls(*args, **kwargs)


class NonconvergedInference(DefaultInference):
    """Inference backend marking one requested model width nonconverged."""

    def __init__(self, design_width: int) -> None:
        super().__init__(n_cpus=1)
        self.design_width = design_width

    def irls(self, *args: Any, **kwargs: Any):
        """Return normal estimates while overriding one convergence status."""
        result = super().irls(*args, **kwargs)
        design_matrix = kwargs.get("design_matrix", args[2] if len(args) > 2 else None)
        assert design_matrix is not None
        if np.asarray(design_matrix).shape[1] == self.design_width:
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


def test_dds_lrt_cache_is_reused_by_stats(counts_df, metadata, monkeypatch):
    """Repeated summaries reuse the prepared LRT without rescanning fitted inputs."""
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

    def unexpected_fit_scan(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("Repeated summaries must not rescan fitted LRT inputs.")

    monkeypatch.setattr(dds, "_lrt_fit_digest", unexpected_fit_scan)
    stats.summary()
    stats.summary()

    assert len(inference.irls_inputs) == calls_after_dds_fit
    np.testing.assert_allclose(stats.statistics, dds.var["_lrt_statistic"])
    np.testing.assert_allclose(stats.p_values, dds.var["_lrt_pvalue"])

    assert stats.reduced_design_matrix is not None
    stats.reduced_design_matrix.iloc[0, 0] += 1
    for action in (stats.run_lrt_test, stats.summary):
        with pytest.raises(RuntimeError, match="LRT inputs changed"):
            action()


def test_lrt_uses_transcript_length_normalization_factors(counts_df, metadata):
    """Both LRT fits and its cache use gene-specific normalization factors."""
    sample_effect = np.linspace(-1.0, 1.0, counts_df.shape[0])
    gene_effect = np.linspace(-1.0, 1.0, counts_df.shape[1])
    lengths = (800.0 + 100.0 * np.arange(counts_df.shape[1]))[None, :]
    lengths = lengths * (1.0 + 0.2 * np.outer(sample_effect, gene_effect))
    transcript_lengths = pd.DataFrame(
        lengths,
        index=counts_df.index,
        columns=counts_df.columns,
    )
    inference = CountingInference()
    dds = DeseqDataSet(
        counts=counts_df.copy(),
        metadata=metadata.copy(),
        transcript_lengths=transcript_lengths,
        design="~condition",
        inference=inference,
        refit_cooks=False,
        quiet=True,
    )

    dds.deseq2(test="LRT", reduced="~1")

    expected = np.asarray(dds.layers["normalization_factors"])[:, dds.non_zero_idx]
    assert len(inference.irls_size_factors) == 2
    for normalization_factors in inference.irls_size_factors:
        assert normalization_factors.ndim == 2
        np.testing.assert_allclose(normalization_factors, expected)

    non_zero = dds.var["non_zero"].to_numpy(dtype=bool)
    counts = _dense_counts(dds.X[:, non_zero])
    factors = np.asarray(dds.layers["normalization_factors"])[:, non_zero]
    dispersions = dds.var.loc[non_zero, "dispersions"].to_numpy()
    full_design = dds.obsm["design_matrix"].to_numpy()
    full_lfcs = dds.varm["LFC"].loc[non_zero].to_numpy()
    full_mu = factors * np.exp(full_design @ full_lfcs.T)
    reduced_design = dds.obsm["_lrt_reduced_design_matrix"].to_numpy()
    _, reduced_mu, _, _ = DefaultInference(n_cpus=1).irls(
        counts=counts,
        size_factors=factors,
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
    assert dds._lrt_fit_state_matches()

    original = np.asarray(dds.layers["normalization_factors"]).copy()
    changed = original.copy()
    changed[0, 0] *= 1.01
    dds.layers["normalization_factors"] = changed
    assert not dds._lrt_fit_state_matches()
    dds.layers["normalization_factors"] = original
    assert dds._lrt_fit_state_matches()


def test_rejected_size_factor_refit_preserves_lrt_state(counts_df, metadata):
    """Request validation must not discard a valid LRT or its effective counts."""
    counts = counts_df.copy()
    counts.loc["sample1", "gene1"] = 200
    transcript_lengths = pd.DataFrame(
        1.0,
        index=counts.index,
        columns=counts.columns,
    )
    dds = DeseqDataSet(
        counts=counts,
        metadata=metadata.copy(),
        transcript_lengths=transcript_lengths,
        design="~condition",
        inference=DefaultInference(n_cpus=1),
        refit_cooks=True,
        quiet=True,
    )
    dds.deseq2(test="LRT", reduced="~1")

    replacement_delta = dds.layers[_REPLACEMENT_DELTA_LAYER].copy()
    fit_generation = int(dds.uns["_fit_generation"])

    with pytest.raises(ValueError, match="iterative size-factor method"):
        dds.fit_size_factors("iterative")

    assert dds._lrt_fit_state_matches()
    assert int(dds.uns["_fit_generation"]) == fit_generation
    assert dds.uns["_deseq2_test"] == "LRT"
    assert dds.uns["_pydeseq2_replace_counts_owned"] is True
    np.testing.assert_array_equal(
        _dense_counts(dds.layers[_REPLACEMENT_DELTA_LAYER]),
        _dense_counts(replacement_delta),
    )


@pytest.mark.parametrize(
    "sparse_constructor",
    [csr_matrix, csc_array],
)
def test_sparse_lrt_matches_dense_with_transcript_normalization(
    counts_df,
    metadata,
    sparse_constructor,
):
    """Sparse LRT inputs remain sparse and reproduce dense normalized results."""
    sample_effect = np.linspace(-1.0, 1.0, counts_df.shape[0])
    gene_effect = np.linspace(-1.0, 1.0, counts_df.shape[1])
    lengths = (800.0 + 100.0 * np.arange(counts_df.shape[1]))[None, :]
    lengths = lengths * (1.0 + 0.2 * np.outer(sample_effect, gene_effect))
    transcript_lengths = pd.DataFrame(
        lengths,
        index=counts_df.index,
        columns=counts_df.columns,
    )
    sparse_adata = ad.AnnData(
        X=sparse_constructor(counts_df.to_numpy()),
        obs=metadata.copy(),
        var=pd.DataFrame(index=counts_df.columns),
    )
    sparse_dds = DeseqDataSet(
        adata=sparse_adata,
        transcript_lengths=transcript_lengths,
        design="~condition",
        inference=DefaultInference(n_cpus=1),
        refit_cooks=False,
        quiet=True,
    )
    dense_dds = DeseqDataSet(
        counts=counts_df.copy(),
        metadata=metadata.copy(),
        transcript_lengths=transcript_lengths,
        design="~condition",
        inference=DefaultInference(n_cpus=1),
        refit_cooks=False,
        quiet=True,
    )

    sparse_dds.deseq2(test="LRT", reduced="~1")
    dense_dds.deseq2(test="LRT", reduced="~1")

    assert issparse(sparse_dds.X)
    assert issparse(sparse_dds._effective_counts())
    assert sparse_dds._lrt_fit_state_matches()
    np.testing.assert_allclose(
        sparse_dds.var["_lrt_statistic"],
        dense_dds.var["_lrt_statistic"],
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        sparse_dds.var["_lrt_pvalue"],
        dense_dds.var["_lrt_pvalue"],
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    )


@pytest.mark.parametrize(
    ("sparse_constructor", "low_memory"),
    [(csc_array, False), (None, True)],
    ids=["sparse_array", "low_memory"],
)
def test_lrt_fits_sparse_and_low_memory_data_in_bounded_blocks(
    counts_df,
    metadata,
    sparse_constructor,
    low_memory,
    monkeypatch,
):
    """Reduced-model IRLS should receive at most one bounded gene block."""
    n_genes = 129
    repeats = int(np.ceil(n_genes / counts_df.shape[1]))
    values = np.tile(counts_df.to_numpy(), (1, repeats))[:, :n_genes]
    genes = [f"gene_{index}" for index in range(n_genes)]
    expanded_counts = pd.DataFrame(values, index=counts_df.index, columns=genes)
    inference = CountingInference()

    if sparse_constructor is not None:
        adata = ad.AnnData(
            X=sparse_constructor(expanded_counts.to_numpy()),
            obs=metadata.copy(),
            var=pd.DataFrame(index=genes),
        )
        dds = DeseqDataSet(
            adata=adata,
            design="~condition",
            inference=inference,
            refit_cooks=False,
            quiet=True,
        )
    else:
        dds = DeseqDataSet(
            counts=expanded_counts,
            metadata=metadata.copy(),
            design="~condition",
            inference=inference,
            low_memory=low_memory,
            refit_cooks=False,
            quiet=True,
        )
    dds.deseq2()
    if low_memory:
        assert "_mu_LFC" not in dds.obsm
    inference.irls_inputs.clear()
    inference.irls_size_factors.clear()

    results = dds._compute_lrt(dds._prepare_lrt_reduced_design("~1"))

    reduced_widths = [
        block_counts.shape[1]
        for block_counts, design in inference.irls_inputs
        if design.shape[1] == 1
    ]
    assert reduced_widths == [128, 1]
    fitted = dds.var["non_zero"] & dds.var["dispersions"].notna()
    assert results[0].loc[fitted].notna().all()

    effective_counts = _dense_counts(dds._effective_counts())
    monkeypatch.setattr(dds, "_effective_counts", lambda: effective_counts)
    dds.low_memory = False
    oracle = dds._compute_lrt(
        dds._prepare_lrt_reduced_design("~1"),
        inference=DefaultInference(n_cpus=1),
    )
    for blocked, unblocked in zip(results[:3], oracle[:3], strict=True):
        np.testing.assert_allclose(
            blocked,
            unblocked,
            rtol=1e-10,
            atol=1e-12,
            equal_nan=True,
        )
    np.testing.assert_allclose(
        results[6],
        oracle[6],
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    )


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
    stats.summary()
    assert len(inference.irls_inputs) == 2

    second_stats = _condition_stats(dds, inference=inference)
    assert len(inference.irls_inputs) == 2
    _assert_same_test_results(stats, second_stats)


def test_lrt_stats_rejects_refit_before_cache_creation(counts_df, metadata):
    """Public refits invalidate stats built from a manually fitted dataset."""
    dds = DeseqDataSet(
        counts=counts_df.copy(),
        metadata=metadata.copy(),
        design="~condition",
        inference=DefaultInference(n_cpus=1),
        refit_cooks=False,
        quiet=True,
    )
    for fit in (
        dds.fit_size_factors,
        dds.fit_genewise_dispersions,
        dds.fit_dispersion_trend,
        dds.fit_dispersion_prior,
        dds.fit_MAP_dispersions,
        dds.fit_LFC,
    ):
        fit()

    stats = DeseqStats(
        dds,
        contrast=["condition", "B", "A"],
        test="LRT",
        reduced="~1",
        cooks_filter=False,
        independent_filter=False,
        quiet=True,
    )
    fit_generation = int(dds.uns["_fit_generation"])
    dds.fit_size_factors()

    assert int(dds.uns["_fit_generation"]) == fit_generation + 1
    with pytest.raises(RuntimeError, match="LRT inputs changed"):
        stats.summary()


def test_recovered_full_fit_is_reused_for_a_different_reduced_model(
    counts_df,
    metadata,
):
    """A recovered full optimum must seed later LRTs with another reduced model."""
    dds = _fit_dds(counts_df, metadata, "~group + condition")
    bad_lfcs = dds.varm["LFC"].copy()
    bad_lfcs.loc["gene1"] = [
        1.4294080342883349,
        0.4225143835704045,
        0.4066283298245054,
    ]
    dds.varm["LFC"] = bad_lfcs

    first_stats = DeseqStats(
        dds,
        contrast=["condition", "B", "A"],
        test="LRT",
        reduced="~group",
        cooks_filter=False,
        independent_filter=False,
        quiet=True,
    )
    first_stats.summary()
    assert not np.allclose(first_stats.LFC.loc["gene1"], bad_lfcs.loc["gene1"])

    second_reduced = dds._prepare_lrt_reduced_design("~condition")
    stats = DeseqStats(
        dds,
        contrast=["condition", "B", "A"],
        test="LRT",
        reduced="~condition",
        cooks_filter=False,
        independent_filter=False,
        quiet=True,
    )
    stats.summary()
    oracle = dds._compute_lrt(second_reduced, refit_full=True)

    np.testing.assert_allclose(
        stats.statistics.loc["gene1"],
        oracle[0].loc["gene1"],
        rtol=1e-8,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        stats.p_values.loc["gene1"],
        oracle[1].loc["gene1"],
        rtol=1e-8,
        atol=1e-12,
    )


def test_run_lrt_test_restores_raw_pvalues_and_rejects_refits(counts_df, metadata):
    """Repeated LRTs restore filtering while public refits invalidate old stats."""
    counts = counts_df.copy()
    counts.loc["sample1", "gene1"] = 200
    inference = CountingInference()
    dds = _fit_dds(
        counts,
        metadata,
        "~condition",
        inference=inference,
        refit_cooks=False,
        test="LRT",
        reduced="~1",
    )
    raw_pvalue = float(dds.var.loc["gene1", "_lrt_pvalue"])
    calls_after_dds_fit = len(inference.irls_inputs)
    stats = _condition_stats(
        dds,
        test=None,
        reduced=None,
        inference=inference,
        cooks_filter=True,
        independent_filter=False,
    )

    assert pd.isna(stats.p_values.loc["gene1"])
    stats.run_lrt_test()
    assert stats.p_values.loc["gene1"] == raw_pvalue
    assert len(inference.irls_inputs) == calls_after_dds_fit
    assert not hasattr(stats, "padj")
    assert not hasattr(stats, "results_df")

    stats.cooks_filter = False
    stats.summary()
    assert stats.results_df.loc["gene1", "pvalue"] == raw_pvalue
    assert np.isfinite(stats.results_df.loc["gene1", "padj"])

    fit_generation = int(dds.uns["_fit_generation"])
    dds.fit_size_factors()
    assert int(dds.uns["_fit_generation"]) == fit_generation + 1
    with pytest.raises(RuntimeError, match="LRT inputs changed"):
        stats.summary()

    assert len(inference.irls_inputs) == calls_after_dds_fit


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
    assert np.isfinite(
        matrix_stats.results_df.loc[matrix_dds.var["non_zero"], "stat"]
    ).all()
    assert matrix_stats.lrt_df == 1


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
    assert _REPLACEMENT_DELTA_LAYER in dds.layers
    assert issparse(dds.layers[_REPLACEMENT_DELTA_LAYER])
    expected_counts = pd.read_csv(
        Path(__file__).parent / "data" / "lrt" / "r_lrt_outlier_replace_counts.csv",
        index_col=0,
    ).T
    expected_counts = expected_counts.loc[dds.obs_names, dds.var_names]
    effective_counts = _dense_counts(dds._effective_counts())
    np.testing.assert_array_equal(effective_counts, expected_counts)

    replaced = dds.var["replaced"].to_numpy(dtype=bool)
    expected_delta = np.zeros(dds.shape, dtype=np.int64)
    expected_delta[:, replaced] = np.asarray(
        dds.counts_to_refit.X,
        dtype=np.int64,
    ) - _dense_counts(dds.X[:, replaced]).astype(np.int64)
    np.testing.assert_array_equal(
        _dense_counts(dds.layers[_REPLACEMENT_DELTA_LAYER]),
        expected_delta,
    )

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
    assert _REPLACEMENT_DELTA_LAYER in dds.layers
    assert not _dense_counts(
        dds._effective_counts()[:, dds.var_names == "single_outlier"]
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


def test_wald_lfc_shrink_keeps_original_counts_after_cooks_refit(
    counts_df,
    metadata,
    monkeypatch,
):
    """Wald shrinkage must retain its established pre-LRT count inputs."""
    counts = counts_df.copy()
    counts.loc["sample1", "gene1"] = 200
    dds = _fit_dds(counts, metadata, "~condition", refit_cooks=True)
    stats = _condition_stats(dds, test=None, reduced=None)
    original_counts = _dense_counts(dds.X[:, dds.non_zero_idx])
    effective_counts = _dense_counts(dds._effective_counts()[:, dds.non_zero_idx])
    assert not np.array_equal(original_counts, effective_counts)

    captured: dict[str, Any] = {}
    original_shrink = stats.inference.lfc_shrink_nbinom_glm
    original_prior = stats._fit_prior_var

    def record_shrink(*args: Any, **kwargs: Any):
        captured["counts"] = _dense_counts(kwargs["counts"]).copy()
        return original_shrink(*args, **kwargs)

    def record_prior(*args: Any, **kwargs: Any) -> float:
        captured["gene_mask"] = kwargs.get("gene_mask")
        return original_prior(*args, **kwargs)

    monkeypatch.setattr(stats.inference, "lfc_shrink_nbinom_glm", record_shrink)
    monkeypatch.setattr(stats, "_fit_prior_var", record_prior)
    stats.lfc_shrink(coeff="condition[T.B]")

    np.testing.assert_array_equal(captured["counts"], original_counts)
    assert captured["gene_mask"] is None


def test_lrt_console_output_distinguishes_contrast_from_omnibus_test(
    counts_df,
    metadata,
    capsys,
):
    """LRT output must not describe its omnibus p-value as contrast-specific."""
    dds = _fit_dds(
        counts_df,
        metadata,
        "~condition",
        test="LRT",
        reduced="~1",
    )
    capsys.readouterr()
    stats = DeseqStats(
        dds,
        contrast=["condition", "B", "A"],
        test=None,
        cooks_filter=False,
        independent_filter=False,
        quiet=False,
    )

    stats.summary()
    stdout = capsys.readouterr().out
    assert "Reported log2 fold-change contrast: condition B vs A" in stdout
    assert "Likelihood-ratio test p-value: full model vs reduced model" in stdout
    assert "Log2 fold change & LRT test p-value" not in stdout

    stats.lfc_shrink(coeff="condition[T.B]", adapt=False)
    stdout = capsys.readouterr().out
    assert (
        "Shrunk log2 fold change: condition[T.B]; likelihood-ratio test "
        "p-value remains the full-vs-reduced omnibus comparison"
    ) in stdout


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
        n_cpus=1,
    )
    explicit = DeseqStats(
        dds,
        ["condition", "B", "A"],
        0.1,
        False,
        False,
        quiet=True,
        test="Wald",
        n_cpus=1,
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
    formula_metadata = metadata.copy()
    formula_metadata["intercept_alias"] = 1.0
    formula_dds = _fit_dds(counts_df, formula_metadata, "~group + condition")
    reduced_matrix = pd.DataFrame(model_matrix("~group", metadata.copy()))
    with pytest.raises(ValueError, match="formula"):
        _summarize_lrt(
            formula_dds,
            test="LRT",
            reduced=reduced_matrix,
        )

    alias_stats = _condition_stats(
        formula_dds,
        reduced="~0 + intercept_alias",
    )
    assert alias_stats.lrt_df == 2

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


def test_picklable_anndata_preserves_user_series(counts_df, metadata):
    """Serialization cleanup must leave user-owned unstructured metadata intact."""
    dds = _fit_dds(counts_df, metadata, "~condition")
    user_series = pd.Series(
        [1.0, 2.0],
        index=["first", "second"],
        name="user_metadata",
    )
    dds.uns["user_series"] = user_series.copy()

    adata = dds.to_picklable_anndata()

    assert isinstance(adata.uns["user_series"], pd.Series)
    pd.testing.assert_series_equal(adata.uns["user_series"], user_series)
    pd.testing.assert_series_equal(dds.uns["user_series"], user_series)


def test_lrt_state_survives_picklable_h5ad_roundtrip(
    counts_df, metadata, tmp_path: Path
):
    """A serialized prepared LRT should be reusable without another model fit."""
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
    expected = _condition_stats(dds, test=None, reduced=None)
    expected_counts = _dense_counts(dds._effective_counts()).copy()
    adata = dds.to_picklable_anndata()
    output_path = tmp_path / "lrt_state.h5ad"

    adata.write_h5ad(output_path)
    restored = ad.read_h5ad(output_path)

    restored_inference = CountingInference()
    restored_dds = DeseqDataSet(
        adata=restored,
        design="~condition",
        inference=restored_inference,
        refit_cooks=True,
        quiet=True,
    )
    stats = _condition_stats(restored_dds, test=None, reduced=None)

    assert not restored_inference.irls_inputs
    _assert_same_test_results(stats, expected)
    np.testing.assert_array_equal(
        _dense_counts(restored_dds._effective_counts()),
        expected_counts,
    )


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


def test_stale_lrt_cache_rejected_after_reduced_design_mutation(
    counts_df,
    metadata,
):
    """Changing the prepared reduced design must invalidate cache restoration."""
    dds = _fit_dds(
        counts_df,
        metadata,
        "~condition",
        test="LRT",
        reduced="~1",
    )
    changed_reduced = dds.obsm["_lrt_reduced_design_matrix"].copy()
    changed_reduced.iloc[0, 0] += 1
    dds.obsm["_lrt_reduced_design_matrix"] = changed_reduced

    with pytest.raises(RuntimeError, match="Cached LRT fit state"):
        DeseqStats(
            dds,
            contrast=["condition", "B", "A"],
            test=None,
            cooks_filter=False,
            independent_filter=False,
            quiet=True,
        )


def test_full_rerun_clears_stale_cook_replacement_state(counts_df, metadata):
    """Changing Cook-refit policy between full runs must rebuild clean state."""
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
    assert dds.var["replaced"].any()
    assert _REPLACEMENT_DELTA_LAYER in dds.layers

    dds.refit_cooks = False
    dds.deseq2(test="LRT", reduced="~1")
    assert "replaced" not in dds.var
    assert "refitted" not in dds.var
    assert _REPLACEMENT_DELTA_LAYER not in dds.layers
    assert "_pydeseq2_replace_counts_owned" not in dds.uns
    assert dds._lrt_fit_state_matches()
    stats = _condition_stats(dds, test=None, reduced=None)
    assert stats.statistics.loc[dds.var["non_zero"]].notna().all()

    dds.refit_cooks = True
    dds.deseq2(test="LRT", reduced="~1")
    assert dds.var["replaced"].any()
    assert _REPLACEMENT_DELTA_LAYER in dds.layers
    assert dds._lrt_fit_state_matches()


@pytest.mark.parametrize(
    ("model", "design_width", "convergence_attribute"),
    [
        ("full", 2, "full_converged"),
        ("reduced", 1, "reduced_converged"),
    ],
)
def test_model_nonconvergence_warns_but_keeps_lrt_results(
    counts_df,
    metadata,
    model,
    design_width,
    convergence_attribute,
):
    """A failed model convergence flag is visible without discarding the test."""
    dds = _fit_dds(counts_df, metadata, "~condition")
    with pytest.warns(
        UserWarning,
        match=rf"{model}-model fit did not converge",
    ) as warning_record:
        stats = _condition_stats(
            dds,
            test="LRT",
            reduced="~1",
            inference=NonconvergedInference(design_width),
        )

    assert any(
        f"DeseqStats.{convergence_attribute}" in str(warning.message)
        for warning in warning_record
    )
    assert not bool(getattr(stats, convergence_attribute).iloc[0])
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


def test_lrt_recovery_subsets_gene_specific_factors_by_fit_position(
    counts_df, metadata, monkeypatch
):
    """Targeted recovery must map fitted positions back to global factor columns."""
    counts = counts_df.copy()
    counts.insert(0, "all_zero", 0)
    sample_effect = np.linspace(-1.0, 1.0, counts.shape[0])
    gene_effect = np.linspace(-1.0, 1.0, counts.shape[1])
    lengths = (800.0 + 100.0 * np.arange(counts.shape[1]))[None, :]
    lengths = lengths * (1.0 + 0.2 * np.outer(sample_effect, gene_effect))
    transcript_lengths = pd.DataFrame(
        lengths,
        index=counts.index,
        columns=counts.columns,
    )
    dds = DeseqDataSet(
        counts=counts,
        metadata=metadata.copy(),
        transcript_lengths=transcript_lengths,
        design="~condition",
        inference=DefaultInference(n_cpus=1),
        refit_cooks=False,
        quiet=True,
    )
    dds.deseq2()

    target = "gene10"
    effective_nonzero = dds._nonzero_count_columns(dds._effective_counts())
    fit_idx = np.flatnonzero(
        effective_nonzero & np.isfinite(dds.var["dispersions"].to_numpy())
    )
    target_idx = dds.var_names.get_loc(target)
    target_fit_position = int(np.flatnonzero(fit_idx == target_idx)[0])
    all_factors = np.asarray(dds.layers["normalization_factors"])
    bad_lfcs = dds.varm["LFC"].copy()
    bad_lfcs.loc[target, :] = [15.0, -30.0]
    dds.varm["LFC"] = bad_lfcs
    inference = CountingInference()
    nll_calls: list[int] = []

    def controlled_nll(counts, mu, alpha):
        n_genes = np.asarray(counts).shape[1]
        call = len(nll_calls)
        nll_calls.append(n_genes)
        if call == 0:
            return np.full(n_genes, 100.0)
        if call == 1:
            reduced_nll = np.full(n_genes, 101.0)
            reduced_nll[target_fit_position] = 99.0
            return reduced_nll
        if call == 2:
            assert n_genes == 1
            return np.array([98.0])
        raise AssertionError("Unexpected extra nb_nll call.")

    monkeypatch.setattr("pydeseq2.dds.nb_nll", controlled_nll)
    results = dds._compute_lrt(
        dds._prepare_lrt_reduced_design("~1"),
        inference=inference,
    )
    statistics, _, _, _, _, _, recovered_lfcs = results

    assert target_idx != target_fit_position
    assert nll_calls == [len(fit_idx), len(fit_idx), 1]
    assert [design.shape[1] for _, design in inference.irls_inputs] == [1, 2]
    np.testing.assert_allclose(
        inference.irls_size_factors[0],
        all_factors[:, fit_idx],
    )
    np.testing.assert_allclose(
        inference.irls_size_factors[1],
        all_factors[:, [target_idx]],
    )
    np.testing.assert_array_equal(
        inference.irls_inputs[1][0],
        _dense_counts(dds.X)[:, [target_idx]],
    )
    assert np.isfinite(statistics.loc[target])
    assert not np.allclose(recovered_lfcs.loc[target], bad_lfcs.loc[target])


def test_nonfinite_cached_full_likelihood_triggers_recovery(counts_df, metadata):
    """A finite cached LFC that underflows its mean should be refitted."""
    dds = _fit_dds(counts_df + 1, metadata, "~condition")
    bad_lfcs = dds.varm["LFC"].copy()
    bad_lfcs.loc["gene1", :] = [-1000.0, 0.0]
    dds.varm["LFC"] = bad_lfcs
    inference = CountingInference()

    with np.errstate(divide="ignore", invalid="ignore"):
        results = dds._compute_lrt(
            dds._prepare_lrt_reduced_design("~1"),
            inference=inference,
        )
    statistics, p_values, full_deviance, _, _, _, recovered_lfcs = results

    assert [design.shape[1] for _, design in inference.irls_inputs] == [1, 2]
    assert np.isfinite(statistics.loc["gene1"])
    assert statistics.loc["gene1"] >= 0
    assert np.isfinite(p_values.loc["gene1"])
    assert np.isfinite(full_deviance.loc["gene1"])
    assert np.isfinite(recovered_lfcs.loc["gene1"]).all()
    assert not np.allclose(recovered_lfcs.loc["gene1"], bad_lfcs.loc["gene1"])


def test_nonfinite_cached_full_lfc_triggers_targeted_recovery(counts_df, metadata):
    """A nonfinite cached coefficient should be refitted rather than skipped."""
    dds = _fit_dds(counts_df + 1, metadata, "~condition")
    bad_lfcs = dds.varm["LFC"].copy()
    bad_lfcs.loc["gene1", :] = np.nan
    dds.varm["LFC"] = bad_lfcs
    inference = CountingInference()

    results = dds._compute_lrt(
        dds._prepare_lrt_reduced_design("~1"),
        inference=inference,
    )
    (
        statistics,
        p_values,
        full_deviance,
        _,
        full_converged,
        _,
        recovered_lfcs,
    ) = results

    assert [design.shape[1] for _, design in inference.irls_inputs] == [1, 2]
    assert inference.irls_inputs[1][0].shape[1] == 1
    assert np.isfinite(statistics.loc["gene1"])
    assert np.isfinite(p_values.loc["gene1"])
    assert np.isfinite(full_deviance.loc["gene1"])
    assert bool(full_converged.loc["gene1"])
    assert np.isfinite(recovered_lfcs.loc["gene1"]).all()


def test_optimizer_scale_negative_roundoff_is_clipped_without_refit(
    counts_df,
    metadata,
    monkeypatch,
):
    """A negative within optimizer tolerance maps to zero without recovery."""
    dds = _fit_dds(counts_df, metadata, "~condition")
    inference = CountingInference()
    small_negative = 1e-8
    calls: list[int] = []

    def controlled_nll(counts, mu, alpha):
        n_genes = np.asarray(counts).shape[1]
        call = len(calls)
        calls.append(n_genes)
        if call == 0:
            return np.full(n_genes, 100.0)
        if call == 1:
            reduced_nll = np.full(n_genes, 101.0)
            reduced_nll[0] = 100.0 - small_negative / 2
            return reduced_nll
        raise AssertionError("Unexpected extra nb_nll call.")

    monkeypatch.setattr("pydeseq2.dds.nb_nll", controlled_nll)
    results = dds._compute_lrt(
        dds._prepare_lrt_reduced_design("~1"),
        inference=inference,
    )
    statistics, p_values = results[:2]
    fitted_genes = dds.var_names[
        dds.var["non_zero"]
        & dds.var["dispersions"].notna()
        & np.isfinite(dds.varm["LFC"]).all(axis=1)
    ]
    recovered_gene = fitted_genes[0]

    assert small_negative > 64 * np.finfo(float).eps * 200
    assert calls == [len(fitted_genes), len(fitted_genes)]
    assert [design.shape[1] for _, design in inference.irls_inputs] == [1]
    assert statistics.loc[recovered_gene] == 0.0
    assert p_values.loc[recovered_gene] == 1.0


def test_nonfinite_likelihood_after_recovery_is_invalid(
    counts_df, metadata, monkeypatch
):
    """Persistent non-finite likelihoods must never become a valid null result."""
    dds = _fit_dds(counts_df, metadata, "~condition")
    inference = CountingInference()
    calls: list[int] = []

    def controlled_nll(counts, mu, alpha):
        n_genes = np.asarray(counts).shape[1]
        call = len(calls)
        calls.append(n_genes)
        if call == 0:
            full_nll = np.full(n_genes, 100.0)
            full_nll[0] = np.inf
            return full_nll
        if call == 1:
            return np.full(n_genes, 101.0)
        if call == 2:
            assert n_genes == 1
            return np.full(n_genes, np.inf)
        raise AssertionError("Unexpected extra nb_nll call.")

    monkeypatch.setattr("pydeseq2.dds.nb_nll", controlled_nll)
    with pytest.warns(RuntimeWarning, match="non-finite likelihoods"):
        results = dds._compute_lrt(
            dds._prepare_lrt_reduced_design("~1"),
            inference=inference,
        )
    statistics, p_values, full_deviance = results[:3]
    fitted_genes = dds.var_names[
        dds.var["non_zero"]
        & dds.var["dispersions"].notna()
        & np.isfinite(dds.varm["LFC"]).all(axis=1)
    ]
    invalid_gene = fitted_genes[0]

    assert calls == [len(fitted_genes), len(fitted_genes), 1]
    assert [design.shape[1] for _, design in inference.irls_inputs] == [1, 2]
    assert pd.isna(statistics.loc[invalid_gene])
    assert pd.isna(p_values.loc[invalid_gene])
    assert pd.isna(full_deviance.loc[invalid_gene])


def test_exact_null_roundoff_does_not_drop_lrt_results():
    """Optimizer-scale negative roundoff under an exact null should map to zero."""
    rng = np.random.default_rng(91)
    n_per_group = 12
    n_genes = 10
    means = np.exp(rng.normal(3.5, 1.0, n_genes))
    dispersion = 0.2
    size = 1 / dispersion
    block = rng.negative_binomial(
        size,
        size / (size + means),
        size=(n_per_group, n_genes),
    )
    sample_names = [f"sample{i}" for i in range(2 * n_per_group)]
    gene_names = [f"g{i}" for i in range(n_genes)]
    counts = pd.DataFrame(
        np.vstack((block, block)),
        index=sample_names,
        columns=gene_names,
    )
    metadata = pd.DataFrame(
        {"condition": ["A"] * n_per_group + ["B"] * n_per_group},
        index=sample_names,
    )
    dds = DeseqDataSet(
        counts=counts,
        metadata=metadata,
        design="~condition",
        fit_type="mean",
        refit_cooks=False,
        inference=DefaultInference(n_cpus=1),
        quiet=True,
    )
    dds.deseq2(test="LRT", reduced="~1")
    stats = _condition_stats(dds, test=None, reduced=None)

    fitted = dds.var["non_zero"] & dds.var["dispersions"].notna()
    assert stats.statistics.loc[fitted].notna().all()
    assert stats.p_values.loc[fitted].notna().all()
    np.testing.assert_allclose(stats.statistics.loc[fitted], 0.0, atol=1e-8)
    assert (stats.p_values.loc[fitted] > 0.999).all()
