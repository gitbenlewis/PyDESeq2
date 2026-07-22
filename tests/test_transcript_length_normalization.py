import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csc_array
from scipy.sparse import csc_matrix
from scipy.sparse import csr_array
from scipy.sparse import csr_matrix

from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from pydeseq2.utils import fit_moments_dispersions
from pydeseq2.utils import load_example_data


def small_tximport_data():
    samples = pd.Index([f"sample{i}" for i in range(1, 7)])
    genes = pd.Index(["gene1", "gene2", "gene3"])
    counts = pd.DataFrame(
        [
            [100.2, 201.7, 302.1],
            [121.8, 181.2, 330.9],
            [89.6, 240.4, 281.2],
            [170.1, 260.8, 410.3],
            [160.7, 290.2, 389.9],
            [190.4, 250.6, 430.8],
        ],
        index=samples,
        columns=genes,
    )
    metadata = pd.DataFrame(
        {"condition": ["A", "A", "A", "B", "B", "B"]},
        index=samples,
    )
    transcript_lengths = pd.DataFrame(
        [
            [900.0, 1400.0, 2100.0],
            [950.0, 1350.0, 2050.0],
            [1000.0, 1300.0, 2000.0],
            [1250.0, 1100.0, 1850.0],
            [1300.0, 1050.0, 1800.0],
            [1350.0, 1000.0, 1750.0],
        ],
        index=samples,
        columns=genes,
    )
    return counts, metadata, transcript_lengths


def small_pytximport_adata():
    counts, metadata, transcript_lengths = small_tximport_data()
    adata = ad.AnnData(
        X=counts.to_numpy(),
        obs=metadata.copy(),
        var=pd.DataFrame(index=counts.columns),
    )
    adata.obsm["length"] = transcript_lengths.to_numpy()
    adata.uns["counts_from_abundance"] = None
    return adata, counts, metadata, transcript_lengths


def varying_transcript_lengths(counts):
    sample_effect = np.linspace(-1.0, 1.0, counts.shape[0])
    gene_effect = np.linspace(-1.0, 1.0, counts.shape[1])
    lengths = (800.0 + 100.0 * np.arange(counts.shape[1]))[None, :]
    lengths = lengths * (1.0 + 0.2 * np.outer(sample_effect, gene_effect))
    return pd.DataFrame(lengths, index=counts.index, columns=counts.columns)


def test_transcript_length_normalization_matches_deseq2_formula():
    counts, metadata, transcript_lengths = small_tximport_data()
    dds = DeseqDataSet(
        counts=counts,
        metadata=metadata,
        transcript_lengths=transcript_lengths,
        design="~condition",
        quiet=True,
    )
    dds.fit_size_factors()

    rounded_counts = np.rint(counts.to_numpy())
    lengths = transcript_lengths.to_numpy()
    length_factors = lengths / np.exp(np.mean(np.log(lengths), axis=0))
    adjusted_counts = rounded_counts / length_factors
    logmeans = np.mean(np.log(adjusted_counts), axis=0)
    size_factors = np.exp(np.median(np.log(adjusted_counts) - logmeans, axis=1))
    size_factors /= np.exp(np.mean(np.log(size_factors)))
    expected_factors = length_factors * size_factors[:, None]
    expected_factors /= np.exp(np.mean(np.log(expected_factors), axis=0))

    np.testing.assert_array_equal(dds.X, rounded_counts)
    np.testing.assert_allclose(dds.layers["avg_tx_length"], lengths)
    np.testing.assert_allclose(dds.obs["size_factors"], size_factors)
    np.testing.assert_allclose(dds.layers["normalization_factors"], expected_factors)
    np.testing.assert_allclose(
        dds.layers["normed_counts"], rounded_counts / expected_factors
    )
    np.testing.assert_allclose(
        np.exp(np.mean(np.log(dds.layers["normalization_factors"]), axis=0)),
        np.ones(dds.n_vars),
    )


def test_poscounts_transcript_length_normalization_matches_deseq2_formula():
    counts, metadata, transcript_lengths = small_tximport_data()
    for idx in range(3):
        counts.iat[idx, idx] = 0
    dds = DeseqDataSet(
        counts=counts,
        metadata=metadata,
        transcript_lengths=transcript_lengths,
        quiet=True,
    )
    dds.fit_size_factors("poscounts")

    rounded_counts = np.rint(counts.to_numpy())
    lengths = transcript_lengths.to_numpy()
    length_factors = lengths / np.exp(np.mean(np.log(lengths), axis=0))
    adjusted_counts = rounded_counts / length_factors
    log_counts = np.zeros_like(rounded_counts)
    np.log(rounded_counts, out=log_counts, where=rounded_counts != 0)
    logmeans = log_counts.mean(axis=0)
    size_factors = np.empty(dds.n_obs)
    for sample_idx in range(dds.n_obs):
        positive = adjusted_counts[sample_idx] > 0
        size_factors[sample_idx] = np.exp(
            np.median(np.log(adjusted_counts[sample_idx, positive]) - logmeans[positive])
        )
    size_factors /= np.exp(np.mean(np.log(size_factors)))
    expected_factors = length_factors * size_factors[:, None]
    expected_factors /= np.exp(np.mean(np.log(expected_factors), axis=0))

    np.testing.assert_allclose(dds.layers["normalization_factors"], expected_factors)


@pytest.mark.parametrize("bad_value", [0.0, -1.0, np.nan, np.inf])
def test_transcript_lengths_must_be_positive_and_finite(bad_value):
    counts, metadata, transcript_lengths = small_tximport_data()
    transcript_lengths.iloc[0, 0] = bad_value

    with pytest.raises(ValueError, match="transcript_lengths"):
        DeseqDataSet(
            counts=counts,
            metadata=metadata,
            transcript_lengths=transcript_lengths,
        )


def test_transcript_length_labels_must_match_counts():
    counts, metadata, transcript_lengths = small_tximport_data()
    transcript_lengths = transcript_lengths.rename(index={"sample1": "wrong_sample"})

    with pytest.raises(ValueError, match="same sample index"):
        DeseqDataSet(
            counts=counts,
            metadata=metadata,
            transcript_lengths=transcript_lengths,
        )


def test_transcript_lengths_reject_iterative_size_factors():
    counts, metadata, transcript_lengths = small_tximport_data()
    dds = DeseqDataSet(
        counts=counts,
        metadata=metadata,
        transcript_lengths=transcript_lengths,
    )

    with pytest.raises(ValueError, match="iterative"):
        dds.fit_size_factors("iterative")


def test_external_vst_transform_requires_matching_transcript_lengths():
    counts, metadata, transcript_lengths = small_tximport_data()
    dds = DeseqDataSet(
        counts=counts,
        metadata=metadata,
        transcript_lengths=transcript_lengths,
    )
    dds.fit_size_factors()

    with pytest.raises(ValueError, match="matching transcript lengths"):
        dds.vst_transform(dds.X)


@pytest.mark.parametrize("sparse_constructor", [csr_matrix, csc_matrix])
def test_transcript_length_factors_are_used_throughout_pipeline(
    monkeypatch, sparse_constructor
):
    counts = load_example_data("raw_counts", "synthetic", debug=False)
    metadata = load_example_data("metadata", "synthetic", debug=False)
    transcript_lengths = varying_transcript_lengths(counts)
    adata = ad.AnnData(
        X=sparse_constructor(counts.to_numpy()),
        obs=metadata,
        var=pd.DataFrame(index=counts.columns),
    )
    adata.obsm["length"] = transcript_lengths.to_numpy()
    adata.uns["counts_from_abundance"] = None

    dds = DeseqDataSet(
        adata=adata,
        design="~0 + condition",
        fit_type="mean",
        refit_cooks=False,
        n_cpus=1,
        quiet=True,
    )
    reference_dds = DeseqDataSet(
        counts=counts,
        metadata=metadata,
        transcript_lengths=transcript_lengths,
        design="~0 + condition",
        fit_type="mean",
        refit_cooks=False,
        n_cpus=1,
        quiet=True,
    )
    dds.deseq2()
    reference_dds.deseq2()

    dispersion_columns = ["_MoM_dispersions", "genewise_dispersions", "dispersions"]
    np.testing.assert_allclose(
        dds.var[dispersion_columns].to_numpy(),
        reference_dds.var[dispersion_columns].to_numpy(),
    )
    np.testing.assert_allclose(dds.varm["LFC"], reference_dds.varm["LFC"])

    non_zero = dds.var["non_zero"].to_numpy()
    normalization_factors = dds.layers["normalization_factors"][:, non_zero]
    design_matrix = dds.obsm["design_matrix"].to_numpy()
    coefficients = dds.varm["LFC"].loc[dds.var_names[non_zero]].to_numpy()
    expected_mu = np.maximum(
        normalization_factors * np.exp(design_matrix @ coefficients.T),
        dds.min_mu,
    )
    np.testing.assert_allclose(dds.obsm["_mu_LFC"], expected_mu)

    stats = DeseqStats(dds, contrast=["condition", "B", "A"], quiet=True)
    reference_stats = DeseqStats(
        reference_dds, contrast=["condition", "B", "A"], quiet=True
    )
    stats.summary()
    reference_stats.summary()
    assert stats.results_df["pvalue"].notna().any()
    np.testing.assert_allclose(
        stats.results_df[["stat", "pvalue", "lfcSE"]],
        reference_stats.results_df[["stat", "pvalue", "lfcSE"]],
        equal_nan=True,
    )

    coefficient = stats.LFC.columns[-1]
    stats.lfc_shrink(coeff=coefficient, adapt=False)
    reference_stats.lfc_shrink(coeff=coefficient, adapt=False)
    assert stats.shrunk_LFCs
    np.testing.assert_allclose(
        stats.LFC[coefficient], reference_stats.LFC[coefficient], equal_nan=True
    )
    monkeypatch.setattr(dds, "fit_size_factors", lambda *args, **kwargs: pytest.fail())
    dds.vst(fit_type="mean")
    reference_dds.vst(fit_type="mean")
    np.testing.assert_allclose(
        dds.layers["vst_counts"], reference_dds.layers["vst_counts"]
    )
    assert isinstance(dds.X, sparse_constructor)


@pytest.mark.parametrize(
    "sparse_constructor", [csr_matrix, csc_matrix, csr_array, csc_array]
)
def test_sparse_anndata_ratio_pipeline_matches_dense(sparse_constructor):
    counts = load_example_data("raw_counts", "synthetic", debug=False)
    metadata = load_example_data("metadata", "synthetic", debug=False)
    adata = ad.AnnData(
        X=sparse_constructor(counts.to_numpy()),
        obs=metadata.copy(),
        var=pd.DataFrame(index=counts.columns),
    )

    dds = DeseqDataSet(
        adata=adata,
        design="~0 + condition",
        fit_type="mean",
        refit_cooks=False,
        n_cpus=1,
        quiet=True,
    )
    reference_dds = DeseqDataSet(
        counts=counts,
        metadata=metadata,
        design="~0 + condition",
        fit_type="mean",
        refit_cooks=False,
        n_cpus=1,
        quiet=True,
    )
    dds.deseq2()
    reference_dds.deseq2()

    np.testing.assert_allclose(
        dds.obs["size_factors"], reference_dds.obs["size_factors"]
    )
    np.testing.assert_allclose(
        dds.layers["normed_counts"], reference_dds.layers["normed_counts"]
    )
    dispersion_columns = ["_MoM_dispersions", "genewise_dispersions", "dispersions"]
    np.testing.assert_allclose(
        dds.var[dispersion_columns], reference_dds.var[dispersion_columns]
    )
    np.testing.assert_allclose(dds.varm["LFC"], reference_dds.varm["LFC"])
    assert isinstance(dds.X, sparse_constructor)


@pytest.mark.parametrize(
    "sparse_constructor", [csr_matrix, csc_matrix, csr_array, csc_array]
)
def test_sparse_anndata_poscounts_size_factors_match_dense(sparse_constructor):
    counts, metadata, _ = small_tximport_data()
    counts = counts.round().astype(int)
    for idx in range(counts.shape[1]):
        counts.iat[idx, idx] = 0
    adata = ad.AnnData(
        X=sparse_constructor(counts.to_numpy()),
        obs=metadata.copy(),
        var=pd.DataFrame(index=counts.columns),
    )
    dds = DeseqDataSet(adata=adata, design="~condition", quiet=True)
    reference_dds = DeseqDataSet(
        counts=counts,
        metadata=metadata,
        design="~condition",
        quiet=True,
    )

    dds.fit_size_factors("poscounts")
    reference_dds.fit_size_factors("poscounts")

    np.testing.assert_allclose(
        dds.obs["size_factors"], reference_dds.obs["size_factors"]
    )
    np.testing.assert_allclose(
        dds.layers["normed_counts"], reference_dds.layers["normed_counts"]
    )
    assert isinstance(dds.layers["normed_counts"], np.ndarray)
    assert isinstance(dds.X, sparse_constructor)


@pytest.mark.parametrize(
    "sparse_constructor", [csr_matrix, csc_matrix, csr_array, csc_array]
)
def test_sparse_anndata_iterative_size_factors_match_dense(sparse_constructor):
    counts = load_example_data("raw_counts", "synthetic", debug=False).iloc[:20]
    metadata = load_example_data("metadata", "synthetic", debug=False).loc[counts.index]
    adata = ad.AnnData(
        X=sparse_constructor(counts.to_numpy()),
        obs=metadata.copy(),
        var=pd.DataFrame(index=counts.columns),
    )
    dds = DeseqDataSet(adata=adata, design="~1", n_cpus=1, quiet=True)
    reference_dds = DeseqDataSet(
        counts=counts,
        metadata=metadata,
        design="~1",
        n_cpus=1,
        quiet=True,
    )
    for dataset in (dds, reference_dds):
        dataset.logmeans = np.ones(dataset.n_vars)
        dataset.filtered_genes = np.ones(dataset.n_vars, dtype=bool)
        dataset._fit_iterate_size_factors(niter=1)

    np.testing.assert_allclose(
        dds.obs["size_factors"], reference_dds.obs["size_factors"]
    )
    np.testing.assert_allclose(
        dds.layers["normed_counts"], reference_dds.layers["normed_counts"]
    )
    assert dds.logmeans is None
    assert dds.filtered_genes is None
    assert isinstance(dds.layers["normed_counts"], np.ndarray)
    assert isinstance(dds.X, sparse_constructor)


@pytest.mark.parametrize(
    "sparse_constructor", [csr_matrix, csc_matrix, csr_array, csc_array]
)
def test_sparse_ratio_falls_back_to_iterative(monkeypatch, sparse_constructor):
    counts, metadata, _ = small_tximport_data()
    counts = counts.round().astype(int)
    for idx in range(counts.shape[1]):
        counts.iat[idx, idx] = 0
    adata = ad.AnnData(
        X=sparse_constructor(counts.to_numpy()),
        obs=metadata.copy(),
        var=pd.DataFrame(index=counts.columns),
    )
    dds = DeseqDataSet(adata=adata, design="~condition", quiet=True)
    iterative_called = False

    def fit_iterative():
        nonlocal iterative_called
        iterative_called = True
        dds.obs["size_factors"] = np.ones(dds.n_obs)
        dds.layers["normed_counts"] = dds.X.toarray()
        dds.logmeans = None
        dds.filtered_genes = None

    monkeypatch.setattr(dds, "_fit_iterate_size_factors", fit_iterative)

    with pytest.warns(UserWarning, match="Switching to iterative mode"):
        dds.fit_size_factors("ratio")

    assert iterative_called


def test_moments_dispersions_match_deseq2_with_matrix_factors():
    counts, metadata, transcript_lengths = small_tximport_data()
    dds = DeseqDataSet(
        counts=counts,
        metadata=metadata,
        transcript_lengths=transcript_lengths,
        quiet=True,
    )
    dds.fit_size_factors()

    normalization_factors = dds.layers["normalization_factors"]
    normed_counts = dds.layers["normed_counts"]
    mean_inverse_factor = np.mean(1 / normalization_factors.mean(axis=1))
    means = normed_counts.mean(axis=0)
    variances = normed_counts.var(axis=0, ddof=1)
    expected = np.nan_to_num((variances - mean_inverse_factor * means) / means**2)

    np.testing.assert_allclose(
        fit_moments_dispersions(normed_counts, normalization_factors),
        expected,
    )


def test_moments_dispersions_align_matrix_factors_after_zero_gene_filtering():
    normed_counts = np.array(
        [
            [0.0, 10.0],
            [0.0, 14.0],
            [0.0, 18.0],
        ]
    )
    normalization_factors = np.array(
        [
            [100.0, 1.0],
            [100.0, 1.0],
            [100.0, 1.0],
        ]
    )

    np.testing.assert_allclose(
        fit_moments_dispersions(normed_counts, normalization_factors),
        np.array([2.0 / 196.0]),
    )


def test_sparse_anndata_with_stored_transcript_lengths():
    counts, metadata, transcript_lengths = small_tximport_data()
    adata = ad.AnnData(
        X=csr_matrix(counts.to_numpy()),
        obs=metadata,
        var=pd.DataFrame(index=counts.columns),
    )
    adata.layers["avg_tx_length"] = transcript_lengths.to_numpy()

    dds = DeseqDataSet(adata=adata, quiet=True)
    dds.fit_size_factors()

    np.testing.assert_array_equal(dds.X.toarray(), np.rint(counts.to_numpy()))
    assert "normalization_factors" in dds.layers
    np.testing.assert_array_equal(adata.X.toarray(), counts.to_numpy())
    np.testing.assert_allclose(
        adata.layers["avg_tx_length"], transcript_lengths.to_numpy()
    )
    assert "design_matrix" not in adata.obsm
    assert "normalization_factors" not in adata.layers
    assert "normed_counts" not in adata.layers


def test_sparse_estimated_counts_round_after_summing_duplicates():
    counts = csr_matrix(
        (np.array([0.4, 0.4, 0.4]), np.array([0, 0, 1]), np.array([0, 2, 3])),
        shape=(2, 2),
    )
    adata = ad.AnnData(X=counts)
    adata.obsm["length"] = np.ones((2, 2))
    adata.uns["counts_from_abundance"] = None

    dds = DeseqDataSet(adata=adata, design="~1", quiet=True)

    np.testing.assert_array_equal(dds.X.toarray(), np.array([[1, 0], [0, 0]]))
    assert dds.X.nnz == 1
    assert adata.X.nnz == 3


def test_matching_integer_transcript_length_labels_are_accepted():
    counts, metadata, transcript_lengths = small_tximport_data()
    integer_samples = pd.Index(range(1, len(counts) + 1))
    integer_genes = pd.Index(range(101, 101 + counts.shape[1]))
    counts.index = integer_samples
    counts.columns = integer_genes
    metadata.index = integer_samples
    transcript_lengths.index = integer_samples
    transcript_lengths.columns = integer_genes

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ad.ImplicitModificationWarning)
        dds = DeseqDataSet(
            counts=counts,
            metadata=metadata,
            transcript_lengths=transcript_lengths,
            quiet=True,
        )

    np.testing.assert_allclose(
        dds.layers["avg_tx_length"], transcript_lengths.to_numpy()
    )


def test_scalar_size_factor_refit_clears_stale_matrix_factors():
    counts, metadata, _ = small_tximport_data()
    counts = counts.round().astype(int)
    dds = DeseqDataSet(counts=counts, metadata=metadata, quiet=True)
    dds.layers["normalization_factors"] = np.full(dds.shape, 2.0)

    dds.fit_size_factors()

    assert "normalization_factors" not in dds.layers
    np.testing.assert_allclose(
        dds.layers["normed_counts"],
        dds.X / dds.obs["size_factors"].to_numpy()[:, None],
    )


def test_explicit_transcript_lengths_clear_inherited_normalization_state():
    counts, metadata, transcript_lengths = small_tximport_data()
    fitted_dds = DeseqDataSet(
        counts=counts,
        metadata=metadata,
        transcript_lengths=transcript_lengths,
        quiet=True,
    )
    fitted_dds.fit_size_factors()
    original_size_factors = fitted_dds.obs["size_factors"].copy()
    original_normalization_factors = fitted_dds.layers["normalization_factors"].copy()
    original_normed_counts = fitted_dds.layers["normed_counts"].copy()

    replacement_lengths = transcript_lengths.copy()
    replacement_lengths.iloc[:, 0] *= np.linspace(1.0, 2.0, len(replacement_lengths))
    dds = DeseqDataSet(
        adata=fitted_dds,
        transcript_lengths=replacement_lengths,
        quiet=True,
    )

    assert "size_factors" not in dds.obs
    assert "normalization_factors" not in dds.layers
    assert "normed_counts" not in dds.layers
    np.testing.assert_allclose(fitted_dds.obs["size_factors"], original_size_factors)
    np.testing.assert_allclose(
        fitted_dds.layers["normalization_factors"],
        original_normalization_factors,
    )
    np.testing.assert_allclose(
        fitted_dds.layers["normed_counts"], original_normed_counts
    )

    reference_dds = DeseqDataSet(
        counts=counts,
        metadata=metadata,
        transcript_lengths=replacement_lengths,
        quiet=True,
    )
    dds.fit_genewise_dispersions()
    reference_dds.fit_size_factors()

    np.testing.assert_allclose(
        dds.obs["size_factors"], reference_dds.obs["size_factors"]
    )
    np.testing.assert_allclose(
        dds.layers["normalization_factors"],
        reference_dds.layers["normalization_factors"],
    )
    np.testing.assert_allclose(
        dds.layers["normed_counts"], reference_dds.layers["normed_counts"]
    )


def test_explicit_transcript_lengths_clear_inherited_fitted_state():
    counts = load_example_data("raw_counts", "synthetic", debug=False)
    metadata = load_example_data("metadata", "synthetic", debug=False)
    transcript_lengths = varying_transcript_lengths(counts)
    fitted_dds = DeseqDataSet(
        counts=counts,
        metadata=metadata,
        transcript_lengths=transcript_lengths,
        design="~0 + condition",
        fit_type="mean",
        refit_cooks=False,
        n_cpus=1,
        quiet=True,
    )
    fitted_dds.deseq2()

    replacement_lengths = transcript_lengths.copy()
    replacement_lengths.iloc[:, 0] *= np.linspace(1.0, 2.0, len(replacement_lengths))
    dds = DeseqDataSet(
        adata=fitted_dds,
        transcript_lengths=replacement_lengths,
        design="~0 + condition",
        fit_type="mean",
        refit_cooks=False,
        n_cpus=1,
        quiet=True,
    )

    assert set(dds.obs.columns).isdisjoint({"size_factors", "replaceable"})
    assert set(dds.var.columns).isdisjoint(
        {
            "_normed_means",
            "non_zero",
            "_MoM_dispersions",
            "genewise_dispersions",
            "vst_genewise_dispersions",
            "_genewise_converged",
            "fitted_dispersions",
            "MAP_dispersions",
            "_MAP_converged",
            "dispersions",
            "_outlier_genes",
            "_LFC_converged",
            "_pvalue_cooks_outlier",
            "replaced",
            "refitted",
        }
    )
    assert set(dds.obsm).isdisjoint({"_mu_LFC", "_hat_diagonals"})
    assert "LFC" not in dds.varm
    assert set(dds.layers).isdisjoint(
        {
            "normalization_factors",
            "normed_counts",
            "_mu_hat",
            "_vst_mu_hat",
            "cooks",
            "replace_cooks",
            "vst_counts",
        }
    )
    assert set(dds.uns).isdisjoint(
        {
            "trend_coeffs",
            "vst_trend_coeffs",
            "mean_disp",
            "disp_function_type",
            "_squared_logres",
            "prior_disp_var",
        }
    )
    assert "size_factors" in fitted_dds.obs
    assert "genewise_dispersions" in fitted_dds.var
    assert "_mu_LFC" in fitted_dds.obsm
    assert "LFC" in fitted_dds.varm
    assert "cooks" in fitted_dds.layers
    assert "mean_disp" in fitted_dds.uns

    reference_dds = DeseqDataSet(
        counts=counts,
        metadata=metadata,
        transcript_lengths=replacement_lengths,
        design="~0 + condition",
        fit_type="mean",
        refit_cooks=False,
        n_cpus=1,
        quiet=True,
    )
    dds.fit_dispersion_trend()
    reference_dds.fit_dispersion_trend()
    dds.fit_LFC()
    reference_dds.fit_LFC()

    np.testing.assert_allclose(
        dds.layers["normalization_factors"],
        reference_dds.layers["normalization_factors"],
    )
    np.testing.assert_allclose(dds.var["dispersions"], reference_dds.var["dispersions"])
    np.testing.assert_allclose(dds.varm["LFC"], reference_dds.varm["LFC"])


@pytest.mark.parametrize("sparse_constructor", [csr_matrix, csc_matrix])
def test_cooks_outlier_refit_preserves_matrix_factors(sparse_constructor):
    counts = load_example_data("raw_counts", "synthetic", debug=False)
    metadata = load_example_data("metadata", "synthetic", debug=False)
    counts.iloc[0, 0] *= 10_000
    transcript_lengths = varying_transcript_lengths(counts)
    adata = ad.AnnData(
        X=sparse_constructor(counts.to_numpy()),
        obs=metadata,
        var=pd.DataFrame(index=counts.columns),
    )
    adata.obsm["length"] = transcript_lengths.to_numpy()
    adata.uns["counts_from_abundance"] = None

    dds = DeseqDataSet(
        adata=adata,
        design="~0 + condition",
        fit_type="mean",
        refit_cooks=True,
        min_replicates=3,
        n_cpus=1,
        quiet=True,
    )
    reference_dds = DeseqDataSet(
        counts=counts,
        metadata=metadata,
        transcript_lengths=transcript_lengths,
        design="~0 + condition",
        fit_type="mean",
        refit_cooks=True,
        min_replicates=3,
        n_cpus=1,
        quiet=True,
    )
    dds.deseq2()
    reference_dds.deseq2()

    assert dds.var["replaced"].sum() == 1
    assert dds.var["refitted"].sum() == 1
    np.testing.assert_array_equal(
        dds.var[["replaced", "refitted"]],
        reference_dds.var[["replaced", "refitted"]],
    )
    np.testing.assert_allclose(
        dds.layers["cooks"],
        reference_dds.layers["cooks"],
        rtol=1e-7,
        atol=1e-10,
        equal_nan=True,
    )
    np.testing.assert_array_equal(dds.counts_to_refit.X, reference_dds.counts_to_refit.X)
    refitted = dds.var["refitted"].to_numpy()
    np.testing.assert_allclose(
        dds.counts_to_refit.layers["normalization_factors"],
        dds.layers["normalization_factors"][:, refitted],
    )
    np.testing.assert_allclose(
        dds.var.loc[refitted, ["genewise_dispersions", "dispersions"]],
        reference_dds.var.loc[refitted, ["genewise_dispersions", "dispersions"]],
    )
    np.testing.assert_allclose(
        dds.varm["LFC"].loc[dds.var_names[refitted]],
        reference_dds.varm["LFC"].loc[reference_dds.var_names[refitted]],
    )
    assert isinstance(dds.X, sparse_constructor)


@pytest.mark.parametrize("sparse_counts", [False, True])
def test_pytximport_anndata_matches_explicit_transcript_lengths(sparse_counts):
    adata, counts, metadata, transcript_lengths = small_pytximport_adata()
    if sparse_counts:
        adata.X = csr_matrix(adata.X)

    pytximport_dds = DeseqDataSet(adata=adata, design="~condition", quiet=True)
    explicit_dds = DeseqDataSet(
        counts=counts,
        metadata=metadata,
        transcript_lengths=transcript_lengths,
        design="~condition",
        quiet=True,
    )
    pytximport_dds.fit_size_factors()
    explicit_dds.fit_size_factors()

    pytximport_counts = pytximport_dds.X.toarray() if sparse_counts else pytximport_dds.X
    np.testing.assert_array_equal(pytximport_counts, explicit_dds.X)
    np.testing.assert_allclose(
        pytximport_dds.layers["avg_tx_length"],
        explicit_dds.layers["avg_tx_length"],
    )
    np.testing.assert_allclose(
        pytximport_dds.obs["size_factors"], explicit_dds.obs["size_factors"]
    )
    np.testing.assert_allclose(
        pytximport_dds.layers["normalization_factors"],
        explicit_dds.layers["normalization_factors"],
    )
    np.testing.assert_allclose(
        pytximport_dds.layers["normed_counts"],
        explicit_dds.layers["normed_counts"],
    )
    assert isinstance(pytximport_dds.X, csr_matrix) is sparse_counts


def test_pytximport_backed_anndata_requires_memory(tmp_path):
    adata = ad.AnnData(X=np.ones((2, 2)))
    adata.obsm["length"] = np.ones((2, 2))
    adata.uns["counts_from_abundance"] = None
    path = tmp_path / "backed.h5ad"
    adata.write_h5ad(path)
    backed = ad.read_h5ad(path, backed="r")

    try:
        with pytest.raises(ValueError, match="to_memory"):
            DeseqDataSet(adata=backed, quiet=True)
    finally:
        backed.file.close()


def test_pytximport_dataframe_length_labels_are_preserved():
    adata, _, _, transcript_lengths = small_pytximport_adata()
    adata.obsm["length"] = transcript_lengths.copy()

    dds = DeseqDataSet(adata=adata, quiet=True)

    assert dds.obs_names.equals(transcript_lengths.index)
    assert dds.var_names.equals(transcript_lengths.columns)
    np.testing.assert_allclose(
        dds.layers["avg_tx_length"], transcript_lengths.to_numpy()
    )


def test_pytximport_dataframe_gene_labels_must_match():
    adata, _, _, transcript_lengths = small_pytximport_adata()
    adata.obsm["length"] = transcript_lengths.rename(columns={"gene1": "wrong_gene"})

    with pytest.raises(ValueError, match="same gene columns"):
        DeseqDataSet(adata=adata, quiet=True)


@pytest.mark.parametrize(
    "mode",
    [
        "scaled_tpm",
        "length_scaled_tpm",
        "dtu_scaled_tpm",
        "scaledTPM",
        "lengthScaledTPM",
    ],
)
def test_pytximport_scaled_count_modes_are_rejected(mode):
    adata, _, _, _ = small_pytximport_adata()
    adata.uns["counts_from_abundance"] = mode

    with pytest.raises(ValueError, match="unscaled estimated counts"):
        DeseqDataSet(adata=adata, quiet=True)


@pytest.mark.parametrize("higher_precedence_source", ["explicit", "canonical"])
def test_scaled_pytximport_mode_rejects_higher_precedence_lengths(
    higher_precedence_source,
):
    adata, _, _, transcript_lengths = small_pytximport_adata()
    adata.uns["counts_from_abundance"] = "length_scaled_tpm"
    del adata.obsm["length"]
    explicit_lengths = None
    if higher_precedence_source == "explicit":
        explicit_lengths = transcript_lengths
    else:
        adata.layers["avg_tx_length"] = transcript_lengths.to_numpy()

    with pytest.raises(ValueError, match="must not be combined"):
        DeseqDataSet(
            adata=adata,
            transcript_lengths=explicit_lengths,
            quiet=True,
        )


@pytest.mark.parametrize("bad_value", [0.0, np.nan, np.inf])
def test_pytximport_lengths_must_be_positive_and_finite(bad_value):
    adata, _, _, _ = small_pytximport_adata()
    lengths = adata.obsm["length"].copy()
    lengths[0, 0] = bad_value
    adata.obsm["length"] = lengths

    with pytest.raises(ValueError, match="transcript_lengths"):
        DeseqDataSet(adata=adata, quiet=True)


def test_pytximport_lengths_reject_unsynchronized_gene_subsetting():
    adata, _, _, _ = small_pytximport_adata()
    subset = adata[:, :2].copy()

    with pytest.raises(ValueError, match="same shape"):
        DeseqDataSet(adata=subset, quiet=True)


@pytest.mark.parametrize("use_explicit_lengths", [False, True])
def test_transcript_length_source_precedence(use_explicit_lengths):
    adata, _, _, transcript_lengths = small_pytximport_adata()
    canonical_lengths = transcript_lengths.to_numpy() + 100.0
    explicit_lengths = transcript_lengths + 200.0
    adata.layers["avg_tx_length"] = canonical_lengths

    dds = DeseqDataSet(
        adata=adata,
        transcript_lengths=explicit_lengths if use_explicit_lengths else None,
        quiet=True,
    )

    expected = explicit_lengths.to_numpy() if use_explicit_lengths else canonical_lengths
    np.testing.assert_allclose(dds.layers["avg_tx_length"], expected)


def test_pytximport_length_without_mode_is_not_auto_detected():
    adata, counts, _, _ = small_pytximport_adata()
    del adata.uns["counts_from_abundance"]
    adata.X = np.rint(counts.to_numpy())

    dds = DeseqDataSet(adata=adata, quiet=True)

    assert "avg_tx_length" not in dds.layers


def test_counts_from_abundance_without_lengths_is_not_pytximport():
    adata, counts, _, _ = small_pytximport_adata()
    del adata.obsm["length"]
    adata.X = np.rint(counts.to_numpy())
    adata.uns["counts_from_abundance"] = "scaled_tpm"

    dds = DeseqDataSet(adata=adata, quiet=True)

    assert "avg_tx_length" not in dds.layers


def test_pytximport_input_anndata_is_not_mutated():
    adata, counts, _, transcript_lengths = small_pytximport_adata()
    original_obs = adata.obs.copy()
    connectivities = np.eye(adata.n_obs)
    adata.uns["neighbors"] = {"connectivities": connectivities}

    dds = DeseqDataSet(adata=adata, quiet=True)
    dds.fit_size_factors()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        picklable_adata = dds.to_picklable_anndata()

    np.testing.assert_array_equal(adata.X, counts.to_numpy())
    pd.testing.assert_frame_equal(adata.obs, original_obs)
    assert dds.var is not adata.var
    assert adata.uns["neighbors"]["connectivities"] is connectivities
    assert "connectivities" in picklable_adata.obsp
    np.testing.assert_array_equal(adata.obsm["length"], transcript_lengths.to_numpy())
    assert "design_matrix" not in adata.obsm
    assert not adata.layers
