import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
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


def test_transcript_length_factors_are_used_throughout_pipeline():
    counts = load_example_data("raw_counts", "synthetic", debug=False)
    metadata = load_example_data("metadata", "synthetic", debug=False)
    transcript_lengths = varying_transcript_lengths(counts)

    dds = DeseqDataSet(
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
    stats.summary()
    assert stats.results_df["pvalue"].notna().any()

    coefficient = stats.LFC.columns[-1]
    stats.lfc_shrink(coeff=coefficient, adapt=False)
    assert stats.shrunk_LFCs
    assert stats.LFC[coefficient].notna().any()


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


def test_cooks_outlier_refit_preserves_matrix_factors():
    counts = load_example_data("raw_counts", "synthetic", debug=False)
    metadata = load_example_data("metadata", "synthetic", debug=False)
    counts.iloc[0, 0] *= 10_000
    transcript_lengths = varying_transcript_lengths(counts)

    dds = DeseqDataSet(
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

    assert dds.var["replaced"].sum() == 1
    assert dds.var["refitted"].sum() == 1
    refitted = dds.var["refitted"].to_numpy()
    np.testing.assert_allclose(
        dds.counts_to_refit.layers["normalization_factors"],
        dds.layers["normalization_factors"][:, refitted],
    )
    assert np.isfinite(dds.varm["LFC"].loc[dds.var_names[refitted]].to_numpy()).all()
