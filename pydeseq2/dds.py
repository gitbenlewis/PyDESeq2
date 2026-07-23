import hashlib
import sys
import time
import warnings
from collections.abc import Mapping
from typing import Any
from typing import Literal
from typing import Protocol
from typing import cast

import anndata as ad  # type: ignore
import numpy as np
import pandas as pd
from formulaic_contrasts import FormulaicContrasts  # type: ignore[import-untyped]
from scipy.optimize import minimize
from scipy.sparse import csr_matrix  # type: ignore
from scipy.sparse import issparse  # type: ignore
from scipy.special import polygamma  # type: ignore
from scipy.stats import chi2  # type: ignore
from scipy.stats import f  # type: ignore
from scipy.stats import trim_mean  # type: ignore

from pydeseq2.default_inference import DefaultInference
from pydeseq2.inference import Inference
from pydeseq2.preprocessing import deseq2_norm_fit
from pydeseq2.preprocessing import deseq2_norm_transform
from pydeseq2.utils import dispersion_trend
from pydeseq2.utils import make_scatter
from pydeseq2.utils import mean_absolute_deviation
from pydeseq2.utils import n_or_more_replicates
from pydeseq2.utils import nb_nll
from pydeseq2.utils import robust_method_of_moments_disp
from pydeseq2.utils import test_valid_counts
from pydeseq2.utils import trimmed_mean

# Ignore AnnData's FutureWarning about implicit data conversion.
warnings.simplefilter("ignore", FutureWarning)


def _rounded_counts(counts: Any) -> Any:
    """Round dense or sparse estimated counts without mutating the input."""
    if isinstance(counts, pd.DataFrame):
        return counts.round()
    if isinstance(counts, np.ndarray):
        return np.rint(counts)
    if counts.format in {"csr", "csc"}:
        rounded = counts.copy()
    else:
        rounded = counts.tocsr()
    rounded.sum_duplicates()
    rounded.data = np.rint(rounded.data)
    rounded.eliminate_zeros()
    return rounded


_LRT_VAR_KEYS = (
    "_lrt_statistic",
    "_lrt_pvalue",
    "_lrt_full_deviance",
    "_lrt_reduced_converged",
    "_lrt_full_converged",
    "_lrt_new_all_zero",
)

_LRT_FULL_LFC_KEY = "_lrt_full_LFC"
_LRT_CACHE_VERSION = 4
_LRT_GENE_BLOCK_SIZE = 128
_REPLACE_COUNTS_LAYER = "_pydeseq2_cook_replacement_delta"
_REPLACE_COUNTS_OWNER_KEY = "_pydeseq2_replace_counts_owned"

_LRTResult = tuple[
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.DataFrame,
]


class _LRTDigest(Protocol):
    """Minimal structural type implemented by hashlib digest objects."""

    def update(self, data: bytes | bytearray | memoryview, /) -> None:
        """Add bytes to the digest state."""


class DeseqDataSet(ad.AnnData):
    r"""A class to implement dispersion and log fold-change (LFC) estimation.

    The DeseqDataSet extends the `AnnData class
    <https://anndata.readthedocs.io/en/latest/generated/anndata.AnnData.html#anndata.AnnData>`_.
    As such, it implements the same methods and attributes, in addition to those that are
    specific to pydeseq2.
    Dispersions and LFCs are estimated following the DESeq2 pipeline
    :cite:p:`DeseqDataSet-love2014moderated`.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData from which to initialize the DeseqDataSet. Must have counts ('X') and
        sample metadata ('obs') fields. If ``None``, both ``counts`` and ``metadata``
        arguments must be provided. Compatible pytximport objects with unscaled
        estimated counts are detected from ``obsm["length"]`` and
        ``uns["counts_from_abundance"] is None``.
        Backed AnnData objects are not supported; call ``adata.to_memory()`` before
        initialization.

    counts : pandas.DataFrame
        Raw counts. One column per gene, rows are indexed by sample barcodes.

    metadata : pandas.DataFrame
        DataFrame containing sample metadata.
        Must be indexed by sample barcodes.

    transcript_lengths : pandas.DataFrame or numpy.ndarray, optional
        Average transcript lengths for each sample and gene, typically imported from
        transcript-level quantification with unscaled estimated counts (tximport's
        ``countsFromAbundance="no"`` mode). Scaled TPM-derived counts must not be paired
        with these offsets. Must have the same sample-by-gene shape and ordering as
        ``counts``. When provided, PyDESeq2
        constructs gene- and sample-specific normalization factors that correct for
        transcript-length changes as well as library size. Estimated counts are rounded
        to the nearest integer, matching DESeq2. Explicit lengths take precedence over
        ``adata.layers["avg_tx_length"]``, which takes precedence over compatible
        pytximport fields. (default: ``None``).

    design : str or pandas.DataFrame
        Model design. Can be either a pandas DataFrame representing a design matrix, or
        a formulaic formula in the format ``'x + z'`` or ``'~x+z'``.
        If a design matrix is provided, DeseqStats built from this DeseqDataSet will
        only support contrasts in the form of numeric vectors.
        (Default: ``'~condition')``.

    design_factors : str or list, optional
        Depecated. An optional list of factors to include in the design matrix.
        Will be removed in a future release. (default: ``None``).

    continuous_factors : list, optional
        Deprecated. Continuous factors are now automatically detected from the design,
        or cast to categorical using the C() operator in the formula.
        (default: ``None``).

    ref_level : list, optional
        Deprecated.

    fit_type: str
        Either ``"parametric"`` or ``"mean"`` for the type of fitting of dispersions to
        the mean intensity. ``"parametric"``: fit a dispersion-mean relation via a
        robust gamma-family GLM. ``"mean"``: use the mean of gene-wise dispersion
        estimates. Will set the fit type for the DEA and the vst transformation. If
        needed, it can be set separately for each method.(default: ``"parametric"``).

    size_factors_fit_type : str
        The normalization method to use: ``"ratio"``, ``"poscounts"`` or ``"iterative"``.
        ``"ratio"``: fit size factors using the median-of-ratios method. ``"poscounts"``:
        fit size factors using the method implemented in DESeq2 for the case where there
        may be few or no genes which have no zero values.
        ``"iterative"``: fit size factors iteratively. (default: ``"ratio"``).

    control_genes : ndarray, list, or pandas.Index, optional
        Genes to use as control genes for size factor fitting. If provided, size factors
        will be fit using only these genes. This is useful when certain genes are known
        to be invariant across conditions (e.g., housekeeping genes). Any valid AnnData
        indexer (bool array, integer positions, or gene name strings) can be used.
        (default: ``None``).

    min_mu : float
        Threshold for mean estimates. (default: ``0.5``).

    min_disp : float
        Lower threshold for dispersion parameters. (default: ``1e-8``).

    max_disp : float
        Upper threshold for dispersion parameters.
        Note: The threshold that is actually enforced is max(max_disp, len(counts)).
        (default: ``10``).

    refit_cooks : bool
        Whether to refit cooks outliers. (default: ``True``).

    min_replicates : int
        Minimum number of replicates a condition should have
        to allow refitting its samples. (default: ``7``).

    beta_tol : float
        Stopping criterion for IRWLS. (default: ``1e-8``).

        .. math:: \vert dev_t - dev_{t+1}\vert / (\vert dev \vert + 0.1) < \beta_{tol}.

    n_cpus : int
        Number of cpus to use.  If ``None`` and if ``inference`` is not provided, all
        available cpus will be used by the ``DefaultInference``. If both are specified
        (i.e., ``n_cpus`` and ``inference`` are not ``None``), it will try to override
        the ``n_cpus`` attribute of the ``inference`` object. (default: ``None``).

    inference : Inference
        Implementation of inference routines object instance.
        (default:
        :class:`DefaultInference <pydeseq2.default_inference.DefaultInference>`).

    quiet : bool
        Suppress deseq2 status updates during fit.

    low_memory : bool
        Remove intermediate data structures from .layers and from .obsm that are no
        longer necessary after they are used during deseq2 run, such as Cook's
        distances. (default: False)

    Attributes
    ----------
    X
        A ‘number of samples’ x ‘number of genes’ count data matrix.

    obs
        Key-indexed one-dimensional observations annotation of length 'number of
        samples". Used to store design factors.

    var
        Key-indexed one-dimensional gene-level annotation of length ‘number of genes’.

    uns
        Key-indexed unstructured annotation.

    obsm
        Key-indexed multi-dimensional observations annotation of length
        ‘number of samples’. Stores "design_matrix" and "size_factors", among others.

    varm
        Key-indexed multi-dimensional gene annotation of length ‘number of genes’.
        Stores "dispersions" and "LFC", among others.

    layers
        Key-indexed multi-dimensional arrays aligned to dimensions of `X`, e.g. "cooks".
        Average transcript lengths and the resulting factors are stored as
        ``"avg_tx_length"`` and ``"normalization_factors"``, respectively.

    n_processes : int
        Number of cpus to use for multiprocessing.

    non_zero_idx : ndarray
        Indices of genes that have non-uniformly zero counts.

    non_zero_genes : pandas.Index
        Index of genes that have non-uniformly zero counts.

    counts_to_refit : anndata.AnnData
        Read counts after replacement, containing only genes
        for which dispersions and LFCs must be fitted again.

    new_all_zeroes_genes : pandas.Index
        Genes which have only zero counts after outlier replacement.

    quiet : bool
        Suppress deseq2 status updates during fit.

    logmeans: numpy.ndarray
        Gene-wise mean log counts, computed in ``preprocessing.deseq2_norm_fit()``.

    filtered_genes: numpy.ndarray
        Genes whose log means are different from -∞, computed in
        preprocessing.deseq2_norm_fit().

    factor_storage : dict
        A dictionary storing metadata for each factor processed by the custom
        materializer (only if ``design`` is input as a formula).

    variable_to_factors : dict
        A dictionary mapping variable names to factor names (only if ``design`` is input
        as a formula).

    References
    ----------
    .. bibliography::
        :keyprefix: DeseqDataSet-

    """

    def __init__(
        self,
        *,
        adata: ad.AnnData | None = None,
        counts: pd.DataFrame | None = None,
        metadata: pd.DataFrame | None = None,
        transcript_lengths: pd.DataFrame | np.ndarray | None = None,
        design: str | pd.DataFrame = "~condition",
        design_factors: str | list[str] | None = None,
        continuous_factors: list[str] | None = None,
        ref_level: list[str] | None = None,
        fit_type: Literal["parametric", "mean"] = "parametric",
        size_factors_fit_type: Literal["ratio", "poscounts", "iterative"] = "ratio",
        control_genes: np.ndarray | list[str] | list[int] | pd.Index | None = None,
        min_mu: float = 0.5,
        min_disp: float = 1e-8,
        max_disp: float = 10.0,
        refit_cooks: bool = True,
        min_replicates: int = 7,
        beta_tol: float = 1e-8,
        n_cpus: int | None = None,
        inference: Inference | None = None,
        quiet: bool = False,
        low_memory: bool = False,
    ) -> None:
        # Initialize the AnnData part
        has_stored_transcript_lengths = (
            adata is not None and "avg_tx_length" in adata.layers
        )
        selected_transcript_lengths: Any = transcript_lengths
        has_transcript_lengths = (
            selected_transcript_lengths is not None or has_stored_transcript_lengths
        )
        preserve_inherited_fit = False
        if adata is not None:
            has_pytximport_fields = (
                "length" in adata.obsm and "counts_from_abundance" in adata.uns
            )
            has_transcript_lengths = has_transcript_lengths or has_pytximport_fields
            counts_from_abundance = adata.uns.get("counts_from_abundance")
            if counts_from_abundance is not None and has_transcript_lengths:
                raise ValueError(
                    "pytximport transcript-length offsets require unscaled "
                    "estimated counts with "
                    "adata.uns['counts_from_abundance'] set to None; got "
                    f"{counts_from_abundance!r}. Abundance-scaled counts must "
                    "not be combined with transcript-length offsets."
                )
            if (
                has_pytximport_fields
                and selected_transcript_lengths is None
                and not has_stored_transcript_lengths
            ):
                selected_transcript_lengths = adata.obsm["length"]

            preserve_inherited_fit = (
                selected_transcript_lengths is None
                and has_stored_transcript_lengths
                and "normalization_factors" in adata.layers
            )

            if adata.isbacked:
                raise ValueError(
                    "DeseqDataSet requires an in-memory AnnData object. Call "
                    "adata.to_memory() before initialization."
                )
            expected_length_index = adata.obs_names
            expected_length_columns = adata.var_names
            if counts is not None:
                warnings.warn(
                    "adata was provided; ignoring counts.", UserWarning, stacklevel=2
                )
            if metadata is not None:
                warnings.warn(
                    "adata was provided; ignoring metadata.", UserWarning, stacklevel=2
                )
            prepared_counts: Any = adata.X
            if has_transcript_lengths:
                prepared_counts = _rounded_counts(prepared_counts)
            elif issparse(prepared_counts):
                sparse_counts = cast(Any, prepared_counts)
                if sparse_counts.format not in {"csr", "csc"}:
                    prepared_counts = sparse_counts.tocsr()
                    prepared_counts.sum_duplicates()
                elif not sparse_counts.has_canonical_format:
                    prepared_counts = sparse_counts.copy()
                    prepared_counts.sum_duplicates()
            # Test counts before going further
            test_valid_counts(prepared_counts)
            integer_counts = prepared_counts.astype(int)
            if has_transcript_lengths:
                # Own the containers PyDESeq2 writes while retaining references to
                # existing large aligned matrices.
                # AnnData migrates legacy neighbor matrices out of uns in place.
                owned_uns = dict(adata.uns)
                if "neighbors" in owned_uns:
                    owned_uns["neighbors"] = dict(owned_uns["neighbors"])
                super().__init__(
                    X=integer_counts,
                    obs=cast(pd.DataFrame, adata.obs).copy(),
                    var=cast(pd.DataFrame, adata.var).copy(),
                    uns=None,
                    obsm=cast(Any, dict(adata.obsm)),
                    varm=cast(Any, dict(adata.varm)),
                    obsp=cast(Any, dict(adata.obsp)),
                    varp=cast(Any, dict(adata.varp)),
                    # AnnData 0.13 can expose X as a None-keyed layer item.
                    layers={
                        key: value
                        for key, value in adata.layers.items()
                        if key is not None
                    },
                    raw=cast(Any, adata.raw),
                )
                self.uns.update(owned_uns)
            else:
                self.__dict__.update(adata.__dict__)
                # AnnData 0.13 stores X under the None layer key. Detach the
                # container so assigning self.X does not modify the input AnnData.
                if None in self.__dict__.get("_layers", {}):
                    self.__dict__["_layers"] = self.__dict__["_layers"].copy()
                self.X = integer_counts
        elif counts is not None and metadata is not None:
            expected_length_index = counts.index
            expected_length_columns = counts.columns
            prepared_counts = (
                _rounded_counts(counts) if has_transcript_lengths else counts
            )
            # Test counts before going further
            test_valid_counts(prepared_counts)
            super().__init__(X=prepared_counts.astype(int), obs=metadata)
        else:
            raise ValueError(
                "Either adata or both counts and metadata arguments must be provided."
            )

        if selected_transcript_lengths is not None:
            if isinstance(selected_transcript_lengths, pd.DataFrame):
                if not selected_transcript_lengths.index.equals(expected_length_index):
                    raise ValueError(
                        "transcript_lengths must have the same sample index as counts."
                    )
                if not selected_transcript_lengths.columns.equals(
                    expected_length_columns
                ):
                    raise ValueError(
                        "transcript_lengths must have the same gene columns as counts."
                    )
                transcript_lengths_array = selected_transcript_lengths.to_numpy(
                    dtype=float, copy=True
                )
            else:
                transcript_lengths_array = np.array(
                    selected_transcript_lengths, dtype=float, copy=True
                )
            self._validate_transcript_lengths(transcript_lengths_array)
            self.layers["avg_tx_length"] = transcript_lengths_array
        elif "avg_tx_length" in self.layers:
            stored_lengths = self.layers["avg_tx_length"]
            transcript_lengths_array = (
                cast(Any, stored_lengths).toarray()
                if hasattr(stored_lengths, "toarray")
                else np.asarray(stored_lengths, dtype=float)
            )
            self._validate_transcript_lengths(transcript_lengths_array)
            self.layers["avg_tx_length"] = transcript_lengths_array

        self.fit_type = fit_type
        self.design = design

        if continuous_factors is not None:
            warnings.warn(
                "continuous_factors is deprecated and will soon be removed."
                "Continuous factors are now automatically detected from the design,"
                "or can be cast to categorical using the C() operator in the formula",
                DeprecationWarning,
                stacklevel=2,
            )

        if ref_level is not None:
            warnings.warn(
                "ref_level is deprecated and no longer has any effect. It will be"
                "removed in a future release.",
                DeprecationWarning,
                stacklevel=2,
            )

        if design_factors is not None:
            warnings.warn(
                "design_factors is deprecated and will soon be removed."
                "Please consider providing a formulaic formula using the design argument"
                "instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            design_factors = (
                design_factors if isinstance(design_factors, list) else [design_factors]
            )
            self.design = "~" + " + ".join(design_factors)

        if not (
            isinstance(self.design, (str | pd.DataFrame)) or isinstance(self.design, str)
        ):
            raise ValueError(
                "design must be a string representing a formulaic formula,"
                "or a pandas DataFrame."
            )

        if isinstance(self.design, str):
            # Keep track of the categorical factors used in the model specification,
            # including variable and factor names, by generating a custom materializer.
            self.formulaic_contrasts = FormulaicContrasts(self.obs, self.design)
            self.obsm["design_matrix"] = self.formulaic_contrasts.design_matrix
        else:
            self.obsm["design_matrix"] = self.design

        if self.obsm["design_matrix"].isna().any().any():
            raise ValueError("NaNs are not allowed in the design.")

        # Check that the design matrix has full rank
        self._check_full_rank_design()

        if preserve_inherited_fit and adata is not None:
            source_design = adata.obsm.get("design_matrix")
            normalization_control_mask = self._make_control_mask(control_genes)
            fit_settings = (
                ("fit_type", fit_type),
                ("min_mu", min_mu),
                ("min_disp", min_disp),
                ("max_disp", np.maximum(max_disp, self.n_obs)),
                ("refit_cooks", refit_cooks),
                ("min_replicates", min_replicates),
                ("beta_tol", beta_tol),
            )
            preserve_inherited_fit = (
                isinstance(source_design, pd.DataFrame)
                and cast(pd.DataFrame, self.obsm["design_matrix"]).equals(source_design)
                and all(
                    hasattr(adata, attr) and np.array_equal(getattr(adata, attr), value)
                    for attr, value in fit_settings
                )
                and getattr(adata, "_normalization_fit_type", None)
                == size_factors_fit_type
                and hasattr(adata, "_normalization_control_mask")
                and np.array_equal(
                    adata._normalization_control_mask,
                    normalization_control_mask,
                )
                and hasattr(adata, "inference")
                and (
                    type(adata.inference) is DefaultInference
                    if inference is None
                    else inference is adata.inference
                )
            )

        if adata is not None and has_transcript_lengths and not preserve_inherited_fit:
            self._invalidate_lrt_cache(discard_replace_counts=True)
            # Invalidate fitting state derived from copied normalization factors.
            for column in ("size_factors", "replaceable"):
                if column in self.obs:
                    del self.obs[column]
            for column in (
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
                "replaced",
                "refitted",
                "_pvalue_cooks_outlier",
            ):
                if column in self.var:
                    del self.var[column]
            for layer in (
                "normalization_factors",
                "normed_counts",
                "_mu_hat",
                "_vst_mu_hat",
                "vst_counts",
                "cooks",
                "replace_cooks",
            ):
                self.layers.pop(layer, None)
            for key in ("_mu_LFC", "_hat_diagonals"):
                self.obsm.pop(key, None)
            self.varm.pop("LFC", None)
            for key in (
                "trend_coeffs",
                "vst_trend_coeffs",
                "mean_disp",
                "disp_function_type",
                "_squared_logres",
                "prior_disp_var",
            ):
                self.uns.pop(key, None)

        self.min_mu = min_mu
        self.min_disp = min_disp
        self.max_disp = np.maximum(max_disp, self.n_obs)
        self.refit_cooks = refit_cooks
        self.min_replicates = min_replicates
        self.beta_tol = beta_tol
        self.quiet = quiet
        self.low_memory = low_memory
        self.size_factors_fit_type = size_factors_fit_type
        self.control_genes = control_genes
        self.logmeans: np.ndarray | None = None
        self.filtered_genes: np.ndarray | None = None

        if inference:
            if n_cpus:
                if hasattr(inference, "n_cpus"):
                    inference.n_cpus = n_cpus
                else:
                    warnings.warn(
                        "The provided inference object does not have an n_cpus "
                        "attribute, cannot override `n_cpus`.",
                        UserWarning,
                        stacklevel=2,
                    )

        # Initialize the inference object.
        self.inference = inference or DefaultInference(n_cpus=n_cpus)

        if preserve_inherited_fit and adata is not None:
            for key in ("LFC", _LRT_FULL_LFC_KEY):
                if key in self.varm:
                    self.varm[key] = self.varm[key].copy()

            self._normalization_fit_type = adata._normalization_fit_type
            self._normalization_control_mask = adata._normalization_control_mask.copy()

            for attr in (
                "non_zero_idx",
                "non_zero_genes",
                "counts_to_refit",
                "new_all_zeroes_genes",
            ):
                if hasattr(adata, attr):
                    setattr(self, attr, getattr(adata, attr).copy())

            if hasattr(adata, "vst_fit_type"):
                self.vst_fit_type = adata.vst_fit_type

    def _validate_transcript_lengths(self, transcript_lengths: np.ndarray) -> None:
        """Validate a sample-by-gene average transcript-length matrix."""
        if transcript_lengths.shape != self.shape:
            raise ValueError(
                "transcript_lengths must have the same shape as counts "
                f"({self.shape}), got {transcript_lengths.shape}."
            )
        if not np.isfinite(transcript_lengths).all():
            raise ValueError("transcript_lengths must only contain finite values.")
        if (transcript_lengths <= 0).any():
            raise ValueError("transcript_lengths must contain only positive values.")

    def _make_control_mask(self, control_genes: Any) -> np.ndarray:
        """Return a gene mask for the selected normalization controls."""
        if control_genes is None:
            return np.ones(self.n_vars, dtype=bool)
        control_mask = np.zeros(self.n_vars, dtype=bool)
        control_mask[self._normalize_indices((slice(None), control_genes))[1]] = True
        return control_mask

    def _get_normalization_factors(self, gene_idx: Any = None) -> np.ndarray:
        """Return normalization factors, optionally restricted to genes."""
        if "normalization_factors" in self.layers:
            factors = np.asarray(self.layers["normalization_factors"])
            return factors if gene_idx is None else factors[:, gene_idx]
        return self.obs["size_factors"].to_numpy()

    def _fit_transcript_length_factors(
        self,
        fit_type: Literal["ratio", "poscounts"],
        control_mask: np.ndarray,
    ) -> None:
        """Fit DESeq2-style normalization factors from average transcript lengths."""
        counts = cast(Any, self.X).toarray() if issparse(self.X) else np.asarray(self.X)
        transcript_lengths = np.asarray(self.layers["avg_tx_length"])

        # DESeq2 centers each gene's average transcript lengths around a geometric
        # mean of one before estimating library-size factors on adjusted counts.
        length_factors = transcript_lengths / np.exp(
            np.mean(np.log(transcript_lengths), axis=0)
        )
        adjusted_counts = counts / length_factors

        if fit_type == "ratio":
            self.logmeans, self.filtered_genes = deseq2_norm_fit(adjusted_counts)
        else:
            log_counts = np.zeros_like(counts, dtype=float)
            np.log(counts, out=log_counts, where=counts != 0)
            self.logmeans = log_counts.mean(axis=0)
            self.filtered_genes = counts.sum(axis=0) > 0

        usable_genes = control_mask & self.filtered_genes
        if not usable_genes.any():
            raise ValueError(
                "No genes are available to estimate size factors after applying "
                "transcript-length and control-gene filters."
            )

        log_size_factors = np.empty(self.n_obs)
        for sample_idx in range(self.n_obs):
            sample_genes = usable_genes & (adjusted_counts[sample_idx] > 0)
            if not sample_genes.any():
                raise ValueError(
                    "At least one sample has no positive counts among the genes "
                    "available for transcript-length normalization."
                )
            log_size_factors[sample_idx] = np.median(
                np.log(adjusted_counts[sample_idx, sample_genes])
                - self.logmeans[sample_genes]
            )

        # estimateNormFactors() in DESeq2 returns a matrix whose gene-wise geometric
        # means are one. Retain the library-size component separately for backwards
        # compatibility, while using the full matrix throughout model fitting.
        size_factors = np.exp(log_size_factors)
        size_factors /= np.exp(np.mean(np.log(size_factors)))
        normalization_factors = length_factors * size_factors[:, None]
        normalization_factors /= np.exp(np.mean(np.log(normalization_factors), axis=0))

        self.obs["size_factors"] = size_factors
        self.layers["normalization_factors"] = normalization_factors
        self.layers["normed_counts"] = counts / normalization_factors

    @staticmethod
    def _validate_test(test: str) -> None:
        """Validate a requested DESeq2 test type."""
        if test not in {"Wald", "LRT"}:
            raise ValueError(f"test must be either 'Wald' or 'LRT'; got {test!r}.")

    def _prepare_lrt_reduced_design(
        self, reduced: str | pd.DataFrame | None
    ) -> pd.DataFrame:
        """Build and validate a reduced design matrix for a classical LRT."""
        if reduced is None:
            if "_lrt_reduced_design_matrix" not in self.obsm:
                raise ValueError(
                    "test='LRT' requires a reduced model. Pass a formula string "
                    "such as '~1' or a reduced design matrix."
                )
            reduced_matrix = self.obsm["_lrt_reduced_design_matrix"]
            if not isinstance(reduced_matrix, pd.DataFrame):
                raise RuntimeError(
                    "Cached LRT reduced design is not DataFrame-backed. "
                    "Rerun dds.deseq2(test='LRT', reduced=...)."
                )
        elif isinstance(self.design, str):
            if not isinstance(reduced, str):
                raise ValueError(
                    "When the full design is formula-based, 'reduced' must also be "
                    "a formula string."
                )
            try:
                reduced_contrasts = FormulaicContrasts(self.obs, reduced)
            except Exception as error:
                raise ValueError(
                    f"Could not build the reduced design from {reduced!r}: {error}"
                ) from error

            reduced_matrix = reduced_contrasts.design_matrix
        else:
            if not isinstance(reduced, pd.DataFrame):
                raise ValueError(
                    "When the full design is a DataFrame, 'reduced' must also be a "
                    "pandas DataFrame."
                )
            reduced_matrix = reduced

        if not pd.Index(reduced_matrix.index).equals(pd.Index(self.obs_names)):
            raise ValueError(
                "The reduced design matrix must have the same sample index, in the "
                "same order, as the DeseqDataSet."
            )
        if not reduced_matrix.columns.is_unique:
            raise ValueError("The reduced design matrix must have unique columns.")

        try:
            reduced_values = np.asarray(reduced_matrix, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "The reduced design matrix must contain only finite numeric values."
            ) from error

        if reduced_values.ndim != 2 or not np.isfinite(reduced_values).all():
            raise ValueError(
                "The reduced design matrix must contain only finite numeric values."
            )
        if reduced_values.shape[1] == 0:
            raise ValueError(
                "The reduced design must contain at least one column; offset-only "
                "reduced models are not currently supported."
            )
        if np.linalg.matrix_rank(reduced_values) < reduced_values.shape[1]:
            raise ValueError("The reduced design matrix is not full column rank.")

        full_values = np.asarray(self.obsm["design_matrix"], dtype=float)
        full_rank = np.linalg.matrix_rank(full_values)
        if full_rank < full_values.shape[1]:
            raise ValueError("The full design matrix is not full column rank.")
        if full_values.shape[1] <= reduced_values.shape[1]:
            raise ValueError(
                "The full model must have more columns than the reduced model "
                f"(got {full_values.shape[1]} and {reduced_values.shape[1]})."
            )

        augmented_rank = np.linalg.matrix_rank(
            np.column_stack((full_values, reduced_values))
        )
        if augmented_rank != full_rank:
            raise ValueError("The reduced model is not nested within the full model.")

        return pd.DataFrame(
            reduced_values,
            index=self.obs_names.copy(),
            columns=reduced_matrix.columns.copy(),
        )

    def _owns_replace_counts(self) -> bool:
        """Return whether the replacement-count layer is managed by PyDESeq2."""
        owner = self.uns.get(_REPLACE_COUNTS_OWNER_KEY, False)
        return (
            isinstance(owner, (bool, np.bool_))
            and bool(owner)
            and _REPLACE_COUNTS_LAYER in self.layers
        )

    def _discard_owned_replace_counts(self) -> None:
        """Delete only an internally managed replacement-count layer."""
        owner = self.uns.get(_REPLACE_COUNTS_OWNER_KEY, False)
        if not isinstance(owner, (bool, np.bool_)) or not bool(owner):
            return
        if _REPLACE_COUNTS_LAYER in self.layers:
            del self.layers[_REPLACE_COUNTS_LAYER]
        del self.uns[_REPLACE_COUNTS_OWNER_KEY]

    def _clear_cooks_refit_state(self) -> None:
        """Remove Cook-refit state before recomputing a complete model fit."""
        var = cast(pd.DataFrame, self.var)
        obs = cast(pd.DataFrame, self.obs)
        for key in ("replaced", "refitted", "_pvalue_cooks_outlier"):
            if key in var:
                del var[key]
        if "replaceable" in obs:
            del obs["replaceable"]
        if "replace_cooks" in self.layers:
            del self.layers["replace_cooks"]
        for attribute in ("counts_to_refit", "new_all_zeroes_genes"):
            if hasattr(self, attribute):
                delattr(self, attribute)

    @staticmethod
    def _nonzero_count_columns(counts: Any) -> np.ndarray:
        """Return a dense mask identifying columns with at least one nonzero count."""
        if issparse(counts):
            return np.asarray((cast(Any, counts) != 0).sum(axis=0)).ravel() > 0
        return np.asarray((np.asarray(counts) != 0).any(axis=0), dtype=bool)

    @staticmethod
    def _lrt_nb_nll(
        counts: Any,
        mu: np.ndarray,
        dispersions: np.ndarray,
    ) -> np.ndarray:
        """Evaluate likelihoods for one already-bounded LRT gene block."""
        if issparse(counts):
            counts = cast(Any, counts).toarray()
        return np.atleast_1d(np.asarray(nb_nll(counts, mu, dispersions), dtype=float))

    @staticmethod
    def _validated_lrt_bool_values(values: object) -> np.ndarray:
        """Return canonical int8 booleans after strict LRT state validation."""
        array = np.asarray(values)
        error = "LRT boolean state must contain only booleans or integers -1, 0, and 1."
        if array.dtype.kind == "b":
            return np.ascontiguousarray(array, dtype="<i1")
        if array.dtype.kind not in {"i", "u"} or not np.isin(array, (-1, 0, 1)).all():
            raise ValueError(error)
        return np.ascontiguousarray(array, dtype="<i1")

    @staticmethod
    def _update_lrt_array_digest(
        digest: _LRTDigest,
        label: str,
        values: object,
        kind: Literal["float", "int", "bool"],
    ) -> None:
        """Stream one canonical labeled LRT array into a digest.

        Integer inputs must already be canonical int64 arrays.
        """
        if kind == "float":
            array = np.asarray(values, dtype="<f8")
            array = np.ascontiguousarray(array)
            if np.isnan(array).any():
                array = array.copy()
                array[np.isnan(array)] = np.nan
        elif kind == "int":
            array = (
                cast(Any, values)
                if issparse(values)
                else np.ascontiguousarray(values, dtype="<i8")
            )
        else:
            array = DeseqDataSet._validated_lrt_bool_values(values)

        encoded_label = label.encode("utf-8")
        shape = np.asarray(array.shape, dtype="<i8").tobytes()
        digest.update(len(encoded_label).to_bytes(8, "little"))
        digest.update(encoded_label)
        digest.update(len(shape).to_bytes(8, "little"))
        digest.update(shape)
        if issparse(array):
            sparse_array = cast(Any, array).tocsr(copy=False)
            digest.update(b"sparse_csr")
            for buffer in (
                np.asarray(sparse_array.indptr, dtype="<i8"),
                np.asarray(sparse_array.indices, dtype="<i8"),
                np.asarray(sparse_array.data, dtype="<i8"),
            ):
                buffer = np.ascontiguousarray(buffer)
                digest.update(buffer.nbytes.to_bytes(8, "little"))
                digest.update(buffer.data.cast("B"))
        else:
            digest.update(array.data.cast("B"))

    @staticmethod
    def _update_lrt_text_digest(
        digest: _LRTDigest, label: str, values: list[str]
    ) -> None:
        """Stream unambiguous canonical labeled text into a digest."""
        encoded_label = label.encode("utf-8")
        digest.update(len(encoded_label).to_bytes(8, "little"))
        digest.update(encoded_label)
        digest.update(len(values).to_bytes(8, "little"))
        for value in values:
            encoded_value = value.encode("utf-8")
            digest.update(len(encoded_value).to_bytes(8, "little"))
            digest.update(encoded_value)

    @staticmethod
    def _lrt_nullable_bool_values(
        values: pd.Series | np.ndarray,
    ) -> np.ndarray:
        """Encode nullable booleans as -1, 0, and 1 for stable hashing."""
        return pd.array(values.tolist(), dtype="boolean").to_numpy(
            dtype=np.int8,
            na_value=-1,
        )

    def _lrt_fit_digest(self) -> str | None:
        """Fingerprint the fitted scientific state shared by every LRT."""
        full_design = self.obsm.get("design_matrix")
        full_lfcs = self.varm.get("LFC")
        if not isinstance(full_design, pd.DataFrame) or not isinstance(
            full_lfcs, pd.DataFrame
        ):
            return None
        if "size_factors" not in self.obs or "dispersions" not in self.var:
            return None
        if "non_zero" not in self.var:
            return None
        if not pd.Index(full_design.index).equals(pd.Index(self.obs_names)):
            return None
        if not pd.Index(full_lfcs.index).equals(pd.Index(self.var_names)):
            return None
        if [str(column) for column in full_lfcs.columns] != [
            str(column) for column in full_design.columns
        ]:
            return None
        digest = hashlib.sha256()
        replacement_owned = self._owns_replace_counts()
        self._update_lrt_array_digest(
            digest, "effective_counts", self._effective_counts(), "int"
        )
        self._update_lrt_array_digest(
            digest,
            "normalization_factors",
            self._get_normalization_factors(),
            "float",
        )
        self._update_lrt_array_digest(
            digest,
            "dispersions",
            self.var["dispersions"].to_numpy(dtype=float),
            "float",
        )
        self._update_lrt_array_digest(
            digest, "full_lfcs", full_lfcs.to_numpy(dtype=float), "float"
        )
        self._update_lrt_array_digest(
            digest, "full_design", full_design.to_numpy(dtype=float), "float"
        )
        self._update_lrt_array_digest(
            digest,
            "non_zero",
            self.var["non_zero"].to_numpy(),
            "bool",
        )
        self._update_lrt_array_digest(
            digest, "replacement_owned", [replacement_owned], "bool"
        )
        self._update_lrt_array_digest(
            digest, "lrt_controls", [self.min_mu, self.beta_tol], "float"
        )
        self._update_lrt_text_digest(
            digest,
            "full_design_columns",
            [str(column) for column in full_design.columns],
        )
        self._update_lrt_text_digest(
            digest,
            "full_lfc_columns",
            [str(column) for column in full_lfcs.columns],
        )
        self._update_lrt_text_digest(
            digest,
            "obs_names",
            [str(name) for name in self.obs_names],
        )
        self._update_lrt_text_digest(
            digest,
            "var_names",
            [str(name) for name in self.var_names],
        )

        if "replaced" in self.var:
            self._update_lrt_array_digest(digest, "has_replaced", [True], "bool")
            self._update_lrt_array_digest(
                digest,
                "replaced",
                self.var["replaced"].to_numpy(),
                "bool",
            )
        else:
            self._update_lrt_array_digest(digest, "has_replaced", [False], "bool")

        if "_LFC_converged" in self.var:
            self._update_lrt_array_digest(digest, "has_lfc_converged", [True], "bool")
            self._update_lrt_array_digest(
                digest,
                "lfc_converged",
                self._lrt_nullable_bool_values(self.var["_LFC_converged"]),
                "bool",
            )
        else:
            self._update_lrt_array_digest(digest, "has_lfc_converged", [False], "bool")
        return digest.hexdigest()

    def _lrt_input_digest(
        self,
        reduced_design: pd.DataFrame,
        reduced_formula: str,
        fit_digest: str | None = None,
    ) -> str | None:
        """Fingerprint a reduced model together with the current fitted state."""
        if not pd.Index(reduced_design.index).equals(pd.Index(self.obs_names)):
            return None
        if fit_digest is None:
            fit_digest = self._lrt_fit_digest()
        if fit_digest is None:
            return None

        digest = hashlib.sha256()
        self._update_lrt_text_digest(digest, "fit_sha256", [fit_digest])
        self._update_lrt_array_digest(
            digest,
            "reduced_design",
            reduced_design.to_numpy(dtype=float),
            "float",
        )
        self._update_lrt_text_digest(
            digest,
            "reduced_design_columns",
            [str(column) for column in reduced_design.columns],
        )
        self._update_lrt_text_digest(
            digest,
            "reduced_formula",
            [reduced_formula],
        )
        return digest.hexdigest()

    def _lrt_fit_state_matches(self, fit_digest: str | None = None) -> bool:
        """Return whether cached LRT provenance matches the current fitted state."""
        try:
            metadata_value = self.uns["_lrt"]
            if not isinstance(metadata_value, Mapping):
                return False
            metadata = cast(Mapping[str, object], metadata_value)
            full_design = self.obsm["design_matrix"]
            reduced_design = self.obsm["_lrt_reduced_design_matrix"]
            if not isinstance(full_design, pd.DataFrame) or not isinstance(
                reduced_design, pd.DataFrame
            ):
                return False
            if not pd.Index(full_design.index).equals(pd.Index(self.obs_names)):
                return False
            if not pd.Index(reduced_design.index).equals(pd.Index(self.obs_names)):
                return False
            cached_lfcs = self.varm.get(_LRT_FULL_LFC_KEY)
            if (
                any(key not in self.var for key in _LRT_VAR_KEYS)
                or not isinstance(cached_lfcs, pd.DataFrame)
                or not pd.Index(cached_lfcs.index).equals(pd.Index(self.var_names))
                or [str(column) for column in cached_lfcs.columns]
                != [str(column) for column in full_design.columns]
            ):
                return False

            stored_cache_version = metadata.get("cache_version")
            if (
                isinstance(stored_cache_version, (bool, np.bool_))
                or not isinstance(stored_cache_version, (int, np.integer))
                or int(stored_cache_version) != _LRT_CACHE_VERSION
            ):
                return False

            stored_generation = metadata.get("fit_generation")
            current_generation = self.uns.get("_fit_generation", 0)
            if any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                for value in (stored_generation, current_generation)
            ) or int(cast(Any, stored_generation)) != int(cast(Any, current_generation)):
                return False

            stored_df = metadata.get("df")
            if (
                isinstance(stored_df, (bool, np.bool_))
                or not isinstance(stored_df, (int, np.integer))
                or int(stored_df) != full_design.shape[1] - reduced_design.shape[1]
            ):
                return False
            reduced_formula = metadata.get("reduced_formula")
            if not isinstance(reduced_formula, str):
                return False

            input_digest = self._lrt_input_digest(
                reduced_design,
                reduced_formula,
                fit_digest,
            )
            return (
                input_digest is not None and metadata.get("input_sha256") == input_digest
            )
        except (
            AttributeError,
            KeyError,
            OverflowError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            return False

    def _invalidate_lrt_cache(self, *, discard_replace_counts: bool = False) -> None:
        """Remove cached test results after changing fitted model state."""
        var = cast(pd.DataFrame, self.var)
        fit_generation = self.uns.get("_fit_generation", 0)
        if isinstance(fit_generation, (bool, np.bool_)) or not isinstance(
            fit_generation, (int, np.integer)
        ):
            fit_generation = 0
        self.uns["_fit_generation"] = int(fit_generation) + 1

        if "_deseq2_test" in self.uns:
            del self.uns["_deseq2_test"]
        if "_lrt" in self.uns:
            del self.uns["_lrt"]
        if "_lrt_reduced_design_matrix" in self.obsm:
            del self.obsm["_lrt_reduced_design_matrix"]
        for key in _LRT_VAR_KEYS:
            if key in var:
                del var[key]
        if _LRT_FULL_LFC_KEY in self.varm:
            del self.varm[_LRT_FULL_LFC_KEY]

        if discard_replace_counts:
            self._discard_owned_replace_counts()

    def _effective_counts(self) -> Any:
        """Return the counts used for final inference, including Cook replacements."""
        if self.X is None:
            raise RuntimeError("A count matrix is required.")
        counts = self.X
        if not self._owns_replace_counts():
            return counts

        replacement_delta = self.layers[_REPLACE_COUNTS_LAYER]
        if not issparse(replacement_delta):
            raise RuntimeError("Internally managed Cook replacement state is malformed.")
        replacement_delta = cast(Any, replacement_delta).tocsr(copy=False)
        if replacement_delta.shape != counts.shape:
            raise RuntimeError("Internally managed Cook replacement state is malformed.")

        if issparse(counts):
            effective_counts = counts + replacement_delta
        else:
            effective_counts = counts.copy()
            delta = replacement_delta.tocoo(copy=False)
            np.add.at(effective_counts, (delta.row, delta.col), delta.data)
        return effective_counts

    def _compute_lrt(
        self,
        reduced_design: pd.DataFrame,
        inference: Inference | None = None,
        *,
        refit_full: bool = False,
        full_lfcs_seed: pd.DataFrame | None = None,
        full_converged_seed: pd.Series | None = None,
    ) -> _LRTResult:
        """Fit a reduced NB GLM and compute a classical unpenalized LRT."""
        if "LFC" not in self.varm or "dispersions" not in self.var:
            raise RuntimeError(
                "LRT requires fitted dispersions and LFCs. Run dds.deseq2() first."
            )
        if "size_factors" not in self.obs:
            raise RuntimeError("LRT requires fitted size factors.")
        full_design_frame = self.obsm.get("design_matrix")
        full_lfcs_value = self.varm.get("LFC")
        if not isinstance(full_design_frame, pd.DataFrame) or not isinstance(
            full_lfcs_value, pd.DataFrame
        ):
            raise RuntimeError("LRT requires DataFrame-backed designs and fitted LFCs.")
        var = cast(pd.DataFrame, self.var)
        if (
            "replaced" in var
            and var["replaced"].fillna(False).astype(bool).any()
            and not self._owns_replace_counts()
        ):
            raise RuntimeError(
                "LRT requires the effective counts used during Cook outlier refitting, "
                "but an internally managed replacement-count layer is unavailable. "
                "Rerun dds.deseq2() before computing an LRT."
            )

        counts = self._effective_counts()
        dispersions = var["dispersions"].to_numpy(dtype=float)
        full_lfcs = (
            full_lfcs_value if full_lfcs_seed is None else full_lfcs_seed
        ).copy()
        effective_nonzero = self._nonzero_count_columns(counts)
        fit_mask = effective_nonzero & np.isfinite(dispersions)
        fit_idx = np.flatnonzero(fit_mask)
        fit_positions = fit_idx.tolist()

        new_all_zero_values = np.zeros(self.n_vars, dtype=bool)
        if self._owns_replace_counts() and "replaced" in var:
            new_all_zero_values = (
                var["non_zero"].to_numpy(dtype=bool)
                & var["replaced"].to_numpy(dtype=bool)
                & ~effective_nonzero
            )
        new_all_zero = pd.Series(
            new_all_zero_values,
            index=self.var_names,
            dtype=bool,
        )

        statistics = pd.Series(np.nan, index=self.var_names, dtype=float)
        p_values = pd.Series(np.nan, index=self.var_names, dtype=float)
        full_deviance = pd.Series(np.nan, index=self.var_names, dtype=float)
        reduced_converged = pd.Series(
            pd.array([pd.NA] * self.n_vars, dtype="boolean"),
            index=self.var_names,
        )
        if refit_full:
            full_converged = pd.Series(
                pd.array([pd.NA] * self.n_vars, dtype="boolean"),
                index=self.var_names,
            )
        elif full_converged_seed is not None:
            full_converged = pd.Series(
                pd.array(full_converged_seed.to_numpy(), dtype="boolean"),
                index=self.var_names,
            )
        elif "_LFC_converged" in var:
            full_converged = pd.Series(
                pd.array(var["_LFC_converged"].to_numpy(), dtype="boolean"),
                index=self.var_names,
            )
        else:
            full_converged = pd.Series(
                pd.array([pd.NA] * self.n_vars, dtype="boolean"),
                index=self.var_names,
            )
        if refit_full:
            full_lfcs = pd.DataFrame(
                np.nan,
                index=self.var_names,
                columns=full_design_frame.columns,
            )
        full_lfcs.loc[new_all_zero] = 0.0
        statistics.loc[new_all_zero] = 0.0
        p_values.loc[new_all_zero] = 1.0

        if len(fit_idx) == 0:
            return (
                statistics,
                p_values,
                full_deviance,
                reduced_converged,
                full_converged,
                new_all_zero,
                full_lfcs,
            )

        full_design = full_design_frame.to_numpy(dtype=float)
        reduced_values = reduced_design.to_numpy(dtype=float)
        fit_inference = inference or self.inference
        block_size = (
            _LRT_GENE_BLOCK_SIZE if self.low_memory or issparse(counts) else len(fit_idx)
        )
        full_nll = np.empty(len(fit_idx), dtype=float)
        reduced_nll = np.empty(len(fit_idx), dtype=float)
        converged = np.empty(len(fit_idx), dtype=bool)

        for start in range(0, len(fit_idx), block_size):
            stop = min(start + block_size, len(fit_idx))
            fit_slice = slice(start, stop)
            gene_idx = fit_idx[fit_slice]
            block_counts = counts[:, gene_idx]
            block_factors = self._get_normalization_factors(gene_idx)
            block_dispersions = dispersions[gene_idx]

            if refit_full:
                full_beta, full_mu, _, block_full_converged = fit_inference.irls(
                    counts=block_counts,
                    size_factors=block_factors,
                    design_matrix=full_design,
                    disp=block_dispersions,
                    min_mu=self.min_mu,
                    beta_tol=self.beta_tol,
                )
                full_lfcs.iloc[gene_idx] = full_beta
                full_converged.iloc[gene_idx] = list(
                    np.asarray(block_full_converged, dtype=bool)
                )
            else:
                full_mu = np.exp(
                    full_design @ full_lfcs.iloc[gene_idx].to_numpy(dtype=float).T
                )
                if block_factors.ndim == 1:
                    full_mu *= block_factors[:, None]
                else:
                    full_mu *= block_factors

            _, reduced_mu, _, block_converged = fit_inference.irls(
                counts=block_counts,
                size_factors=block_factors,
                design_matrix=reduced_values,
                disp=block_dispersions,
                min_mu=self.min_mu,
                beta_tol=self.beta_tol,
            )
            converged[fit_slice] = np.asarray(block_converged, dtype=bool)
            full_nll[fit_slice] = self._lrt_nb_nll(
                block_counts,
                full_mu,
                block_dispersions,
            )
            reduced_nll[fit_slice] = self._lrt_nb_nll(
                block_counts,
                reduced_mu,
                block_dispersions,
            )
            del full_mu, reduced_mu, _

        reduced_failed = ~converged
        if reduced_failed.any():
            warnings.warn(
                "The reduced-model fit did not converge for "
                f"{int(reduced_failed.sum())} gene(s); inspect "
                "DeseqStats.reduced_converged before interpreting those results.",
                UserWarning,
                stacklevel=2,
            )

        lrt_statistics = 2 * (reduced_nll - full_nll)

        def negative_tolerances() -> tuple[np.ndarray, np.ndarray]:
            full_deviance_scale = np.abs(2 * full_nll)
            reduced_deviance_scale = np.abs(2 * reduced_nll)
            likelihood_scale = np.maximum.reduce(
                (
                    np.ones_like(full_deviance_scale),
                    full_deviance_scale,
                    reduced_deviance_scale,
                )
            )
            machine_tolerance = 64 * np.finfo(float).eps * likelihood_scale
            optimization_tolerance = machine_tolerance + abs(self.beta_tol) * (
                full_deviance_scale + reduced_deviance_scale + 0.2
            )
            return machine_tolerance, optimization_tolerance

        machine_tolerance, optimization_tolerance = negative_tolerances()
        full_status = full_converged.iloc[fit_idx]
        both_models_converged = full_status.fillna(False).to_numpy(
            dtype=bool
        ) & np.asarray(converged, dtype=bool)
        recovery_tolerance = np.where(
            both_models_converged,
            optimization_tolerance,
            machine_tolerance,
        )
        needs_full_recovery = ~np.isfinite(full_nll) | (
            np.isfinite(lrt_statistics) & (lrt_statistics < -recovery_tolerance)
        )
        if needs_full_recovery.any() and not refit_full:
            recovery_positions = np.flatnonzero(needs_full_recovery)
            for start in range(0, len(recovery_positions), block_size):
                block_positions = recovery_positions[start : start + block_size]
                recovery_idx = fit_idx[block_positions]
                recovery_counts = counts[:, recovery_idx]
                recovery_factors = self._get_normalization_factors(recovery_idx)
                (
                    recovery_beta,
                    recovery_mu,
                    _,
                    recovery_converged,
                ) = fit_inference.irls(
                    counts=recovery_counts,
                    size_factors=recovery_factors,
                    design_matrix=full_design,
                    disp=dispersions[recovery_idx],
                    min_mu=self.min_mu,
                    beta_tol=self.beta_tol,
                )
                full_lfcs.iloc[recovery_idx] = recovery_beta
                full_converged.iloc[recovery_idx] = list(
                    np.asarray(recovery_converged, dtype=bool)
                )
                full_nll[block_positions] = self._lrt_nb_nll(
                    recovery_counts,
                    recovery_mu,
                    dispersions[recovery_idx],
                )
                del recovery_mu, _

            lrt_statistics = 2 * (reduced_nll - full_nll)

        full_status = full_converged.iloc[fit_idx]
        full_failed = full_status.notna().to_numpy() & ~full_status.fillna(
            True
        ).to_numpy(dtype=bool)
        if full_failed.any():
            warnings.warn(
                "The full-model fit did not converge for "
                f"{int(full_failed.sum())} gene(s); inspect "
                "DeseqStats.full_converged before interpreting those results.",
                UserWarning,
                stacklevel=2,
            )

        machine_tolerance, optimization_tolerance = negative_tolerances()
        both_models_converged = full_status.fillna(False).to_numpy(
            dtype=bool
        ) & np.asarray(converged, dtype=bool)
        allowed_negative = np.where(
            both_models_converged,
            optimization_tolerance,
            machine_tolerance,
        )
        nonfinite_likelihood = (
            ~np.isfinite(full_nll)
            | ~np.isfinite(reduced_nll)
            | ~np.isfinite(lrt_statistics)
        )
        materially_negative = ~nonfinite_likelihood & (
            lrt_statistics < -allowed_negative
        )
        invalid_lrt = nonfinite_likelihood | materially_negative
        lrt_statistics[(lrt_statistics < 0) & ~invalid_lrt] = 0.0
        if nonfinite_likelihood.any():
            warnings.warn(
                f"{int(nonfinite_likelihood.sum())} gene(s) had non-finite "
                "likelihoods after full-model fitting or recovery; their "
                "statistics, p-values, and full-model deviances were set to NaN.",
                RuntimeWarning,
                stacklevel=2,
            )
        if materially_negative.any():
            warnings.warn(
                f"{int(materially_negative.sum())} gene(s) had materially negative "
                "LRT statistics after full-model fitting or recovery; their "
                "statistics and p-values were set to NaN.",
                RuntimeWarning,
                stacklevel=2,
            )
        lrt_statistics[invalid_lrt] = np.nan
        degrees_of_freedom = full_design.shape[1] - reduced_values.shape[1]

        statistic_values = statistics.to_numpy(copy=True)
        p_value_values = p_values.to_numpy(copy=True)
        full_deviance_values = full_deviance.to_numpy(copy=True)
        statistic_values[fit_idx] = lrt_statistics
        p_value_values[fit_idx] = chi2.sf(lrt_statistics, df=degrees_of_freedom)
        fitted_full_deviance = 2 * full_nll
        fitted_full_deviance[nonfinite_likelihood] = np.nan
        full_deviance_values[fit_idx] = fitted_full_deviance
        statistics = pd.Series(statistic_values, index=self.var_names, dtype=float)
        p_values = pd.Series(p_value_values, index=self.var_names, dtype=float)
        full_deviance = pd.Series(
            full_deviance_values,
            index=self.var_names,
            dtype=float,
        )
        reduced_converged.iloc[fit_positions] = list(np.asarray(converged, dtype=bool))
        return (
            statistics,
            p_values,
            full_deviance,
            reduced_converged,
            full_converged,
            new_all_zero,
            full_lfcs,
        )

    def _store_lrt_results(
        self,
        reduced_design: pd.DataFrame,
        reduced: str | pd.DataFrame,
        results: _LRTResult,
        fit_digest: str | None = None,
    ) -> None:
        """Store the default LRT result in serialization-safe AnnData slots."""
        (
            statistics,
            p_values,
            full_deviance,
            reduced_converged,
            full_converged,
            new_all_zero,
            full_lfcs,
        ) = results
        self.obsm["_lrt_reduced_design_matrix"] = pd.DataFrame(
            reduced_design.to_numpy(copy=True),
            index=self.obs_names.copy(),
            columns=reduced_design.columns.copy(),
        )
        self.var["_lrt_statistic"] = statistics
        self.var["_lrt_pvalue"] = p_values
        self.var["_lrt_full_deviance"] = full_deviance
        self.var["_lrt_reduced_converged"] = reduced_converged.array
        self.var["_lrt_full_converged"] = full_converged.array
        self.var["_lrt_new_all_zero"] = new_all_zero.to_numpy(dtype=bool)
        self.varm[_LRT_FULL_LFC_KEY] = full_lfcs.copy()
        fit_generation = self.uns.get("_fit_generation", 0)
        if isinstance(fit_generation, (bool, np.bool_)) or not isinstance(
            fit_generation, (int, np.integer)
        ):
            fit_generation = 0
        fit_generation = int(fit_generation)
        self.uns["_fit_generation"] = fit_generation
        reduced_formula = reduced if isinstance(reduced, str) else ""
        input_digest = self._lrt_input_digest(
            cast(pd.DataFrame, self.obsm["_lrt_reduced_design_matrix"]),
            reduced_formula,
            fit_digest,
        )
        if input_digest is None:
            raise RuntimeError("Could not construct serialization-safe LRT provenance.")
        lrt_metadata: dict[str, str | np.int64] = {
            "cache_version": np.int64(_LRT_CACHE_VERSION),
            "df": np.int64(
                self.obsm["design_matrix"].shape[1] - reduced_design.shape[1]
            ),
            "reduced_formula": reduced_formula,
            "fit_generation": np.int64(fit_generation),
            "input_sha256": input_digest,
        }
        self.uns["_lrt"] = lrt_metadata

    def _lrt_cache_matches(
        self,
        reduced_design: pd.DataFrame,
    ) -> bool:
        """Return whether serialized LRT results use this reduced design."""
        if "_lrt_reduced_design_matrix" not in self.obsm:
            return False

        stored = self.obsm["_lrt_reduced_design_matrix"]
        if not isinstance(stored, pd.DataFrame):
            return False
        return (
            pd.Index(stored.index).equals(pd.Index(reduced_design.index))
            and [str(column) for column in stored.columns]
            == [str(column) for column in reduced_design.columns]
            and np.array_equal(
                stored.to_numpy(dtype=float),
                reduced_design.to_numpy(dtype=float),
                equal_nan=True,
            )
        )

    def _load_lrt_results(
        self,
    ) -> _LRTResult:
        """Load a cached LRT result from gene-aligned AnnData fields."""
        var = cast(pd.DataFrame, self.var)
        full_lfcs = self.varm[_LRT_FULL_LFC_KEY]
        if not isinstance(full_lfcs, pd.DataFrame):
            raise RuntimeError("Cached LRT LFCs are not DataFrame-backed.")
        return (
            var["_lrt_statistic"].copy(),
            var["_lrt_pvalue"].copy(),
            var["_lrt_full_deviance"].copy(),
            var["_lrt_reduced_converged"].copy(),
            var["_lrt_full_converged"].copy(),
            var["_lrt_new_all_zero"].copy(),
            full_lfcs.copy(),
        )

    @property
    def variables(self):
        """Get the names of the variables used in the model definition."""
        try:
            return self.formulaic_contrasts.variables
        except AttributeError:
            raise ValueError(
                """Retrieving variables is only possible if the model was initialized
                using a formula."""
            ) from None

    def vst(
        self,
        use_design: bool = False,
        fit_type: Literal["parametric", "mean"] | None = None,
    ) -> None:
        """Fit a variance stabilizing transformation, and apply it to normalized counts.

        Results are stored in ``dds.layers["vst_counts"]``.

        Parameters
        ----------
        use_design : bool
            Whether to use the full design matrix to fit dispersions and the trend curve.
            If False, only an intercept is used. (default: ``False``).

        fit_type: str
            * ``None``: fit_type provided at initialization to fit
              the dispersions trend curve.
            * ``"parametric"``: fit a dispersion-mean relation via a robust
              gamma-family GLM.
            * ``"mean"``: use the mean of gene-wise dispersion estimates.

            (default: ``None``).
        """
        if fit_type is not None:
            self.vst_fit_type = fit_type
        else:
            self.vst_fit_type = self.fit_type

        if not self.quiet:
            print(f"Fit type used for VST : {self.vst_fit_type}")

        self.vst_fit(use_design=use_design)
        self.layers["vst_counts"] = self.vst_transform()

    def vst_fit(
        self,
        use_design: bool = False,
    ) -> None:
        """Fit a variance stabilizing transformation.

        This method should be called before `vst_transform`.

        Results are stored in ``dds.layers["vst_counts"]``.

        Parameters
        ----------
        use_design : bool
            Whether to use the full design matrix to fit dispersions and the trend curve.
            If False, only an intercept is used.
            Only useful if ``fit_type = "parametric"`.
            (default: ``False``).
        """
        # Start by fitting median-of-ratio size factors if not already present,
        # or if they were computed iteratively
        if "size_factors" not in self.obs or self.logmeans is None:
            self.fit_size_factors(
                fit_type=self.size_factors_fit_type
            )  # by default, fit_type != "iterative"

        if not hasattr(self, "vst_fit_type"):
            self.vst_fit_type = self.fit_type

        if use_design:
            if self.vst_fit_type == "parametric":
                self._fit_parametric_dispersion_trend(vst=True)
            else:
                warnings.warn(
                    "use_design=True is only useful when fit_type='parametric'. ",
                    UserWarning,
                    stacklevel=2,
                )
                self.fit_genewise_dispersions(vst=True)

        else:
            # Reduce the design matrix to an intercept and reconstruct at the end
            self.obsm["design_matrix_buffer"] = self.obsm["design_matrix"].copy()
            self.obsm["design_matrix"] = pd.DataFrame(
                1, index=self.obs_names, columns=["Intercept"]
            )
            # Fit the trend curve with an intercept design
            self.fit_genewise_dispersions(vst=True)
            if self.vst_fit_type == "parametric":
                self._fit_parametric_dispersion_trend(vst=True)

            # Restore the design matrix and free buffer
            self.obsm["design_matrix"] = self.obsm["design_matrix_buffer"].copy()
            del self.obsm["design_matrix_buffer"]

    def vst_transform(self, counts: np.ndarray | None = None) -> np.ndarray:
        """Apply the variance stabilizing transformation.

        Uses the results from the ``vst_fit`` method.

        Parameters
        ----------
        counts : numpy.ndarray
            Counts to transform. If ``None``, use the counts from the current dataset.
            (default: ``None``).

        Returns
        -------
        numpy.ndarray
            Variance stabilized counts.

        Raises
        ------
        RuntimeError
            If the size factors were not fitted before calling this method.
        """
        if "size_factors" not in self.obs:
            raise RuntimeError(
                "The vst_fit method should be called prior to vst_transform."
            )

        if counts is not None and "avg_tx_length" in self.layers:
            raise ValueError(
                "Transforming external counts with transcript-length normalization "
                "requires matching transcript lengths, which are not yet supported."
            )

        if counts is None:
            # the transformed counts will be the current ones
            normed_counts = self.layers["normed_counts"]
        else:
            if self.logmeans is None:
                # the size factors were still computed iteratively
                warnings.warn(
                    "The size factors were fitted iteratively. They will "
                    "be re-computed with the counts to be transformed. In a train/test "
                    "setting with a downstream task, this would result in a leak of "
                    "data from test to train set.",
                    UserWarning,
                    stacklevel=2,
                )
                logmeans, filtered_genes = deseq2_norm_fit(counts)
            elif self.filtered_genes is not None:
                logmeans, filtered_genes = self.logmeans, self.filtered_genes
            else:
                raise RuntimeError(
                    "Logmeans is set but filtered_genes is None. This should not happen."
                )

            normed_counts, _ = deseq2_norm_transform(counts, logmeans, filtered_genes)

        if self.vst_fit_type == "parametric":
            if "vst_trend_coeffs" not in self.uns:
                raise RuntimeError("Fit the dispersion curve prior to applying VST.")

            a0, a1 = self.uns["vst_trend_coeffs"]
            return np.log2(
                (
                    1
                    + a1
                    + 2 * a0 * normed_counts
                    + 2 * np.sqrt(a0 * normed_counts * (1 + a1 + a0 * normed_counts))
                )
                / (4 * a0)
            )
        elif self.vst_fit_type == "mean":
            gene_dispersions = self.var["vst_genewise_dispersions"]
            use_for_mean = gene_dispersions > 10 * self.min_disp
            mean_disp = trim_mean(gene_dispersions[use_for_mean], proportiontocut=0.001)
            return (
                2 * np.arcsinh(np.sqrt(mean_disp * normed_counts))
                - np.log(mean_disp)
                - np.log(4)
            ) / np.log(2)
        else:
            raise NotImplementedError(
                f"Found fit_type '{self.vst_fit_type}'. Expected 'parametric' or 'mean'."
            )

    def deseq2(
        self,
        fit_type: Literal["parametric", "mean"] | None = None,
        *,
        test: Literal["Wald", "LRT"] = "Wald",
        reduced: str | pd.DataFrame | None = None,
    ) -> None:
        """Perform dispersion and log fold-change (LFC) estimation.

        Wrapper for the first part of the PyDESeq2 pipeline.


        Parameters
        ----------
        fit_type : str
            Either None, ``"parametric"`` or ``"mean"`` for the type of fitting of
            dispersions to the mean intensity.``"parametric"``: fit a dispersion-mean
            relation via a robust gamma-family GLM. ``"mean"``: use the mean of
            gene-wise dispersion estimates.

            If None, the fit_type provided at class initialization is used.
            (default: ``None``).

        test : str
            Statistical test to prepare. ``"Wald"`` preserves the standard pipeline;
            ``"LRT"`` additionally fits a reduced model and caches a likelihood-ratio
            test. (default: ``"Wald"``).

        reduced : str or pandas.DataFrame, optional
            Reduced formula or design matrix for ``test="LRT"``. Its columns must span
            a strict subspace of the full design. Must be omitted for a Wald fit.
        """
        self._validate_test(test)
        reduced_design: pd.DataFrame | None = None
        if test == "Wald":
            if reduced is not None:
                raise ValueError("'reduced' is only supported when test='LRT'.")
        else:
            if reduced is None:
                raise ValueError(
                    "test='LRT' requires a reduced model. Pass a formula string "
                    "such as '~1' or a reduced design matrix."
                )
            reduced_design = self._prepare_lrt_reduced_design(reduced)

        self._invalidate_lrt_cache(discard_replace_counts=True)
        self._clear_cooks_refit_state()

        if fit_type is not None:
            self.fit_type = fit_type
            if not self.quiet:
                print(f"Using {self.fit_type} fit type.")

        # Compute DESeq2 normalization factors using the Median-of-ratios method
        self.fit_size_factors(
            fit_type=self.size_factors_fit_type, control_genes=self.control_genes
        )
        # Fit an independent negative binomial model per gene
        self.fit_genewise_dispersions()
        # Fit a parameterized trend curve for dispersions, of the form
        # f(\mu) = \alpha_1/\mu + a_0
        self.fit_dispersion_trend()
        # Compute prior dispersion variance
        self.fit_dispersion_prior()
        # Refit genewise dispersions a posteriori (shrinks estimates towards trend curve)
        self.fit_MAP_dispersions()
        # Fit log-fold changes (in natural log scale)
        self.fit_LFC()
        # Compute Cooks distances to find outliers
        self.calculate_cooks()

        if self.refit_cooks:
            # Replace outlier counts, and refit dispersions and LFCs
            # for genes that had outliers replaced
            self.refit()

        # Compute gene mask for cooks outliers
        self.cooks_outlier()

        if test == "LRT":
            assert reduced_design is not None
            assert reduced is not None
            if not self.quiet:
                print("Running likelihood ratio tests...", file=sys.stderr)
            start = time.time()
            lrt_results = self._compute_lrt(reduced_design, inference=self.inference)
            self._store_lrt_results(reduced_design, reduced, lrt_results)
            if not self.quiet:
                print(
                    f"... done in {time.time() - start:.2f} seconds.\n",
                    file=sys.stderr,
                )

        self.uns["_deseq2_test"] = test

    def cond(self, **kwargs):
        """
        Get a contrast vector representing a specific condition.

        Parameters
        ----------
        **kwargs
            Column/value pairs.

        Returns
        -------
        ndarray
            A contrast vector that aligns to the columns of the design matrix.
        """
        try:
            return self.formulaic_contrasts.cond(**kwargs)
        except AttributeError:
            raise AttributeError(
                "The cond() method requires a formula-based design. "
                "When using a precomputed design matrix (DataFrame), "
                "pass the contrast vector directly instead."
            ) from None

    def contrast(self, *args, **kwargs):
        """Get a contrast for a simple pairwise comparison."""
        try:
            return self.formulaic_contrasts.contrast(*args, **kwargs)
        except AttributeError:
            raise AttributeError(
                "The contrast() method requires a formula-based design. "
                "When using a precomputed design matrix (DataFrame), "
                "pass the contrast vector directly instead."
            ) from None

    def fit_size_factors(
        self,
        fit_type: Literal["ratio", "poscounts", "iterative"] | None = None,
        control_genes: np.ndarray | list[str] | list[int] | pd.Index | None = None,
    ) -> None:
        """Fit sample-wise deseq2 normalization (size) factors.

        Uses the median-of-ratios method: see :func:`pydeseq2.preprocessing.deseq2_norm`,
        unless each gene has at least one sample with zero read counts, in which case it
        switches to the ``iterative`` method.

        Also available is the 'poscounts' method implemented in DESeq2 for the
        single-cell or metagenomics use case where there may be few or no features which
        have no zero values. In this situation, size factors can depend on a very small
        number of features (or only one feature) leading to incorrect inference. This
        method for calculating size factors will only exclude genes which have all-0
        values (and are not amenable to inference anyway).

        The "poscounts" method calculates the n-th root of the product of the non-zero
        (positive) counts.

        Control genes can be optionally provided; if so, size factors will be fit to
        only the genes in this argument. This is the same functionality as controlGenes
        in R DESeq2. Any valid AnnData indexer (bool, int position, var_name string) is
        accepted.

        Parameters
        ----------
        fit_type : str
            The normalization method to use: "ratio", "poscounts" or "iterative".
            (default: ``"ratio"``).
        control_genes : ndarray, list, or pandas.Index, optional
            Genes to use as control genes for size factor fitting. If None, all genes
            are used. Note that manually passing control genes here will override the
            `DeseqDataSet` `control_genes` attribute.
            (default: ``None``).
        """
        if fit_type is None:
            fit_type = self.size_factors_fit_type
        if not self.quiet:
            print("Fitting size factors...", file=sys.stderr)

        start = time.time()

        if control_genes is None:
            # Check whether control genes were specified at initialization
            if hasattr(self, "control_genes"):
                control_genes = self.control_genes
                if not self.quiet:
                    print(
                        f"Using {control_genes} as control genes, passed at"
                        " DeseqDataSet initialization"
                    )

        _control_mask = self._make_control_mask(control_genes)
        normalization_control_mask = _control_mask.copy()

        transcript_fit_type: Literal["ratio", "poscounts"] | None = None
        if "avg_tx_length" in self.layers:
            if fit_type == "iterative":
                raise ValueError(
                    "The iterative size-factor method does not support "
                    "transcript-length normalization. Use 'ratio' or 'poscounts'."
                )
            transcript_fit_type = fit_type

        self._invalidate_lrt_cache(discard_replace_counts=True)

        if transcript_fit_type is None and "normalization_factors" in self.layers:
            del self.layers["normalization_factors"]

        if transcript_fit_type is not None:
            self._fit_transcript_length_factors(
                fit_type=transcript_fit_type,
                control_mask=_control_mask,
            )

        elif fit_type == "iterative":
            self._fit_iterate_size_factors()

        elif fit_type == "poscounts":
            counts = (
                cast(Any, self.X).toarray() if issparse(self.X) else np.asarray(self.X)
            )

            # Calculate logcounts for x > 0 and take the mean for each gene
            log_counts = np.zeros_like(counts, dtype=float)
            np.log(counts, out=log_counts, where=counts != 0)
            logmeans = log_counts.mean(axis=0)

            # Determine which genes are usable (finite logmeans)
            self.filtered_genes = (~np.isinf(logmeans)) & (logmeans > 0)
            _control_mask &= self.filtered_genes

            # Calculate size factor per sample
            def sizeFactor(x):
                _mask = np.logical_and(_control_mask, x > 0)
                return np.exp(np.median(np.log(x[_mask]) - logmeans[_mask]))

            sf = np.apply_along_axis(sizeFactor, 1, counts)
            del log_counts

            # Normalize size factors to a geometric mean of 1 to match DESeq
            self.obs["size_factors"] = sf / (np.exp(np.mean(np.log(sf))))
            self.layers["normed_counts"] = (
                counts / self.obs["size_factors"].values[:, None]
            )
            self.logmeans = logmeans

        # Test whether it is possible to use median-of-ratios.
        elif (
            (np.asarray((cast(Any, self.X) != 0).sum(axis=0)).ravel() < self.n_obs).all()
            if issparse(self.X)
            else (self.X == 0).any(0).all()
        ):
            # There is at least a zero for each gene
            warnings.warn(
                "Every gene contains at least one zero, "
                "cannot compute log geometric means. Switching to iterative mode.",
                UserWarning,
                stacklevel=2,
            )
            self._fit_iterate_size_factors()

        elif self.X is not None:
            counts = (
                cast(Any, self.X).toarray() if issparse(self.X) else np.asarray(self.X)
            )
            self.logmeans, self.filtered_genes = deseq2_norm_fit(counts)
            _control_mask &= self.filtered_genes

            (
                self.layers["normed_counts"],
                self.obs["size_factors"],
            ) = deseq2_norm_transform(
                counts, cast(np.ndarray, self.logmeans), _control_mask
            )
        else:
            raise ValueError("Counts matrix 'X' is None, cannot fit size factors.")

        end = time.time()
        self.var["_normed_means"] = self.layers["normed_counts"].mean(axis=0)
        self._normalization_fit_type = fit_type
        self._normalization_control_mask = normalization_control_mask

        if not self.quiet:
            print(f"... done in {end - start:.2f} seconds.\n", file=sys.stderr)

    def fit_genewise_dispersions(self, vst=False) -> None:
        """Fit gene-wise dispersion estimates.

        Fits a negative binomial per gene, independently.

        Parameters
        ----------
        vst : bool
            Whether the dispersion estimates are being fitted as part of the VST
            pipeline. (default: ``False``).
        """
        if not vst:
            self._invalidate_lrt_cache(discard_replace_counts=True)

        # Check that size factors are available. If not, compute them.
        if "size_factors" not in self.obs:
            self.fit_size_factors(fit_type=self.size_factors_fit_type)

        # Exclude genes with all zeroes
        self.var["non_zero"] = np.asarray((self.X != 0).sum(axis=0)).ravel() > 0
        self.non_zero_idx = np.arange(self.n_vars)[self.var["non_zero"]]
        self.non_zero_genes = self.var_names[self.var["non_zero"]]

        if isinstance(self.non_zero_genes, pd.MultiIndex):
            raise ValueError("non_zero_genes should not be a MultiIndex")

        # Fit "method of moments" dispersion estimates
        self._fit_MoM_dispersions()

        # Convert design_matrix to numpy for speed
        design_matrix = self.obsm["design_matrix"].values
        size_factors = self._get_normalization_factors(self.non_zero_idx)

        # mu_hat is initialized differently depending on the number of different factor
        # groups. If there are as many different factor combinations as design factors
        # (intercept included), it is fitted with a linear model, otherwise it is fitted
        # with a GLM (using rough dispersion estimates).
        if (
            len(self.obsm["design_matrix"].value_counts())
            == self.obsm["design_matrix"].shape[-1]
        ):
            mu_hat_ = self.inference.lin_reg_mu(
                counts=self.X[:, self.non_zero_idx],
                size_factors=size_factors,
                design_matrix=design_matrix,
                min_mu=self.min_mu,
            )
        else:
            _, mu_hat_, _, _ = self.inference.irls(
                counts=self.X[:, self.non_zero_idx],
                size_factors=size_factors,
                design_matrix=design_matrix,
                disp=self.var.loc[self.var["non_zero"], "_MoM_dispersions"].values,
                min_mu=self.min_mu,
                beta_tol=self.beta_tol,
            )

        mu_param_name = "_vst_mu_hat" if vst else "_mu_hat"
        disp_param_name = "vst_genewise_dispersions" if vst else "genewise_dispersions"

        self.layers[mu_param_name] = np.full((self.n_obs, self.n_vars), np.nan)
        self.layers[mu_param_name][:, self.var["non_zero"]] = mu_hat_

        del mu_hat_

        if not self.quiet:
            print("Fitting dispersions...", file=sys.stderr)
        start = time.time()
        dispersions_, l_bfgs_b_converged_ = self.inference.alpha_mle(
            counts=self.X[:, self.non_zero_idx],
            design_matrix=design_matrix,
            mu=self.layers[mu_param_name][:, self.non_zero_idx],
            alpha_hat=self.var.loc[self.var["non_zero"], "_MoM_dispersions"].values,
            min_disp=self.min_disp,
            max_disp=self.max_disp,
        )
        end = time.time()

        if not self.quiet:
            print(f"... done in {end - start:.2f} seconds.\n", file=sys.stderr)

        self.var[disp_param_name] = np.full(self.n_vars, np.nan)
        self.var.loc[self.var["non_zero"], disp_param_name] = np.clip(
            dispersions_, self.min_disp, self.max_disp
        )

        self.var["_genewise_converged"] = pd.array(
            [pd.NA] * self.n_vars, dtype="boolean"
        )
        self.var.loc[self.var["non_zero"], "_genewise_converged"] = l_bfgs_b_converged_

    def fit_dispersion_trend(self, vst: bool = False) -> None:
        """Fit the dispersion trend curve.

        Parameters
        ----------
        vst : bool
            Whether the dispersion trend curve is being fitted as part of the VST
            pipeline. (default: ``False``).
        """
        if not vst:
            self._invalidate_lrt_cache(discard_replace_counts=True)

        disp_param_name = "vst_genewise_dispersions" if vst else "genewise_dispersions"
        fit_type = self.vst_fit_type if vst else self.fit_type

        # Check that genewise dispersions are available. If not, compute them.
        if disp_param_name not in self.var:
            self.fit_genewise_dispersions(vst)

        if not self.quiet:
            print("Fitting dispersion trend curve...", file=sys.stderr)
        start = time.time()

        if fit_type == "parametric":
            self._fit_parametric_dispersion_trend(vst)
        elif fit_type == "mean":
            self._fit_mean_dispersion_trend(vst)
        else:
            raise NotImplementedError(
                f"Expected 'parametric' or 'mean' trend curve fit "
                f"types, received {fit_type}"
            )
        end = time.time()

        if not self.quiet:
            print(f"... done in {end - start:.2f} seconds.\n", file=sys.stderr)

    def disp_function(self, x):
        """Return the dispersion trend function at x."""
        if self.uns["disp_function_type"] == "parametric":
            return dispersion_trend(x, self.uns["trend_coeffs"])
        elif self.uns["disp_function_type"] == "mean":
            return np.full_like(x, self.uns["mean_disp"])

    def fit_dispersion_prior(self) -> None:
        """Fit dispersion variance priors and standard deviation of log-residuals.

        The computation is based on genes whose dispersions are above 100 * min_disp.

        Note: when the design matrix has fewer than 3 degrees of freedom, the
        estimate of log dispersions is likely to be imprecise.
        """
        self._invalidate_lrt_cache(discard_replace_counts=True)

        # Check that the dispersion trend curve was fitted. If not, fit it.
        if "fitted_dispersions" not in self.var:
            self.fit_dispersion_trend()

        # Exclude genes with all zeroes
        num_samples = self.n_obs
        num_vars = self.obsm["design_matrix"].shape[-1]

        # Check the degrees of freedom
        if (num_samples - num_vars) <= 3:
            warnings.warn(
                "As the residual degrees of freedom is less than 3, the distribution "
                "of log dispersions is especially asymmetric and likely to be poorly "
                "estimated by the MAD.",
                UserWarning,
                stacklevel=2,
            )

        # Fit dispersions to the curve, and compute log residuals
        disp_residuals = np.log(
            self[:, self.non_zero_genes].var["genewise_dispersions"]
        ) - np.log(self[:, self.non_zero_genes].var["fitted_dispersions"])

        # Compute squared log-residuals and prior variance based on genes whose
        # dispersions are above 100 * min_disp. This is to reproduce DESeq2's behaviour.
        above_min_disp = self[:, self.non_zero_genes].var["genewise_dispersions"] >= (
            100 * self.min_disp
        )

        self.uns["_squared_logres"] = (
            mean_absolute_deviation(disp_residuals[above_min_disp]) ** 2
        )

        self.uns["prior_disp_var"] = np.maximum(
            self.uns["_squared_logres"] - polygamma(1, (num_samples - num_vars) / 2),
            0.25,
        )

    def fit_MAP_dispersions(self) -> None:
        """Fit Maximum a Posteriori dispersion estimates.

        After MAP dispersions are fit, filter genes for which we don't apply shrinkage.
        """
        self._invalidate_lrt_cache(discard_replace_counts=True)

        # Check that the dispersion prior variance is available. If not, compute it.
        if "prior_disp_var" not in self.uns:
            self.fit_dispersion_prior()

        # Convert design matrix to numpy for speed
        design_matrix = self.obsm["design_matrix"].values

        if not self.quiet:
            print("Fitting MAP dispersions...", file=sys.stderr)
        start = time.time()
        dispersions_, l_bfgs_b_converged_ = self.inference.alpha_mle(
            counts=self.X[:, self.non_zero_idx],
            design_matrix=design_matrix,
            mu=self.layers["_mu_hat"][:, self.non_zero_idx],
            alpha_hat=self.var.loc[self.var["non_zero"], "fitted_dispersions"].values,
            min_disp=self.min_disp,
            max_disp=self.max_disp,
            prior_disp_var=self.uns["prior_disp_var"].item(),
            cr_reg=True,
            prior_reg=True,
        )
        end = time.time()

        if not self.quiet:
            print(f"... done in {end - start:.2f} seconds.\n", file=sys.stderr)

        self.var["MAP_dispersions"] = np.full(self.n_vars, np.nan)
        self.var.loc[self.var["non_zero"], "MAP_dispersions"] = np.clip(
            dispersions_, self.min_disp, self.max_disp
        )

        self.var["_MAP_converged"] = pd.array([pd.NA] * self.n_vars, dtype="boolean")
        self.var.loc[self.var["non_zero"], "_MAP_converged"] = l_bfgs_b_converged_

        # Filter outlier genes for which we won't apply shrinkage
        self.var["dispersions"] = self.var["MAP_dispersions"].copy()
        self.var["_outlier_genes"] = np.log(self.var["genewise_dispersions"]) > np.log(
            self.var["fitted_dispersions"]
        ) + 2 * np.sqrt(self.uns["_squared_logres"])
        self.var.loc[self.var["_outlier_genes"], "dispersions"] = self.var.loc[
            self.var["_outlier_genes"], "genewise_dispersions"
        ]

        if self.low_memory:
            del self.layers["_mu_hat"]

    def fit_LFC(self) -> None:
        """Fit log fold change (LFC) coefficients.

        In the 2-level setting, the intercept corresponds to the base mean,
        while the second is the actual LFC coefficient, in natural log scale.
        """
        self._invalidate_lrt_cache(discard_replace_counts=True)

        # Check that MAP dispersions are available. If not, compute them.
        if "dispersions" not in self.var:
            self.fit_MAP_dispersions()

        # Convert design matrix to numpy for speed
        design_matrix = self.obsm["design_matrix"].values

        if not self.quiet:
            print("Fitting LFCs...", file=sys.stderr)
        start = time.time()
        normalization_factors = self._get_normalization_factors(self.non_zero_idx)
        mle_lfcs_, mu_, hat_diagonals_, converged_ = self.inference.irls(
            counts=self.X[:, self.non_zero_idx],
            size_factors=normalization_factors,
            design_matrix=design_matrix,
            disp=self.var.loc[self.var["non_zero"], "dispersions"].values,
            min_mu=self.min_mu,
            beta_tol=self.beta_tol,
        )
        end = time.time()

        if not self.quiet:
            print(f"... done in {end - start:.2f} seconds.\n", file=sys.stderr)

        self.varm["LFC"] = pd.DataFrame(
            np.nan,
            index=self.var_names,
            columns=self.obsm["design_matrix"].columns,
        )

        self.varm["LFC"].update(
            pd.DataFrame(
                mle_lfcs_,
                index=self.non_zero_genes,
                columns=self.obsm["design_matrix"].columns,
            )
        )

        self.obsm["_mu_LFC"] = mu_
        self.obsm["_hat_diagonals"] = hat_diagonals_

        self.var["_LFC_converged"] = pd.array([pd.NA] * self.n_vars, dtype="boolean")
        self.var.loc[self.var["non_zero"], "_LFC_converged"] = converged_

    def calculate_cooks(self) -> None:
        """Compute Cook's distance for outlier detection.

        Measures the contribution of a single entry to the output of LFC estimation.
        """
        # Check that MAP dispersions are available. If not, compute them.
        if "dispersions" not in self.var:
            self.fit_MAP_dispersions()

        if not self.quiet:
            print("Calculating cook's distance...", file=sys.stderr)

        start = time.time()
        num_vars = self.obsm["design_matrix"].shape[-1]

        # Calculate dispersion
        dispersions = robust_method_of_moments_disp(
            self.layers["normed_counts"][:, self.var["non_zero"]],
            self.obsm["design_matrix"],
        )

        # Calculate the squared pearson residuals for non-zero features
        counts = self.X[:, self.non_zero_idx]
        if hasattr(counts, "tocoo"):
            counts = counts.tocoo()
            squared_pearson_res = -self.obsm["_mu_LFC"]
            np.add.at(squared_pearson_res, (counts.row, counts.col), counts.data)
        else:
            squared_pearson_res = counts - self.obsm["_mu_LFC"]
        squared_pearson_res **= 2

        # Calculate the overdispersion parameter tau
        V = self.obsm["_mu_LFC"] ** 2
        V *= dispersions[None, :]
        V += self.obsm["_mu_LFC"]

        # Calculate r^2 / (tau * num_vars)
        squared_pearson_res /= V
        squared_pearson_res /= num_vars

        del V

        # Calculate leverage modifier H / (1 - H)^2
        diag_mul = 1 - self.obsm["_hat_diagonals"]
        diag_mul **= 2
        diag_mul = self.obsm["_hat_diagonals"] / diag_mul

        # Multiply r^2 / (tau * num_vars) by H / (1 - H)^2 to get cook's distance
        squared_pearson_res *= diag_mul

        del diag_mul

        if self.low_memory:
            del self.obsm["_mu_LFC"]
            del self.obsm["_hat_diagonals"]

        self.layers["cooks"] = np.full((self.n_obs, self.n_vars), np.nan)
        self.layers["cooks"][:, self.var["non_zero"]] = squared_pearson_res

        if not self.quiet:
            print(f"... done in {time.time() - start:.2f} seconds.\n", file=sys.stderr)

    def refit(self) -> None:
        """Refit Cook outliers.

        Replace values that are filtered out based on the Cooks distance with imputed
        values, and then re-run the whole DESeq2 pipeline on replaced values.
        """
        self._invalidate_lrt_cache(discard_replace_counts=True)
        self._clear_cooks_refit_state()

        # Replace outlier counts
        self._replace_outliers()
        if not self.quiet:
            print(
                f"Replacing {sum(self.var['replaced'])} outlier genes.\n",
                file=sys.stderr,
            )

        if sum(self.var["replaced"]) > 0:
            # Refit dispersions and LFCs for genes that had outliers replaced
            self._refit_without_outliers()
        else:
            # Store the fact that no sample was refitted
            self.var["refitted"] = np.full(
                self.n_vars,
                False,
            )

    def cooks_outlier(self):
        """Filter p-values based on Cooks outliers."""
        if "_pvalue_cooks_outlier" in self.var.keys():
            return self.var["_pvalue_cooks_outlier"]

        num_samples = self.n_obs
        num_vars = self.obsm["design_matrix"].shape[-1]
        cooks_cutoff = f.ppf(0.99, num_vars, num_samples - num_vars)

        # As in DESeq2, only take samples with 3 or more replicates when looking for
        # max cooks.
        use_for_max = n_or_more_replicates(self.obsm["design_matrix"], 3)

        # If for a gene there are 3 samples or more that have more counts than the
        # maximum cooks sample, don't count this gene as an outlier.

        # Take into account whether we already replaced outliers
        if (
            self.refit_cooks
            and (self.var["refitted"].sum() > 0)
            and "replace_cooks" in self.layers.keys()
        ):
            cooks_outlier = (
                self.layers["replace_cooks"][use_for_max, :] > cooks_cutoff
            ).any(axis=0)

        else:
            cooks_outlier = (self.layers["cooks"][use_for_max, :] > cooks_cutoff).any(
                axis=0
            )

        pos = np.asarray(self.layers["cooks"][:, cooks_outlier].argmax(0)).ravel()
        for gene_idx, sample_idx in zip(np.flatnonzero(cooks_outlier), pos, strict=True):
            gene_counts = (
                cast(Any, self.X)[:, [gene_idx]].toarray().ravel()
                if issparse(self.X)
                else np.asarray(self.X[:, gene_idx]).ravel()
            )
            cooks_outlier[gene_idx] = (gene_counts > gene_counts[sample_idx]).sum() < 3

        if self.low_memory:
            del self.layers["cooks"]

        if self.low_memory and "replace_cooks" in self.layers.keys():
            del self.layers["replace_cooks"]

        self.var["_pvalue_cooks_outlier"] = cooks_outlier
        return self.var["_pvalue_cooks_outlier"]

    def to_picklable_anndata(self) -> ad.AnnData:
        """Convert the DESeqDataSet to a picklable AnnData object.

        Builds an AnnData object from the DESeqDataSet with the same data, but converts
        the design matrix to a DataFrame to remove the formulaic model_spec attribute
        and internal dispersion-trend Series to NumPy arrays. These conversions do not
        mutate the DeseqDataSet or user-provided unstructured metadata.

        Returns
        -------
        anndata.AnnData
            The AnnData object, without DeseqDataSet unpicklable attributes.
        """
        serializable_uns = dict(self.uns)
        for key in ("trend_coeffs", "vst_trend_coeffs"):
            value = serializable_uns.get(key)
            if isinstance(value, pd.Series):
                serializable_uns[key] = value.to_numpy(copy=True)

        # Initialize an AnnData object
        adata = ad.AnnData(
            X=self.X,
            obs=self.obs,
            var=self.var,
            obsm=self.obsm,
            varm=self.varm,
            uns=serializable_uns,
            layers=self.layers,
        )

        # Convert the design matrix to a DataFrame to remove model_spec
        adata.obsm["design_matrix"] = pd.DataFrame(adata.obsm["design_matrix"])

        return adata

    def _fit_MoM_dispersions(self) -> None:
        """Rough method of moments initial dispersions fit.

        Estimates are the max of "robust" and "method of moments" estimates.
        """
        # Check that size_factors are available. If not, compute them.
        if "normed_counts" not in self.layers:
            self.fit_size_factors(fit_type=self.size_factors_fit_type)

        normed_counts = self.layers["normed_counts"][:, self.non_zero_idx]
        rde = self.inference.fit_rough_dispersions(
            normed_counts,
            self.obsm["design_matrix"].values,
        )
        normalization_factors = self._get_normalization_factors(self.non_zero_idx)
        mde = self.inference.fit_moments_dispersions(
            normed_counts, normalization_factors
        )
        alpha_hat = np.minimum(rde, mde)

        self.var["_MoM_dispersions"] = np.full(self.n_vars, np.nan)
        self.var.loc[self.var["non_zero"], "_MoM_dispersions"] = np.clip(
            alpha_hat, self.min_disp, self.max_disp
        )

    def plot_dispersions(
        self, log: bool = True, save_path: str | None = None, **kwargs
    ) -> None:
        """Plot dispersions.

        Make a scatter plot with genewise dispersions, trend curve and final (MAP)
        dispersions.

        Parameters
        ----------
        log : bool
            Whether to log scale x and y axes (``default=True``).

        save_path : str, optional
            The path where to save the plot. If left None, the plot won't be saved
            (``default=None``).

        **kwargs
            Keyword arguments for the scatter plot.
        """
        disps = [
            self.var["genewise_dispersions"],
            self.var["dispersions"],
            self.var["fitted_dispersions"],
        ]
        legend_labels = ["Estimated", "Final", "Fitted"]
        make_scatter(
            disps,
            legend_labels=legend_labels,
            x_val=self.var["_normed_means"],
            log=log,
            save_path=save_path,
            **kwargs,
        )

    def _fit_parametric_dispersion_trend(self, vst: bool = False):
        r"""Fit the dispersion curve according to a parametric model.

        :math:`f(\mu) = \alpha_1/\mu + a_0`.

        Parameters
        ----------
        vst : bool
            Whether the dispersion trend curve is being fitted as part of the VST
            pipeline. (default: ``False``).
        """
        disp_param_name = "vst_genewise_dispersions" if vst else "genewise_dispersions"

        if disp_param_name not in self.var:
            self.fit_genewise_dispersions(vst)

        # Exclude all-zero counts
        targets = pd.Series(
            self.var.loc[self.non_zero_genes, disp_param_name].copy(),
            index=self.non_zero_genes,
        )
        covariates = pd.Series(
            1 / self.var.loc[self.non_zero_genes, "_normed_means"],
            index=self.non_zero_genes,
        )

        for gene in self.non_zero_genes:
            if (
                np.isinf(covariates.loc[gene]).any()
                or np.isnan(covariates.loc[gene]).any()
            ):
                targets.drop(labels=[gene], inplace=True)
                covariates.drop(labels=[gene], inplace=True)

        # Initialize coefficients
        old_coeffs: np.ndarray | pd.Series = pd.Series([0.1, 0.1])
        coeffs: np.ndarray | pd.Series = pd.Series([1.0, 1.0])
        while (coeffs > 1e-10).all() and (
            np.log(np.abs(coeffs / old_coeffs)) ** 2
        ).sum() >= 1e-6:
            old_coeffs = coeffs
            coeffs, predictions, converged = self.inference.dispersion_trend_gamma_glm(
                covariates, targets
            )
            if not converged or (coeffs <= 1e-10).any():
                warnings.warn(
                    "The dispersion trend curve fitting did not converge. "
                    "Switching to a mean-based dispersion trend.",
                    UserWarning,
                    stacklevel=2,
                )

                self._fit_mean_dispersion_trend(vst)
                return

            # Filter out genes that are too far away from the curve before refitting
            pred_ratios = self.var.loc[covariates.index, disp_param_name] / predictions

            targets.drop(
                targets[(pred_ratios < 1e-4) | (pred_ratios >= 15)].index,
                inplace=True,
            )
            covariates.drop(
                covariates[(pred_ratios < 1e-4) | (pred_ratios >= 15)].index,
                inplace=True,
            )

        if vst:
            self.uns["vst_trend_coeffs"] = pd.Series(coeffs, index=["a0", "a1"])
        else:
            self.uns["trend_coeffs"] = pd.Series(coeffs, index=["a0", "a1"])

            self.var["fitted_dispersions"] = np.full(self.n_vars, np.nan)
            self.uns["disp_function_type"] = "parametric"
            self.var.loc[self.var["non_zero"], "fitted_dispersions"] = (
                self.disp_function(self.var.loc[self.var["non_zero"], "_normed_means"])
            )

    def _fit_mean_dispersion_trend(self, vst: bool = False):
        """Use the mean of dispersions as trend curve.

        Parameters
        ----------
        vst : bool
            Whether the dispersion trend curve is being fitted as part of the VST
            pipeline. (default: ``False``).
        """
        disp_param_name = "vst_genewise_dispersions" if vst else "genewise_dispersions"

        self.uns["mean_disp"] = trim_mean(
            self.var.loc[
                self.var[disp_param_name] > 10 * self.min_disp, disp_param_name
            ].values,
            proportiontocut=0.001,
        )

        if vst:
            self.vst_fit_type = "mean"
        else:
            self.uns["disp_function_type"] = "mean"
        self.var["fitted_dispersions"] = np.full(self.n_vars, self.uns["mean_disp"])

    def _replace_outliers(self) -> None:
        """Replace values that are filtered out (based on Cooks) with imputed values."""
        self._discard_owned_replace_counts()

        # Check that cooks distances are available. If not, compute them.
        if "cooks" not in self.layers:
            self.calculate_cooks()

        num_samples = self.n_obs
        num_vars = self.obsm["design_matrix"].shape[1]

        # Check whether cohorts have enough samples to allow refitting
        self.obs["replaceable"] = n_or_more_replicates(
            self.obsm["design_matrix"], self.min_replicates
        ).values

        if self.obs["replaceable"].sum() == 0:
            # No sample can be replaced. Set self.replaced to False and exit.
            self.var["replaced"] = np.full(
                self.n_vars,
                False,
            )
            return

        # Get positions of counts with cooks above threshold
        cooks_cutoff = f.ppf(0.99, num_vars, num_samples - num_vars)
        idx = self.layers["cooks"] > cooks_cutoff
        self.var["replaced"] = idx.any(axis=0)
        replaced = self.var["replaced"].to_numpy(dtype=bool)

        if replaced.any():
            if _REPLACE_COUNTS_LAYER in self.layers:
                raise ValueError(
                    "Cook replacement cannot create its private delta layer because "
                    "that key is already used by state not owned by PyDESeq2."
                )

            # Compute replacement counts: trimmed means * normalization factors
            self.counts_to_refit = self[:, replaced].copy()
            if hasattr(self.counts_to_refit.X, "toarray"):
                self.counts_to_refit.X = self.counts_to_refit.X.toarray()
            refit_counts = np.asarray(self.counts_to_refit.X)
            normalization_factors = self._get_normalization_factors(
                self.var["replaced"].to_numpy()
            )
            if normalization_factors.ndim == 1:
                normalization_factors = normalization_factors[:, None]

            trim_base_mean = np.asarray(
                trimmed_mean(
                    refit_counts / normalization_factors,
                    trim=0.2,
                    axis=0,
                )
            )
            replacement_mask = (
                self.obs["replaceable"].to_numpy(dtype=bool)[:, None] & idx[:, replaced]
            )
            changed_rows, changed_local_columns = np.nonzero(replacement_mask)
            original_values = refit_counts[changed_rows, changed_local_columns].astype(
                np.int64, copy=True
            )
            if normalization_factors.shape[1] == 1:
                replacement_factors = normalization_factors[changed_rows, 0]
            else:
                replacement_factors = normalization_factors[
                    changed_rows, changed_local_columns
                ]
            replacement_values = np.asarray(
                trim_base_mean[changed_local_columns] * replacement_factors,
                dtype=np.int64,
            )
            refit_counts[changed_rows, changed_local_columns] = replacement_values

            delta_values = replacement_values - original_values
            changed = delta_values != 0
            changed_rows = changed_rows[changed]
            changed_columns = np.flatnonzero(replaced)[changed_local_columns[changed]]
            self.layers[_REPLACE_COUNTS_LAYER] = csr_matrix(
                (
                    delta_values[changed],
                    (changed_rows, changed_columns),
                ),
                shape=self.shape,
                dtype=np.int64,
            )
            self.uns[_REPLACE_COUNTS_OWNER_KEY] = True

    def _refit_without_outliers(
        self,
    ) -> None:
        """Re-run the whole DESeq2 pipeline with replaced outliers."""
        assert self.refit_cooks, (
            "Trying to refit Cooks outliers but the 'refit_cooks' flag is set to False"
        )

        # Check that _replace_outliers() was previously run.
        if "replaced" not in self.var:
            self._replace_outliers()

        # Only refit genes for which replacing outliers hasn't resulted in all zeroes
        new_all_zeroes = (self.counts_to_refit.X == 0).all(axis=0)
        self.new_all_zeroes_genes = self.counts_to_refit.var_names[new_all_zeroes]

        self.var["refitted"] = self.var["replaced"].copy()
        # Only replace if genes are not all zeroes after outlier replacement
        self.var.loc[self.var["refitted"], "refitted"] = ~new_all_zeroes

        # Take into account new all-zero genes
        if new_all_zeroes.sum() > 0:
            self.var.loc[self.new_all_zeroes_genes, "_normed_means"] = 0
            self.varm["LFC"].loc[self.new_all_zeroes_genes, :] = 0
            cast(pd.DataFrame, self.var).loc[
                self.new_all_zeroes_genes, "_LFC_converged"
            ] = pd.NA

        if self.var["refitted"].sum() == 0:  # if no gene can be refitted, we can skip
            return

        self.counts_to_refit = self.counts_to_refit[:, ~new_all_zeroes].copy()
        if isinstance(self.new_all_zeroes_genes, pd.MultiIndex):
            raise ValueError

        sub_dds = DeseqDataSet(
            counts=pd.DataFrame(
                self.counts_to_refit.X,
                index=self.counts_to_refit.obs_names,
                columns=self.counts_to_refit.var_names,
            ),
            metadata=self.obs,
            design=self.design,
            min_mu=self.min_mu,
            min_disp=self.min_disp,
            max_disp=self.max_disp,
            refit_cooks=self.refit_cooks,
            min_replicates=self.min_replicates,
            beta_tol=self.beta_tol,
            inference=self.inference,
            quiet=self.quiet,
        )

        # Use the same normalization factors
        sub_dds.obs["size_factors"] = self.counts_to_refit.obs["size_factors"]
        if "normalization_factors" in self.counts_to_refit.layers:
            sub_dds.layers["normalization_factors"] = self.counts_to_refit.layers[
                "normalization_factors"
            ]
        normalization_factors = sub_dds._get_normalization_factors()
        if normalization_factors.ndim == 1:
            normalization_factors = normalization_factors[:, None]
        sub_dds.layers["normed_counts"] = sub_dds.X / normalization_factors

        # Estimate gene-wise dispersions.
        sub_dds.fit_genewise_dispersions()

        # Compute trend dispersions.
        # Note: the trend curve is not refitted.
        sub_dds.uns["disp_function_type"] = self.uns["disp_function_type"]
        if sub_dds.uns["disp_function_type"] == "parametric":
            sub_dds.uns["trend_coeffs"] = self.uns["trend_coeffs"]
        elif sub_dds.uns["disp_function_type"] == "mean":
            sub_dds.uns["mean_disp"] = self.uns["mean_disp"]
        sub_dds.var["_normed_means"] = sub_dds.layers["normed_counts"].mean(axis=0)
        # Reshape in case there's a single gene to refit
        sub_dds.var["fitted_dispersions"] = sub_dds.disp_function(
            sub_dds.var["_normed_means"]
        )

        # Estimate MAP dispersions.
        # Note: the prior variance is not recomputed.
        sub_dds.uns["_squared_logres"] = self.uns["_squared_logres"]
        sub_dds.uns["prior_disp_var"] = self.uns["prior_disp_var"]

        sub_dds.fit_MAP_dispersions()

        # Estimate log-fold changes (in natural log scale)
        sub_dds.fit_LFC()

        # Replace values in main object
        self.var.loc[self.var["refitted"], "_normed_means"] = sub_dds.var[
            "_normed_means"
        ]
        self.varm["LFC"][self.var["refitted"]] = sub_dds.varm["LFC"]
        cast(pd.DataFrame, self.var).loc[self.var["refitted"], "_LFC_converged"] = (
            sub_dds.var["_LFC_converged"].array
        )
        self.var.loc[self.var["refitted"], "genewise_dispersions"] = sub_dds.var[
            "genewise_dispersions"
        ]
        self.var.loc[self.var["refitted"], "fitted_dispersions"] = sub_dds.var[
            "fitted_dispersions"
        ]
        self.var.loc[self.var["refitted"], "dispersions"] = sub_dds.var["dispersions"]

        self.layers["replace_cooks"] = self.layers["cooks"].copy()

        for col in np.where(self.var["refitted"])[0]:
            self.layers["replace_cooks"][self.obs["replaceable"], col] = 0.0

    def _fit_iterate_size_factors(self, niter: int = 10, quant: float = 0.95) -> None:
        """
        Fit size factors using the ``iterative`` method.

        Used when each gene has at least one zero.

        Parameters
        ----------
        niter : int
            Maximum number of iterations to perform (default: ``10``).

        quant : float
            Quantile value at which negative likelihood is cut in the optimization
            (default: ``0.95``).

        """
        self.logmeans = None
        self.filtered_genes = None
        counts = cast(Any, self.X).toarray() if issparse(self.X) else np.asarray(self.X)

        # Initialize size factors and normed counts fields
        self.obs["size_factors"] = np.ones(self.n_obs)
        self.layers["normed_counts"] = counts

        # Reduce the design matrix to an intercept and reconstruct at the end
        self.obsm["design_matrix_buffer"] = self.obsm["design_matrix"].copy()
        self.obsm["design_matrix"] = pd.DataFrame(
            1, index=self.obs_names, columns=["Intercept"]
        )

        # Fit size factors using MLE
        def objective(p):
            sf = np.exp(p - np.mean(p))
            nll = nb_nll(
                counts=counts[:, self.non_zero_idx],
                mu=self[:, self.non_zero_genes].layers["_mu_hat"]
                / self.obs["size_factors"].values[:, None]
                * sf[:, None],
                alpha=self.var.loc[self.non_zero_genes, "dispersions"].values,
            )
            # Take out the lowest likelihoods (highest neg) from the sum
            return np.sum(nll[nll < np.quantile(nll, quant)])

        for i in range(niter):
            # Estimate dispersions based on current size factors
            self.fit_genewise_dispersions()

            # Use a mean trend curve
            use_for_mean_genes = self.var_names[
                (self.var["genewise_dispersions"] > 10 * self.min_disp)
                & self.var["non_zero"]
            ]

            if len(use_for_mean_genes) == 0:
                print(
                    "No genes have a dispersion above 10 * min_disp in "
                    "_fit_iterate_size_factors.",
                    file=sys.stderr,
                )
                break

            mean_disp = trim_mean(
                self[:, use_for_mean_genes].var["genewise_dispersions"],
                proportiontocut=0.001,
            )

            self.var["fitted_dispersions"] = np.ones(self.n_vars) * mean_disp
            self.fit_dispersion_prior()
            self.fit_MAP_dispersions()
            old_sf = self.obs["size_factors"].copy()

            # Fit size factors using MLE
            res = minimize(objective, np.log(old_sf), method="Powell")

            self.obs["size_factors"] = np.exp(res.x - np.mean(res.x))

            if not res.success:
                print("A size factor fitting iteration failed.", file=sys.stderr)
                break

            if (i > 1) and np.sum(
                (np.log(old_sf) - np.log(self.obs["size_factors"])) ** 2
            ) < 1e-4:
                break
            elif i == niter - 1:
                print("Iterative size factor fitting did not converge.", file=sys.stderr)

        # Restore the design matrix and free buffer
        self.obsm["design_matrix"] = self.obsm["design_matrix_buffer"].copy()
        del self.obsm["design_matrix_buffer"]

        # Store normalized counts
        self.layers["normed_counts"] = counts / self.obs["size_factors"].values[:, None]

    def _check_full_rank_design(self):
        """Check that the design matrix has full column rank."""
        rank = np.linalg.matrix_rank(self.obsm["design_matrix"])
        num_vars = self.obsm["design_matrix"].shape[1]

        if rank < num_vars:
            warnings.warn(
                "The design matrix is not full rank, so the model cannot be "
                "fitted, but some operations like design-free VST remain possible. "
                "To perform differential expression analysis, please remove the design "
                "variables that are linear combinations of others.",
                UserWarning,
                stacklevel=2,
            )
