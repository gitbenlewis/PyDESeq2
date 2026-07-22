import sys
import time
import warnings
from typing import Literal

# import anndata as ad
import numpy as np
import pandas as pd
from scipy.optimize import root_scalar  # type: ignore
from scipy.stats import false_discovery_control  # type: ignore

from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.inference import Inference
from pydeseq2.utils import lowess
from pydeseq2.utils import make_MA_plot


class DeseqStats:
    """PyDESeq2 statistical tests for differential expression.

    Implements p-value estimation for differential gene expression according
    to the DESeq2 pipeline :cite:p:`DeseqStats-love2014moderated`.

    Also supports apeGLM log-fold change shrinkage :cite:p:`DeseqStats-zhu2019heavy`.

    Parameters
    ----------
    dds : DeseqDataSet
        DeseqDataSet for which dispersion and LFCs were already estimated.

    contrast : list or ndarray
        Either a list of three strings or a numpy array.
        If a list of three strings, it must be in the following format:
        ``['variable_of_interest', 'tested_level', 'ref_level']``.
        Names must correspond to the metadata data passed to the DeseqDataSet.
        E.g., ``['condition', 'B', 'A']`` will measure the LFC of 'condition B' compared
        to 'condition A'.
        If a numpy array, it must be a contrast vector of the same length as the design
        matrix.

    alpha : float
        P-value and adjusted p-value significance threshold (usually 0.05).
        (default: ``0.05``).

    cooks_filter : bool
        Whether to filter p-values based on cooks outliers. (default: ``True``).

    independent_filter : bool
        Whether to perform independent filtering to correct p-value trends.
        (default: ``True``).

    prior_LFC_var : ndarray
        Prior variance for LFCs, used for ridge regularization. (default: ``None``).

    lfc_null : float
        The (log2) log fold change under the null hypothesis. (default: ``0``).

    alt_hypothesis : str, optional
        The alternative hypothesis for computing wald p-values. By default, the normal
        Wald test assesses deviation of the estimated log fold change from the null
        hypothesis, as given by ``lfc_null``.
        One of ``["greaterAbs", "lessAbs", "greater", "less"]`` or ``None``.
        The alternative hypothesis corresponds to what the user wants to find rather
        than the null hypothesis. (default: ``None``).

    inference : Inference
        Implementation of inference routines object instance.
        (default:
        :class:`DefaultInference <pydeseq2.default_inference.DefaultInference>`).

    quiet : bool
        Suppress deseq2 status updates during fit.

    n_cpus : int, optional
        Number of CPUs to use. If an ``inference`` object is provided and exposes an
        ``n_cpus`` attribute, this value overrides it. Otherwise, it configures the
        default inference implementation. (default: ``None``).

    test : {"Wald", "LRT"}, optional
        Statistical test to run. ``None`` inherits the test prepared on ``dds`` (or
        defaults to ``"Wald"`` when no test was prepared). Test names are
        case-sensitive. (default: ``None``).

    reduced : str or pandas.DataFrame, optional
        Nested reduced design for ``test="LRT"``. A formula is supported when the full
        design is formula-based, and an explicit matrix is supported when the full
        design is matrix-based. If omitted for an inherited, prepared LRT, the cached
        reduced design is reused. (default: ``None``).

    Attributes
    ----------
    base_mean : pandas.Series
        Genewise means of normalized counts.

    lfc_null : float
        The (log2) log fold change under the null hypothesis.

    alt_hypothesis : str, optional
        The alternative hypothesis for computing wald p-values.

    test : {"Wald", "LRT"}
        Statistical test selected for this result object.

    reduced : str or pandas.DataFrame, optional
        User-provided reduced LRT design, when applicable.

    reduced_design_matrix : pandas.DataFrame, optional
        Validated reduced LRT design matrix, when applicable.

    lrt_df : int
        Degrees of freedom for an LRT, equal to the full-minus-reduced model rank.

    contrast_vector : ndarray
        Vector encoding the contrast (variable being tested).

    contrast_idx : int
        Index of the LFC column corresponding to the variable being tested.

    design_matrix : pandas.DataFrame
        A DataFrame with experiment design information (to split cohorts).
        Indexed by sample barcodes. Depending on the contrast that is provided to the
        DeseqStats object, it may differ from the DeseqDataSet design matrix, as the
        reference level may need to be adapted.

    LFC : pandas.DataFrame
        Estimated log-fold change between conditions and intercept, in natural log scale.

    SE : pandas.Series
        Standard LFC error.

    statistics : pandas.Series
        Wald statistics or omnibus full-versus-reduced LRT statistics.

    p_values : pandas.Series
        P-values estimated from the selected Wald or LRT statistics.

    reduced_converged : pandas.Series
        Per-gene reduced-model convergence flags for an LRT.

    full_converged : pandas.Series
        Per-gene full-model convergence flags for an LRT.

    full_deviance : pandas.Series
        Per-gene full-model deviance used by an LRT.

    new_all_zero : pandas.Series
        Per-gene flags identifying genes made all-zero by Cook's outlier replacement.

    padj : pandas.Series
        P-values adjusted for multiple testing.

    results_df : pandas.DataFrame
        Summary of the statistical analysis.

    shrunk_LFCs : bool
        Whether LFCs are shrunk.

    n_processes : int
        Number of threads to use for multiprocessing.

    quiet : bool
        Suppress deseq2 status updates during fit.

    References
    ----------
    .. bibliography::
        :keyprefix: DeseqStats-
    """

    def __init__(
        self,
        dds: DeseqDataSet,
        contrast: list[str] | np.ndarray,
        alpha: float = 0.05,
        cooks_filter: bool = True,
        independent_filter: bool = True,
        prior_LFC_var: np.ndarray | None = None,
        lfc_null: float = 0.0,
        alt_hypothesis: (
            Literal["greaterAbs", "lessAbs", "greater", "less"] | None
        ) = None,
        inference: Inference | None = None,
        quiet: bool = False,
        n_cpus: int | None = None,
        *,
        test: Literal["Wald", "LRT"] | None = None,
        reduced: str | pd.DataFrame | None = None,
    ) -> None:
        assert "LFC" in dds.varm, (
            "Please provide a fitted DeseqDataSet by first running the `deseq2` method."
        )

        self.dds = dds

        stored_test = self.dds.uns.get("_deseq2_test", "Wald")
        self.test = str(stored_test) if test is None else test
        self.dds._validate_test(self.test)

        self.alpha = alpha
        self.cooks_filter = cooks_filter
        self.independent_filter = independent_filter
        self.base_mean = self.dds.var["_normed_means"].copy()
        self.prior_LFC_var = prior_LFC_var

        if self.test == "Wald":
            if reduced is not None:
                raise ValueError("'reduced' is only supported when test='LRT'.")
            self.reduced = None
            self.reduced_design_matrix = None
        else:
            if prior_LFC_var is not None or lfc_null != 0 or alt_hypothesis is not None:
                raise ValueError(
                    "prior_LFC_var, lfc_null, and alt_hypothesis are only supported "
                    "for Wald tests."
                )
            if "_lrt" in self.dds.uns and not self.dds._lrt_fit_state_matches():
                raise RuntimeError(
                    "Cached LRT fit state no longer matches this DeseqDataSet. "
                    "Rerun dds.deseq2(test='LRT', reduced=...) before creating "
                    "DeseqStats."
                )
            self.reduced = reduced
            self.reduced_design_matrix = self.dds._prepare_lrt_reduced_design(reduced)
            self.lrt_df = int(
                self.dds.obsm["design_matrix"].shape[1]
                - self.reduced_design_matrix.shape[1]
            )

        if lfc_null < 0 and alt_hypothesis in {"greaterAbs", "lessAbs"}:
            raise ValueError(
                f"The alternative hypothesis being {alt_hypothesis}, please provide a",
                f"positive lfc_null value (got {lfc_null}).",
            )
        self.lfc_null = lfc_null
        self.alt_hypothesis = alt_hypothesis

        # Initialize the design matrix and LFCs. If the chosen reference level are the
        # same as in dds, keep them unchanged. Otherwise, change reference level.
        self.design_matrix = self.dds.obsm["design_matrix"].copy()
        self.LFC = self.dds.varm["LFC"].copy()

        # Check the validity of the contrast (if provided) or build it.
        self.contrast: list[str] | np.ndarray
        if contrast is None:
            raise ValueError(
                """Default contrasts are no longer supported.
                The "contrast" argument must be provided."""
            )
        elif isinstance(contrast, np.ndarray):
            if contrast.shape[0] != self.dds.obsm["design_matrix"].shape[1]:
                raise ValueError(
                    "The contrast vector must have the same length as the design matrix."
                )
            self.contrast = contrast
            self.contrast_vector = contrast
        else:
            self.contrast = contrast
            self._build_contrast_vector()

        # Set a flag to indicate that LFCs are unshrunk
        self.shrunk_LFCs = False
        self.quiet = quiet

        # Initialize the inference object.
        if inference is not None:
            self.inference = inference
        elif self.test == "LRT":
            self.inference = self.dds.inference
        else:
            self.inference = DefaultInference(n_cpus=n_cpus)

        self._force_local_lrt_refit = (
            inference is not None and inference is not self.dds.inference
        )

        if n_cpus is not None and (inference is not None or self.test == "LRT"):
            if hasattr(self.inference, "n_cpus"):
                self.inference.n_cpus = n_cpus
            else:
                warnings.warn(
                    "The selected inference object does not have an n_cpus "
                    "attribute, cannot override `n_cpus`.",
                    UserWarning,
                    stacklevel=2,
                )

        # If the `refit_cooks` attribute of the dds object is True, check that outliers
        # were actually refitted.
        if self.dds.refit_cooks and "replaced" not in self.dds.var:
            raise AttributeError(
                "dds has 'refit_cooks' set to True but Cooks outliers have not been "
                "refitted. Please run 'dds.refit()' first or set 'dds.refit_cooks' "
                "to False."
            )

    @property
    def variables(self):
        """Get the names of the variables used in the model definition."""
        return self.dds.variables

    def summary(
        self,
        **kwargs,
    ) -> None:
        """Run the statistical analysis.

        The results are stored in the ``results_df`` attribute.

        Parameters
        ----------
        **kwargs
            Keyword arguments: providing new values for ``lfc_null`` or
            ``alt_hypothesis`` will override the corresponding ``DeseqStat`` attributes.
        """
        new_lfc_null = kwargs.get("lfc_null", "default")
        new_alt_hypothesis = kwargs.get("alt_hypothesis", "default")

        rerun_summary = False
        if new_lfc_null == "default":
            lfc_null = self.lfc_null
        else:
            lfc_null = new_lfc_null
        if new_alt_hypothesis == "default":
            alt_hypothesis = self.alt_hypothesis
        else:
            alt_hypothesis = new_alt_hypothesis
        if lfc_null < 0 and alt_hypothesis in {"greaterAbs", "lessAbs"}:
            raise ValueError(
                f"The alternative hypothesis being {alt_hypothesis}, please provide a",
                f"positive lfc_null value (got {lfc_null}).",
            )

        if self.test == "LRT" and (lfc_null != 0 or alt_hypothesis is not None):
            raise ValueError(
                "lfc_null and alt_hypothesis are only supported for Wald tests."
            )

        if (
            not hasattr(self, "p_values")
            or self.lfc_null != lfc_null
            or self.alt_hypothesis != alt_hypothesis
        ):
            self.lfc_null = lfc_null
            self.alt_hypothesis = alt_hypothesis
            rerun_summary = True
            self._run_selected_test()

        if self.cooks_filter:
            # Filter p-values based on Cooks outliers
            self._cooks_filtering()

        if not hasattr(self, "padj") or rerun_summary:
            if self.independent_filter:
                # Compute adjusted p-values and correct p-value trend
                self._independent_filtering()
            else:
                # Compute adjusted p-values using the Benjamini-Hochberg method, without
                # correcting the p-value trend.
                self._p_value_adjustment()

        # Store the results in a DataFrame, in log2 scale for LFCs.
        self.results_df = pd.DataFrame(index=self.dds.var_names)
        self.results_df["baseMean"] = self.base_mean
        self.results_df["log2FoldChange"] = self.LFC @ self.contrast_vector / np.log(2)
        self.results_df["lfcSE"] = self.SE / np.log(2)
        self.results_df["stat"] = self.statistics
        self.results_df["pvalue"] = self.p_values
        self.results_df["padj"] = self.padj

        if not self.quiet:
            if isinstance(self.contrast, np.ndarray):
                # The contrast vector was directly provided
                print(
                    f"Log2 fold change & {self.test} test p-value, contrast vector: "
                    f"{self.contrast}"
                )
            else:
                # The factor is categorical
                print(
                    f"Log2 fold change & {self.test} test p-value: "
                    f"{self.contrast[0]} {self.contrast[1]} vs {self.contrast[2]}"
                )
            print(self.results_df)

    def _run_wald_inference(
        self, *, announce: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute Wald quantities without changing the selected test result."""
        num_vars = self.design_matrix.shape[1]
        normalization_factors = self.dds._get_normalization_factors()
        if normalization_factors.ndim == 1:
            normalization_factors = normalization_factors[:, None]
        mu = np.exp(self.design_matrix @ self.LFC.T).to_numpy() * normalization_factors

        if self.prior_LFC_var is not None:
            ridge_factor = np.diag(1 / self.prior_LFC_var**2)
        else:
            ridge_factor = np.diag(np.repeat(1e-6, num_vars))

        if announce and not self.quiet:
            print("Running Wald tests...", file=sys.stderr)
        start = time.time()
        pvals, stats, se = self.inference.wald_test(
            design_matrix=self.design_matrix.values,
            disp=self.dds.var["dispersions"].values,
            lfc=self.LFC.values,
            mu=mu,
            ridge_factor=ridge_factor,
            contrast=self.contrast_vector,
            lfc_null=np.log(2) * self.lfc_null,
            alt_hypothesis=self.alt_hypothesis,
        )
        if announce and not self.quiet:
            print(f"... done in {time.time() - start:.2f} seconds.\n", file=sys.stderr)
        return pvals, stats, se

    def _run_selected_test(self) -> None:
        """Run the statistical test selected at construction."""
        if self.test == "Wald":
            self.run_wald_test()
        else:
            self.run_lrt_test()

    def run_wald_test(self) -> None:
        """Perform a Wald test.

        Get gene-wise p-values for gene over/under-expression.
        """
        if self.test != "Wald":
            raise ValueError("run_wald_test() requires test='Wald'.")

        # Raise a warning if LFCs are shrunk.
        if self.shrunk_LFCs:
            if not self.quiet:
                print(
                    "Note: running Wald test on shrunk LFCs. "
                    "Some sequencing datasets show better performance with the testing "
                    "separated from the use of the LFC prior.",
                    file=sys.stderr,
                )

        pvals, stats, se = self._run_wald_inference()

        self.p_values: pd.Series = pd.Series(pvals, index=self.dds.var_names)
        self.statistics: pd.Series = pd.Series(stats, index=self.dds.var_names)
        self.SE: pd.Series = pd.Series(se, index=self.dds.var_names)

        # Account for possible all_zeroes due to outlier refitting in DESeqDataSet
        if self.dds.refit_cooks and self.dds.var["replaced"].sum() > 0:
            self.SE.loc[self.dds.new_all_zeroes_genes] = 0.0
            self.statistics.loc[self.dds.new_all_zeroes_genes] = 0.0
            self.p_values.loc[self.dds.new_all_zeroes_genes] = 1.0

    def run_lrt_test(self) -> None:
        """Perform the selected full-versus-reduced likelihood-ratio test."""
        if self.test != "LRT" or self.reduced_design_matrix is None:
            raise ValueError("run_lrt_test() requires test='LRT' and a reduced model.")
        if getattr(self, "_lrt_has_run", False):
            return

        use_dds_cache = not self._force_local_lrt_refit and self.dds._lrt_cache_matches(
            self.reduced_design_matrix
        )
        if use_dds_cache:
            results = self.dds._load_lrt_results()
        else:
            if not self.quiet:
                print("Running likelihood ratio tests...", file=sys.stderr)
            start = time.time()
            results = self.dds._compute_lrt(
                self.reduced_design_matrix,
                inference=self.inference,
                refit_full=self._force_local_lrt_refit,
            )
            if not self.quiet:
                print(
                    f"... done in {time.time() - start:.2f} seconds.\n",
                    file=sys.stderr,
                )

        (
            statistics,
            p_values,
            full_deviance,
            reduced_converged,
            full_converged,
            new_all_zero,
            full_lfcs,
        ) = results
        self.statistics = statistics.copy()
        self.p_values = p_values.copy()
        self.full_deviance = full_deviance.copy()
        self.reduced_converged = reduced_converged.copy()
        self.full_converged = full_converged.copy()
        self.new_all_zero = new_all_zero.copy()
        self.LFC = full_lfcs.copy()
        _, _, se = self._run_wald_inference(announce=False)
        self.SE = pd.Series(se, index=self.dds.var_names)
        self.SE.loc[self.new_all_zero.fillna(False).astype(bool)] = 0.0
        self._lrt_has_run = True

    # TODO update this to reflect the new contrast format
    def lfc_shrink(self, coeff: str, adapt: bool = True) -> None:
        """LFC shrinkage with an apeGLM prior :cite:p:`DeseqStats-zhu2019heavy`.

        Shrinks LFCs using a heavy-tailed Cauchy prior, leaving p-values unchanged.

        Parameters
        ----------
        coeff : str
            The LFC coefficient to shrink. Must be one of the columns of the LFC matrix.
            (default: ``None``).

        adapt: bool
            Whether to use the MLE estimates of LFC to adapt the prior. If False, the
            prior scale is set to 1. (``default=True``)
        """
        if coeff not in self.LFC.columns:
            raise KeyError(
                f"The coeff argument '{coeff}' should be one the LFC columns. "
                f"The available LFC coeffs are {self.LFC.columns[1:]}."
            )

        coeff_idx = self.LFC.columns.get_loc(coeff)

        design_matrix = self.design_matrix.values
        size = 1.0 / self.dds.var["dispersions"].values
        effective_counts = self.dds._effective_counts()
        effective_nonzero = (effective_counts != 0).any(axis=0)
        shrink_mask = self.dds.var["non_zero"].to_numpy(dtype=bool) & effective_nonzero
        shrink_idx = np.flatnonzero(shrink_mask)
        shrink_genes = self.dds.var_names[shrink_idx]
        offset = np.log(self.dds._get_normalization_factors(shrink_idx))

        # Set priors
        prior_no_shrink_scale = 15
        prior_scale = 1
        if adapt:
            prior_var = self._fit_prior_var(coeff_idx=coeff_idx, gene_mask=shrink_mask)
            prior_scale = np.minimum(np.sqrt(prior_var), 1)

        if not self.quiet:
            print("Fitting MAP LFCs...", file=sys.stderr)
        start = time.time()
        lfcs, inv_hessians, l_bfgs_b_converged_ = self.inference.lfc_shrink_nbinom_glm(
            design_matrix=design_matrix,
            counts=effective_counts[:, shrink_idx],
            size=size[shrink_idx],
            offset=offset,
            prior_no_shrink_scale=prior_no_shrink_scale,
            prior_scale=prior_scale,
            optimizer="L-BFGS-B",
            shrink_index=coeff_idx,
        )
        end = time.time()
        if not self.quiet:
            print(f"... done in {end - start:.2f} seconds.\n", file=sys.stderr)

        new_lfc_values = np.array(lfcs)[:, coeff_idx]
        new_se_values = np.array(
            [
                np.sqrt(np.abs(inv_hess[coeff_idx, coeff_idx]))
                for inv_hess in inv_hessians
            ]
        )
        nan_mask = ~np.isfinite(new_lfc_values) | ~np.isfinite(new_se_values)

        if nan_mask.any():
            warnings.warn(
                f"{nan_mask.sum()} gene(s) had NaN/infinite values during LFC shrinkage,"
                " their LFCs and SEs were not updated.",
                UserWarning,
                stacklevel=2,
            )

        # Only update genes with valid (non-NaN) shrinkage results
        valid_genes = shrink_genes[~nan_mask]
        self.LFC.loc[valid_genes, coeff] = new_lfc_values[~nan_mask]
        self.SE.loc[valid_genes] = new_se_values[~nan_mask]

        self._LFC_shrink_converged = pd.Series(
            pd.array([pd.NA] * len(self.dds.var_names), dtype="boolean"),
            index=self.dds.var_names,
        )
        self._LFC_shrink_converged.loc[shrink_genes] = l_bfgs_b_converged_

        # Set a flag to indicate that LFCs were shrunk
        self.shrunk_LFCs = True

        # Replace in results dataframe, if it exists
        if hasattr(self, "results_df"):
            self.results_df["log2FoldChange"] = self.LFC.iloc[:, coeff_idx] / np.log(2)
            self.results_df["lfcSE"] = self.SE / np.log(2)
            if not self.quiet:
                print(f"Shrunk log2 fold change & {self.test} test p-value: {coeff}")
                print(self.results_df)

    def plot_MA(self, log: bool = True, save_path: str | None = None, **kwargs):
        """
        Create an log ratio (M)-average (A) plot using matplotlib.

        Useful for looking at log fold-change versus mean expression
        between two groups/samples/etc.
        Uses matplotlib to emulate the ``make_MA()`` function in DESeq2 in R.

        Parameters
        ----------
        log : bool
            Whether or not to log scale x and y axes (``default=True``).

        save_path : str, optional
            The path where to save the plot. If left None, the plot won't be saved
            (``default=None``).

        **kwargs
            Matplotlib keyword arguments for the scatter plot.
        """
        # Raise an error if results_df are missing
        if not hasattr(self, "results_df"):
            raise AttributeError(
                "Trying to make an MA plot but p-values were not computed yet. "
                "Please run the summary() method first."
            )

        make_MA_plot(
            self.results_df,
            padj_thresh=self.alpha,
            log=log,
            save_path=save_path,
            lfc_null=self.lfc_null,
            alt_hypothesis=self.alt_hypothesis,
            **kwargs,
        )

    def _independent_filtering(self) -> None:
        """Compute adjusted p-values using independent filtering.

        Corrects p-value trend (see :cite:p:`DeseqStats-love2014moderated`)
        """
        # Check that p-values are available. If not, compute them.
        if not hasattr(self, "p_values"):
            self._run_selected_test()

        lower_quantile = np.mean(self.base_mean == 0)

        if lower_quantile < 0.95:
            upper_quantile = 0.95
        else:
            upper_quantile = 1

        theta = np.linspace(lower_quantile, upper_quantile, 50)
        cutoffs = np.quantile(self.base_mean, theta)

        result = pd.DataFrame(
            np.nan, index=self.dds.var_names, columns=np.arange(len(theta))
        )

        for i, cutoff in enumerate(cutoffs):
            use = (self.base_mean >= cutoff) & (~self.p_values.isna())
            U2 = self.p_values[use]
            if not U2.empty:
                result.loc[use, i] = false_discovery_control(U2, method="bh")
        num_rej = (result < self.alpha).sum(axis=0).to_numpy().astype(int)
        lowess_res = lowess(theta, num_rej, frac=1 / 5)

        if num_rej.max() <= 10:
            j = 0
        else:
            residual = num_rej[num_rej > 0] - lowess_res[num_rej > 0]
            thresh = lowess_res.max() - np.sqrt(np.mean(residual**2))
            if np.any(num_rej > thresh):
                j = np.where(num_rej > thresh)[0][0]
            else:
                j = 0

        self.padj = result.loc[:, j]

    def _p_value_adjustment(self) -> None:
        """Compute adjusted p-values using the Benjamini-Hochberg method.

        Does not correct the p-value trend.
        This method and the `_independent_filtering` are mutually exclusive.
        """
        if not hasattr(self, "p_values"):
            self._run_selected_test()

        self.padj = pd.Series(np.nan, index=self.dds.var_names)
        self.padj.loc[~self.p_values.isna()] = false_discovery_control(
            self.p_values.dropna(), method="bh"
        )

    def _cooks_filtering(self) -> None:
        """Filter p-values based on Cooks outliers."""
        # Check that p-values are available. If not, compute them.
        if not hasattr(self, "p_values"):
            self._run_selected_test()

        cooks_outlier = self.dds.cooks_outlier().copy()
        if self.test == "LRT":
            new_all_zero = self.new_all_zero.fillna(False).astype(bool)
            cooks_outlier.loc[new_all_zero] = False
        self.p_values[cooks_outlier] = np.nan

    def _fit_prior_var(
        self,
        coeff_idx: int,
        min_var: float = 1e-6,
        max_var: float = 400.0,
        *,
        gene_mask: np.ndarray | pd.Series | None = None,
    ) -> float:
        """Estimate the prior variance of the apeGLM model.

        Returns shrinkage factors.

        Parameters
        ----------
        coeff_idx : int
            Index of the coefficient to shrink.

        min_var : float
            Lower bound for prior variance. (default: ``1e-6``).

        max_var : float
            Upper bound for prior variance. (default: ``400``).

        Returns
        -------
        float
            Estimated prior variance.
        """
        keep = ~self.LFC.iloc[:, coeff_idx].isna()
        if gene_mask is not None:
            keep &= np.asarray(gene_mask, dtype=bool)

        S = self.LFC[keep].iloc[:, coeff_idx] ** 2
        D = self.SE[keep] ** 2

        def objective(a: float) -> float:
            # Equation to solve
            coeff = 1 / (2 * (a + D) ** 2)
            return ((S - D) * coeff).sum() / coeff.sum() - a

        # The prior variance is the zero of the above function.
        if objective(min_var) < 0:
            return min_var
        else:
            return root_scalar(objective, bracket=(min_var, max_var)).root

    def _build_contrast_vector(self) -> None:
        """
        Build a vector corresponding to the desired contrast.

        Allows to test any pair of levels without refitting LFCs.
        """
        factor = self.contrast[0]
        alternative = self.contrast[1]
        ref = self.contrast[2]
        self.contrast_vector = self.dds.contrast(
            column=factor, baseline=ref, group_to_compare=alternative
        )
