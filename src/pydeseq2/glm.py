"""Fitting and testing of the negative binomial GLM."""

from typing import Literal
from typing import cast

import numpy as np
from scipy.linalg import solve  # type: ignore
from scipy.optimize import minimize  # type: ignore
from scipy.stats import norm  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore

from pydeseq2.distributions import nb_nll
from pydeseq2.grid_search import grid_fit_beta


def irls_solver(
    counts: np.ndarray,
    size_factors: np.ndarray,
    design_matrix: np.ndarray,
    disp: float,
    min_mu: float = 0.5,
    beta_tol: float = 1e-8,
    min_beta: float = -30,
    max_beta: float = 30,
    optimizer: Literal["BFGS", "L-BFGS-B"] = "L-BFGS-B",
    maxiter: int = 250,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    r"""Fit a NB GLM wit log-link to predict counts from the design matrix.

    See equations (1-2) in the DESeq2 paper.

    Parameters
    ----------
    counts
        Raw counts for a given gene.
    size_factors
        Sample-wise scaling factors (obtained from median-of-ratios).
    design_matrix
        Design matrix.
    disp
        Gene-wise dispersion prior.
    min_mu
        Lower bound on estimated means, to ensure numerical stability. (default: ``0.5``).
    beta_tol
        Stopping criterion for IRWLS: :math:`\vert dev - dev_{old}\vert / \vert dev + 0.1 \vert < \beta_{tol}`. (default: ``1e-8``).
    min_beta
        Lower-bound on LFC. (default: ``-30``).
    max_beta
        Upper-bound on LFC. (default: ``-30``).
    optimizer
        Optimizing method to use in case IRLS starts diverging.
        Accepted values: 'BFGS' or 'L-BFGS-B'.
        NB: only 'L-BFGS-B' ensures that LFCS will lay in the [min_beta, max_beta] range. (default: ``'L-BFGS-B'``).
    maxiter
        Maximum number of IRLS iterations to perform before switching to L-BFGS-B. (default: ``250``).

    Returns
    -------
    beta
        Fitted (basemean, lfc) coefficients of negative binomial GLM.
    mu
        Means estimated from size factors and beta: :math:`\mu = s_{ij} \exp(\beta^t X)`.
    H
        Diagonal of the :math:`W^{1/2} X (X^t W X)^-1 X^t W^{1/2}` covariance matrix.
    converged
        Whether IRLS or the optimizer converged.
        If not and if dimension allows it, perform grid search.
    """
    assert optimizer in ["BFGS", "L-BFGS-B"]

    num_vars = design_matrix.shape[1]
    X = design_matrix

    # if full rank, estimate initial betas for IRLS below
    if np.linalg.matrix_rank(X) == num_vars:
        Q, R = np.linalg.qr(X)
        y = np.log(counts / size_factors + 0.1)
        beta_init = solve(R, Q.T @ y)
        beta = beta_init
    else:  # Initialise intercept with log base mean
        beta_init = np.zeros(num_vars)
        beta_init[0] = np.log(counts / size_factors).mean()
        beta = beta_init

    dev = 1000.0
    dev_ratio = 1.0

    ridge_factor = np.diag(np.repeat(1e-6, num_vars))
    mu = np.maximum(size_factors * np.exp(X @ beta), min_mu)

    converged = True
    i = 0
    while dev_ratio > beta_tol:
        W = mu / (1.0 + mu * disp)
        z = np.log(mu / size_factors) + (counts - mu) / mu
        H = (X.T * W) @ X + ridge_factor
        beta_hat = solve(H, X.T @ (W * z), assume_a="pos")
        i += 1

        if sum(np.abs(beta_hat) > max_beta) > 0 or i >= maxiter:
            # If IRLS starts diverging, use L-BFGS-B
            def f(beta: np.ndarray) -> float:
                # closure to minimize
                mu_ = np.maximum(size_factors * np.exp(X @ beta), min_mu)
                return nb_nll(counts, mu_, disp) + 0.5 * (ridge_factor @ beta**2).sum()

            def df(beta: np.ndarray) -> np.ndarray:
                mu_ = np.maximum(size_factors * np.exp(X @ beta), min_mu)
                return (
                    -X.T @ counts
                    + ((1 / disp + counts) * mu_ / (1 / disp + mu_)) @ X
                    + ridge_factor @ beta
                )

            res = minimize(
                f,
                beta_init,
                jac=df,
                method=optimizer,
                bounds=(
                    [(min_beta, max_beta)] * num_vars
                    if optimizer == "L-BFGS-B"
                    else None
                ),
            )

            beta = res.x
            mu = np.maximum(size_factors * np.exp(X @ beta), min_mu)
            converged = res.success

            if not res.success and num_vars <= 2:
                beta = grid_fit_beta(
                    counts,
                    size_factors,
                    X,
                    disp,
                )
                mu = np.maximum(size_factors * np.exp(X @ beta), min_mu)
            break

        beta = beta_hat
        mu = np.maximum(size_factors * np.exp(X @ beta), min_mu)
        # Compute deviation
        old_dev = dev
        # Replaced deviation with -2 * nll, as in the R code
        dev = -2 * cast(float, nb_nll(counts, mu, disp))
        dev_ratio = np.abs(dev - old_dev) / (np.abs(dev) + 0.1)

    # Compute H diagonal (useful for Cook distance outlier filtering)
    # Calculate only the diagonal for X(XTWX)-1XT using einsum
    # This is numerically equivalent to the more expensive calculation
    # np.diag(X @ (X^T @ np.inv(X^T @ np.diag(W) @ X + lambda) @ X^T)
    W = mu / (1.0 + mu * disp)
    H = np.einsum(
        "ij,jk,ki->i", X, np.linalg.inv((X.T * W[None, :]) @ X + ridge_factor), X.T
    )

    W_sq = np.sqrt(W)
    H = W_sq * H * W_sq

    # Return an UNthresholded mu (as in the R code)
    # Previous quantities are estimated with a threshold though
    mu = size_factors * np.exp(X @ beta)
    return beta, mu, H, converged


def fit_lin_mu(
    counts: np.ndarray,
    size_factors: np.ndarray,
    design_matrix: np.ndarray,
    min_mu: float = 0.5,
) -> np.ndarray:
    """Estimate mean of negative binomial model using a linear regression.

    Used to initialize genewise dispersion models.

    Parameters
    ----------
    counts
        Raw counts for a given gene.
    size_factors
        Sample-wise scaling factors (obtained from median-of-ratios).
    design_matrix
        Design matrix.
    min_mu
        Lower threshold for fitted means, for numerical stability. (default: ``0.5``).

    Returns
    -------
    Estimated mean.
    """
    reg = LinearRegression(fit_intercept=False)
    reg.fit(design_matrix, counts / size_factors)
    mu_hat = size_factors * reg.predict(design_matrix)
    # Threshold mu_hat as 1/mu_hat will be used later on.
    return np.maximum(mu_hat, min_mu)


def wald_test(
    design_matrix: np.ndarray,
    disp: float,
    lfc: np.ndarray,
    mu: np.ndarray,
    ridge_factor: np.ndarray,
    contrast: np.ndarray,
    lfc_null: float,
    alt_hypothesis: (
        Literal[
            "greaterAbs",
            "greaterAbs2014",
            "greaterAbsUPSHOT",
            "lessAbs",
            "greater",
            "less",
        ]
        | None
    ),
) -> tuple[float, float, float]:
    """Run Wald test for differential expression.

    Computes Wald statistics, standard error and p-values from dispersion and LFC estimates.

    Parameters
    ----------
    design_matrix
        Design matrix.
    disp
        Dispersion estimate.
    lfc
        Log-fold change estimate (in natural log scale).
    mu
        Mean estimation for the NB model.
    ridge_factor
        Regularization factors.
    contrast
        Vector encoding the contrast that is being tested.
    lfc_null
        The log fold change under the null hypothesis, in natural log scale.
    alt_hypothesis
        The alternative hypothesis for computing wald p-values.

    Returns
    -------
    wald_p_value
        Estimated p-value.
    wald_statistic
        Wald statistic.
    wald_se
        Standard error of the Wald statistic.
    """
    # Build covariance matrix estimator
    W = mu / (1 + mu * disp)
    M = (design_matrix.T * W[None, :]) @ design_matrix
    H = np.linalg.inv(M + ridge_factor)
    Hc = H @ contrast
    # Evaluate the full contrast estimate and its standard error before applying any
    # threshold. Thresholding individual coefficients first is not invariant to model
    # parameterization and gives incorrect results for multi-coefficient contrasts.
    contrast_lfc = float(contrast @ lfc)
    wald_se: float = np.sqrt(Hc.T @ M @ Hc)

    def greater(lfc_null: float) -> tuple[float, float]:
        standardized_distance = (contrast_lfc - lfc_null) / wald_se
        stat = max(standardized_distance, 0.0)
        # The p-value uses the unclipped distance. On the null side it may be > 0.5.
        pval = norm.sf(standardized_distance)
        return stat, pval

    def less(lfc_null: float) -> tuple[float, float]:
        standardized_distance = (contrast_lfc - lfc_null) / wald_se
        stat = min(standardized_distance, 0.0)
        # PyDESeq2 uses a signed null for this test (for example, -0.5 for beta < -0.5).
        pval = norm.cdf(standardized_distance)
        return stat, pval

    def greater_abs(lfc_null: float) -> tuple[float, float]:
        abs_lfc = abs(contrast_lfc)
        stat = contrast_lfc / wald_se
        pval = norm.sf((abs_lfc - lfc_null) / wald_se) + norm.sf(
            (abs_lfc + lfc_null) / wald_se
        )
        return stat, pval

    def greater_abs_2014(lfc_null: float) -> tuple[float, float]:
        standardized_distance = (abs(contrast_lfc) - lfc_null) / wald_se
        stat = np.sign(contrast_lfc) * max(standardized_distance, 0.0)
        pval = min(1.0, 2 * norm.sf(standardized_distance))
        return stat, pval

    def greater_abs_upshot(lfc_null: float) -> tuple[float, float]:
        stat = contrast_lfc / wald_se
        threshold_stat = lfc_null / wald_se
        # The closed form below suffers from cancellation as its integration interval
        # collapses. In that limit, UPSHOT is the ordinary two-sided Wald test.
        if threshold_stat <= np.sqrt(np.finfo(float).eps):
            return stat, 2 * norm.sf(abs(stat))

        a = abs(stat) + threshold_stat
        b = abs(stat) - threshold_stat
        if a == b:
            return stat, 2 * norm.sf(abs(stat))
        pval = (2 / (b - a)) * (
            -a * norm.cdf(-a) + norm.pdf(a) + b * norm.cdf(-b) - norm.pdf(b)
        )
        return stat, float(np.clip(pval, 0.0, 1.0))

    def less_abs(lfc_null: float) -> tuple[float, float]:
        distance_from_upper = (lfc_null - contrast_lfc) / wald_se
        distance_from_lower = (contrast_lfc + lfc_null) / wald_se
        stat = min(max(distance_from_upper, 0.0), max(distance_from_lower, 0.0))
        pval = max(norm.sf(distance_from_upper), norm.sf(distance_from_lower))
        return stat, pval

    wald_statistic: float
    wald_p_value: float
    if alt_hypothesis is not None:
        wald_test_fn = {
            "greaterAbs": greater_abs,
            "greaterAbs2014": greater_abs_2014,
            "greaterAbsUPSHOT": greater_abs_upshot,
            "lessAbs": less_abs,
            "greater": greater,
            "less": less,
        }[alt_hypothesis]
        wald_statistic, wald_p_value = wald_test_fn(lfc_null)
    else:
        wald_statistic = (contrast_lfc - lfc_null) / wald_se
        wald_p_value = 2 * norm.sf(np.abs(wald_statistic))

    return wald_p_value, wald_statistic, wald_se
