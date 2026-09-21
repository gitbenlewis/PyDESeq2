import numpy as np
import pandas as pd
import pytest

from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats


def make_dds(n_genes, **kwargs):
    samples = [f"s{i}" for i in range(6)]
    return DeseqDataSet(
        counts=pd.DataFrame(
            np.ones((6, n_genes), dtype=int),
            index=samples,
            columns=[f"g{i}" for i in range(n_genes)],
        ),
        metadata=pd.DataFrame({"condition": ["A"] * 3 + ["B"] * 3}, index=samples),
        design="~condition",
        n_cpus=1,
        quiet=True,
        refit_cooks=False,
        **kwargs,
    )


@pytest.mark.parametrize("vst", [False, True])
def test_trend_reconsiders_excluded_genes(monkeypatch, vst):
    dds = make_dds(4)
    dds.non_zero_genes = dds.var_names
    dds.var["non_zero"] = True
    dds.var["_normed_means"] = 1.0
    name = "vst_genewise_dispersions" if vst else "genewise_dispersions"
    dds.var[name] = [1.0, 2.0, 20.0, dds.min_disp]
    seen = []

    def fit(covariates, targets):
        seen.append(list(targets.index))
        return np.array([1.0, 1.0]), np.full(len(targets), 2.0), True

    monkeypatch.setattr(dds.inference, "dispersion_trend_gamma_glm", fit)
    dds._fit_parametric_dispersion_trend(vst)
    assert seen == [list(dds.var_names[:2]), list(dds.var_names[:3])]
    key = "vst_trend_coeffs" if vst else "trend_coeffs"
    np.testing.assert_array_equal(dds.uns[key], [1.0, 1.0])


def test_parametric_trend_known_curve():
    dds = make_dds(100)
    dds.non_zero_genes = dds.var_names
    dds.var["non_zero"] = True
    dds.var["_normed_means"] = np.geomspace(0.1, 1000, 100)
    dds.var["genewise_dispersions"] = 0.2 + 3.0 / dds.var["_normed_means"]
    dds._fit_parametric_dispersion_trend()
    np.testing.assert_allclose(dds.uns["trend_coeffs"], [0.2, 3.0], rtol=1e-4)


def test_trend_iteration_limit(monkeypatch):
    dds = make_dds(3)
    dds.non_zero_genes = dds.var_names
    dds.var["non_zero"] = True
    dds.var["_normed_means"] = 1.0
    dds.var["genewise_dispersions"] = 1.0
    calls = []

    def fit(covariates, targets):
        calls.append(True)
        coeffs = np.array([1.0, 1.0]) * (1 + len(calls) % 2)
        return coeffs, np.full(len(targets), coeffs.sum()), True

    monkeypatch.setattr(dds.inference, "dispersion_trend_gamma_glm", fit)
    with pytest.warns(UserWarning, match="Switching to a mean-based"):
        dds._fit_parametric_dispersion_trend()
    assert len(calls) == 11
    assert dds.uns["disp_function_type"] == "mean"


def test_dispersion_outlier_uses_unadjusted_residual_variance(monkeypatch):
    dds = make_dds(3)
    dds.non_zero_idx = np.arange(3)
    dds.var["non_zero"] = True
    dds.var["fitted_dispersions"] = 0.1
    dds.var["genewise_dispersions"] = 0.1 * np.exp([1.9, 2.0, 2.1])
    dds.uns["_squared_logres"] = 1.0
    dds.uns["prior_disp_var"] = np.float64(0.25)
    dds.layers["_mu_hat"] = np.ones(dds.shape)
    monkeypatch.setattr(
        dds.inference,
        "alpha_mle",
        lambda **kwargs: (np.full(3, 0.2), np.ones(3, dtype=bool)),
    )
    dds.fit_MAP_dispersions()
    np.testing.assert_array_equal(dds.var["_outlier_genes"], [False, False, True])
    np.testing.assert_allclose(dds.var["dispersions"], [0.2, 0.2, 0.1 * np.exp(2.1)])


@pytest.mark.parametrize("gene_specific", [False, True])
@pytest.mark.parametrize("min_mu", [0.5, 0.2])
def test_wald_fitted_mean_floor(gene_specific, min_mu):
    dds = make_dds(2, min_mu=min_mu)
    factors = np.array([0.7, 1.0, 1.3, 0.8, 1.0, 1.2])
    dds.obs["size_factors"] = factors
    if gene_specific:
        dds.layers["normalization_factors"] = factors[:, None] * np.array([[1.0, 2.0]])
    dds.var["_normed_means"] = [1.0, 2.0]
    dds.var["dispersions"] = [0.7, 0.2]
    dds.varm["LFC"] = pd.DataFrame(
        [[np.log(0.05), np.log(20.0)], [np.log(2.0), np.log(2.0)]],
        index=dds.var_names,
        columns=dds.obsm["design_matrix"].columns,
    )
    stats = DeseqStats(dds, contrast=np.array([0.0, 1.0]), n_cpus=1, quiet=True)
    stats.run_wald_test()
    # Independent sandwich covariance calculation using the floored count means.
    design = stats.design_matrix.to_numpy()
    for j in range(2):
        scale = factors * (2.0 if gene_specific and j == 1 else 1.0)
        means = np.maximum(scale * np.exp(design @ stats.LFC.iloc[j].to_numpy()), min_mu)
        weights = means / (1 + means * dds.var["dispersions"].iloc[j])
        information = design.T @ (weights[:, None] * design)
        inverse = np.linalg.inv(information + np.eye(2) * 1e-6)
        expected = np.sqrt((inverse @ information @ inverse)[1, 1])
        np.testing.assert_allclose(stats.SE.iloc[j], expected, rtol=1e-12)


@pytest.mark.parametrize("cr_reg", [False, True])
@pytest.mark.parametrize("prior_reg", [False, True])
def test_dispersion_fallback_preserves_objective(monkeypatch, cr_reg, prior_reg):
    from types import SimpleNamespace

    from pydeseq2 import dispersions

    counts = np.array([1.0, 5.0, 2.0, 20.0, 3.0, 30.0])
    design = np.column_stack([np.ones(6), [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])
    mu = np.array([3.0, 3.0, 3.0, 18.0, 18.0, 18.0])
    kwargs = {
        "counts": counts,
        "design_matrix": design,
        "mu": mu,
        "alpha_hat": 0.1,
        "min_disp": 1e-8,
        "max_disp": 10.0,
        "prior_disp_var": 0.25,
        "cr_reg": cr_reg,
        "prior_reg": prior_reg,
    }
    expected = np.exp(dispersions.grid_fit_alpha(**kwargs))
    monkeypatch.setattr(
        dispersions, "minimize", lambda *args, **kwargs: SimpleNamespace(success=False)
    )
    actual, converged = dispersions.fit_alpha_mle(**kwargs)
    assert not converged
    np.testing.assert_allclose(actual, expected)
