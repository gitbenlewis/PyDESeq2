import pathlib
from unittest import mock

import numpy as np
import pytest
from scipy.sparse import bsr_array
from scipy.sparse import bsr_matrix
from scipy.sparse import coo_array
from scipy.sparse import coo_matrix
from scipy.sparse import dia_array
from scipy.sparse import dia_matrix
from scipy.sparse import dok_array
from scipy.sparse import dok_matrix
from scipy.sparse import lil_array
from scipy.sparse import lil_matrix
from scipy.stats import norm

from pydeseq2.utils import load_example_data
from pydeseq2.utils import nb_nll
from pydeseq2.utils import test_valid_counts as validate_counts
from pydeseq2.utils import wald_test


@pytest.mark.parametrize("mu, alpha", [(10, 0.5), (10, 0.1), (3, 0.5), (9, 0.05)])
def test_nb_nll_moments(mu, alpha):
    # get the probability of many points
    y = np.arange(int(10 * (mu + mu**2 / alpha)))
    probas = np.zeros(y.shape)
    for i in range(y.size):
        # crude trapezoidal interpolation
        probas[i] = np.exp(-nb_nll(np.array([y[i]]), mu, alpha))
    # check that probas sums very close to 1
    assert np.allclose(probas.sum(), 1.0)
    # Re-sample according to probas
    n_montecarlo = int(1e6)
    rng = np.random.default_rng(42)
    sample = rng.choice(y, size=(n_montecarlo,), p=probas)
    # Get the theoretical values
    mean_th = mu
    var_th = (mu * alpha + 1) * mu
    # Check that the mean is in an acceptable range, up to stochasticity
    diff = sample.mean() - mean_th
    deviation = var_th / np.sqrt(n_montecarlo)
    assert np.abs(diff) < 0.2 * deviation
    error_var = np.abs(sample.var() - var_th) / var_th
    assert error_var < 1 / np.sqrt(n_montecarlo)


# Test data loading from outside the package (e.g. on RTF)
@pytest.mark.parametrize("modality", ["raw_counts", "metadata"])
@pytest.mark.parametrize("mocked_dir_flag", [True, False])
@mock.patch("pathlib.Path.is_dir")
def test_rtd_example_data_loading(mocked_function, modality, mocked_dir_flag):
    """
    Test that load_example_data still works when run from a place where the ``datasets``
    directory is not accessible, as is when the documentation is built on readthedocs.
    """

    # Mock the output of is_dir() as False to emulate not having access to the
    # ``datasets`` directory
    pathlib.Path.is_dir.return_value = mocked_dir_flag

    # Try loading data.
    load_example_data(
        modality=modality,
        dataset="synthetic",
        debug=False,
    )


@pytest.mark.parametrize(
    "sparse_constructor",
    [
        bsr_matrix,
        coo_matrix,
        dia_matrix,
        dok_matrix,
        lil_matrix,
        bsr_array,
        coo_array,
        dia_array,
        dok_array,
        lil_array,
    ],
)
def test_valid_sparse_count_formats(sparse_constructor):
    validate_counts(sparse_constructor([[1, 0], [0, 2]]))


@pytest.mark.parametrize("sparse_constructor", [dia_matrix, dia_array])
def test_valid_dia_counts_ignore_padding(sparse_constructor):
    counts = sparse_constructor(
        (np.array([[-1, 2]]), np.array([1])),
        shape=(2, 2),
    )
    np.testing.assert_array_equal(counts.toarray(), [[0, 2], [0, 0]])
    validate_counts(counts)


@pytest.mark.parametrize(
    ("value", "message"),
    [(np.nan, "NaNs"), (1.5, "integers"), (-1, "non-negative")],
)
def test_invalid_general_sparse_counts(value, message):
    with pytest.raises(ValueError, match=message):
        validate_counts(dok_matrix([[value]]))


_WALD_DESIGN = np.eye(3)
_WALD_MU = np.ones(3)
_WALD_RIDGE = np.zeros((3, 3))
_WALD_CONTRAST = np.array([0.0, -1.0, 1.0])


def _run_wald_test(lfc, lfc_null, alt_hypothesis, contrast=_WALD_CONTRAST):
    return wald_test(
        design_matrix=_WALD_DESIGN,
        disp=0.0,
        lfc=np.asarray(lfc, dtype=float),
        mu=_WALD_MU,
        ridge_factor=_WALD_RIDGE,
        contrast=contrast,
        lfc_null=lfc_null,
        alt_hypothesis=alt_hypothesis,
    )


def test_wald_threshold_is_applied_to_the_full_contrast():
    lfc = np.array([0.0, 1.0, 3.0])
    lfc_null = 0.5
    expected_se = np.sqrt(2.0)
    expected_stat = (2.0 - lfc_null) / expected_se
    expected_pvalue = 2 * norm.sf(abs(expected_stat))

    result = _run_wald_test(lfc, lfc_null, None)
    shifted_result = _run_wald_test(lfc + np.array([0.0, 10.0, 10.0]), lfc_null, None)

    np.testing.assert_allclose(result, (expected_pvalue, expected_stat, expected_se))
    np.testing.assert_allclose(shifted_result, result)


def test_wald_threshold_handles_fractional_contrasts():
    contrast = np.array([-1.0, 0.5, 0.5])
    lfc = np.array([1.0, 2.0, 4.0])
    lfc_null = 0.5
    expected_se = np.sqrt(1.5)
    expected_stat = (2.0 - lfc_null) / expected_se

    pvalue, statistic, se = _run_wald_test(lfc, lfc_null, None, contrast=contrast)

    assert se == pytest.approx(expected_se)
    assert statistic == pytest.approx(expected_stat)
    assert pvalue == pytest.approx(2 * norm.sf(abs(expected_stat)))


@pytest.mark.parametrize(
    "alt_hypothesis,lfc_null,expected_stat,expected_pvalue",
    [
        ("greater", 1.0, 0.0, norm.sf(-1 / np.sqrt(2))),
        ("less", -1.0, 0.0, norm.cdf(1 / np.sqrt(2))),
        (
            "lessAbs",
            1.0,
            1 / np.sqrt(2),
            norm.sf(1 / np.sqrt(2)),
        ),
    ],
)
def test_wald_one_sided_pvalues_use_unclipped_distance(
    alt_hypothesis, lfc_null, expected_stat, expected_pvalue
):
    pvalue, statistic, _ = _run_wald_test([0.0, 1.0, 1.0], lfc_null, alt_hypothesis)

    assert statistic == pytest.approx(expected_stat)
    assert pvalue == pytest.approx(expected_pvalue)
    assert pvalue != pytest.approx(0.5)


def test_wald_current_and_legacy_greater_abs_methods():
    lfc = [0.0, 1.0, 1.5]
    lfc_null = 1.0
    se = np.sqrt(2.0)
    effect = 0.5

    current_pvalue, current_statistic, _ = _run_wald_test(lfc, lfc_null, "greaterAbs")
    legacy_pvalue, legacy_statistic, _ = _run_wald_test(lfc, lfc_null, "greaterAbs2014")

    expected_current_pvalue = norm.sf((abs(effect) - lfc_null) / se) + norm.sf(
        (abs(effect) + lfc_null) / se
    )
    assert current_statistic == pytest.approx(effect / se)
    assert current_pvalue == pytest.approx(expected_current_pvalue)
    assert legacy_statistic == 0.0
    assert legacy_pvalue == 1.0


def test_wald_upshot_matches_deseq2_formula():
    lfc = [0.0, 1.0, 3.0]
    lfc_null = 0.5
    se = np.sqrt(2.0)
    effect = 2.0
    a = (abs(effect) + lfc_null) / se
    b = (abs(effect) - lfc_null) / se
    expected_pvalue = (2 / (b - a)) * (
        -a * norm.cdf(-a) + norm.pdf(a) + b * norm.cdf(-b) - norm.pdf(b)
    )

    pvalue, statistic, _ = _run_wald_test(lfc, lfc_null, "greaterAbsUPSHOT")
    zero_threshold_pvalue, zero_threshold_statistic, _ = _run_wald_test(
        lfc, 0.0, "greaterAbsUPSHOT"
    )
    near_zero_pvalue, _, _ = _run_wald_test(lfc, 1e-12, "greaterAbsUPSHOT")
    null_pvalue, _, _ = _run_wald_test([0.0, 1.0, 1.0], 1e-9, "greaterAbsUPSHOT")

    assert statistic == pytest.approx(effect / se)
    assert pvalue == pytest.approx(expected_pvalue)
    assert zero_threshold_statistic == pytest.approx(effect / se)
    assert zero_threshold_pvalue == pytest.approx(2 * norm.sf(abs(effect / se)))
    assert near_zero_pvalue == pytest.approx(zero_threshold_pvalue)
    assert 0 <= null_pvalue <= 1


def test_wald_dispatch_only_evaluates_the_selected_alternative():
    with mock.patch(
        "pydeseq2.utils.norm.pdf",
        side_effect=AssertionError("UPSHOT should not be evaluated"),
    ):
        pvalue, statistic, _ = _run_wald_test([0.0, 1.0, 3.0], 0.5, "greater")

    assert statistic > 0
    assert 0 <= pvalue <= 1


def test_wald_upshot_handles_a_collapsed_floating_point_interval():
    lfc = [0.0, 0.0, 1e20]

    greater_pvalue, greater_statistic, _ = _run_wald_test(lfc, 0.5, "greater")
    upshot_pvalue, upshot_statistic, _ = _run_wald_test(lfc, 0.5, "greaterAbsUPSHOT")

    assert np.isfinite(greater_statistic)
    assert np.isfinite(upshot_statistic)
    assert greater_pvalue == 0.0
    assert upshot_pvalue == 0.0


@pytest.mark.parametrize(
    "alt_hypothesis", ["greaterAbs", "greaterAbs2014", "greaterAbsUPSHOT", "lessAbs"]
)
def test_wald_absolute_tests_are_invariant_to_contrast_direction(alt_hypothesis):
    lfc = [0.0, 1.0, 3.0]
    forward = _run_wald_test(lfc, 0.5, alt_hypothesis)
    reverse = _run_wald_test(lfc, 0.5, alt_hypothesis, contrast=-_WALD_CONTRAST)

    assert forward[0] == pytest.approx(reverse[0])
    assert abs(forward[1]) == pytest.approx(abs(reverse[1]))
    assert forward[2] == pytest.approx(reverse[2])


def test_wald_directional_tests_reverse_with_the_contrast():
    lfc = [0.0, 1.0, 3.0]
    greater = _run_wald_test(lfc, 0.5, "greater")
    less = _run_wald_test(lfc, -0.5, "less", contrast=-_WALD_CONTRAST)

    assert greater[0] == pytest.approx(less[0])
    assert greater[1] == pytest.approx(-less[1])
    assert greater[2] == pytest.approx(less[2])
