"""Conformance tests for egamma, built from the test vectors published with
the paper. Run with `pytest test_conformance.py`.

Each vector is an elicited three-point estimate at P_L = 0.10 together with
the parameters a conforming implementation must return, and the largest
deviation permitted when the elicited values are recovered from them,
expressed as a fraction of the elicited range H - L.
"""
import numpy as np
import pytest

import egamma as eg

P_LOW = 0.10

# The acceptance tolerance is itself the reproduction bound: the search stops on
# |t_alpha - t|, and that difference is the maximum normalised error in the three
# reproduced values. There is no longer a separate bound to derive.
BOUND = eg.TOLERANCE

# case, low, mode, high, alpha, beta, delta, max deviation / (H - L)
VECTORS = [
    ("right-skewed",       100,   140.0,     300, 2.772463261303343,    49.50562468366566,    52.25309902033068,  3.79e-13),
    ("right-skewed, mild", 100,   180.0,     300, 22.727720919870098,   16.475502606082216, -177.9751226395469,   5.73e-13),
    ("symmetric",          100,   200.0,     300, 1.0e9,                 0.002467538369628579, -2467338.3671610407, 1.50e-05),
    ("left-skewed, mild",  100,   220.0,     300, 22.727720919870098,  -16.475502606082216,  577.975122639547,    5.74e-13),
    ("left-skewed",        100,   260.0,     300, 2.772463261303343,   -49.50562468366566,   347.7469009796693,   3.79e-13),
    ("mode at low",        100,   100.0,     300, 1.1563104087145744,   82.90485327962145,    87.04110849944055,  1.97e-11),
    ("near low",           100,   100.00001, 300, 1.1563105945690988,   82.90484483478599,    87.04110441122212,  2.28e-11),
    ("near high",          100,   299.99999, 300, 1.1563105945690988,  -82.90484483478599,   312.95889558877790,  2.28e-11),
    ("mode at high",       100,   300.0,     300, 1.1563104087145744,  -82.90485327962145,   312.95889150055945,  1.97e-11),
    ("project element",   4000, 10000.0,   13000, 8.364944725412242, -1236.0477382059835,  19103.42326985789,    4.40e-12),
]


@pytest.mark.parametrize("case,low,mode,high,alpha,beta,delta,max_dev", VECTORS)
def test_parameters_match(case, low, mode, high, alpha, beta, delta, max_dev):
    a, b, d = eg.params(low, mode, high, P_LOW)
    assert a == pytest.approx(alpha, rel=1e-9)
    assert b == pytest.approx(beta, rel=1e-9)
    assert d == pytest.approx(delta, rel=1e-9, abs=1e-6)


@pytest.mark.parametrize("case,low,mode,high,alpha,beta,delta,max_dev", VECTORS)
def test_elicited_values_recovered(case, low, mode, high, alpha, beta, delta, max_dev):
    a, b, d = eg.params(low, mode, high, P_LOW)
    span = high - low
    got = (eg.ppf(P_LOW, a, b, d), eg.mode(a, b, d), eg.ppf(1 - P_LOW, a, b, d))
    dev = max(abs(g - w) for g, w in zip(got, (low, mode, high))) / span
    assert dev <= max_dev * 1.5


def test_reproduction_bound_holds_across_the_admissible_range():
    """Every ordinarily converged fit must satisfy the tolerance, not merely the
    tabulated cases."""
    worst = 0.0
    for mode in range(100, 301):
        r = eg.params(100, mode, 300, P_LOW)
        if r.ceiling:            # symmetric: governed by the ceiling instead
            continue
        a, b, d = r
        got = (eg.ppf(P_LOW, a, b, d), eg.mode(a, b, d), eg.ppf(1 - P_LOW, a, b, d))
        worst = max(worst, max(abs(g - w) for g, w in zip(got, (100, mode, 300))) / 200)
    assert worst < BOUND


def test_mode_near_an_outer_value_is_fitted_as_supplied():
    """A mode close to an outer value must keep its own target position rather
    than being rounded to the endpoint. Under the relative stopping rule used up
    to 1.1.1 these estimates could be refused outright."""
    endpoint = eg.params(100, 100, 300, P_LOW)
    for offset in (1e-3, 1e-5, 1e-7, 1e-9):
        r = eg.params(100, 100 + offset, 300, P_LOW)
        assert not r.ceiling
        a, b, d = r
        assert a >= endpoint[0]
        got = (eg.ppf(P_LOW, a, b, d), eg.mode(a, b, d), eg.ppf(1 - P_LOW, a, b, d))
        dev = max(abs(g - w) for g, w in zip(got, (100, 100 + offset, 300))) / 200
        assert dev < BOUND

    # Where the displacement is resolvable at the requested tolerance, the fit
    # must actually differ from the endpoint fit. Below that the two coincide,
    # which is correct: the tolerance does not promise to distinguish positions
    # finer than itself.
    for offset in (1e-3, 1e-5, 1e-7):
        assert eg.params(100, 100 + offset, 300, P_LOW)[0] > endpoint[0]


def test_ceiling_is_reported_rather_than_returned_silently():
    """A caller must be able to tell a converged fit from the ceiling
    approximation."""
    symmetric = eg.params(100, 200, 300, P_LOW)
    assert symmetric.ceiling
    assert symmetric[0] == eg.ALPHA_MAX
    assert symmetric.position_error > BOUND

    ordinary = eg.params(100, 140, 300, P_LOW)
    assert not ordinary.ceiling
    assert ordinary.position_error < BOUND


def test_fit_result_is_a_plain_tuple():
    """The diagnostics must not change the shape of the public return value."""
    r = eg.params(100, 140, 300, P_LOW)
    a, b, d = r
    assert len(r) == 3
    assert (a, b, d) == tuple(r)
    assert isinstance(r, tuple)


def test_fitted_mode_is_the_elicited_one():
    """The fit constructs the location as delta = most_likely - (alpha - 1)*beta,
    so the mode of the fitted distribution is the elicited most likely value.
    This is what makes the identity mean = mode + beta hold in the fitted
    context; the distribution's own moments stay defined from alpha, beta and
    delta alone."""
    for mode_value in (100, 140, 180, 220, 260, 300):
        a, b, d = eg.params(100, mode_value, 300, P_LOW)
        assert eg.mode(a, b, d) == pytest.approx(mode_value, abs=1e-8)
        assert eg.mean(a, b, d) == pytest.approx(mode_value + b, rel=1e-9)


def test_numerical_controls_validated():
    for bad in (0.0, -1e-12, 0.5, 1.0, float('nan')):
        with pytest.raises(ValueError):
            eg.params(100, 140, 300, P_LOW, tolerance=bad)
    for bad in (1.0, 0.5, -1.0, float('inf')):
        with pytest.raises(ValueError):
            eg.params(100, 140, 300, P_LOW, alpha_max=bad)
    for bad in (0, -1, 2.5):
        with pytest.raises(ValueError):
            eg.params(100, 140, 300, P_LOW, max_iter=bad)


def test_ceiling_below_the_endpoint_shape_is_rejected():
    """A ceiling that does not reach past the endpoint shape cannot fit anything
    and must be refused rather than used."""
    with pytest.raises(ValueError):
        eg.params(100, 140, 300, P_LOW, alpha_max=1.0000001)


def test_reflection_is_symmetric():
    """Mirror-image estimates must give mirror-image parameters. An
    implementation that has dropped the reflection passes the right-skewed
    cases and fails here."""
    for offset in (10, 20, 40, 60, 80):
        ra, rb, rd = eg.params(100, 100 + offset, 300, P_LOW)
        la, lb, ld = eg.params(100, 300 - offset, 300, P_LOW)
        assert la == pytest.approx(ra, rel=1e-12)
        assert lb == pytest.approx(-rb, rel=1e-12)
        assert ld == pytest.approx(400 - rd, rel=1e-12)


@pytest.mark.parametrize("bad", [
    (300, 140, 100),            # low > mode
    (100, 400, 300),            # mode > high
    (100, 100, 100),            # degenerate
    (float("inf"), 140, 300),   # not finite
    (100, float("nan"), 300),
])
def test_invalid_estimates_rejected(bad):
    with pytest.raises(ValueError):
        eg.params(*bad)


@pytest.mark.parametrize("p", [0.0, 0.5, 0.6, 1.0, -0.1])
def test_invalid_percentile_rejected(p):
    with pytest.raises(ValueError):
        eg.params(100, 140, 300, p)


def test_pdf_is_zero_outside_support():
    a, b, d = eg.params(100, 140, 300, P_LOW)
    assert eg.pdf(d - 1.0, a, b, d) == 0.0
    la, lb, ld = eg.params(100, 260, 300, P_LOW)
    assert eg.pdf(ld + 1.0, la, lb, ld) == 0.0


@pytest.mark.parametrize("beta", [49.5, -49.5])
def test_cdf_inverts_ppf(beta):
    for p in (0.01, 0.1, 0.37, 0.5, 0.9, 0.99):
        x = eg.ppf(p, 2.77, beta, 52.25)
        assert eg.cdf(x, 2.77, beta, 52.25) == pytest.approx(p, abs=1e-12)


def test_moments_agree_with_sampling():
    """A loose check that rvs draws from the distribution the closed-form
    moments describe. The tolerance is Monte Carlo slack, not a precision
    claim, and is kept generous so the test does not depend on the exact
    random stream of a particular SciPy version."""
    a, b, d = eg.params(4000, 10000, 13000, P_LOW)
    x = eg.rvs(a, b, d, size=200000, random_state=12345)
    assert np.mean(x) == pytest.approx(eg.mean(a, b, d), rel=1e-2)
    assert np.std(x) == pytest.approx(eg.std(a, b, d), rel=1e-2)


@pytest.mark.parametrize("method", ["mle", "mom"])
def test_fit_recovers_both_skew_directions(method):
    rng = np.random.default_rng(0)
    sample = rng.gamma(3.0, 2.0, 5000)
    a, b, d = eg.fit(sample, method=method)
    assert b > 0 and a == pytest.approx(3.0, rel=0.15)
    a, b, d = eg.fit(-sample, method=method)
    assert b < 0 and a == pytest.approx(3.0, rel=0.15)


def test_mom_matches_the_published_spreadsheet_formulas():
    """The method of moments must reproduce the formulas given for Excel, so
    that the two reference implementations agree."""
    from scipy.stats import skew as sample_skew
    rng = np.random.default_rng(7)
    x = rng.gamma(3.0, 2.0, 5000) + 50
    g = sample_skew(x, bias=False)
    alpha = 4 / g ** 2
    beta = np.std(x, ddof=1) * g / 2
    delta = np.mean(x) - alpha * beta
    assert eg.fit(x, method='mom') == (alpha, beta, delta)


def test_mom_and_mle_agree_on_a_large_sample():
    rng = np.random.default_rng(11)
    x = rng.gamma(4.0, 3.0, 20000) + 20
    mle = eg.fit(x, method='mle')
    mom = eg.fit(x, method='mom')
    for got, want in zip(mom, mle):
        assert got == pytest.approx(want, rel=0.08, abs=0.6)


def test_unknown_fit_method_rejected():
    with pytest.raises(ValueError):
        eg.fit([1.0, 2.0, 3.0], method='ols')
