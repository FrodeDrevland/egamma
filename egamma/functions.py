"""Expanded gamma distribution: the Pearson Type III family written in the
gamma distribution's shape, scale and location parameters, with the scale
permitted to take negative values."""

import numpy as np
from scipy.special import gammainc, gammaincinv
from scipy.stats import gamma, skew as _sample_skew

#: Largest shape parameter the fitting routines will return.
#:
#: A symmetric three-point estimate, where the mode sits exactly midway
#: between the outer values, is only reproducible in the limit as the shape
#: parameter tends to infinity, so a finite stand-in is needed. At this value
#: the distribution is close enough to normal that the elicited values are
#: reproduced to about 1.5e-5 of the elicited range. Raising it buys little,
#: because recovery of the shape parameter is increasingly ill-conditioned as
#: an estimate approaches symmetry while the elicited values continue to be
#: reproduced within the same bound.
ALPHA_MAX = 1e9

#: Default acceptance tolerance on the normalised mode position.
#:
#: The search stops when the fitted mode position differs from the elicited one
#: by less than this, and that difference is itself the maximum normalised error
#: in the three reproduced values, so the tolerance is stated directly in the
#: quantity a user cares about. It replaces the relative threshold on the
#: half-range ratio used up to version 1.1.1, which demanded ever tighter
#: absolute agreement as the mode approached an outer value and could refuse
#: admissible estimates for that reason alone. The value preserves the
#: reproduction accuracy the old relative threshold of 1e-10 implied.
TOLERANCE = 2.5e-11

#: Deprecated alias for :data:`TOLERANCE`, retained so that code written against
#: version 1.1.1 keeps importing. Note that the quantity it bounds has changed.
THRESHOLD = TOLERANCE

#: Maximum bisection steps before reporting failure.
MAX_ITER = 100

#: Smallest distance from 0 and 1 at which a percentile is evaluated.
EPS = np.finfo(float).eps


def pdf(x, alpha, beta=1.0, delta=0.0):
    """
    Calculate the probability density function for the expanded gamma distribution.

    :param x: The point at which the pdf is evaluated.
    :type x: float
    :param alpha: The shape parameter of the expanded gamma distribution.
    :type alpha: float
    :param beta: The scale parameter of the expanded gamma distribution, defaults to 1.0.
    :type beta: float, optional
    :param delta: The location parameter of the expanded gamma distribution, defaults to 0.0.
    :type delta: float, optional
    :returns: The probability density function evaluated at x, or zero outside the support.
    :rtype: float
    """
    if beta == 0:
        return np.nan
    if (x - delta) / beta < 0:
        return 0.0
    return gamma.pdf(abs(x - delta), a=alpha, loc=0.0, scale=abs(beta))


def cdf(x, alpha, beta=1.0, delta=0.0):
    """
    Calculate the cumulative distribution function for the expanded gamma distribution.

    :param x: The point at which the cdf is evaluated.
    :type x: float
    :param alpha: The shape parameter of the expanded gamma distribution.
    :type alpha: float
    :param beta: The scale parameter of the expanded gamma distribution, defaults to 1.0.
    :type beta: float, optional
    :param delta: The location parameter of the expanded gamma distribution, defaults to 0.0.
    :type delta: float, optional
    :returns: The cumulative distribution function evaluated at x.
    :rtype: float
    """
    if beta == 0:
        return np.nan
    if beta > 0:
        if x <= delta:
            return 0.0
        return float(gammainc(alpha, (x - delta) / beta))
    if x >= delta:
        return 1.0
    return float(1.0 - gammainc(alpha, (delta - x) / abs(beta)))


def ppf(percentile, alpha, beta=1.0, delta=0.0):
    """
    Calculate the percent point function (inverse of cdf) at a given percentile for the expanded gamma distribution.

    :param percentile: The percentile at which to evaluate (0-1).
    :type percentile: float
    :param alpha: The shape parameter of the expanded gamma distribution.
    :type alpha: float
    :param beta: The scale parameter of the expanded gamma distribution, defaults to 1.0.
    :type beta: float, optional
    :param delta: The location parameter of the expanded gamma distribution, defaults to 0.0.
    :type delta: float, optional
    :returns: The value of the distribution at the given percentile.
    :rtype: float
    """
    if beta == 0:
        return np.nan
    p = float(np.clip(percentile, EPS, 1.0 - EPS))
    if beta > 0:
        return delta + beta * float(gammaincinv(alpha, p))
    return delta - abs(beta) * float(gammaincinv(alpha, 1.0 - p))


def rvs(alpha, beta=1, delta=0, size=1, random_state=None):
    """
    Generate random variates of the expanded gamma distribution.

    :param alpha: The shape parameter of the expanded gamma distribution.
    :type alpha: float
    :param beta: The scale parameter of the expanded gamma distribution, defaults to 1.
    :type beta: float, optional
    :param delta: The location parameter of the expanded gamma distribution, defaults to 0.
    :type delta: float, optional
    :param size: The number of random variates to generate, defaults to 1.
    :type size: int, optional
    :param random_state: If int or RandomState, use it for drawing the random variates. If None, rely on self.random_state, defaults to None.
    :type random_state: int, RandomState instance or None, optional
    :returns: Random variates of the expanded gamma distribution.
    :rtype: ndarray or scalar
    """
    if beta == 0:
        raise ValueError("beta must be nonzero")
    y = gamma.rvs(alpha, loc=0.0, scale=abs(beta), size=size, random_state=random_state)
    return delta + y if beta > 0 else delta - y


def mean(alpha, beta=1, delta=0):
    """
    Calculate the mean of the expanded gamma distribution.

    :param alpha: The shape parameter of the expanded gamma distribution.
    :type alpha: float
    :param beta: The scale parameter of the expanded gamma distribution, defaults to 1.
    :type beta: float, optional
    :param delta: The location parameter of the expanded gamma distribution, defaults to 0.
    :type delta: float, optional
    :returns: The mean of the expanded gamma distribution.
    :rtype: float
    """
    return alpha * beta + delta


def mode(alpha, beta=1, delta=0):
    """
    Calculate the mode of the expanded gamma distribution.

    Holds for alpha > 1; for alpha <= 1 the density is monotone on its support
    and the mode lies at delta.

    :param alpha: The shape parameter of the expanded gamma distribution.
    :type alpha: float
    :param beta: The scale parameter of the expanded gamma distribution, defaults to 1.
    :type beta: float, optional
    :param delta: The location parameter of the expanded gamma distribution, defaults to 0.
    :type delta: float, optional
    :returns: The mode of the expanded gamma distribution.
    :rtype: float
    """
    return (alpha - 1) * beta + delta


def median(alpha, beta=1, delta=0):
    """
    Calculate the median of the expanded gamma distribution.

    :param alpha: The shape parameter of the expanded gamma distribution.
    :type alpha: float
    :param beta: The scale parameter of the expanded gamma distribution, defaults to 1.
    :type beta: float, optional
    :param delta: The location parameter of the expanded gamma distribution, defaults to 0.
    :type delta: float, optional
    :returns: The median of the expanded gamma distribution.
    :rtype: float
    """
    return ppf(0.50, alpha, beta, delta)


def var(alpha, beta=1, delta=0):
    """
    Calculate the variance of the expanded gamma distribution.

    :param alpha: The shape parameter of the expanded gamma distribution.
    :type alpha: float
    :param beta: The scale parameter of the expanded gamma distribution, defaults to 1.
    :type beta: float, optional
    :param delta: The location parameter of the expanded gamma distribution, defaults to 0.
    :type delta: float, optional
    :returns: The variance of the expanded gamma distribution.
    :rtype: float
    """
    return alpha * (beta ** 2)


def std(alpha, beta=1, delta=0):
    """
    Calculate the standard deviation of the expanded gamma distribution.

    :param alpha: The shape parameter of the expanded gamma distribution.
    :type alpha: float
    :param beta: The scale parameter of the expanded gamma distribution, defaults to 1.
    :type beta: float, optional
    :param delta: The location parameter of the expanded gamma distribution, defaults to 0.
    :type delta: float, optional
    :returns: The standard deviation of the expanded gamma distribution.
    :rtype: float
    """
    return np.sqrt(var(alpha, beta, delta))


def skew(alpha, beta=1, delta=0):
    """
    Calculate the skewness of the expanded gamma distribution.

    :param alpha: The shape parameter of the expanded gamma distribution.
    :type alpha: float
    :param beta: The scale parameter of the expanded gamma distribution, defaults to 1.
    :type beta: float, optional
    :param delta: The location parameter of the expanded gamma distribution, defaults to 0.
    :type delta: float, optional
    :returns: The skewness of the expanded gamma distribution.
    :rtype: float
    """
    if beta == 0:
        return np.nan
    return (2.0 / np.sqrt(alpha)) * np.sign(beta)


def kurtosis(alpha, beta=1, delta=0):
    """
    Calculate the excess kurtosis of the expanded gamma distribution.

    :param alpha: The shape parameter of the expanded gamma distribution.
    :type alpha: float
    :param beta: The scale parameter of the expanded gamma distribution, defaults to 1.
    :type beta: float, optional
    :param delta: The location parameter of the expanded gamma distribution, defaults to 0.
    :type delta: float, optional
    :returns: The *excess* kurtosis of the expanded gamma distribution.
    :rtype: float
    """
    return 6 / alpha


def fit(data, method='mle'):
    r"""
    Fit the expanded gamma distribution to a sample.

    Two estimators are available. ``'mle'`` maximises the likelihood, via
    SciPy's gamma fit applied to the sample in whichever direction it is
    skewed. ``'mom'`` is the method of moments, inverting the expressions for
    the mean, variance and skewness:

    .. math::
        \alpha = \frac{4}{g^2}, \qquad
        \beta = \frac{s\,g}{2}, \qquad
        \delta = \bar{x} - \alpha\beta

    where :math:`\bar{x}`, :math:`s` and :math:`g` are the sample mean,
    standard deviation and skewness. The sign of the skewness carries into the
    scale parameter, so the direction of skew is handled automatically.

    Maximum likelihood is the default and is generally the better estimator.
    The method of moments is closed-form and needs no iteration, which makes it
    useful as a starting point or where an optimiser is unavailable, but it
    rests on the sample skewness, whose sampling variance is large; estimates
    from small samples can depart substantially from the parent distribution.
    It is the estimator the companion Excel library uses, so results from the
    two agree.

    :param data: The data to fit.
    :type data: array_like
    :param method: ``'mle'`` (default) or ``'mom'``.
    :type method: str, optional
    :returns: The estimated shape, scale and location parameters.
    :rtype: tuple
    :raises ValueError: If ``method`` is not recognised, if fewer than three
        values are supplied, or if the method of moments is asked for and the
        sample skewness is zero.
    """
    data = np.asarray(data, dtype=float)
    if data.size < 3:
        raise ValueError('At least three observations are required to fit.')
    if not np.all(np.isfinite(data)):
        raise ValueError('Data must be finite.')

    if method == 'mle':
        if _sample_skew(data) > 0:
            p = gamma.fit(data)
            return p[0], p[2], p[1]
        p = gamma.fit(-data)
        return p[0], -p[2], -p[1]

    if method == 'mom':
        g = _sample_skew(data, bias=False)
        if g == 0:
            raise ValueError(
                'Method of moments is undefined for a sample with zero '
                'skewness: the shape parameter would be infinite. Use '
                "method='mle', or treat the sample as symmetric.")
        alpha = 4 / g ** 2
        if alpha > ALPHA_MAX:
            alpha = ALPHA_MAX
        beta = np.std(data, ddof=1) * g / 2
        delta = np.mean(data) - alpha * beta
        return alpha, beta, delta

    raise ValueError("method must be 'mle' or 'mom', not %r" % (method,))


class FitResult(tuple):
    """The fitted ``(alpha, beta, delta)``, with diagnostics attached.

    This is a plain 3-tuple, so ``alpha, beta, delta = params(...)`` and
    ``params(...)[0]`` behave exactly as they did in earlier versions. The
    extra attributes are there because returning the shape ceiling silently is
    unhelpful: a caller that wants to know whether it received a fitted shape or
    the ceiling approximation can ask.

    :ivar bool ceiling: True when the estimate was too near symmetry for the
        shape ceiling to resolve, so ``alpha_max`` was returned. The elicited
        values are then reproduced to ``position_error`` rather than to the
        requested tolerance.
    :ivar float position_error: ``|t_alpha - t|`` at the returned shape, the
        maximum normalised error in the reproduced values in exact arithmetic.
    :ivar int iterations: Bisection steps used; zero when the ceiling was taken.
    """

    def __new__(cls, alpha, beta, delta, ceiling=False, position_error=0.0,
                iterations=0):
        self = super().__new__(cls, (alpha, beta, delta))
        self.ceiling = ceiling
        self.position_error = position_error
        self.iterations = iterations
        return self

    def __repr__(self):
        return ('FitResult(alpha=%r, beta=%r, delta=%r, ceiling=%r, '
                'position_error=%r)' % (self[0], self[1], self[2],
                                        self.ceiling, self.position_error))


def params(low, most_likely, high, low_prob=0.1, tolerance=TOLERANCE,
           alpha_max=ALPHA_MAX, max_iter=MAX_ITER):
    """
    Find the parameters of the expanded gamma distribution given a three-point estimate.

    This function is designed to work with estimates of uncertain quantities based on an
    optimistic, most likely, and pessimistic scenario. The terms 'low' and 'high' are used
    instead of 'optimistic' and 'pessimistic' to accommodate contexts where the meaning of
    these terms may be reversed, such as costs (where high is pessimistic) versus revenues
    (where high is optimistic).

    The shape parameter is found by matching the mode's position within the
    elicited range; scale and location then follow in closed form. The scale is
    taken from the full elicited span rather than from a mode-to-outer distance,
    because the span is better conditioned and one expression covers every case,
    including a mode coinciding with an outer value where a mode-to-outer
    distance degenerates to zero over zero.

    The returned object is a 3-tuple of ``(alpha, beta, delta)`` and unpacks as
    one. It additionally carries ``ceiling``, ``position_error`` and
    ``iterations``; see :class:`FitResult`.

    :param float low: The low estimate.
    :param float most_likely: The most likely estimate.
    :param float high: The high estimate.
    :param float low_prob: The probability associated with the low estimate. Defaults to
        0.1. Note: the converse high_prob is calculated automatically as (1 - low_prob).
    :param float tolerance: Acceptance tolerance on the normalised mode position,
        which is also the bound on the normalised error in the reproduced values.
        Defaults to :data:`TOLERANCE`.
    :param float alpha_max: The shape ceiling. Defaults to :data:`ALPHA_MAX`.
    :param int max_iter: Maximum bisection steps. Defaults to :data:`MAX_ITER`.

    :return: The shape (alpha), scale (beta) and location (delta) parameters.
    :rtype: FitResult

    :raises ValueError: If the estimates are not finite, if low_prob is outside (0, 0.5),
        if a numerical control is out of range, or if the provided three-point estimate
        does not form a valid range.
    :raises RuntimeError: If the shape search does not converge.
    """
    for name, value in (('low', low), ('most_likely', most_likely), ('high', high)):
        if not np.isfinite(value):
            raise ValueError("Invalid three-point-estimate: '%s' must be finite" % name)

    if not 0 < low_prob < 0.5:
        raise ValueError('Invalid low_prob: must satisfy 0 < low_prob < 0.5')

    if not (np.isfinite(tolerance) and 0 < tolerance < 0.5):
        raise ValueError('Invalid tolerance: must satisfy 0 < tolerance < 0.5')

    if not (np.isfinite(alpha_max) and alpha_max > 1):
        raise ValueError('Invalid alpha_max: must be finite and greater than 1')

    if not (isinstance(max_iter, (int, np.integer)) and max_iter > 0):
        raise ValueError('Invalid max_iter: must be a positive integer')

    if low > most_likely or high < most_likely or low >= high:
        msg = 'Invalid three-point-estimate: '
        if low > most_likely:
            msg += "'low' must be less than or equal to 'most_likely'"
        elif high < most_likely:
            msg += "'high' must be greater than or equal to 'most_likely'"
        else:
            msg += "'high' must be greater than 'low'"
        raise ValueError(msg)

    alpha, iterations, at_ceiling, position_error = __find_alpha(
        low, most_likely, high, low_prob, tolerance, alpha_max, max_iter)

    if alpha is None:
        raise RuntimeError(
            'Fit did not converge: the requested tolerance of %g could not be '
            'met within %d iterations. Try a looser tolerance, or check that '
            'the arithmetic can represent the elicited values and their '
            'differences.' % (tolerance, max_iter))

    beta = (high - low) / (ppf(1 - low_prob, alpha) - ppf(low_prob, alpha))

    if (most_likely - low) > (high - most_likely):
        beta = -beta

    delta = most_likely - (alpha - 1) * beta

    return FitResult(alpha, beta, delta, at_ceiling, position_error, iterations)


def __find_alpha(low, mode, high, low_prob=0.1, tolerance=TOLERANCE,
                 alpha_max=ALPHA_MAX, max_iter=MAX_ITER):
    """Find the shape parameter by bisection on the magnitude of skewness.

    The search matches the mode's position within the elicited range,

        t         = min(mode - low, high - mode) / (high - low)
        t_alpha   = (alpha - 1 - q_low) / (q_high - q_low)

    and stops when ``|t_alpha - t| < tolerance``. Both skew directions reduce to
    the same target, so no separate treatment of left skew is needed here, and a
    mode at an outer value is simply the case ``t = 0`` rather than a separate
    routine.

    Matching positions rather than the half-range ratio matters near an outer
    value. The ratio tends to zero there, so a relative test on it would demand
    progressively finer absolute agreement even though the accuracy required of
    the elicited values had not changed.

    :returns: ``(alpha, iterations, at_ceiling, position_error)``, with ``alpha``
        None if the search did not converge.
    """
    high_prob = 1 - low_prob
    span = high - low
    target_position = min(mode - low, high - mode) / span

    def _position(a, q_low, q_high):
        return ((a - 1) - q_low) / (q_high - q_low)

    q_low = ppf(low_prob, alpha_max)
    q_high = ppf(high_prob, alpha_max)
    max_position = _position(alpha_max, q_low, q_high)

    # The greatest position the search can reach is the one alpha_max produces,
    # and it depends on the percentile convention: about 0.499985 at P_L = 0.10
    # but 0.457948 at P_L = 0.4999. Deriving the cut from alpha_max rather than
    # fixing it keeps the root bracketed under any convention.
    if max_position <= 0:
        raise ValueError(
            'alpha_max=%g is too small for low_prob=%g: it does not reach '
            'beyond the endpoint shape, so no admissible estimate can be '
            'fitted with this ceiling.' % (alpha_max, low_prob))

    if target_position >= max_position:
        return alpha_max, 0, True, abs(max_position - target_position)

    skew_low = 2 / np.sqrt(alpha_max)
    skew_high = 2
    for iteration in range(1, max_iter + 1):
        skew_mid = (skew_low + skew_high) / 2
        if skew_mid == skew_low or skew_mid == skew_high:
            return None, iteration, False, np.nan
        alpha_candidate = 4 / (skew_mid ** 2)
        q_low = ppf(low_prob, alpha_candidate)
        q_high = ppf(high_prob, alpha_candidate)
        candidate_position = _position(alpha_candidate, q_low, q_high)
        position_error = abs(candidate_position - target_position)

        if position_error < tolerance:
            return alpha_candidate, iteration, False, position_error
        elif candidate_position < target_position:
            skew_high = skew_mid
        else:
            skew_low = skew_mid
    return None, max_iter, False, np.nan
