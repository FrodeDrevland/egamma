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

#: Default relative acceptance threshold on the target ratio. The reproduction
#: error of the elicited values is bounded by THRESHOLD / (2 * (2 - THRESHOLD)).
THRESHOLD = 1e-10

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


def params(low, most_likely, high, low_prob=0.1):
    """
    Find the parameters of the expanded gamma distribution given a three-point estimate.

    This function is designed to work with estimates of uncertain quantities based on a optimistic ,
    most likely , and pessimistic  scenario. The terms 'low' and 'high' are used instead of
    'optimistic' and 'pessimistic' to accommodate contexts where the meaning of these terms may be reversed,
    such as costs (where high is pessimistic) versus revenues (where high is optimistic).

    The scale parameter is taken from the full elicited span rather than from a
    mode-to-outer distance. The span is better conditioned, and one expression
    covers every case including the mode coinciding with an outer value, where
    a mode-to-outer distance degenerates to zero over zero.

    :param float low: The low estimate.
    :param float most_likely: The most likely estimate.
    :param float high: The high estimate.
    :param float low_prob:  The probability associated with the low estimate. Defaults to 0.1. Note: the converse high_prob is calculated automatically as (1 - low_prob)

    :return: A tuple containing the estimated shape (alpha), scale (beta), and location (delta) parameters of the gamma distribution.
    :rtype: tuple

    :raises ValueError: If the estimates are not finite, if low_prob is outside (0, 0.5), or if the provided three-point estimates do not form a valid range.
    :raises RuntimeError: If the shape search does not converge.
    """
    for name, value in (('low', low), ('most_likely', most_likely), ('high', high)):
        if not np.isfinite(value):
            raise ValueError("Invalid three-point-estimate: '%s' must be finite" % name)

    if not 0 < low_prob < 0.5:
        raise ValueError('Invalid low_prob: must satisfy 0 < low_prob < 0.5')

    if low > most_likely or high < most_likely or low == high:
        msg = 'Invalid three-point-estimate: '
        if low > most_likely:
            msg += "'low' must be less than or equal to 'mode'"
        if high < most_likely:
            msg += "'High' must be greater than or equal 'mode'"
        if low == high:
            msg += "'High' must be greater than low"
        raise ValueError(msg)

    if low == most_likely or high == most_likely:
        alpha = __find_alpha_at_mode_equals_probability(low_prob)
    else:
        alpha = __find_alpha(low, most_likely, high, low_prob)

    if alpha is None:
        raise RuntimeError(
            'Fit did not converge: the requested threshold could not be met. '
            'This usually means the estimate is more skewed than the elicited '
            'percentiles admit.')

    beta = (high - low) / (ppf(1 - low_prob, alpha) - ppf(low_prob, alpha))

    if abs(beta) == 0:
        beta = np.finfo(np.float64).tiny

    if (most_likely - low) > (high - most_likely):
        beta = -beta

    delta = most_likely - (alpha - 1) * beta

    return alpha, beta, delta


def __find_alpha(low, mode, high, low_prob=0.1, high_prob=None, return_itertations=False,
                 threshold=THRESHOLD, alpha_max=ALPHA_MAX, max_iter=MAX_ITER):
    """Solve the single equation in the shape parameter by bisection on skewness.

    Returns ``None`` (or ``(None, iterations)``) if the requested threshold
    cannot be met, rather than looping or returning an unconverged value.
    """
    if high_prob is None:
        high_prob = 1 - low_prob

    def _ratio(a):
        return ((a - 1) - ppf(low_prob, a)) / (ppf(high_prob, a) - (a - 1))

    target_ratio = (mode - low) / (high - mode)
    if target_ratio > 1:
        target_ratio = 1 / target_ratio

    # The largest ratio the search can actually reach is the one alpha_max
    # produces. Deriving the symmetry shortcut from it rather than fixing it
    # keeps the root bracketed for any percentile convention.
    ratio_max = _ratio(alpha_max)
    if ratio_max <= 0:
        raise ValueError(
            'alpha_max is too small for low_prob=%g: no admissible estimate '
            'can be fitted with this ceiling.' % low_prob)

    if target_ratio >= ratio_max:
        return (alpha_max, 0) if return_itertations else alpha_max

    skew_low = 2 / np.sqrt(alpha_max)
    skew_high = 2
    for iteration in range(1, max_iter + 1):
        skew_mid = (skew_low + skew_high) / 2
        if skew_mid == skew_low or skew_mid == skew_high:
            return (None, iteration) if return_itertations else None
        alpha_candidate = 4 / (skew_mid ** 2)
        current_ratio = _ratio(alpha_candidate)

        if abs((current_ratio / target_ratio) - 1) < threshold:
            return (alpha_candidate, iteration) if return_itertations else alpha_candidate
        elif current_ratio < target_ratio:
            skew_high = skew_mid
        else:
            skew_low = skew_mid
    return (None, max_iter) if return_itertations else None


def __find_alpha_at_mode_equals_probability(probability, return_itertations=False,
                                            threshold=THRESHOLD, alpha_max=ALPHA_MAX,
                                            max_iter=MAX_ITER):
    """Shape parameter whose mode falls exactly at the given probability.

    Stops on the residual normalised by the standard percentile span, so the
    reproduction bound of the general search covers this case too.
    """
    if probability > 0.5:
        probability = 1 - probability

    skew_low = 2 / np.sqrt(alpha_max)
    skew_high = 2
    for iteration in range(1, max_iter + 1):
        skew_mid = (skew_low + skew_high) / 2
        if skew_mid == skew_low or skew_mid == skew_high:
            return (None, iteration) if return_itertations else None
        alpha_candidate = 4 / (skew_mid ** 2)
        mode_standard = alpha_candidate - 1
        mode_candidate = ppf(probability, alpha_candidate)
        span = ppf(1 - probability, alpha_candidate) - mode_candidate

        if abs(mode_candidate - mode_standard) / span < threshold / 4:
            return (alpha_candidate, iteration) if return_itertations else alpha_candidate
        elif mode_candidate > mode_standard:
            skew_high = skew_mid
        else:
            skew_low = skew_mid
    return (None, max_iter) if return_itertations else None
