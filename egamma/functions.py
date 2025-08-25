import scipy.special as sc
from scipy.stats import gamma
import numpy as np
import numpy as np
from scipy.special import gammainc, gammaincinv
from scipy.stats import gamma, skew as skew_stat


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
    :returns: The probability density function evaluated at x.
    :rtype: float
    """
    if beta == 0:
        return np.nan  # or raise ValueError
    t = (x - delta) / beta
    if t < 0:
        return 0.0
    # map to standard 2p Gamma with loc=0, scale=|beta|
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
            return np.nan  # or raise ValueError
    if beta > 0:
        if x <= delta:
            return 0.0
        t = (x - delta) / beta  # t >= 0
        return float(gammainc(alpha, t))
    else:
        if x >= delta:
            return 1.0
        t = (delta - x) / abs(beta)  # t >= 0
        return float(1.0 - gammainc(alpha, t))


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
        return np.nan  # or raise ValueError
    p = float(np.clip(percentile, eps, 1.0 - eps))
    if beta > 0:
        return delta + beta * float(gammaincinv(alpha, p))
    else:
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
    Calculate the kurtosis of the expanded gamma distribution.

    :param alpha: The shape parameter of the expanded gamma distribution.
    :type alpha: float
    :param beta: The scale parameter of the expanded gamma distribution, defaults to 1.
    :type beta: float, optional
    :param delta: The location parameter of the expanded gamma distribution, defaults to 0.
    :type delta: float, optional
    :returns: The kurtosis of the expanded gamma distribution.
    :rtype: float
    """
    return 6 / alpha


def fit(data):
    """
    Fit the expanded gamma distribution to data using maximum likelihood estimation.

    :param data: The data to fit.
    :type data: array_like
    :returns: The estimated shape, location, and scale parameters of the expanded gamma distribution.
    :rtype: tuple
    """
    if sc.stats.skew(data) > 0:
        p = gamma.fit(data)
        return p[0], p[2], p[1]
    else:
        data = data * -1
        p = gamma.fit(data)
        return p[0], -p[2], -p[1]


def params(low, most_likely, high, low_prob=0.1):
    """
    Find the parameters of the expanded gamma distribution given a three-point estimate.

    This function is designed to work with estimates of uncertain quantities based on a optimistic ,
    most likely , and pessimistic  scenario. The terms 'low' and 'high' are used instead of
    'optimistic' and 'pessimistic' to accommodate contexts where the meaning of these terms may be reversed,
    such as costs (where high is pessimistic) versus revenues (where high is optimistic).

    :param float low: The low estimate.
    :param float most_likely: The most likely estimate.
    :param float high: The high estimate.
    :param float low_prob:  The probability associated with the low estimate. Defaults to 0.1. Note: the converse high_prob is calculated automatically as (1 - low_prob)

    :return: A tuple containing the estimated shape (alpha), scale (beta), and location (delta) parameters of the gamma distribution.
    :rtype: tuple

    :raises ValueError: If the provided three-point estimates do not form a valid range (i.e., if 'low' is greater than 'most_likely', 'high' is less than 'most_likely', or 'low' is equal to 'high').

    """

    if low > most_likely or high < most_likely or low == high:
        msg = 'Invalid three-point-estimate: '
        if low > most_likely:
            msg += "'low' must be less than or equal to 'mode'"
        if high < most_likely:
            msg += "'High' must be greater than or equal 'mode'"
        if low == high:
            msg += "'High' must be greater than low"
        raise ValueError(msg)
    
    # Solve alpha first 
    if low == most_likely or high == most_likely:
        alpha = __find_alpha_at_mode_equals_probability(low_prob)
    else:
        alpha = __find_alpha(low, most_likely, high, low_prob)

    # Decide skew by the side lengths around the mode
    dL = most_likely - low
    dH = high - most_likely
    right_skew = (dH >= dL)

    q_lo = float(gammaincinv(alpha, low_prob))
    q_hi = float(gammaincinv(alpha, 1.0 - low_prob))

    if right_skew:  # beta > 0
        denom = q_lo - (alpha - 1.0)
        if denom == 0:
            denom = np.copysign(np.finfo(float).tiny, denom)
        beta = (low - most_likely) / denom
    else:  # beta < 0, use q(1-p)
        denom = q_hi - (alpha - 1.0)
        if denom == 0:
            denom = np.copysign(np.finfo(float).tiny, denom)
        beta = (low - most_likely) / denom  # this will be negative

    # Guard against +0.0 / -0.0
    if beta == 0:
        beta = np.copysign(np.finfo(float).tiny, -1.0 if not right_skew else 1.0)
    delta = most_likely - (alpha - 1) * beta

    return alpha, beta, delta


def __find_alpha(low, mode, high, low_prob=0.1, high_prob=0.9, return_itertations=False, threshold=1e-10):
    iter = 0
    target_ratio = (mode - low) / (high - mode)
    if abs(target_ratio) > 1:
        target_ratio = 1 / target_ratio

    if target_ratio > 0.99999:
        if (return_itertations):
            return 1e9, 0
        else:
            return 1e9

    skew_low = 2 / (np.sqrt(1e15))
    skew_high = 2
    while skew_low <= skew_high:
        iter += 1
        skew_mid = (skew_low + skew_high) / 2
        alpha_candidate = 4 / (skew_mid ** 2)
        current_ratio = ((alpha_candidate - 1) - ppf(low_prob, alpha_candidate)) / (
                ppf(high_prob, alpha_candidate) - (alpha_candidate - 1))

        if abs((current_ratio / target_ratio) - 1) < threshold:
            if (return_itertations):
                return alpha_candidate, iter
            else:
                return alpha_candidate
        elif current_ratio < target_ratio:
            skew_high = skew_mid
        else:
            skew_low = skew_mid
    return None


def __find_alpha_at_mode_equals_probability(probability, decimals=10, return_itertations=False):
    if probability > 0.5:
        probability = 1 - probability
    skew_low = 0.000001
    skew_high = 2
    iter = 0
    while skew_low <= skew_high:
        iter += 1
        skew_mid = (skew_low + skew_high) / 2
        alpha_candidate = 4 / (skew_mid ** 2)
        mode = alpha_candidate - 1
        mode_candidate = ppf(probability, alpha_candidate)

        if round(mode_candidate, decimals) == round(mode, decimals):
            if (return_itertations):
                return alpha_candidate, iter
            else:
                return alpha_candidate
        elif mode_candidate > mode:
            skew_high = skew_mid
        else:
            skew_low = skew_mid
    return None
