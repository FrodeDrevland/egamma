import random
import statistics as stat
import scipy.special as sc
from scipy.stats import gamma
import numpy as np


# Probabilty density function
def pdf(x, alpha, beta=1.0, delta=0.0):
    if (delta-x)/beta > 0:
        return np.nan
    else:
        return gamma.pdf(abs(x - delta), alpha, 0, abs(beta))

# Cumulative distribution function
def cdf(x, alpha, beta=1.0, delta=0.0):
    if beta > 0:
        return sc.gammainc(alpha, (x - delta) / beta)
    else:
        return 1 - sc.gammainc(alpha, (x - delta) / beta)


# Percent point function (Inverse CDF)
def ppf(percentile, alpha, beta=1.0, delta=0.0):
    if beta > 0:
        return sc.gammaincinv(alpha, percentile) * beta + delta
    else:
        return sc.gammaincinv(alpha, 1 - percentile) * beta + delta



# Random variates
def rvs(alpha, beta=1, delta=0, size=1, random_state=None):
    result = gamma.rvs(alpha, scale=abs(beta), loc=0, size=size, random_state=random_state)
    return result + delta if beta > 0 else delta - result


def mean(alpha, beta=1, delta=0):
    return alpha * beta + delta


def mode(alpha, beta=1, delta=0):
    return (alpha - 1) * beta + delta


# Median of the distribution
def median(alpha, beta=1, delta=0):
    return ppf(0.50, alpha, beta, delta)


# Variance of the distribution
def var(alpha, beta=1, delta=0):
    return alpha * beta ** 2


# Standard deviation of the distribution.
def std(alpha, beta=1, delta=0):
    return np.sqrt(alpha) * beta


# Skew deviation of the distribution.
def skew(alpha, beta=1, delta=0):
    return 2 / np.sqrt(alpha) * (beta / abs(beta))

def kurtosis(alpha, beta=1, delta=0):
    return 6 / alpha

def fit(data):
    if sc.stats.skew(data)>0:
       p=gamma.fit(data)
       return p[0],p[2],p[1]
    else:
        data = data*-1
        p = gamma.fit(data)
        return p[0], -p[2], -p[1]


def params(low, mode, high, low_prob=0.1):
    """Return the parameters alpha, beta, delta of the Expanded Gamma distribution from a three-point-estimate.

       Parameters
       ----------
       low : float
           The low or optimistic value of the three-point-estimate. Must be less than or equal to mode, and less than high

       mode : float
           The mode or most-likely value of the three-point-estimate. Must be greater than or equal to modem and be less
           than or equal to high

       high : float
           The high or pessimistic value of the three-point-estimate. Must be greater than or equal to mode, and greater
            than low

       low_prob : float, optional
            the probability of the low estimate. The probability of the high estimate is set as 1-low_prob

       Raises
       ------
       ValueError
           If the three-point-estimate is invalid: low>mode, mode>high, or low==high.
       """

    if low > mode or high < mode or low == high:
        msg = 'Invalid three-point-estimate: '
        if low > mode:
            msg += "'low' must be less than or equal to 'mode'"
        if high < mode:
            msg += "'High' must be greater than or equal 'mode'"
        if low == high:
            msg += "'High' must be greater than low"
        raise ValueError(msg)

    if low == mode or high == mode:
        alpha = __find_alpha_at_mode_equals_probability(low_prob)
    else:
        alpha = __find_alpha(low, mode, high, low_prob)

    beta = (mode - low) / ((alpha - 1) - ppf(low_prob, alpha))
    if (high - mode < mode - low):
        beta *= -1

    delta = mode - (alpha - 1) * beta

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

