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

