import numpy as np
from egamma.core import ppf


def alpha(low, mode, high, low_prob=0.1):
    """Return the alpha parameter of the Expanded Gamma distribution from a three-point-estimate.

    Parameters
    ----------
    low : float
        The low or optimistic value of the three-point-estimate. Must be less than or equal to mode, and less than high

    mode : float
        The mode or most-likely value of the three-point-estimate. Must be greater than or equal to modem and be less t
        han or equal to high

    high : float
        The high or pessimistic value of the three-point-estimate. Must be greater than or equal to mode, and greater
         than low

    low_prob : float, optional
         the probability of the low estimate. The probability of the high estimate is set as 1-low_prob

    Raises
    ------
    NotImplementedError
        If no sound is set for the animal or passed in as a
        param
    """

    high_prob = 1 - low_prob

    # Check for invalid three-point estimate
    if (low == mode and high == mode) or low > mode or high < mode:
        return np.NaN
    elif low == mode or high == mode:
        return __find_alpha_at_mode_equals_probability(low_prob)
    else:
        return __find_alpha(low, mode, high, low_prob)



def beta(low, mode, high, low_prob, p_alpha: float = None):
    high_prob = 1 - low_prob

    if p_alpha is None or p_alpha < 1:
        p_alpha = alpha(low, mode, high, low_prob)

    _beta=(mode-low)/((p_alpha-1)- ppf(low_prob, p_alpha))

    if(high-mode>mode-low):
        return _beta
    else:
        return -_beta



def delta(low, mode, high, low_prob, p_alpha=None, p_beta=None):
    if p_alpha is None or p_alpha < 1:
        p_alpha = alpha(low, mode, high, low_prob)

    if p_beta is None or p_beta == 0:
        p_beta = beta(low, mode, high, low_prob, alpha)

    return mode - (p_alpha - 1) * p_beta


def params(low, mode, high, low_prob=0.1):
    """Return the parameters of the Expanded Gamma distribution from a three-point-estimate.

       Parameters
       ----------
       low : float
           The low or optimistic value of the three-point-estimate. Must be less than or equal to mode, and less than high

       mode : float
           The mode or most-likely value of the three-point-estimate. Must be greater than or equal to modem and be less t
           han or equal to high

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

    if low == high:
        msg = 'Invalid three-point-estimate: '
        msg += "'High' must be greater than low"
        raise ValueError(msg)

    _alpha = alpha(low, mode, high, low_prob)
    _beta = beta(low, mode, high, low_prob, _alpha)
    _delta = delta(low, mode, high, low_prob, _alpha, _beta)

    return _alpha, _beta, _delta





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


