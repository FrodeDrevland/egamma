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
#: parameter tends to infinity, so a finite stand-in is needed. At this value,
#: and at the default 10th/90th percentile convention, the elicited values are
#: reproduced to about 1.5e-5 of the elicited range. That figure is specific to
#: the convention: the ceiling error grows as the elicited percentiles approach
#: the median, reaching about 4.2e-2 at a low probability of 0.4999.
#:
#: Raising the ceiling does reduce the error at a symmetric estimate, but
#: recovery of the shape parameter becomes increasingly ill-conditioned as an
#: estimate approaches symmetry. The ceiling is therefore an implementation
#: choice rather than a property of the method, and should be chosen together
#: with the numerical functions and precision the implementation uses.
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

#: Default approximation tolerance for a fit that reaches the shape ceiling.
#:
#: A ceiling fit is accepted only if its position residual ``|t_max - t|``,
#: which is the normalised reproduction error in exact arithmetic, is at most
#: this. It is deliberately separate from :data:`TOLERANCE`: the ceiling error
#: is set by ``alpha_max`` and the percentile convention, not by the search,
#: and at ``alpha_max = 1e9`` it is about 1.5e-5 at P_L = 0.10. All five
#: supported conventions (P_L = 0.01, 0.05, 0.10, 0.20, 0.25) meet this
#: default for a symmetric estimate; a convention near the median, such as
#: P_L = 0.4999, does not, and its symmetric estimate is refused.
CEILING_TOLERANCE = 2e-5

#: Factor in the estimate of rounding uncertainty in the reconstruction.
#:
#: The fit is judged on its position residual, which involves no location or
#: scale and is not degraded by large values. The low, mode and high values are
#: then reconstructed from the fitted parameters as a check on the floating-point
#: computation, and that check carries rounding uncertainty of its own. It is
#: estimated as ``ROUNDING_FACTOR * EPS * m / (high - low)``, where ``m`` is the
#: largest magnitude among the elicited values, the location and the scaled
#: quantiles. A first-order count of the roughly eight rounded operations in the
#: reconstruction, each contributing at most half of ``EPS`` times a magnitude
#: no larger than ``m``, gives a factor of about 4. That is an estimate, not a
#: proven bound: it ignores higher-order terms and assumes IEEE double
#: arithmetic. Empirically, across about 38,000 ordinary fits the largest
#: discrepancy observed was 0.97 times ``EPS * m / (high - low)``.
ROUNDING_FACTOR = 4

#: Machine epsilon. Retained as a public name for code written against
#: earlier versions; it no longer bounds the percentiles :func:`ppf` accepts,
#: which are tested against their own complements instead.
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
    :raises ValueError: If the percentile does not lie strictly between 0 and 1,
        or lies so close to one of them that its complement no longer remains
        distinct from that endpoint in double precision.

    .. note::
       Up to version 1.2.0 a percentile was silently moved into
       ``[eps, 1 - eps]``, so a request for the 1e-20 quantile returned the
       quantile at machine epsilon instead and the distribution evaluated was
       not the one asked for. Such a percentile is now rejected: the library
       either computes what was asked for or says it cannot. Every percentile
       convention used in practice, and all those tabulated in the
       accompanying paper, are unaffected.
    """
    if beta == 0:
        return np.nan
    p = float(percentile)
    if not 0.0 < p < 1.0:
        raise ValueError(
            'percentile must lie strictly between 0 and 1, not %r' % (percentile,))
    # The reflected branch evaluates the complement, so both p and 1 - p must be
    # distinguishable from their endpoints for the two skew directions to agree.
    if 1.0 - p == 1.0 or 1.0 - p == 0.0:
        raise ValueError(
            'percentile %r is too close to 0 or 1 for its complement to remain '
            'distinct from the endpoint in double precision, so the reflected '
            'branch could not evaluate the quantile the caller asked for'
            % (percentile,))
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

    For alpha > 1 the density has an interior maximum at
    ``(alpha - 1) * beta + delta``. For 0 < alpha <= 1 it is monotone on its
    support and the mode lies at the support boundary, delta; the interior
    expression would place it outside the support entirely.

    A three-point fit never produces a shape at or below 1 — the smallest
    admissible shape is about 1.156 at the default percentile convention — so
    this distinction does not arise for parameters obtained from
    :func:`params`. It arises for a distribution constructed directly.

    :param alpha: The shape parameter of the expanded gamma distribution.
    :type alpha: float
    :param beta: The scale parameter of the expanded gamma distribution, defaults to 1.
    :type beta: float, optional
    :param delta: The location parameter of the expanded gamma distribution, defaults to 0.
    :type delta: float, optional
    :returns: The mode of the expanded gamma distribution, or nan if alpha is
        not positive, since the distribution is then undefined.
    :rtype: float
    """
    if not alpha > 0:
        return np.nan
    if alpha <= 1:
        return delta
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

    A sample skewness small enough to imply a shape above :data:`ALPHA_MAX`
    is capped there, and the scale is then taken as
    :math:`|\beta| = s / \sqrt{\alpha}` with the sign of :math:`g`. Both forms
    agree where the shape is not capped; the second is what preserves the
    sample variance once it is. The sample mean is preserved either way,
    because the location is computed last. The skewness is what a cap gives up.

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
        s = np.std(data, ddof=1)
        alpha = 4 / g ** 2
        if alpha > ALPHA_MAX:
            # Capping the shape without recomputing the scale would keep the
            # scale that belongs to the uncapped shape, so the fitted variance
            # alpha * beta ** 2 would fall short of the sample variance by the
            # factor by which the shape was reduced. Taking |beta| = s /
            # sqrt(alpha) from the capped shape preserves the sample variance;
            # delta below then preserves the mean. The skewness is what the cap
            # gives up, which is unavoidable once a ceiling is imposed.
            alpha = ALPHA_MAX
            beta = np.copysign(s / np.sqrt(alpha), g)
        else:
            beta = s * g / 2
        delta = np.mean(data) - alpha * beta
        return alpha, beta, delta

    raise ValueError("method must be 'mle' or 'mom', not %r" % (method,))


class FitError(RuntimeError):
    """A three-point fit that could not be completed to the requested accuracy.

    Raised when the shape search does not converge, when a numerical step
    produces a non-finite or invalid value, when a ceiling fit's position
    residual exceeds the ceiling tolerance, when the reconstructed values
    disagree with the fit by more than the rounding uncertainty can explain, or
    when non-negative support was required and the fitted distribution extends
    below zero. It subclasses :class:`RuntimeError`, which is what ``params``
    raised for non-convergence in earlier versions, so existing handlers still
    catch it.

    Where they were computed, the diagnostics are attached as
    ``position_error``, ``max_error`` and ``verification_uncertainty``;
    otherwise they are None.
    """

    def __init__(self, message, position_error=None, max_error=None,
                 verification_uncertainty=None):
        super().__init__(message)
        self.position_error = position_error
        self.max_error = max_error
        self.verification_uncertainty = verification_uncertainty


class PrecisionError(FitError):
    """The fit cannot be verified to the requested accuracy in double precision.

    The shape search may have succeeded, but the rounding uncertainty of the
    reconstruction check, ``verification_uncertainty``, exceeds the tolerance
    being verified, so a passing check would not establish the accuracy asked
    for. This typically means the elicited values are very large relative to
    their range: at the default tolerance, more than about 28,000 times it.
    Either request a tolerance of at least ``verification_uncertainty``, or fit
    the estimate in shifted units (for example after subtracting ``low``) and
    shift the location back.
    """


class FitResult(tuple):
    """The fitted ``(alpha, beta, delta)``, with diagnostics attached.

    This is a plain 3-tuple, so ``alpha, beta, delta = params(...)`` and
    ``params(...)[0]`` behave exactly as they did in earlier versions. The
    extra attributes are there because returning the shape ceiling silently is
    unhelpful: a caller that wants to know whether it received a fitted shape or
    the ceiling approximation can ask.

    :ivar bool ceiling: True when the estimate was too near symmetry for the
        shape ceiling to resolve, so ``alpha_max`` was returned. Its position
        residual was then accepted against the ceiling tolerance rather than
        the ordinary one.
    :ivar str status: ``'converged'`` for an ordinary fit, ``'approximation'``
        for a ceiling fit.
    :ivar float position_error: ``|t_alpha - t|`` at the returned shape, the
        maximum normalised error in the reproduced values in exact arithmetic.
        This is the quantity the fit is accepted on.
    :ivar float max_error: The largest absolute difference between the elicited
        and recovered low, mode and high values, divided by the elicited range,
        as actually computed in floating point. A diagnostic of the computation,
        not the acceptance criterion.
    :ivar float verification_uncertainty: The estimated rounding uncertainty of
        ``max_error``, ``ROUNDING_FACTOR * EPS * m / (high - low)``. An
        accepted result has this no larger than the tolerance applied, and
        ``max_error`` within that tolerance plus this.
    :ivar bool support_below_zero: True when the fitted distribution has support
        below zero: always for a negative scale, whose support is unbounded
        below, and for a positive scale with a negative location. A converged
        fit can still be unsuitable for a quantity that cannot be negative.
    :ivar int iterations: Bisection steps used; zero when the ceiling was taken.
    """

    def __new__(cls, alpha, beta, delta, ceiling=False, position_error=0.0,
                iterations=0, max_error=0.0, support_below_zero=None,
                verification_uncertainty=0.0):
        self = super().__new__(cls, (alpha, beta, delta))
        self.ceiling = ceiling
        self.status = 'approximation' if ceiling else 'converged'
        self.position_error = position_error
        self.max_error = max_error
        self.verification_uncertainty = verification_uncertainty
        self.support_below_zero = bool(
            (beta < 0 or delta < 0) if support_below_zero is None
            else support_below_zero)
        self.iterations = iterations
        return self

    def __repr__(self):
        return ('FitResult(alpha=%r, beta=%r, delta=%r, status=%r, '
                'position_error=%r, max_error=%r, support_below_zero=%r)'
                % (self[0], self[1], self[2], self.status, self.position_error,
                   self.max_error, self.support_below_zero))


def _acceptance_failure(position_error, max_error, uncertainty, acceptance,
                        strict):
    """Apply the three acceptance rules in order; return the first one failed.

    1. ``fit``: the position residual must be below ``acceptance`` (``strict``)
       or at most it. This is the accuracy claim, in exact arithmetic, and no
       rounding allowance is added to it.
    2. ``precision``: the reconstruction's rounding uncertainty must be no
       larger than ``acceptance``, or the check below cannot establish the
       accuracy asked for.
    3. ``consistency``: the reconstructed error must lie within ``acceptance``
       plus that uncertainty, confirming that the floating-point computation
       agrees with the fit.

    :returns: ``None`` if all three hold, else ``'fit'``, ``'precision'`` or
        ``'consistency'``.
    """
    if not np.isfinite(position_error) or (
            position_error >= acceptance if strict
            else position_error > acceptance):
        return 'fit'
    if not np.isfinite(uncertainty) or uncertainty > acceptance:
        return 'precision'
    if not np.isfinite(max_error) or max_error > acceptance + uncertainty:
        return 'consistency'
    return None


def params(low, most_likely, high, low_prob=0.1, tolerance=TOLERANCE,
           alpha_max=ALPHA_MAX, max_iter=MAX_ITER,
           ceiling_tolerance=CEILING_TOLERANCE, require_nonnegative=False):
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

    A result is accepted only if all of the following hold, where the
    tolerance applied is ``tolerance`` for an ordinary fit and
    ``ceiling_tolerance`` for a fit at the shape ceiling:

    1. The position residual ``|t_alpha - t|``, which is the normalised
       reproduction error in exact arithmetic, is within the tolerance. No
       allowance is added to it.
    2. The estimated rounding uncertainty of the reconstruction,
       ``verification_uncertainty``, is no larger than the tolerance.
       Otherwise :class:`PrecisionError` is raised: the accuracy asked for
       cannot be verified in double precision.
    3. The low, mode and high values reconstructed from the parameters agree
       with those elicited to within the tolerance plus that uncertainty.

    The tolerance is never relaxed silently.

    The returned object is a 3-tuple of ``(alpha, beta, delta)`` and unpacks as
    one. It additionally carries ``status``, ``ceiling``, ``position_error``,
    ``max_error``, ``verification_uncertainty``, ``support_below_zero`` and
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
    :param float ceiling_tolerance: Approximation tolerance for a fit that
        reaches the shape ceiling, as a fraction of the elicited range.
        Defaults to :data:`CEILING_TOLERANCE`.
    :param bool require_nonnegative: If True, a fit whose distribution has
        support below zero raises :class:`FitError` instead of being returned
        with ``support_below_zero`` set. Defaults to False.

    :return: The shape (alpha), scale (beta) and location (delta) parameters.
    :rtype: FitResult

    :raises ValueError: If the estimates are not finite, if low_prob is outside (0, 0.5),
        if a numerical control is out of range, or if the provided three-point estimate
        does not form a valid range.
    :raises PrecisionError: If the requested accuracy cannot be verified in
        double precision; see rule 2 above.
    :raises FitError: If the shape search does not converge, if a numerical step
        fails, if a ceiling fit misses ``ceiling_tolerance``, if the
        reconstruction disagrees with the fit, or if ``require_nonnegative`` is
        set and the support extends below zero.
    """
    for name, value in (('low', low), ('most_likely', most_likely), ('high', high)):
        if not np.isfinite(value):
            raise ValueError("Invalid three-point-estimate: '%s' must be finite" % name)

    if not 0 < low_prob < 0.5:
        raise ValueError('Invalid low_prob: must satisfy 0 < low_prob < 0.5')

    # The fit places the high value at 1 - low_prob. Below about one ulp of 1
    # that complement rounds to 1 exactly, whose quantile is infinite, so the
    # convention asked for is not the one that would be fitted.
    if 1.0 - low_prob == 1.0:
        raise ValueError(
            'Invalid low_prob: %r is too small for its complement 1 - low_prob '
            'to be distinguishable from 1 in double precision, so the high '
            'percentile could not be placed as requested' % (low_prob,))

    if not (np.isfinite(tolerance) and 0 < tolerance < 0.5):
        raise ValueError('Invalid tolerance: must satisfy 0 < tolerance < 0.5')

    if not (np.isfinite(ceiling_tolerance) and 0 < ceiling_tolerance < 0.5):
        raise ValueError(
            'Invalid ceiling_tolerance: must satisfy 0 < ceiling_tolerance < 0.5')

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

    span = high - low
    if not np.isfinite(span):
        raise ValueError(
            'Invalid three-point-estimate: high - low overflows double precision')

    alpha, iterations, at_ceiling, position_error = __find_alpha(
        low, most_likely, high, low_prob, tolerance, alpha_max, max_iter)

    if alpha is None:
        raise FitError(
            'Fit did not converge: the requested tolerance of %g could not be '
            'met within %d iterations. Try a looser tolerance, or check that '
            'the arithmetic can represent the elicited values and their '
            'differences.' % (tolerance, max_iter))

    q_low = float(gammaincinv(alpha, low_prob))
    q_high = float(gammaincinv(alpha, 1 - low_prob))
    if not (np.isfinite(q_low) and np.isfinite(q_high)):
        raise FitError('Gamma quantile evaluation failed at alpha=%r' % alpha)
    quantile_span = q_high - q_low
    if not (np.isfinite(quantile_span) and quantile_span > 0):
        raise FitError('Invalid Gamma quantile span at alpha=%r' % alpha)

    beta = span / quantile_span
    if (most_likely - low) > (high - most_likely):
        beta = -beta
    delta = most_likely - (alpha - 1) * beta

    if not (np.isfinite(alpha) and np.isfinite(beta) and np.isfinite(delta)
            and alpha > 1 and beta != 0):
        raise FitError('Invalid fitted parameters (%r, %r, %r)'
                       % (alpha, beta, delta))

    # Recover the elicited values from the fitted parameters. Under reflection
    # the lower elicited value comes from the upper standard quantile.
    if beta > 0:
        rec_low, rec_high = delta + beta * q_low, delta + beta * q_high
    else:
        rec_low, rec_high = delta + beta * q_high, delta + beta * q_low
    rec_mode = delta + (alpha - 1) * beta
    if not (np.isfinite(rec_low) and np.isfinite(rec_mode)
            and np.isfinite(rec_high)):
        raise FitError('Reconstruction of the elicited values failed')
    if rec_low >= rec_high:
        raise FitError('Reconstructed low and high values are not ordered')

    max_error = max(abs(rec_low - low), abs(rec_mode - most_likely),
                    abs(rec_high - high)) / span

    # Estimated rounding uncertainty of max_error. The magnitude includes the
    # location and the scaled quantiles, not only the elicited values, because
    # at large shapes delta and (alpha - 1) * beta can be far larger than the
    # elicited values and nearly cancel.
    magnitude = max(abs(low), abs(most_likely), abs(high), abs(delta),
                    abs(beta * q_low), abs(beta * q_high),
                    abs((alpha - 1) * beta))
    uncertainty = ROUNDING_FACTOR * EPS * magnitude / span

    acceptance = ceiling_tolerance if at_ceiling else tolerance
    failure = _acceptance_failure(position_error, max_error, uncertainty,
                                  acceptance, strict=not at_ceiling)
    diagnostics = dict(position_error=position_error, max_error=max_error,
                       verification_uncertainty=uncertainty)
    if failure == 'fit':
        if at_ceiling:
            raise FitError(
                'The estimate is too near symmetry for alpha_max=%g to resolve '
                'within ceiling_tolerance=%g: the elicited values would be '
                'reproduced only to %.3g of the range at low_prob=%g. Raise '
                'ceiling_tolerance if that is acceptable, or use a Normal '
                'distribution for a symmetric estimate.'
                % (alpha_max, ceiling_tolerance, position_error, low_prob),
                **diagnostics)
        raise FitError('The position residual %.3g does not meet the '
                       'tolerance of %g.' % (position_error, tolerance),
                       **diagnostics)
    if failure == 'precision':
        raise PrecisionError(
            'The fit cannot be verified to %g of the range in double precision: '
            'the reconstruction carries an estimated rounding uncertainty of '
            '%.3g of the range, because the values involved (up to %.3g) are '
            'large relative to the range (%.3g). Request a tolerance of at '
            'least %.3g, or fit in shifted units, for example after '
            'subtracting low.'
            % (acceptance, uncertainty, magnitude, span, uncertainty),
            **diagnostics)
    if failure == 'consistency':
        raise FitError(
            'The reconstructed values disagree with the fit: maximum error '
            '%.3g of the range, against a tolerance of %g and an estimated '
            'rounding uncertainty of %.3g.' % (max_error, acceptance, uncertainty),
            **diagnostics)

    support_below_zero = bool(beta < 0 or delta < 0)
    if require_nonnegative and support_below_zero:
        raise FitError(
            'The fitted distribution has support below zero (beta=%r, '
            'delta=%r), and non-negative support was required.' % (beta, delta))

    return FitResult(alpha, beta, delta, at_ceiling, position_error, iterations,
                     max_error, support_below_zero, uncertainty)


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
