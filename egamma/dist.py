import egamma.functions as core

class EgammaDistribution:
    """
    A class representing the expanded gamma distribution.

    :param alpha: The shape parameter of the expanded gamma distribution.
    :type alpha: float
    :param beta: The scale parameter of the expanded gamma distribution.
    :type beta: float
    :param delta: The location parameter of the expanded gamma distribution.
    :type delta: float
    """

    def __init__(self, alpha, beta, delta):
        """
        Initialize the EgammaDistribution instance.

        :param alpha: The shape parameter of the expanded gamma distribution.
        :type alpha: float
        :param beta: The scale parameter of the expanded gamma distribution.
        :type beta: float
        :param delta: The location parameter of the expanded gamma distribution.
        :type delta: float
        """
        self.alpha = alpha
        self.beta = beta
        self.delta = delta

    @classmethod
    def from_tpe(cls, low, most_likely, high, low_prob=0.1):
        """
        Create an EgammaDistribution instance from a three-point estimate.

        :param low: The low estimate of the distribution.
        :type low: float
        :param most_likely: The most likely estimate of the distribution.
        :type most_likely: float
        :param high: The high estimate of the distribution.
        :type high: float
        :param low_prob: The probability associated with the low estimate, defaults to 0.1.
        :type low_prob: float, optional
        :returns: An instance of EgammaDistribution.
        :rtype: EgammaDistribution
        """
        a, b, d = core.params(low, most_likely, high, low_prob)
        return cls(a, b, d)

    @classmethod
    def from_fit(cls, data):
        """
        Create an EgammaDistribution instance by fitting to data.

        :param data: The data to fit.
        :type data: array_like
        :returns: An instance of EgammaDistribution.
        :rtype: EgammaDistribution
        """
        a, b, d = core.fit(data)
        return cls(a, b, d)

    def pdf(self, x):
        """
        Calculate the probability density function for the expanded gamma distribution.

        :param x: The point at which the pdf is evaluated.
        :type x: float
        :returns: The probability density function evaluated at x.
        :rtype: float
        """
        return core.pdf(x, self.alpha, self.beta, self.delta)

    def cdf(self, x):
        """
        Calculate the cumulative distribution function for the expanded gamma distribution.

        :param x: The point at which the cdf is evaluated.
        :type x: float
        :returns: The cumulative distribution function evaluated at x.
        :rtype: float
        """
        return core.cdf(x, self.alpha, self.beta, self.delta)

    def ppf(self, probability):
        """
        Calculate the percent point function (inverse of cdf) at a given probability for the expanded gamma distribution.

        :param probability: The probability at which to evaluate.
        :type probability: float
        :returns: The value of the distribution at the given probability.
        :rtype: float
        """
        return core.ppf(probability, self.alpha, self.beta, self.delta)

    def rvs(self, size=1, random_state=None):
        """
        Generate random variates of the expanded gamma distribution.

        :param size: The number of random variates to generate, defaults to 1.
        :type size: int, optional
        :param random_state: If int or RandomState, use it for drawing the random variates. If None, rely on self.random_state, defaults to None.
        :type random_state: int, RandomState instance or None, optional
        :returns: Random variates of the expanded gamma distribution.
        :rtype: ndarray or scalar
        """
        return core.rvs(self.alpha, self.beta, self.delta, size=size, random_state=random_state)

    def mean(self):
        """
        Calculate the mean of the expanded gamma distribution.

        :returns: The mean of the expanded gamma distribution.
        :rtype: float
        """
        return core.mean(self.alpha, self.beta, self.delta)

    def mode(self):
        """
        Calculate the mode of the expanded gamma distribution.

        :returns: The mode of the expanded gamma distribution.
        :rtype: float
        """
        return core.mode(self.alpha, self.beta, self.delta)

    def median(self):
        """
        Calculate the median of the expanded gamma distribution.

        :returns: The median of the expanded gamma distribution.
        :rtype: float
        """
        return core.median(self.alpha, self.beta, self.delta)

    def var(self):
        """
        Calculate the variance of the expanded gamma distribution.

        :returns: The variance of the expanded gamma distribution.
        :rtype: float
        """
        return core.var(self.alpha, self.beta, self.delta)

    def std(self):
        """
        Calculate the standard deviation of the expanded gamma distribution.

        :returns: The standard deviation of the expanded gamma distribution.
        :rtype: float
        """
        return core.std(self.alpha, self.beta, self.delta)

    def skew(self):
        """
        Calculate the skewness of the expanded gamma distribution.

        :returns: The skewness of the expanded gamma distribution.
        :rtype: float
        """
        return core.skew(self.alpha, self.beta, self.delta)

    def kurtosis(self):
        """
        Calculate the kurtosis of the expanded gamma distribution.

        :returns: The kurtosis of the expanded gamma distribution.
        :rtype: float
        """
        return core.kurtosis(self.alpha, self.beta, self.delta)

    def _get_params(self):
        """
        Get the parameters of the expanded gamma distribution.

        :returns: The alpha, beta, and delta parameters.
        :rtype: tuple
        """
        return self.alpha, self.beta, self.delta

    def _set_params(self, value):
        """
        Set the parameters of the expanded gamma distribution.

        :param value: A tuple containing the alpha, beta, and delta parameters.
        :type value: tuple
        """
        self.alpha, self.beta, self.delta = value

    params = property(fget=_get_params, fset=_set_params, doc="Gets or sets the alpha, beta, and delta parameters.")
