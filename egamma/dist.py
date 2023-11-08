import egamma.functions as core

class EgammaDistribution:

    def __init__(self, alpha, beta, delta):
        self.alpha = alpha
        self.beta = beta
        self.delta = delta



    @classmethod
    def from_tpe(cls, low, most_likely, high, low_prob=0.1):
        a,b,d = core.params(low, most_likely, high, low_prob)
        return cls(a,b,d)

    @classmethod
    def from_fit(cls,data):
        a,b,d= core.fit(data)
        return cls(a, b, d)

    # Probabilty density function
    def pdf(self, x, alpha, beta=1.0, delta=0.0):
        return core.pdf(self.alpha, self.beta, self.delta)

    # Cumulative distribution function
    def cdf(self, x, probability):
        return core.ppf(x, self.alpha, self.beta, self.delta)

    # Percent point function (Inverse CDF)
    def ppf(self, probability):
        return core.ppf(probability, self.alpha, self.beta, self.delta)

    # Random variates
    def rvs(self, size=1, random_state=None):
        return core.rvs(self.alpha, self.beta, self.delta, size=size, random_state=random_state)

    def mean(self):
        return core.mean(self.alpha, self.beta, self.delta)

    def mode(self):
        return core.mode(self.alpha, self.beta, self.delta)

    # Median of the distribution
    def median(self):
        return core.median(self.alpha, self.beta, self.delta)

    def var(self):
        return core.var(self.alpha, self.beta, self.delta)

    def std(self):
        return core.std(self.alpha, self.beta, self.delta)

    def skew(self):
        return core.skew(self.alpha, self.beta, self.delta)

    def kurtosis(self):
        return core.kurtosis(self.alpha, self.beta, self.delta)

    def kurtosis(alpha, beta=1, delta=0):
        return 6 / alpha


    def _get_params(self):
        return self.alpha, self.beta, self.delta

    def _set_params(self, value):
        self.alpha = value[0]
        self.beta = value[1]
        self.delta = value[2]

    params = property(fget=_get_params, fset=_set_params, doc="Gets or sets the alpha, beta and delta parameters")

