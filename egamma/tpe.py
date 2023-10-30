import egamma.core as egamma

def pdf(x, low, most_likely, high, low_prob=0.1):
    a,b,d = egamma.params(low,most_likely,high,low_prob)
    return egamma.pdf(x, a, b, d)

def cdf(probability, low, most_likely, high, low_prob=0.1):
    a,b,d = egamma.params(low,most_likely,high,low_prob)
    return egamma.cdf(probability, a, b, d)
def ppf(probability, low, most_likely, high, low_prob=0.1):
    a,b,d = egamma.params(low,most_likely,high,low_prob)
    return egamma.ppf(probability, a, b, d)

def rvs(low, most_likely, high, low_prob=0.1, size=1, random_state=None):
    a,b,d = egamma.params(low,most_likely,high,low_prob)
    return egamma.rvs(a,b,d)

def mean(low, most_likely, high, low_prob=0.1):
    a,b,d = egamma.params(low,most_likely,high,low_prob)
    return egamma.mean(a, b, d)

def mode(low, most_likely, high, low_prob=0.1):
    a,b,d = egamma.params(low,most_likely,high,low_prob)
    return egamma.mode(a, b, d)


# Median of the distribution
def median(low, most_likely, high, low_prob=0.1):
    a,b,d = egamma.params(low,most_likely,high,low_prob)
    return egamma.median(a, b, d)


# Variance of the distribution
def var(low, most_likely, high, low_prob=0.1):
    a,b,d = egamma.params(low,most_likely,high,low_prob)
    return egamma.var(a, b, d)

# Standard deviation of the distribution.
def std(low, most_likely, high, low_prob=0.1):
    a,b,d = egamma.params(low,most_likely,high,low_prob)
    return egamma.std(a, b, d)


# Skew deviation of the distribution.
def skew(low, most_likely, high, low_prob=0.1):
    a,b,d = egamma.params(low,most_likely,high,low_prob)
    return egamma.skew(a, b, d)

def kurtosis(low, most_likely, high, low_prob=0.1):
    a,b,d = egamma.params(low,most_likely,high,low_prob)
    return egamma.kurtosis(a,b,d)



