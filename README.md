# `egamma` — gamma distributions from three-point estimates

Turn an expert's **low / most likely / high** estimate into a probability
distribution you can simulate from — including the left-skewed estimates that
an ordinary gamma distribution cannot represent.

```python
import egamma

dist = egamma.EgammaDistribution.from_tpe(low=100, most_likely=140, high=300)

dist.mean()        # 189.5
dist.ppf(0.5)      # median
dist.rvs(10000)    # a sample for a Monte Carlo run
```

The fitted distribution's mode is exactly 140, its 10th percentile exactly
100, and its 90th percentile exactly 300. That is the point: the numbers the
estimator gave you come back out.

## Is this the library you are looking for?

**Yes, if** you have three-point estimates — from PERT, from the Successive
Principle, or from any expert-judgement exercise — and you need actual
distribution parameters, because your Monte Carlo tool needs something to
sample from.

**Yes, if** some of your estimates lean the wrong way. When the most likely
value sits closer to the high end than the low end, the estimate is
left-skewed, and `scipy.stats.gamma` cannot fit it at all: a gamma
distribution is always right-skewed. Most tooling either silently ignores this
or makes you reverse your data by hand.

**Yes, if** you are fitting an ordinary right-skewed gamma too. That is the
common case and it works the same way.

**Probably not, if** you have a sample of observations rather than a
judgement. `scipy.stats.gamma.fit` is the right tool for that, though
`egamma.fit` will do it in either skew direction if you need that.

## Why not just use SciPy?

SciPy has the distribution but not the fit. `scipy.stats.gamma` will not take
a negative scale, so it cannot go left. `scipy.stats.pearson3` covers both
directions, but neither will fit to a mode plus two percentiles: the
elicitation packages that exist work from quantiles alone, and a mode is not a
quantile.

`egamma` builds on SciPy's gamma implementation and adds the missing piece.

## What the distribution is

The ordinary three-parameter gamma distribution is strictly right-skewed
because its scale parameter must be positive. Let the scale go negative and
the density reflects about the location parameter: the support runs downward
from it instead of upward, and the skewness turns negative.

That family is not new — it is the Pearson Type III distribution, long
established in hydrology and elsewhere. What `egamma` gives you is that family
in the shape, scale and location parameters you already use for the gamma
distribution, so the switch costs nothing.

| Parameter | Symbol | Meaning |
|---|---|---|
| shape    | `alpha` | positive; skewness is `2/sqrt(alpha)` |
| scale    | `beta`  | non-zero; **the sign sets the direction of skew** |
| location | `delta` | the bound: lower when `beta > 0`, upper when `beta < 0` |

## Installation

```bash
pip install egamma
```

## Usage

Every function works standalone or through an `EgammaDistribution` instance.

```python
import egamma

# standalone
probability = egamma.cdf(x=200, alpha=10, beta=40, delta=-100)

# or as an object
dist = egamma.EgammaDistribution(alpha=10, beta=40, delta=-100)
probability = dist.cdf(x=200)
```

### From a three-point estimate

```python
dist = egamma.EgammaDistribution.from_tpe(low=100, most_likely=200,
                                          high=400, low_prob=0.1)
```

`low_prob` is the probability attached to the low estimate. The high estimate
gets the complement, so the default of `0.1` treats the outer values as the
10th and 90th percentiles. Use `0.01` if your estimators were asked for the
1st and 99th. It must satisfy `0 < low_prob < 0.5`.

A left-skewed estimate needs nothing special — the position of the mode
decides:

```python
dist = egamma.EgammaDistribution.from_tpe(low=100, most_likely=300, high=400)
assert dist.skew() < 0
```

If you just want the parameters:

```python
alpha, beta, delta = egamma.params(low=100, most_likely=140, high=300)
```

### From data

```python
dist = egamma.EgammaDistribution.from_fit(data=data)

alpha, beta, delta = egamma.fit(data)                  # maximum likelihood
alpha, beta, delta = egamma.fit(data, method='mom')    # method of moments
```

Either way the skew direction is taken from the sample, so left- and
right-skewed data are both handled.

Maximum likelihood is the default and is generally the better estimator. The
method of moments is closed-form and needs no iteration, which makes it useful
as a starting point or where an optimiser is unavailable, but it rests on the
sample skewness, whose sampling variance is large; estimates from small samples
can be well off. It is the same estimator the companion Excel library uses, so
the two agree.

### What is available

- **Distribution functions**: `pdf`, `cdf`, `ppf`, `rvs`
- **Statistical measures**: `mean`, `mode`, `median`, `var`, `std`, `skew`,
  `kurtosis` (excess)
- **Fitting**: `params` from a three-point estimate, `fit` from data by
  maximum likelihood or the method of moments

## Things worth knowing

**Not every estimate can be fitted.** The fitted mode cannot fall outside the
interval your two percentiles enclose, which puts a limit on how lopsided an
estimate can be. At the default `low_prob=0.1` the skewness cannot exceed
about 1.86 in magnitude, which corresponds to a mode sitting right on top of
one of the outer values. Anything more extreme is rejected with a
`ValueError`. Eliciting a wider interval — 5th and 95th, say — admits more.

**A perfectly symmetric estimate is a special case.** If the mode sits exactly
midway, no finite gamma-shaped distribution matches it, because the family is
always skewed one way or the other. `egamma` returns a very large shape
parameter, which is close to a normal distribution; the elicited values then
come back to within about 15 parts per million of the range rather than
exactly. In practice this is far below the precision of any cost estimate.

**Otherwise the fit is essentially exact.** The elicited values are reproduced
to within about `2.5e-11` of the range. That is a guaranteed bound rather than
a best case.

**Errors are loud.** Invalid input raises `ValueError`; a fit that cannot
converge raises `RuntimeError`. Nothing silently returns a wrong answer.

## Testing

```bash
pip install pytest
pytest tests/
```

The suite runs the published conformance vectors, checks the accuracy bound
across the whole admissible range, and verifies that left- and right-skewed
fits are exact mirror images of each other.

## Documentation

https://frodedrevland.github.io/egamma/

The [mathematical definitions](https://frodedrevland.github.io/egamma/mathematical_definition.html)
page gives the density, the moments, and the derivation of the fitting
procedure.

## Support

Please open an issue on the
[GitHub repository issue tracker](https://github.com/FrodeDrevland/egamma/issues).

## Contact

- **Associate Professor Frode Drevland**
- **Affiliation**: Norwegian University of Science and Technology (NTNU)
- **Email**: [frode.drevland@ntnu.no](mailto:frode.drevland@ntnu.no)

Feedback, suggestions and contributions are welcome.

## License

MIT. See [LICENSE](https://github.com/FrodeDrevland/egamma/blob/main/LICENSE).
