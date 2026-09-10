# Changelog

## 1.1.0

Corrections to the fitting procedure. **Results change for two cases**, so
figures produced with 1.0.2 should be recomputed if either applies.

### Fixed

- **`ppf` referenced an undefined name and raised `NameError` on every call.**
  Because the shape search calls `ppf`, `params` was unusable, as were
  `median` and anything built on them. This affected the state of `main` and
  so any install from source; the last release predates it.
- **`params` returned a scale of effectively zero when the mode equalled the
  low value.** At the endpoint shape the mode-to-low distance and its
  denominator are both exactly zero, and the zero-guard turned 0/0 into the
  smallest representable double. The scale is now taken from the full elicited
  span, which is never degenerate, covers both endpoints with one expression,
  and is better conditioned: on the first published test vector it reproduces
  the elicited values to 3.8e-13 of the range against 5.7e-11 for a
  mode-to-outer distance.
- **`params` ignored `low_prob` when placing the high percentile.** The high
  probability was fixed at 0.90 regardless of the value passed, so any
  non-default percentile convention produced a silently wrong fit. Fitting
  (100, 140, 300) with `low_prob=0.05` gave a 95th percentile of 347.2 instead
  of 300.
- **Both bisection loops could not terminate.** The loop condition never
  became false, so a fit that could not meet its tolerance ran forever;
  `params(100, 140, 300, low_prob=0.5)` hung. Both are now bounded, with an
  explicit check for the interval collapsing, and report failure rather than
  returning an unconverged value.
- **`fit` raised on every call**, because it referenced `scipy.special.stats`,
  which does not exist. `EgammaDistribution.from_fit` was unusable as a
  result.
- **`pdf` returned `nan` outside the support** rather than zero, which
  propagated through downstream aggregation.

### Changed

- The symmetry shortcut is now derived from the ratio the shape ceiling
  actually produces, rather than fixed at 0.99999. A fixed cut left a band of
  targets that passed the shortcut but could not be bracketed by the search;
  the band is empty at the default percentile convention but not at others.
- The endpoint search stops on the residual normalised by the percentile span,
  at a quarter of the threshold, so the reproduction bound covers endpoint
  cases as well as the general search. It previously compared values rounded
  to ten decimals.
- A single module-level `ALPHA_MAX` replaces three inconsistent ceilings
  (1e15 in the general search, 1e9 in its shortcut, 4e12 in the endpoint
  search).
- Inputs are validated: values must be finite and `0 < low_prob < 0.5`.
  Invalid input raises `ValueError`; a fit that cannot converge raises
  `RuntimeError`.

### Added

- `fit(data, method='mom')` for the method of moments, alongside the existing
  maximum likelihood fit, which remains the default. This is the estimator the
  companion Excel library uses, so results from the two now agree.
- A conformance test suite in `tests/`, covering the published test vectors,
  the reproduction bound across the admissible range, the mirror-image
  property of left- and right-skewed fits, input rejection, and agreement
  between the distribution functions.
- `test_vectors.csv`, the published vectors at full double precision, shipped
  with the package.
- `__version__`.

### Documentation

- The claim that the distribution had "only recently been formally defined"
  is removed. The family is the Pearson Type III distribution and is long
  established; what the library provides is that family in the shape, scale
  and location parameterisation, together with the three-point fit.
- The claim that `fit` offered the method of moments is now true.
- `kurtosis` is documented as excess kurtosis.
- The mathematical documentation gains the derivation of the three-point fit,
  the admissible range of estimates, and the accuracy bound.

## 1.0.2

Earlier releases.
