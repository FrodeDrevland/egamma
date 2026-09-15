# Changelog

## 1.2.0

The fitting procedure now searches on the mode's **position** within the elicited
range rather than on the half-range ratio, and stops on an absolute tolerance
rather than a relative one. **Fitted parameters change** in the last few digits
for most inputs, and estimates with a mode very close to an outer value are now
fitted where 1.1.1 could refuse them. Results produced with 1.1.1 should be
recomputed if either matters.

Only the fitting routine changes. The distribution functions are untouched, and
so are the moments: `mean` remains `alpha * beta + delta` and `std` remains a
function of `alpha` and `beta` alone, as they must be for a distribution
instantiated directly from parameters rather than fitted.

### Changed

- **The shape search matches mode position.** The target is
  `t = min(mode - low, high - mode) / (high - low)` and the model quantity is
  `t_alpha = (alpha - 1 - q_low) / (q_high - q_low)`. The search stops when
  `|t_alpha - t| < tolerance`. That difference is itself the maximum normalised
  error in the three reproduced values, so the tolerance is now stated directly
  in the quantity users care about rather than implying it through a bound.
- **`TOLERANCE = 2.5e-11` replaces `THRESHOLD = 1e-10`.** This is not a rename:
  the old value bounded `|r_hat/r - 1|` on the half-range ratio, the new one
  bounds `|t_alpha - t|` directly. 2.5e-11 is the reproduction accuracy the old
  relative threshold implied, so the intended accuracy is unchanged. `THRESHOLD`
  remains as a deprecated alias.
- **`params` accepts `tolerance`, `alpha_max` and `max_iter`** and validates
  them: the tolerance must lie in (0, 1/2), the ceiling must exceed 1, and the
  iteration limit must be a positive integer.

### Fixed

- **A mode very close to an outer value is no longer refused.** The relative
  stopping rule demanded an absolute agreement of `threshold * r`, which tends to
  zero with the ratio, so an estimate such as (100, 100.00001, 300) could exhaust
  the search and report failure although it is admissible. The absolute rule
  imposes the same reproduction requirement near an outer value as at it.
- **The shape ceiling is no longer returned silently.** `params` returns a
  `FitResult`, which is still a plain `(alpha, beta, delta)` tuple and unpacks as
  one, but also carries `ceiling`, `position_error` and `iterations`. A caller can
  now distinguish a fit that met the tolerance from the ceiling approximation
  returned for an estimate too near symmetry to resolve.

### Removed

- **The separate endpoint solver.** A mode at an outer value gives
  `target_position = 0` and goes through the ordinary search, so
  `__find_alpha_at_mode_equals_probability` is gone. There is no longer a
  different code path, or a different tolerance, for the endpoint cases.

### Conformance vectors

`test_vectors.csv` is regenerated and now carries ten cases rather than eight,
adding a mode immediately inside each outer value, and a `ceiling` column. These
supersede the 1.1.1 vectors.

## 1.1.1

Packaging only. The library, its behaviour and its results are unchanged from
1.1.0; the two are identical as installed.

### Fixed

- **Generated Sphinx output was tracked in the repository, and so was included
  in the release archive.** `docs/build/` was listed in `.gitignore`, but had
  been committed before that rule was added, so it stayed tracked and
  `git archive` — which produces the GitHub release archive, and therefore the
  Zenodo deposit — carried it. The archive held a 282 kB pickled Sphinx
  environment, seven doctrees, and a stale, partially generated HTML site that
  no longer matched `docs/source/`. The directory is no longer tracked, and a
  `.gitattributes` marks it `export-ignore` so that build products stay out of
  release archives if it is ever committed again.

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
