Mathematical Definitions
========================

Introduction
------------

This page is the reference for the mathematics behind the library: the density
and its cumulative distribution function, the statistical measures, and the
procedure by which the parameters are recovered from a three-point estimate.
Nothing here is needed to use the library; see :doc:`usage` for that.

Parameters
----------
The distribution is characterised by three parameters:

- :math:`\alpha` (alpha): The shape parameter, which must be positive (:math:`\alpha > 0`).
- :math:`\beta` (beta): The scale parameter, which must be non-zero (:math:`\beta \neq 0`). Its sign determines the direction of skewness: positive values give right skew, negative values left skew.
- :math:`\delta` (delta): The location parameter, which may be any real number. It is the bound of the support, lower when :math:`\beta > 0` and upper when :math:`\beta < 0`.

The support is :math:`[\delta, \infty)` for a positive scale and
:math:`(-\infty, \delta]` for a negative one.

The distribution is strictly skewed in one direction or the other; exact
symmetry is not attainable at any finite :math:`\alpha`. As :math:`\alpha`
grows the shape converges to that of a normal distribution, so a large value
such as :math:`10^9` approximates a symmetric case closely.

Distribution Functions
----------------------
The probability density function (PDF) and cumulative distribution function
(CDF) are defined as follows.

Probability Density Function (PDF):

.. math::
    f(x; \alpha, \beta, \delta) = \left\{
        \begin{array}{ll}
            \dfrac{|x - \delta|^{\alpha - 1} e^{-(x - \delta) / \beta}}{|\beta|^\alpha \Gamma(\alpha)}, & \text{for } \dfrac{x - \delta}{\beta} > 0 \\
            0, & \text{otherwise}
        \end{array}
    \right.

where :math:`\Gamma(\alpha)` is the gamma function :math:`\displaystyle\int_{0}^{\infty}t^{\alpha-1}e^{-t}\,dt`

Cumulative Density Function (CDF):

.. math::
    F(x; \alpha, \beta, \delta) = \left\{
        \begin{array}{ll}
            \dfrac{\gamma(\alpha, \dfrac{x - \delta}{\beta})}{\Gamma(\alpha)}, & \text{for } \beta > 0, \dfrac{x - \delta}{\beta} > 0 \\
            1 - \dfrac{\gamma(\alpha, \dfrac{x - \delta}{\beta})}{\Gamma(\alpha)}, & \text{for } \beta < 0, \dfrac{x - \delta}{\beta} > 0
        \end{array}
    \right.

where :math:`\gamma \left(\alpha, \frac{x-\delta}{\beta}\right)` is the lower incomplete gamma function :math:`\displaystyle\int_{0}^{\frac{x-\delta}{\beta}}t^{\alpha-1}e^{-t}\,dt`.

The CDF has no closed form, so the percent point function must be obtained
numerically. That single fact shapes the fitting procedure below.

Statistical measures
--------------------
Expected value:

.. math::
    E[X] = \alpha \beta + \delta

Variance:

.. math::
    \text{Var}(X) = \alpha \beta^2

Skewness:

.. math::
    \text{Skew}(X) = \dfrac{\beta}{|\beta|}\times\dfrac{2}{\sqrt{\alpha}}

Excess kurtosis:

.. math::
    \text{ExKurt}(X) = \dfrac{6}{\alpha}

Mode, for :math:`\alpha > 1`:

.. math::
    \text{Mode}(X) = (\alpha - 1) \beta + \delta

For :math:`\alpha \leq 1` the density is monotone on its support and the mode
lies at :math:`\delta`.

Fitting to a three-point estimate
---------------------------------

Given elicited values :math:`L \leq M \leq H`, where :math:`L` and :math:`H`
are the :math:`P_L` and :math:`P_H = 1 - P_L` percentiles and :math:`M` is the
mode, the problem is to find :math:`\alpha`, :math:`\beta` and :math:`\delta`
reproducing all three.

Writing :math:`G^{-1}(p; \alpha)` for the percent point function of the
standard gamma distribution with unit scale and zero location, and taking
:math:`\beta > 0` for the moment, the ratio of the two half-ranges eliminates
both the scale and the location and leaves a single equation in the shape
parameter:

.. math::
    \frac{M-L}{H-M} = \frac{(\alpha-1) - G^{-1}(P_L; \alpha)}{G^{-1}(P_H; \alpha) - (\alpha-1)}

This has no closed-form solution and is solved numerically. The library
bisects on skewness, :math:`2/\sqrt{\alpha}`, rather than on :math:`\alpha`
itself, since skewness is bounded on :math:`(0, 2]` and needs no interval
derived from the ceiling.

With :math:`\alpha` known, the remaining parameters follow in closed form. The
scale comes from the full elicited span, which is better conditioned than
either half:

.. math::
    \beta = \frac{H - L}{G^{-1}(P_H; \alpha) - G^{-1}(P_L; \alpha)}

and the location from the mode relation:

.. math::
    \delta = M - (\alpha - 1)\beta

A left-skewed estimate, one for which :math:`M - L > H - M`, is the mirror
image. It is solved by inverting the ratio, solving the same equation, and
negating the scale before computing the location.

Admissible estimates
--------------------

Not every ordering :math:`L \leq M \leq H` can be fitted. The mode of the
fitted distribution cannot fall outside the interval the elicited percentiles
enclose, and requiring :math:`L \leq M` therefore requires
:math:`G^{-1}(P_L; \alpha) \leq \alpha - 1`. The shape at which the two
coincide is the smallest admissible one. For :math:`P_L = 0.10` it is
:math:`\alpha = 1.15631`, corresponding to a skewness of :math:`1.85991`, so
no valid estimate implies more skewness than that. The bound depends on
:math:`P_L`; a wider elicited interval admits more.

The boundary at :math:`\alpha = 1`, where the mode reaches the support
endpoint, lies strictly inside the inadmissible region and is never reached
from a valid estimate.

A symmetric estimate is reproducible only in the limit
:math:`\alpha \to \infty`. The library returns a ceiling of :math:`10^9`,
which reproduces the elicited values to about :math:`1.5 \times 10^{-5}` of
the elicited range rather than exactly.

Accuracy
--------

For any admissible asymmetric estimate, the deviation of a recovered elicited
value, normalised by the elicited range, is bounded by

.. math::
    E < \frac{\tau}{2(2 - \tau)} = \frac{\tau}{4} + O(\tau^2)

where :math:`\tau` is the relative acceptance threshold on the ratio. At the
default :math:`\tau = 10^{-10}` this is :math:`2.5 \times 10^{-11}`. The bound
holds for every admissible input rather than for tested cases, and it excludes
the symmetric branch, where the ceiling rather than the threshold governs.

Recovery of :math:`\alpha` itself is a weaker guarantee. The ratio above
approaches unity as :math:`\alpha^{-1/2}`, so the recovery becomes
increasingly ill-conditioned near symmetry and a fixed threshold buys
progressively less accuracy in the shape parameter. This is a property of the
problem and not of the solver; tightening the threshold improves the recovered
shape at the cost of more iterations.
