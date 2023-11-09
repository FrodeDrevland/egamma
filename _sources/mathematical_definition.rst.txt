Mathematical Definitions
========================

Introduction
------------

This page provides the mathematical definitions of the expanded gamma distribution, detailing its probability density function, cumulative distribution function, and key statistical measures. These definitions serve as a foundation for understanding the distribution's properties and for applying it to real-world data analysis scenarios.

Parameters
----------
The expanded gamma distribution is characterized by three parameters:

- :math:`\alpha` (alpha): The shape parameter, which must be positive (:math:\alpha > 0).
- :math:`\beta` (beta): The scale parameter, which must be non-zero (:math:\beta \neq 0). It determines the direction of skewness; positive values lead to right-skewness, while negative values result in left-skewness.
- :math:`\delta` (delta): The location parameter, which can be any real number and shifts the distribution along the x-axis.

The parameters :math:`\alpha` and :math:`\beta` primarily control the shape and scale of the distribution, respectively. The parameter :math:`\delta` shifts the distribution along the x-axis.

Note that while the distribution is strictly either left- or right-skewed, as the parameter :math:`\alpha` approaches infinity, the shape of the distribution converges to that of a normal distribution. Thus, for practical purposes, an expanded gamma distribution defined with a high :math:`\alpha` value, such as :math:`1e9`, can approximate symmetrical scenarios quite well.



Distribution Functions
----------------------
The probability density function (PDF) and cumulative distribution function (CDF) are defined as follows:

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

where :math:`\gamma \left(\alpha, \frac{x-\delta}{\beta}\right)` is the lower incomplete gamma function :math:`\displaystyle\int_{0}^{\frac{x-\delta}{\beta}}t^{\alpha-1}e^{-t}\,dt\\.`

Please note that the cumulative distribution function (CDF) of the gamma distribution does not have a closed-form solution. Consequently, the computation of the inverse CDF, or percent-point function, cannot be performed directly and requires the use of numerical techniques or approximation methods for its calculation.


Statistical measures
--------------------
Expected value:

.. math::
    \begin{align}
    E[X] &= \alpha \beta + \delta
    \end{align}


Variance:

.. math::
    \begin{align}
    \text{Var}(X) &= \alpha \beta^2
    \end{align}

Skewness:

.. math::
    \begin{align}
    \text{Skew}(X) &= \dfrac{\beta}{|\beta|}\times\dfrac{2}{\sqrt{\alpha}}
    \end{align}

Kurtosis:

.. math::
    \begin{align}
    \text{Kurt}(X) &= \dfrac{6}{\alpha}
    \end{align}

Mode:

.. math::
    \begin{align}
    \text{Mode}(X)  &= (\alpha - 1) \beta + \delta
    \end{align}