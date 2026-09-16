**********************
Installation and Usage
**********************

Installation
============

To install the egamma library, execute the following command in your terminal:

.. code-block:: bash

    pip install egamma

This command will download and install the egamma library along with any necessary dependencies.

Usage
=====

After installation, import the egamma library into your Python scripts:

.. code-block:: python

    import egamma

The most common thing to do with this library is to turn a three-point
estimate into a distribution:

.. code-block:: python

    dist = egamma.EgammaDistribution.from_tpe(low=100, most_likely=140, high=300)

    dist.mean()        # 189.5
    dist.ppf(0.5)      # median
    dist.rvs(10000)    # a sample for a Monte Carlo run

The fitted distribution's mode is exactly 140, its 10th percentile exactly
100, and its 90th percentile exactly 300.

Quick Calculations
-------------------

If you already know the parameters, access the library's functions directly:

.. code-block:: python

    # Calculate the cumulative probability for x=200
    probability = egamma.cdf(x=200, alpha=10, beta=40, delta=-100)

Using the EgammaDistribution Class
----------------------------------

For a more comprehensive usage, create an instance of the EgammaDistribution class:

.. code-block:: python

    # Create an EgammaDistribution object with specified parameters
    dist = egamma.EgammaDistribution(alpha=10, beta=40, delta=-100)
    # Calculate the cumulative probability for x=200 using the object's method
    probability = dist.cdf(x=200)

From a three-point estimate
---------------------------

.. code-block:: python

    dist = egamma.EgammaDistribution.from_tpe(low=100, most_likely=200,
                                              high=400, low_prob=0.1)

``low_prob`` is the probability attached to the low estimate; the high
estimate takes its complement, ``1 - low_prob``. It must satisfy
``0 < low_prob < 0.5``. The default is 0.1, so the outer values are treated as
the 10th and 90th percentiles.

A left-skewed estimate requires nothing special. The position of the mode
decides the direction:

.. code-block:: python

    dist = egamma.EgammaDistribution.from_tpe(low=100, most_likely=300, high=400)
    assert dist.skew() < 0

Not every estimate can be fitted. The mode cannot fall outside the interval
the elicited percentiles enclose, which limits how asymmetric an estimate may
be: at ``low_prob=0.1`` the skewness cannot exceed 1.85991 in magnitude. See
:doc:`mathematical_definition` for the bound and how it varies with
``low_prob``.

Invalid input raises ``ValueError``:

.. code-block:: python

    egamma.params(300, 140, 100)      # ValueError: low above mode
    egamma.params(100, 140, 300, 0.7) # ValueError: low_prob outside (0, 0.5)

A fit that cannot meet its tolerance raises ``RuntimeError`` rather than
returning an unconverged result.

Fitting to data
---------------

.. code-block:: python

    # Instantiate the distribution from a sample
    dist = egamma.EgammaDistribution.from_fit(data=data)

    # Or get the parameters directly, choosing the estimator
    alpha, beta, delta = egamma.fit(data)                # maximum likelihood
    alpha, beta, delta = egamma.fit(data, method='mom')  # method of moments

The skew direction is taken from the sample, so both left- and right-skewed
data are handled.

Maximum likelihood is the default and is generally the better estimator. The
method of moments is closed-form and needs no iteration, which makes it useful
as a starting point or where an optimiser is unavailable, but it rests on the
sample skewness, whose sampling variance is large, so estimates from small
samples can depart substantially from the parent distribution. It is the same
estimator used by the companion Excel library, so results from the two agree.

For further details, consult the :ref:`functions` and :ref:`classes` documentation.
