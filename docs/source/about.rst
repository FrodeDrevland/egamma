About
======

What this library is for
________________________

``egamma`` turns a three-point estimate — a low value, a most likely value and
a high value — into a probability distribution you can sample from. It is
meant for the situation where an expert has given a judgement rather than data
has been collected, and a Monte Carlo simulation needs actual distribution
parameters before it can run.

.. code-block:: python

    import egamma

    dist = egamma.EgammaDistribution.from_tpe(low=100, most_likely=140, high=300)
    dist.rvs(10000)

The fitted distribution's mode is exactly the most likely value, and its 10th
and 90th percentiles are exactly the two outer values. The numbers the
estimator gave you come back out.

Why not SciPy alone
___________________

SciPy has the gamma distribution but not this fit. Two things are missing.

First, ``scipy.stats.gamma`` will not accept a negative scale parameter, so it
cannot represent a left-skewed estimate — one where the most likely value sits
closer to the high end than the low end. Such estimates are common in cost and
schedule work, and the usual workaround is to reverse the data by hand.

Second, neither ``scipy.stats.gamma`` nor ``scipy.stats.pearson3`` will fit to
a mode plus two percentiles. The elicitation tools that exist work from
quantiles alone, and a mode is not a quantile: its position depends on the
shape of the density rather than on an area under it.

``egamma`` builds on SciPy's gamma implementation and supplies both.

What the distribution is
________________________

The ordinary three-parameter gamma distribution is strictly right-skewed,
because its scale parameter must be positive. Allowing the scale to take
negative values reflects the density about the location parameter, so the
support runs downward from it instead of upward and the skewness turns
negative. One family then covers estimates leaning either way.

That family is not new. It is the Pearson Type III distribution, long
established in hydrology and elsewhere, and the reflected form has been
studied in the statistics literature. What this library provides is that
family in the shape, scale and location parameters already familiar from the
gamma distribution, together with the fitting procedure described in the
mathematical definitions.

Documentation Content
________________________
- **About:** The current section.
- **Mathematical Definitions:** The density, the moments, and how the parameters are recovered from a three-point estimate.
- **Installation and Usage:** Installing the library and using it.
- **Functions:** Descriptions of the library's functions, their parameters and return values.
- **Classes:** The library's classes and their structure.

Support
________
If you encounter any problems or have any questions, please open an issue on the `GitHub repository issue tracker <https://github.com/FrodeDrevland/egamma/issues>`_.

Contact Information
______________________

For inquiries or collaboration on the ``egamma`` library, contact:

- **Associate Professor Frode Drevland**
- **Affiliation**: Norwegian University of Science and Technology (NTNU)
- **Email**: `frode.drevland@ntnu.no <mailto:frode.drevland@ntnu.no>`_

Dr. Drevland is dedicated to the continuous development of the ``egamma`` library and welcomes feedback, suggestions, and contributions from the community.

License
-------
This library is distributed under the MIT License. See `LICENSE <https://github.com/FrodeDrevland/egamma/blob/main/LICENSE>`_ for more information.
