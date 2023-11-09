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

Quick Calculations
-------------------

For immediate calculations, directly access the library's functions:

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


Alternatively, from a three-point estimate:

.. code-block:: python

    # Instantiate the distribution with a three-point estimate
    dist = egamma.EgammaDistribution.from_tpe(low=100, most_likely=200, high=400, low_prob=0.1)
    # Note: low_prob specifies the probability of the low estimate


or by fitting the expanded gamma distribution to your data:

.. code-block:: python

    # Instantiate the distribution with fitting to data
    dist = egamma.EgammaDistribution.from_fit(data=data)


For further details, consult the :ref:`functions` and :ref:`classes` documentation.


