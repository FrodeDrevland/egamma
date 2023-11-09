# `egamma` -  Expanded Gamma Distribution Library

Welcome to the `egamma` - Expanded Gamma Distribution Library, an open-source initiative to provide a python implementation of the expanded gamma distribution.

## Overview
The `egamma` library provides an implementation of the expanded gamma distribution, a variant of the traditional three-parameter gamma distribution also allowing for left-skewed data. This distribution, while only recently receiving a formal definition, has been utilized informally in Scandinavian cost estimation practices since the 1970s.

Similar to the conventional gamma distribution, the expanded version is defined by three parameters: alpha (shape), beta (scale), and delta (location). The key distinction of the expanded gamma distribution is its allowance for the scale parameter, beta, to take on negative values. This enables the distribution to pivot around the location parameter, delta, making it adept at also modeling left-skewed datasets.

`egamma` builds upon the solid foundation of SciPy's implementation of the gamma distribution, enhancing it to incorporate the specialized attributes of the expanded version. Additionally, `egamma` introduces capabilities beyond those offered by SciPy, such as three-point estimation. This method is used with expert judgment to determine the most likely, optimistic, and pessimistic scenarios. While standard practice often places optimistic and pessimistic estimates at the 10th and 90th percentiles, `egamma` offers the flexibility to choose any percentile range, thus allowing for customized three-point estimation according to user preference.

## Features


- **Distribution Functions**: Implements the probability density function (PDF), cumulative density function (CDF), inverse cumulative density or percent point function (PPF).
- **Statistical measures**: Implements functions for mean, mode, median, variance, standard deviation, skew and kurtosis 
- **Parameter Estimation**: Either through data fitting or three-point-estimation
  - **Data Fitting**: Functions to fit the expanded gamma distribution to your data using maximum likelihood estimation or method of moments.
  - **Three-point-estimation**: Implements algorithms to find the shape, scale and location parameter of the expanded gamma distribution from a three-point-estimate. 
- **Documentation**: Comprehensive documentation for each function, including usage examples and parameter explanations.


## Installation

To install the library, use the following command:

```bash
pip install egamma
```

## Usage

After installation, you can import the library into your Python scripts as follows:

```python
import egamma
```

The library's functions are accessible either directly for quick calculations or through creating an instance of the EgammaDistribution class for more complex analyses. For instance, to calculate the cumulative probability of a value using the cumulative density function (CDF), you can do the following:

```python
probability = egamma.cdf(x=200,alpha=10,beta=40,delta=-100)
```
Alternatively, you can create an instance of the EgammaDistribution class with the specified parameters and then call the CDF method:
```python
dist = egamma.EgammaDistribution(alpha=10,beta=40,delta=-100)
probability = dist.cdf(x=200)
```

For scenarios where you have a three-point estimate—comprising a low, most likely, and high value—you can instantiate the distribution as follows:

```python
dist = egamma.EgammaDistribution.from_tpe(low=100,most_likely=200,high=400,low_prob=0.1)
# low_prob specifies the probability of the low estimate; the probability of the high estimate is the complement (1 - low_prob).
```
Similarly, if you have a dataset and wish to fit the expanded gamma distribution to it:
```python
dist = egamma.EgammaDistribution.from_fit(data=data)
```

## Documentation
Comprehensive documentation of the library and its functions can be found at:
https://frodedrevland.github.io/egamma/

## Support
If you encounter any problems or have any questions, please open an issue on the [GitHub repository issue tracker](https://github.com/FrodeDrevland/egamma/issues).

## Contact Information

For inquiries, support, or collaboration on the `egamma` library, please reach out to:

- **Associate Professor Frode Drevland**
- **Affiliation**: Norwegian University of Science and Technology (NTNU)
- **Email**: [frode.drevland@ntnu.no](mailto:frode.drevland@ntnu.no)

Dr. Drevland is dedicated to the continuous development of the `egamma` library and welcomes feedback, suggestions, and contributions from the community. 

## License
This library is distributed under the MIT License. See [LICENSE](https://github.com/FrodeDrevland/egamma/blob/main/LICENSE) for more information.

