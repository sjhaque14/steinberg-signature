Sabina J. Haque

Last updated: Jan 13 2024

Contact: sabina_haque@fas.harvard.edu, sjhaque14@gmail.com

This repository contains the code used for the analysis performed in the upcoming manuscript "Anomalous behaviour of the Steinberg signature for detecting departure from thermodynamic equilibrium". 

The Steinberg signature for detecting departure from thermodynamic equilibrium was introduced by I. Z. Steinberg in the 1986 Biophysical Journal paper ["On the time reversal of noise signals"](https://www.sciencedirect.com/science/article/pii/S000634958683449X?via%3Dihub). It exploits specialized "higher-order" autocorrelation functions to detect time-reversal asymmetry in stochastic signals emitted by an underlying continuous-time Markov process. These higher-order autocorrelation functions take the following form:

$$\mathcal{A}^{\alpha,\beta}(\tau) = \lim_{T \rightarrow 0} \frac{1}{T - \tau}\int_{0}^{T - \tau} f^{\alpha}(t)f^{\beta}(t + \tau) dt.$$

This repository contains code to calculate $\mathcal{A}^{\alpha,\beta}(\tau)$ and its time-reverse $\mathcal{A}^{\beta,\alpha}(\tau)$.

--------------------------------

The [**Steinberg signature**](https://www.sciencedirect.com/science/article/pii/S000634958683449X?via%3Dihub) is a signature of non-equilibrium conditions in Markovian systems. Using higher-order autocorrelation functions of the form $G^{\alpha,\beta}$ and $G^{\beta,\alpha}$, this method allows one to detect whether or not a given system obeys detailed balance. Here, we present computational tools for calculating the Steinberg signature on a 3-state Markov process. We represent Markov processes using a graph-theoretic approach called the [**linear framework**](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0036321), which has been developed by the Gunawardena lab to model time-scale separation in biochemical systems. Specifically, for a linear framework graph $G$ with any given parameterization, we want to be able to calculate the Steinberg signature and determine how it varies in response to increasing thermodynamic force of the underlying system.

![My Image](3vertex-software-infographic.png)

To investigate the relationship between the Steinberg signature and thermodynamic force, we initialize a linear framework graph in an equilibrium steady state. In such a steady state, the thermodynamic force of the system, measured here by the cycle affinity, is equal to 0. The Steinberg signature, measured by the area difference between $G^{\alpha,\beta}$ and $G^{\beta,\alpha}$, is also equal to 0. To increase thermodynamic force, we select a single parameter to perturb from its equilibrium value. Once the system has been driven very far from equilibrium, we plot the area difference against the force of the system. We call the resulting a plot a force-area curve.

The main results are the observations of a non-monotonic relationship between the Steinberg signature and thermodynamic force, as well as how the Steinberg signature asymptotes to 0, the equilibrium value, in the limit of high thermodynamic force.

## Prerequisites

The following Python3 libraries are used in this library. All of the below can be installed using `pip` or `conda`.

* [Numpy](https://numpy.org/install/)

* [Scipy.linalg](https://scipy.org/install/)

* [Matplotlib](https://matplotlib.org/stable/users/installing/index.html)

* [Tqdm](https://pypi.org/project/tqdm/) (optional)

## Installation

The relevant functions for calculating the Steinberg signature and generating force-area curves are included in the file `steinberg_utils_3vertex.py`. 

```
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from tqdm import tqdm

from steinberg_utils_3vertex import *
```

For an example of how to use the functions in `steinberg_utils_3vertex.py`, see the Jupyter notebook `steinberg_3vertex_official.ipynb`. You may also [clone this repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) to obtain the relevant files.

```
git clone https://github.com/sjhaque14/steinberg-signature
```

## Contact

For more details, please contact Sabina J Haque at sabina_haque@fas.harvard.edu.
