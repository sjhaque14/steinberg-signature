This repository contains the code used to perform quantiative analysis in the upcoming manuscript "Anomalous behaviour of the Steinberg signature for detecting departure from thermodynamic equilibrium". 

The Steinberg signature for detecting departure from thermodynamic equilibrium was introduced by I. Z. Steinberg in the 1986 Biophysical Journal paper ["On the time reversal of noise signals"](https://www.sciencedirect.com/science/article/pii/S000634958683449X?via%3Dihub). It exploits specialized "higher-order" autocorrelation functions to detect time-reversal asymmetry in stochastic signals emitted by an underlying continuous-time Markov process. These higher-order autocorrelation functions take the following form:

$$\mathcal{A}^{\alpha,\beta}(\tau) = \lim_{T \rightarrow 0} \frac{1}{T - \tau}\int_{0}^{T - \tau} f^{\alpha}(t)f^{\beta}(t + \tau) dt.$$

This repository contains code to calculate $\mathcal{A}^{\alpha,\beta}(\tau)$ and its time-reverse $\mathcal{A}^{\beta,\alpha}(\tau)$ for any continuous time, finite space, time-homogeneous Markov process. We represent Markov processes using the [**linear framework**](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0036321), a graph-theoretic approach to modeling biochemical networks at steady-state.

![My Image](3vertex-software-infographic.png)

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
