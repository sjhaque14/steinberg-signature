# Steinberg-3vertex

The [**Steinberg signature**](https://www.sciencedirect.com/science/article/pii/S000634958683449X?via%3Dihub) is a signature of non-equilibrium conditions in Markovian systems. Using higher-order autocorrelation functions of the form $G^{\alpha,\beta}$ and $G^{\beta,\alpha}$, this method allows one to detect whether or not a given system obeys detailed balance. Here, we present computational tools for calculating the Steinberg signature on a 3-state Markov process, represented by the [**linear framework**](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0036321). Specifically, for a linear framework graph $G$ with any given parameterization, we want to be able to calculate the Steinberg signature and determine how it varies in response to increasing thermodynamic force of the underlying system.

To investigate the relationship between the Steinberg signature and thermodynamic force, we initialize a linear framework graph in an equilibrium steady state. In such a steady state, the thermodynamic force of the system, measured here by the cycle affinity, is equal to 0. The Steinberg signature, measured by the area difference between $G^{\alpha,\beta}$ and $G^{\beta,\alpha}$, is also equal to 0. To increase thermodynamic force, we select a single parameter to perturb from its equilibrium value. Once the system has been driven very far from equilibrium, we plot the area difference against the force of the system. We call the resulting a plot a force-area curve.


## Installation
