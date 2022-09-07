# steinberg-signature

The Steinberg signature was first introduced by I.Z. Steinberg as a theoretical method of detecting non-equilibrium steady states in Markovian systems, potentially in biochemical contexts. The Steinberg signature is computed using higher-order autocorrelation functions $G^{\alpha, \beta}(\tau)$ and $G^{\beta, \alpha}(\tau)$ of a time-dependent signal $f(t)$ emitted from the underlying Markovian system and observed over the time interval $[t_1, t_2]$. We define the higher-order autocorrelation function as follows,

$$G^{\alpha, \beta}(\tau) = \dfrac{1}{t_2-\tau-t_1} \int_{t_1}^{t_2-\tau} f^\alpha (t) f^\beta (t+\tau) dt.$$

The Steinberg signature uses $G^{\alpha, \beta}(\tau)$ and $G^{\beta, \alpha}(\tau)$, which is equal to the reverse of $G^{\alpha, \beta}(\tau)$, to detect broken time-reversal symmetry. The details of how the Steinberg signature does this can be found in the original publication (IZ Steinberg 1986).

Here, we provide software that calculates the forward and reverse higher-order autocorrelation functions off a Markovian system. We model Markov processes using a graph-theoretic approach called the linear framework, which has been developed by the Gunawardena lab to analyze time-scale separation in biochemical systems and molecular information processing (J Estrada et al 2016)

1. Steinberg, I. Z. On the time reversal of noise signals. Biophys. J. 50, 171–179 (1986)

2. Estrada, J., Wong, F., DePace, A. & Gunawardena, J. Information Integration and Energy Expenditure in Gene Regulation. Cell 166, 234–244 (2016)
