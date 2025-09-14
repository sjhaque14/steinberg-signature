import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx
import scipy.linalg

def round_sig(x, sig_figs=4):
    """
    Rounds a number to a given number of significant figures.
    
    Parameters
    ----------
    x : float
        number to round
        
    sig_figs : integer (default=4)
        number of sig figs to round to
        
    Returns
    ----------
    x rounded to the desired number of significant figures
    """
    if x == 0:
        return 0.0
    return round(x, sig_figs - int(np.floor(np.log10(abs(x)))) - 1)

def eq_params_k4():
    """
    Samples equilibrium parameter sets for a 4-state single-cycle graph
    (k12, k23, k34, k41) and (k21, k32, k43, k14), ensuring zero cycle affinity.

    Returns
    -------
    labels_f : list of 4 floats
        Forward rates: [k12, k23, k34, k41]
    labels_r : list of 4 floats
        Reverse rates: [k21, k32, k43, k14]
    """

    # Sample 7 of the 8 parameters freely
    k12 = round_sig(10 ** np.random.uniform(-3, 3), sig_figs=4)
    k23 = round_sig(10 ** np.random.uniform(-3, 3), sig_figs=4)
    k34 = round_sig(10 ** np.random.uniform(-3, 3), sig_figs=4)
    k41 = round_sig(10 ** np.random.uniform(-3, 3), sig_figs=4)

    k21 = round_sig(10 ** np.random.uniform(-3, 3), sig_figs=4)
    k32 = round_sig(10 ** np.random.uniform(-3, 3), sig_figs=4)
    k43 = round_sig(10 ** np.random.uniform(-3, 3), sig_figs=4)

    # Solve for k14 to make the cycle affinity zero
    # A(C) = log[(k12 k23 k34 k41)/(k21 k32 k43 k14)] = 0 → solve for k14
    k14 = (k12 * k23 * k34 * k41) / (k21 * k32 * k43)

    labels_f = [k12, k23, k34, k41]
    labels_r = [k21, k32, k43, k14]

    return labels_f, labels_r

def lap_k4(labels_f, labels_r):
    """
    Computes the Laplacian matrix for a 4-state single-cycle graph
    (k12, k23, k34, k41) and (k21, k32, k43, k14), ensuring zero cycle affinity.

    Returns
    -------
    labels_f : list of 4 floats
        Forward rates: [k12, k23, k34, k41]
    labels_r : list of 4 floats
        Reverse rates: [k21, k32, k43, k14]
    """
    
    # labels_f = k12, k23, k34, k41
    k12, k23, k34, k41 = labels_f[0], labels_f[1], labels_f[2], labels_f[3]
    # labels_r = k21, k32, k43, k14
    k21, k32, k43, k14 = labels_r[0], labels_r[1], labels_r[2], labels_r[3]
    
    lap = np.array([[-k14-k12, k21, 0, k41], [k12, -k21-k23, k32, 0], [0, k23, -k32-k34, k43], [k14, 0, k34, -k41-k43]],dtype=float)
    
    return lap

def spec_any(lap):
    """
    Computes the steady-state distribution directly from the spectrum of a given Laplacian matrix. This function works for a linear framework graph of any size.
    """
    eigvals, eigvecs = scipy.linalg.eig(lap)
    v = eigvecs[:, np.argmin(np.abs(eigvals))].real
    pi = v / v.sum()
    return pi

def define_tau_range(L, max_points=500, cap_factor=10.0):
    """
    Computes the appropriate tau range based on the mixing time of the Markov process specified by the graph G. This function works for a linear framework graph of any size.
    """
    # infer the slowest rate
    eigs = np.real(np.linalg.eigvals(-L))
    eigs.sort()
    lambda_1 = eigs[1]
    
    # set upper bound but don’t let it explode
    tau_max = min(cap_factor/lambda_1, 100.0)   # never longer than 100 time-units
    tau = np.linspace(0.01, tau_max, num=max_points)
    return tau, tau_max

# analytical autocorrelation function from Eq. 21 in paper

def asymmetric_autocorrelation(signal,L,tau,alpha=1,beta=3):
    """
    Numerically calculates the asymmetric autocorrelation functions A^{1,3}(tau) and A^{3,1}(tau) for a particular Laplacian matrix. This function works for a linear framework graph of any size.
    
    Parameters
    ----------
    signal : 1D array
        vector of possible values of signal S = (S(1), ..., S(N))
        
    L : NxN array
        column-based Laplacian matrix of linear framework graph with N vertices
    
    tau : 1D array
        range of intervals between values of signal along integration interval
    
    alpha, beta : scalar
        asymmetric exponents applied to signal (default: alpha=1, beta=3)
    
    Returns
    -------
    a_13 : 1D array
        forward autocorrelation function values
    
    a_31 : 1D array
        reverse autocorrelation function values
    
    """
    # initialize forward and reverse autocorrelation function arrays
    a_13 = np.zeros(len(tau),dtype=float)
    a_31 = np.zeros(len(tau),dtype=float)
    
    # define the signal vectors
    s_t = np.array([signal],dtype=float) # row vector
    s = s_t.T # column vector
    
    # create the diagonal steady state matrix
    # calculate the stationary distribution of the Markov process
    pi = np.array(spec_any(L))
    delta_u_star = np.diag(pi)
    
    # vectorize the Laplacian matrix multiplied by each value in the vector tau
    list_result = list(map(lambda i: scipy.linalg.expm(L*i), tau))
    
    # populate arrays with analytical solution to autocorrelation function
    for i in range(len(tau)):
        a_13[i] = ((s_t**beta) @ list_result[i]) @ (delta_u_star @ (s ** alpha))
        a_31[i] = ((s_t**alpha) @ list_result[i]) @ (delta_u_star @ (s ** beta))
        
    return a_13, a_31
    
# older version of the autocorrelation function from previous simulations (compares well with above version)

def autocorrelation_analytical(observable,L,tau_n,alpha=1,beta=3):
    """
    Calculates the analytical solution for forward and reverse higher-order autocorrelation functions for a particular Laplacian matrix
    
    Parameters
    ----------
    observable : 1D array
        possible values of observable (which is a state function on the Markov process)
    L : NxN array
        column-based Laplacian matrix of linear framework graph with N vertices
    tau_n : 1D array
        range of intervals between values of observable taken by system
    alpha : scalar
        exponent applied to observable
    beta : scalar
        exponent applied to transpose of observable
    
    Returns
    -------
    t : 1D array
        forward autocorrelation function values
    t_rev : 1D array
        reverse autocorrelation function values
    
    """
    f = np.array([observable],dtype=)
    fstar = f.T
    
    # calculate the stationary distribution of the Markov process
    eigvals, eigvecs = scipy.linalg.eig(L)
    pi = np.array([eigvecs[:,np.argmin(np.abs(eigvals))].real/sum(eigvecs[:,np.argmin(np.abs(eigvals))].real)]).T
    
    # initialize forward and reverse autocorrelation function arrays
    t = np.zeros(len(tau_n),dtype=float)
    t_rev = np.zeros(len(tau_n),dtype=float)
    
    list_result = list(map(lambda i: scipy.linalg.expm(L*i), tau_n))
    
    # populate arrays with analytical solution to autocorrelation function
    for i in range(len(tau_n)):
        t[i] = f**alpha @ list_result[i] @(fstar ** beta * pi)
        t_rev[i] = f**beta @ list_result[i] @(fstar ** alpha * pi)
        
    return t, t_rev
