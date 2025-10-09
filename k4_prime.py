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

def eq_params_k4_2cycle():
    """
    Samples equilibrium parameter sets for a 4-state 2-cycle graph
    (k12, k23, k34, k41) and (k21, k32, k43, k14), ensuring zero cycle affinity.

    Returns
    -------
    labels_f : list of 4 floats
        Forward rates: [[k12, k24, k41], [k23, k34, k42]]
    labels_r : list of 4 floats
        Reverse rates: [[k21, k42, k14], [k32, k43, k24]]
    """

    # Sample 8 of the 10 parameters freely
    k24 = round_sig(10 ** np.random.uniform(-3, 3), sig_figs=4)
    k41 = round_sig(10 ** np.random.uniform(-3, 3), sig_figs=4)
    k21 = round_sig(10 ** np.random.uniform(-3, 3), sig_figs=4)
    k14 = round_sig(10 ** np.random.uniform(-3, 3), sig_figs=4)
    
    k23 = round_sig(10 ** np.random.uniform(-3, 3), sig_figs=4)
    k42 = round_sig(10 ** np.random.uniform(-3, 3), sig_figs=4)
    k32 = round_sig(10 ** np.random.uniform(-3, 3), sig_figs=4)
    k43 = round_sig(10 ** np.random.uniform(-3, 3), sig_figs=4)
    
    # Back-calculate k12
    
    k12 = (k21*k42*k14)/(k24 * k41)
    
    # Back-calculate k34
    
    k34 = (k32*k43*k24)/(k23*k42)

    labels_f = [[k12, k24, k41], [k23, k34, k42]]
    labels_r = [[k21, k42, k14], [k32, k43, k24]]

    return labels_f, labels_r

def lap_k4_2cycle(labels_f, labels_r):
    # labels_f = [[k12, k24, k41], [k23, k34, k42]]
    k12 = labels_f[0][0]
    k24 = labels_f[0][1]
    k41 = labels_f[0][2]
    
    k23 = labels_f[1][0]
    k34 = labels_f[1][1]
    k42 = labels_f[1][2]
    
    # labels_r = [[k21, k42, k14], [k32, k43, k24]]
    
    k21 = labels_r[0][0]
    k14 = labels_r[0][2]
    k32 = labels_r[1][0]
    k43 = labels_r[1][1]
    
    lap = np.array([[-k14-k12, k21, 0, k41], [k12, -k21-k23-k24, k32, k42], [0, k23, -k32-k34, k43], [k14, k24, k34, -k41-k43-k42]],dtype=float)
    
    return lap

def define_tau_range(L, max_points=500, cap_factor=10.0):
    # infer the slowest rate
    eigs = np.real(np.linalg.eigvals(-L))
    eigs.sort()
    lambda_1 = eigs[1]
    
    # set upper bound but don’t let it explode
    tau_max = min(cap_factor/lambda_1, 100.0)   # never longer than 100 time-units
    tau = np.linspace(0.01, tau_max, num=max_points)
    return tau, tau_max

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
    f = np.array([observable])
    fstar = f.T
    
    # calculate the stationary distribution of the Markov process
    eigvals, eigvecs = scipy.linalg.eig(L)
    pi = np.array([eigvecs[:,np.argmin(np.abs(eigvals))].real/sum(eigvecs[:,np.argmin(np.abs(eigvals))].real)]).T
    
    # initialize forward and reverse autocorrelation function arrays
    t = np.zeros(len(tau_n))
    t_rev = np.zeros(len(tau_n))
    
    list_result = list(map(lambda i: scipy.linalg.expm(L*i), tau_n))
    
    # populate arrays with analytical solution to autocorrelation function
    for i in range(len(tau_n)):
        t[i] = f**alpha @ list_result[i] @(fstar ** beta * pi)
        t_rev[i] = f**beta @ list_result[i] @(fstar ** alpha * pi)
        
    return t, t_rev