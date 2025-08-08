import numpy as np
import scipy.linalg

# steinberg_utils_3vertex.py

# This library allows the user to compute higher-order autocorrelation functions and the Steinberg signature for any linear framework graph (which represent continuous-time, finite-state, time-homogeneous Markov process). For relevant mathematical details, please refer to the accompanying manuscript Haque, Cetiner, Gunawardena 2024. The code in this file is designed with the 3-vertex graph, K, as our analysis was primarily performed for this system. That being said, some of the functions in this library are able to be used on linear framework graphs of any size. For code optimized for handling general graphs, please see the library `general_graphs.py`. 

## PARAMETER SAMPLING ##

# Randomly sampling transition rates for the 3-vertex graph, K. These parameters are defined as 10^x, where x is randomly drawn from the uniform distribution on (-3, 3). Here, we provide a function for randomly sampling parameters that satisfy detailed balance and one for which the parameters do not necessarily satisfy detailed balance.

# See Figure 1A. in Haque, Cetiner, Gunawardena 2024 for symbolic edge label assignments. The array params lists these edge labels in the following order: a, b, d, c, f, e.

# Vectorized sig-fig rounding function
def round_to_sigfigs(array, sig_figs=4):
    """
    Rounds an array to a given number of significant figures.
    """
    def _round(x):
        if x == 0:
            return 0.0
        return round(x, sig_figs - int(np.floor(np.log10(abs(x)))) - 1)
    
    vect = np.vectorize(_round)
    
    return vect(array)

def equilibrium_parameters(min_val=-3,max_val=3,num_params=6,sig_figs=4):
    """
    Randomly samples transition rates for a 3-vertex graph, K, which satisfy detailed balance. These parameters are defined as 10^x, where x is randomly drawn from the uniform distribution on (min_val, max_val). 
    
    Parameters
    ----------
    min_val : scalar
        minimum value of sampling range (default=-3)
    max_val : scalar
        maximum value of sampling range (default=3)
    num_params: integer
        number of transition rates in graph (default=6)
        
    Returns
    -------
    params : 1D array
             transition rates in the 3-vertex graph K satisfy the cycle condition
             order of parameters: a, b, d, c, f, e = params[0], params[1], params[2], params[3], params[4], params[5]
    """
    params = np.zeros(num_params)
    
    # randomly sample the first 5 parameters
    params[:-1] = 10**(np.random.uniform(min_val,max_val, size = num_params-1))
    
    # allow the 6th parameter (e = params[-1]) to be a free parameter
    params[-1] = (params[1]*params[3]*params[4])/(params[0]*params[2])
                       
    return round_to_sigfigs(params, sig_figs)

def random_parameters(min_val=-3,max_val=3,num_params=6,sig_figs=4):
    """
    Randomly samples transition rates for a 3-vertex graph, K, which do not necessarily satisfy detailed balance. These parameters are defined as 10^x, where x is randomly drawn from the uniform distribution on (min_val, max_val).
    
    Parameters
    ----------
    min_val : scalar
        minimum value of sampling range (default=-3)
    max_val : scalar
        maximum value of sampling range (default=3)
    num_params: integer
        number of transition rates in graph (default=6)
               
    Returns
    -------
    params : 1D array
             transition rates in the 3-vertex graph K
             order of parameters: a, b, d, c, f, e = params[0], params[1], params[2], params[3], params[4], params[5]
    """
    
    params = np.zeros(num_params)
    
    params[:] = 10**(np.random.uniform(min_val,max_val, size = num_params))
    
    return round_to_sigfigs(params, sig_figs)

## LAPLACIAN MATRIX & STEADY STATE DISTRIBUTION ##

# See Figure 1B. in Haque, Cetiner, Gunawardena 2024 for the Laplacian matrix of the 3-vertex graph K.

def Laplacian_K(params):
    """
    Calculates the Laplacian matrix for the 3-vertex graph K.
    
    Parameters
    ----------
    params : 1D array
             transition rates in the 3-vertex graph K
             order of parameters: a, b, d, c, f, e = params[0], params[1], params[2], params[3], params[4], params[5]
    
    Returns
    -------
    L : 3x3 array
        column-based Laplacian matrix of 3-vertex graph K
    """
    a = params[0]
    b = params[1]
    d = params[2]
    c = params[3]
    f = params[4]
    e = params[5]
    
    L = np.array([[-(a+f), b, e], [a, -(b+d), c], [f, d, -(e+c)]])
    
    return L

def steady_state_MTT_K(params):
    """
    Calculates the steady-state distribution for the 3-vertex graph K using the linear framework and the Matrix-Tree Theorem.
    
    Parameters
    ----------
    params : 1D array
             parameter values of rate constants in 3-vertex graph K
             params = [a,b,d,c,f,e]
    
    Returns
    -------
    pi : 1D array
         the steady state distribution for a 3-vertex graph K.
    
    """
    a = params[0]
    b = params[1]
    d = params[2]
    c = params[3]
    f = params[4]
    e = params[5]
    
    rho_1 = b*c + b*e + d*e
    
    rho_2 = a*e + a*c + c*f
    
    rho_3 = a*d + b*f + d*f
    
    rho_tot = rho_1 + rho_2 + rho_3
    
    pi = np.array([rho_1/rho_tot, rho_2/rho_tot, rho_3/rho_tot])
    
    return pi

## CYCLE AFFINITY ##

# See equation 15 in Haque, Cetiner, Gunawardena 2024. For the 3-vertex graph K, A(C) = ln(ade/bfc).

def cycle_affinity_K(params):
    """
    Calculates the cycle affinity for a 3-vertex graph K.
    
    Parameters
    ----------
    params : 1D array
             parameter values of rate constants in 3-vertex graph K
             params = [a,b,d,c,f,e]
             
    Returns
    -------
    affinity : scalar
               cycle affinity
    """
    
    a = params[0]
    b = params[1]
    d = params[2]
    c = params[3]
    f = params[4]
    e = params[5]
    
    # calculate the forward and reverse cycle products
    forward = a*d*e
    reverse = b*c*f
    
    # calculate the cycle affinity
    affinity = np.abs(np.log(forward/reverse))
    
    return affinity

## HIGHER-ORDER AUTOCORRELATION FUNCTIONS ##

def define_tau_range(L, max_points=500, cap_factor=10.0):
    # infer the slowest rate
    eigs = np.real(np.linalg.eigvals(-L))
    eigs.sort()
    lambda_1 = eigs[1]
    
    # set upper bound but don’t let it explode
    tau_max = min(cap_factor/lambda_1, 100.0)   # never longer than 100 time-units
    tau = np.linspace(0.01, tau_max, num=max_points)
    return tau, tau_max

def asymmetric_autocorrelation(signal,L,tau,pi,alpha=1,beta=3):
    """
    Numerically calculates the asymmetric autocorrelation functions A^{1,3}(\tau) and A^{3,1}(\tau) for a particular Laplacian matrix. This function works for a linear framework graph of any size.
    
    Parameters
    ----------
    signal : 1D array
        vector of possible values of signal S = (S(1), ..., S(N))
        
    L : NxN array
        column-based Laplacian matrix of linear framework graph with N vertices
    
    tau : 1D array
        range of intervals between values of signal along integration interval
        
    pi : 1D array
         the steady state distribution for a linear framework graph with N vertices
    
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
    a_13 = np.zeros(len(tau))
    a_31 = np.zeros(len(tau))
    
    # define the signal vectors
    s_t = np.array([signal]) # row vector
    s = s_t.T # column vector
    
    # create the diagonal steady state matrix 
    delta_u_star = np.diag(pi)
    
    # vectorize the Laplacian matrix multiplied by each value in the vector tau
    list_result = list(map(lambda i: scipy.linalg.expm(L*i), tau))
    
    # populate arrays with analytical solution to autocorrelation function
    for i in range(len(tau)):
        a_13[i] = ((s_t**beta) @ list_result[i]) @ (delta_u_star @ (s ** alpha))
        a_31[i] = ((s_t**alpha) @ list_result[i]) @ (delta_u_star @ (s ** beta))
        
    return a_13, a_31

## THE STEINBERG SIGNATURE ##

def make_observable(node_list):
    """
    Create the observable vector f for a graph with size num_nodes = len(node_list). The observable vector is a function on the states of the Markov process defined for the linear framework graph: when the system exists in state k, f takes vaue f_k.
    
    Parameters
    ----------
    node_list : 1D array
        list of nodes in the graph
    
    Returns
    -------
    f : 1D array
        list of values that the observable f assumes based on the state the Markov process exists in at a given time t 
    
    """
    
    num_nodes = len(node_list)
    
    f = np.zeros(num_nodes)
    
    for i in range(0,num_nodes):
        f[i] = 3+(2*i)
    
    return f

def numerical_area(t,t_rev):
    """
    Calculates the analytical solution for forward and reverse higher-order autocorrelation functions for a particular Laplacian matrix
    
    Parameters
    ----------
    signal : 1D array
        possible values of signal (which is a state function on the Markov process)
    L : NxN array
        column-based Laplacian matrix of linear framework graph with N vertices
    tau : 1D array
        range of intervals between values of signal taken by system
    alpha : scalar
        exponent applied to signal
    beta : scalar
        exponent applied to transpose of signal
    
    Returns
    -------
    t : 1D array
        forward autocorrelation function values
    t_rev : 1D array
        reverse autocorrelation function values
    """
    return np.abs(np.trapz(t)-np.trapz(t_rev))