import numpy as np
import scipy.linalg

###################################
# steinberg-utils.py
# This library allows the user to calculate the Steinberg signature for any linear framework graph and produce force-area curves. 
###################################

"""
Functions for sampling parameters for the 3-vertex graph in equilibrium or non-equilibrium steady states. Functions are included for logarithmic or unifrom sampling of parameter values.
"""

def log_eqparamsample_3vertex(min_val=-3,max_val=3,num_params=6):
    """
    Logarithmically samples equilibrium parameters for the 3-vertex graph from the range [10^min_val, 10^max_val].
    
    Parameters
    ----------
    min_val : scalar
        minimum value of sampling range (10^min_val)
    max_val : scalar
        maximum value of sampling range (10^min_val)
    num_params: scalar
        number of rate constants in the Markov process (default=6)
        
    Returns
    -------
    omegas : 1D array
             parameter values in 3-state Markov process that satisfy the cycle condition
             order of parameters: 
    """
    omegas = np.zeros(num_params,dtype=np.float128)
    
    # choose the first 5 parameters at random
    omegas[:-1] = 10**(np.random.uniform(min_val,max_val, size = num_params-1))
    
    # allow the 6th parameter (omega_31) to be a free parameter
    omegas[-1] = (omegas[1]*omegas[3]*omegas[4])/(omegas[0]*omegas[2])
                       
    return omegas

def log_noneqparamsample_3vertex(min_val=-3,max_val=3,num_params=6):
    """
    Logarithmically samples non-equilibrium parameters for the 3-vertex graph from the range [10^min_val, 10^max_val].
    
    Parameters
    ----------
    min_val : scalar›
        minimum value of sampling range (10^min_val)
    max_val : scalar
        maximum value of sampling range (10^max_val)
    num_params: scalar
        number of states in the Markov process (default=6)
               
    Returns
    -------
    omegas : 1D array
             non-equilibrium values of parameters in Markovian system
    """
    omegas = np.array([],dtype=np.float128)
    
    while omegas.size == 0:
        
        # choose 6 random integers betweem 0 and 200
        
        # vals.fill(np.random.choice(np.arange(min_val, max_val,step=min_val)))
        vals = 10**(np.random.uniform(min_val,max_val, size = num_params))

        # calculate the forward and reverse cycle products
        forward = vals[0]*vals[2]*vals[5]
        reverse = vals[1]*vals[3]*vals[4]
        
        if (forward != reverse) and (reverse != 0):
            omegas = vals
    
    return omegas

def lin_eqparamsample_3vertex(min_val=0.001,max_val=100,num_params=6):
    """
    Randomly samples equilibrium parameter sets in the range [0.01,100] for a 3-state Markov process.
    
    Parameters
    ----------
    max_val : scalar
        maximum value of sampling range (default=100)
    num_params: scalar
        number of rate constants in the Markov process (default=6)
        
    Returns
    -------
    omegas : 1D array
             equilibrium values of parameters in Markovian system
    """
    omegas = np.zeros(num_params)
    
    # choose the first 5 parameters at random
    omegas[:-1] = np.around(np.random.choice(np.arange(min_val,max_val,step = 0.001),size=num_params-1),3)
    
    # allow the 6th parameter (omega_31) to be a free parameter
    omegas[-1] = (omegas[1]*omegas[3]*omegas[4])/(omegas[0]*omegas[2])
                       
    return omegas

def lin_noneqparamsample_3vertex(min_val=0.001, max_val=100,num_params=6):
    """
    Randomly samples non-equilibrium parameter sets in the range [0.001,100] for a 3-state Markov process.
    
    Parameters
    ----------
    max_val : scalar
        maximum value of sampling range (default=100)
    num_params: scalar
        number of states in the Markov process (default=6)
        
    Returns
    -------
    omegas : 1D array
             non-equilibrium values of parameters in Markovian system
    """
    omegas = np.array([],dtype=np.float128)
    
    while omegas.size == 0:
        
        # choose 6 random integers betweem 0 and 200
        vals = np.around(np.random.choice(np.arange(min_val,max_val,step = 0.001),size=num_params),3)

        # calculate the forward and reverse cycle products
        forward = vals[0]*vals[2]*vals[5]
        reverse = vals[1]*vals[3]*vals[4]
        
        if (forward != reverse) and (reverse != 0):
            omegas = vals
    
    return omegas

"""
Function to choose the parameter to perturb
"""

def param_choice(num_params=6):
    return np.random.choice(np.arange(num_params-1))

"""
Functions to calculate the Laplacian matrix from a given set of parameters
"""

def Laplacian_3state(omegas):
    """
    Randomly samples equilibrium parameter sets in the range [0.01,100] for a 4-state Markov process.
    
    Parameters
    ----------
    omegas : 1D array
            parameter values of rate constants in 4-state Markovian system
            omegas = [a,b,d,c,f,e] = [0,1,2,3,4,5]
        
    Returns
    -------
    L : 2D array
        column-based Laplacian matrix of 4-state Markovian system
    """
    
    L = np.array([[-(omegas[0]+omegas[4]), omegas[1], omegas[5]], [omegas[0], -(omegas[1]+omegas[2]), omegas[3]], [omegas[4], omegas[2], -(omegas[5]+omegas[3])]],dtype=np.float128)
    
    return L

def Laplacian_4state(omegas):
    """
    Randomly samples equilibrium parameter sets in the range [0.01,100] for a 4-state Markov process.
    
    Parameters
    ----------
    omegas : 1D array
            parameter values of rate constants in 4-state Markovian system
            omegas = [k_12,k_21,k_14,k_41,k_42,k_24,k_32,k_23,k_34,k_43]
        
    Returns
    -------
    L : 2D array
        column-based Laplacian matrix of 4-state Markovian system
    """
    
    L = np.array([[-omegas[0]-omegas[2], omegas[1], 0, omegas[3]],
                  [omegas[0], -omegas[1]-omegas[7]-omegas[5], omegas[6], omegas[4]],
                  [0, omegas[7], -omegas[6]-omegas[8], omegas[9]],
                  [omegas[2], omegas[5], omegas[8], -omegas[3]-omegas[4]-omegas[9]]],dtype=np.float128)
    
    return L

"""
Functions calculating the affinity from a set of parameters.
"""

def cycle_affinity_3state(omegas):
    """
    Calculates the cycle affinity (or the thermodynamic force) for a single cycle, 3 state Markov process
    
    Parameters
    ----------
    omegas : 1D array
             parameter values of the system
    
    Returns
    -------
    affinity : scalar
               value of the thermodynamic foce of the system
    """
    
    # calculate the forward and reverse cycle products
    forward = omegas[0]*omegas[2]*omegas[5]
    reverse = omegas[1]*omegas[3]*omegas[4]
    
    # calculate the cycle affinity
    affinity = np.log(forward/reverse)
    
    return affinity

"""
Functions calculating the higher order autocorrelation functions
"""

def NG_III_autocorrelation_analytical(observable,L,tau_n,alpha=1,beta=3):
    """
    Calculates the analytical solution for autocorrelation function given a Laplacian matrix
    
    Parameters
    ----------
    observable : 1D array
        possible values of observable (which is a state function on the Markov process)
    L : 2D array
        column-based Laplacian matrix of system (including diagonal entries)
    tau_n : 1D array
        range of intervals between values of observable taken by system
    alpha : scalar
        exponent
    beta : scalar
        exponent
    
    Returns
    -------
    t : 1D array
        forward autocorrelation function values
    t_rev : 1D array
        reverse autocorrelation function values
    
    """
    f = np.array([observable],dtype=np.float128)
    fstar = f.T
    
    eigvals, eigvecs = scipy.linalg.eig(L)
    pi = np.array([eigvecs[:,np.argmin(np.abs(eigvals))].real/sum(eigvecs[:,np.argmin(np.abs(eigvals))].real)],dtype=np.float128).T
    
    # initialize forward and reverse autocorrelation function arrays
    t = np.zeros(len(tau_n),dtype=np.float128)
    t_rev = np.zeros(len(tau_n),dtype=np.float128)
    
    list_result = list(map(lambda i: scipy.linalg.expm(L*i), tau_n))
    
    # populate arrays with analytical solution to autocorrelation function
    for i in range(len(tau_n)):
        t[i] = f**alpha @ list_result[i] @(fstar ** beta * pi)
        t_rev[i] = f**beta @ list_result[i] @(fstar ** alpha * pi)
        
    return t, t_rev

"""
Functions calculating the Steinberg signature
"""

def steinberg_signature(t,t_rev):
    return np.abs(np.trapz(t)-np.trapz(t_rev))

"""
Force area analysis
"""

def peturbation(omegas, param_choice, m=1.01):
    
    omegas[param_choice] = omegas[param_choice]*m
    
    return omegas

def force_area(num_perturbations, omegas, param_choice, observable, tau_n,m=1.01):
    
    forces = np.zeros(num_perturbations,dtype=np.float128)
    areas = np.zeros(num_perturbations,dtype=np.float128)
    
    for i in range(num_perturbations):

        # calculate the cycle affinity        
        forces[i] = cycle_affinity_3state(omegas)
        
        L = np.array([[-(omegas[0]+omegas[4]), omegas[1], omegas[5]], [omegas[0], -(omegas[1]+omegas[2]), omegas[3]], [omegas[4], omegas[2], -(omegas[5]+omegas[3])]],dtype=np.float128)
        
        t, t_rev = NG_III_autocorrelation_analytical(observable,L,tau_n,alpha=1,beta=3)
        
        areas[i] = np.abs(np.trapz(t)-np.trapz(t_rev))
        
        # modify the value of one parameter
        omegas = omegas[param_choice]*m
    
    return forces, areas