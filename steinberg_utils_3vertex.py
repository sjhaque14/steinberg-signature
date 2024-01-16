import numpy as np
import scipy.linalg

# steinberg_utils_3vertex.py

# This library allows the user to compute higher-order autocorrelation functions and the Steinberg signature for any linear framework graph (which represent continuous-time, finite-state, time-homnogeneous Markov process). Much of the code in this file is designed with the 3-vertex graph in mind, as the analysis was primarily performed for this specific system. For relevant mathematical details, please refer to the accompanying manuscript Haque, Cetiner, and Gunawardena 2024.

## PARAMETER SAMPLING ##

# Sampling parameter values for the 3-vertex graph. Parameters are sampled logarithmically from the range (10^{-3}, 10^3). Here, we provide a function for randomly sampling parameters that satisfy detailed balance and one for which the parameters need not satisfy detailed balance.

# See Figure 1. in Haque, Cetiner, and Gunawardena 2024 for label assignments. The parameters are listed in the following order: a, b, d, c, f, e.

def log_eqparamsample_3vertex(min_val=-3,max_val=3,num_params=6):
    """
    Logarithmically samples equilibrium parameters for the 3-vertex graph from the range [10^min_val, 10^max_val].
    
    Parameters
    ----------
    min_val : scalar
        minimum value of sampling range (10^min_val)
    max_val : scalar
        maximum value of sampling range (10^min_val)
    num_params: integer
        number of rate constants in the Markov process (default=6)
        
    Returns
    -------
    params : 1D array
             parameter values in 3-state Markov process that satisfy the cycle condition
             order of parameters: a, b, d, c, f, e = params[0], params[1], params[2], params[3], params[4], params[5]
    """
    omegas = np.zeros(num_params,dtype=np.float128)
    
    # choose the first 5 parameters at random
    omegas[:-1] = 10**(np.random.uniform(min_val,max_val, size = num_params-1))
    
    # allow the 6th parameter (omega_31) to be a free parameter
    # back-calculated with the cycle condition from the 3-vertex graph
    omegas[-1] = (omegas[1]*omegas[3]*omegas[4])/(omegas[0]*omegas[2])
                       
    return omegas

def log_noneqparamsample_3vertex(min_val=-3,max_val=3,num_params=6):
    """
    Logarithmically samples non-equilibrium parameters for the 3-vertex graph from the range [10^min_val, 10^max_val].
    
    Parameters
    ----------
    min_val : scalar
        minimum value of sampling range (10^min_val)
    max_val : scalar
        maximum value of sampling range (10^max_val)
    num_params: integer
        number of rate constants in the Markov process (default=6)
               
    Returns
    -------
    omegas : 1D array
             non-equilibrium values of parameters in Markovian system
    """
    omegas = np.array([],dtype=np.float128)
    
    while omegas.size == 0:
                
        # choose 6 random parameters logarithmically
        vals = 10**(np.random.uniform(min_val,max_val, size = num_params))

        # calculate the forward and reverse cycle products
        forward = vals[0]*vals[2]*vals[5]
        reverse = vals[1]*vals[3]*vals[4]
        
        # if they don't satisfy detailed balance (fat chance), let them be the omegas
        if (forward != reverse) and (reverse != 0):
            omegas = vals
    
    return omegas

#########################################################################################################################################################################################
# Functions to calculate the Laplacian matrix from a given set of parameters
#########################################################################################################################################################################################

def Laplacian_3state(omegas):
    """
    Calculates the Laplacian matrix for any graph. The entries of the Laplacian are computed using the following mathematical formula:
    
    L_{ij}(G) = e_{ij} if i \neq j
    L_{ij}(G) = -\sum_{v \neq j} e_{vj} if i = j.
    
    Parameters
    ----------
    omegas : 1D array
             parameter values of rate constants in 3-state Markovian system
             omegas = [a,b,d,c,f,e] = [0,1,2,3,4,5]
        
    Returns
    -------
    L : 3x3 array
        column-based Laplacian matrix of 3-state Markovian system
    """
    
    L = np.array([[-(omegas[0]+omegas[4]), omegas[1], omegas[5]], [omegas[0], -(omegas[1]+omegas[2]), omegas[3]], [omegas[4], omegas[2], -(omegas[5]+omegas[3])]],dtype=np.float128)
    
    return L

#########################################################################################################################################################################################
# Functions calculating the affinity from a set of parameters.
#########################################################################################################################################################################################

def cycle_affinity_3state(omegas):
    """
    Calculates the cycle affinity (or the thermodynamic force) for a 3-state Markov process using the following mathematical formula:
    
    A(C) = log(ade/bfc)
    
    Parameters
    ----------
    omegas : 1D array
             parameter values of rate constants in 3-state Markovian system
             omegas = [a,b,d,c,f,e] = [0,1,2,3,4,5]
             
    Returns
    -------
    affinity : scalar
               value of the thermodynamic foce of the system
    """
    
    # calculate the forward and reverse cycle products
    forward = omegas[0]*omegas[2]*omegas[5]
    reverse = omegas[1]*omegas[3]*omegas[4]
    
    # calculate the cycle affinity
    affinity = np.abs(np.log(forward/reverse))
    
    return affinity

#########################################################################################################################################################################################
# Determine the appropriate tau range
#########################################################################################################################################################################################

def tau_check(L, observable, start_0=0.01,stop_0=2.0,step_0=0.01):
    """
    Determines a range of tau values for a particular set of parameters for an n-state Markov process
    
    Parameters
    ----------

    L : 3x3 array
        column-based Laplacian matrix of 3-vertex linear framework graph
    
    observable : 1D array
        possible values of observable (which is a state function on the Markov process)
        
    start_0 : scalar (default = 0.01)
        initial start value of the tau array
    
    stop_0 : scalar (default = 2.0)
        initial stop value of the tau array
    
    step_0 : scalar (default = 0.01)
        initial step value of the tau array
    
    Returns
    -------
    tau_n : 1D array
        range of intervals between values of observable assumed by system
        
    t : 1D array
        forward autocorrelation function values
        
    t_rev : 1D array
        reverse autocorrelation function values
    """
    
    stop = stop_0
    tau_n = np.arange(start_0,stop_0,step=step_0)
    t, t_rev = autocorrelation_analytical(observable,L,tau_n,alpha=1,beta=3)
    
    while round(t[-1], 10) != round(t[-2], 10):
        stop = stop + 10
        tau_n = np.arange(start_0,stop,step=step_0)
        t, t_rev = autocorrelation_analytical(observable,L,tau_n,alpha=1,beta=3)
    
    return tau_n, t, t_rev

#########################################################################################################################################################################################
# Functions calculating the higher order autocorrelation functions
#########################################################################################################################################################################################

def autocorrelation_analytical(observable,L,tau_n,alpha=1,beta=3):
    """
    Calculates the analytical solution for forward and reverse higher-order autocorrelation functions for a particular Laplacian matrix using the following formula (in LaTeX):
    
    $$G^{\alpha,\beta}(\tau) = f^\alpha e^{L \tau} f^{* \beta} \pi$$
    
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
    f = np.array([observable],dtype=np.float128)
    fstar = f.T
    
    # calculate the stationary distribution of the Markov process
    eigvals, eigvecs = scipy.linalg.eig(L)
    pi = np.array([eigvecs[:,np.argmin(np.abs(eigvals))].real/sum(eigvecs[:,np.argmin(np.abs(eigvals))].real)]).T
    
    # initialize forward and reverse autocorrelation function arrays
    t = np.zeros(len(tau_n),dtype=np.float128)
    t_rev = np.zeros(len(tau_n),dtype=np.float128)
    
    list_result = list(map(lambda i: scipy.linalg.expm(L*i), tau_n))
    
    # populate arrays with analytical solution to autocorrelation function
    for i in range(len(tau_n)):
        t[i] = f**alpha @ list_result[i] @(fstar ** beta * pi)
        t_rev[i] = f**beta @ list_result[i] @(fstar ** alpha * pi)
        
    return t, t_rev

#########################################################################################################################################################################################
# Functions calculating the Steinberg signature (area between higher-order autocorrelation functions)
#########################################################################################################################################################################################

def numerical_area(t,t_rev):
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
    return np.abs(np.trapz(t)-np.trapz(t_rev))

def Laplacian_spectrum(L):
    """
    Returns the eigenvalues, left eigenvectors, and right eigenvectors (otherwise known as the spectrum) of the Laplacian matrix for a 3-vertex graph. The eigenvectors are normalized with the following conditions. 
    
    u_1 (right eigenvector of the 0 eigenvalue): normalized such that all elements sum to 1
    v_1 (left eigenvector of the 0 eigenvalue): normalized such that all elements = 1
    
    u_n, v_n (n>1): normalized such that they satisfy a bi-orthonormal condition (see below)
    
    v_i * u_j = \delta_ij (Kronecker delta function)
    
    Parameters
    ----------

    L : 3x3 array
        column-based Laplacian matrix of 3-vertex linear framework graph
        
    Returns
    -------
    
    lambda_1 : np.complex128
        0 eigenvalue (largest eigenvalue)
    
    u_1 : numpy.ndarray
        right eigenvector for the 0 eigenvalue. Also known as the steady state or invariant distribution of the Laplacian
        
    v_1 : numpy.ndarray
        left eigenvector for the 0 eigenvalue. Also known as the all ones vector
        
    lambda_2 : np.complex128
        smallest eigenvalue (largest negative eigenvalue)
    
    u_2 : numpy.ndarray
        right eigenvector for \lambda_2. Normalized such that v_2 * u_2 = 1
        
    v_2 : numpy.ndarray
        left eigenvector for the \lambda_2. Normalized such that v_2 * u_2 = 1
        
    lambda_3 : np.complex128
        middle eigenvalue
    
    u_3 : numpy.ndarray
        right eigenvector for \lambda_3. Normalized such that v_3 * u_3 = 1
        
    v_3 : numpy.ndarray
        left eigenvector for the \lambda_3. Normalized such that v_3 * u_3 = 1
    
    """
    # Define eigenvalues, left eigenvectors (v_i) and right eigenvectors (u_i)
    eigvals, left_eigvecs, right_eigvecs = scipy.linalg.eig(L, left=True, right=True)
    
    # Identify the index of the 0 eigenvalue and define it as lambda_1
    idx_1 = np.argmax(eigvals)
    lambda_1 = eigvals[idx_1]

    # Delete this eigenvalue from eigvals
    eigvals_other = np.delete(eigvals, idx_1)

    # Define the right u_1 and left v_1 eigenvectors associated with lambda_1
    u_1 = right_eigvecs[:,idx_1]
    v_1 = left_eigvecs[:,idx_1]

    # Normalize u_1 and v_1
    normalization_1 = sum(u_1)
    u_1 = u_1/normalization_1
    v_1 = v_1/v_1

    # Delete eigenvectors u_1 and v_1 from their respective arrays
    right_eigvecs_other = np.delete(right_eigvecs, idx_1, axis=1)
    left_eigvecs_other = np.delete(left_eigvecs, idx_1, axis=1)

    # Identify indexes for second and third eigenvalues
    idx_2 = np.argmin(eigvals_other)
    idx_3 = np.argmax(eigvals_other)
    
    # Define lambda_2 and lambda_3
    lambda_2 = eigvals_other[idx_2]
    lambda_3 = eigvals_other[idx_3]

    # Define left and right eigenvectors for lambda_2 and normalize appropriately
    u_2 = right_eigvecs_other[:,idx_2]
    v_2 = left_eigvecs_other[:,idx_2]
    normalization_2 = np.dot(u_2, v_2)
    u_2 = u_2/normalization_2
    v_2 = v_2/normalization_2

    # Define left and right eigenvectors for lambda_3 and normalize appropriately
    u_3 = right_eigvecs_other[:,idx_3]
    v_3 = left_eigvecs_other[:,idx_3]

    normalization_3 = np.dot(u_3, v_3)
    u_3 = u_3/normalization_3
    v_3 = v_3/normalization_3

    return lambda_1, u_1, v_1, lambda_2, u_2, v_2, lambda_3, u_3, v_3

def analytical_area(lambda_2, lambda_3, u_1, u_2, v_2, u_3, v_3, f, alpha=1, beta=3):
    """
    Compute the closed form area between higher-order autocorrelation functions (a.k.a. the Steinberg signature) for a particular 3x3 Laplacian matrix.
    
    In LaTeX, here is the area formula for the 3-vertex graph:
    
    $$ A^{\alpha,\beta} = \sum_{n>1} \dfrac{\chi^{\beta,\alpha}_n-\chi^{\alpha,\beta}_n}{\lambda_n} = \dfrac{\chi^{\beta,\alpha}_2-\chi^{\alpha,\beta}_2}{\lambda_2} + \dfrac{\chi^{\beta,\alpha}_3-\chi^{\alpha,\beta}_3}{\lambda_3}$$
    
    $$ \chi^{\alpha,\beta}_n = \left ( f^\alpha \cdot u_n \right ) \cdot \left (v_n \cdot f^{\beta *} \pi \right )$$
    
    Parameters
    ----------

    lambda_2 : np.complex128
        smallest eigenvalue (largest negative eigenvalue)
    
    lambda_3 : np.complex128
        middle eigenvalue
        
    u_1 : numpy.ndarray
        right eigenvector for the 0 eigenvalue. Also known as the steady state or invariant distribution of the Laplacian
        
    u_2 : numpy.ndarray
        right eigenvector for \lambda_2. Normalized such that v_2 * u_2 = 1
        
    v_2 : numpy.ndarray
        left eigenvector for the \lambda_2. Normalized such that v_2 * u_2 = 1
        
    u_3 : numpy.ndarray
        right eigenvector for \lambda_3. Normalized such that v_3 * u_3 = 1
        
    v_3 : numpy.ndarray
        left eigenvector for the \lambda_3. Normalized such that v_3 * u_3 = 1
    
    f : 1D array
        possible values of observable (which is a state function on the Markov process)
    
    alpha : scalar (default = 1)
        exponent applied to observable
    
    beta : scalar (default = 3)
        exponent applied to transpose of observable
        
    Returns
    -------
    
    area_ab : scalar
        the closed form area between higher order autocorrelation functions
    
    """
    
    coefficient_2ab = sum((f**alpha)*u_2) * sum(v_2*(f**beta * u_1))
    coefficient_2ba = sum((f**beta)*u_2) * sum(v_2*(f**alpha * u_1))
    
    coefficient_3ab = sum((f**alpha)*u_3) * sum(v_3*(f**beta * u_1))
    coefficient_3ba = sum((f**beta)*u_3) * sum(v_3*(f**alpha * u_1))
    
    term_1 = ((coefficient_2ba - coefficient_2ab)/lambda_2.real)
    term_2 = ((coefficient_3ba - coefficient_3ab)/lambda_3.real)
    
    area_ab = np.abs(((coefficient_2ba - coefficient_2ab)/lambda_2.real) + ((coefficient_3ba - coefficient_3ab)/lambda_3.real))
    
    return area_ab, coefficient_2ab, coefficient_2ba, coefficient_3ab, coefficient_3ba, term_1, term_2

def force_area(omegas, f, alpha=1, beta=3, N=1000):
    """
    Computes a force-area curve for a given inital equilibrium paramterization of the 3-vertex graph.
    
    Parameters
    ----------
    
    omegas : 1D array
        parameter values of rate constants in 3-state Markovian system
        omegas = [a,b,d,c,f,e] = [0,1,2,3,4,5]
    
    f : 1D array
        possible values of observable (which is a state function on the Markov process)
    
    alpha : scalar (default = 1)
        exponent applied to observable
    
    beta : scalar (default = 3)
        exponent applied to transpose of observable
        
    N : integer (default = 1000)
        number of times a particular parameter is perturbed 
    
    Returns
    ----------
    
    forces : 1D array
        values of thermodynamic force of the system as it is perturbed from equilibrium
    
    areas : 1D array
        list of areas computed via the Steinberg signature for each perturbation
    
    param_choice : integer
        index of parameter chosen to perturb from equilibrium
    
    param_changes : 1D array
        list of parameter values assumed by the perturbed parameter
        
    initial_params : 1D array
        equilibrium parameters for the system
    
    """
    
    forces = np.zeros(N)
    areas = np.zeros(N)
    param_changes = np.zeros(N)
    initial_params = omegas
    
    # select a paramter to perturb from equilibrium
    param_choice = np.random.choice(np.arange(0,5))

    for i in range(0,N):
        
        # record the value of the chosen parameter
        param_changes[i] = omegas[param_choice]
        
        # record the thermodynamic force of the system
        forces[i] = cycle_affinity_3state(omegas)
        
        # compute the Laplacian for the current parameter set
        L = Laplacian_3state(omegas)
        
        # obtain the spectrum of the Laplacian
        lambda_1, u_1, v_1, lambda_2, u_2, v_2, lambda_3, u_3, v_3 = Laplacian_spectrum(L)
        
        # compute the closed form area
        areas[i] = analytical_area(lambda_2, lambda_3, u_1, u_2, v_2, u_3, v_3, f, alpha=1, beta=3)
        
        # perturb the chosen parameter
        omegas[param_choice] = omegas[param_choice]*1.01
    
    return forces, areas, param_choice, param_changes, initial_params

def is_noisy(array):
    """
    Determines if the values in an array are noisy.
    
    Args:
        array: The array to be checked. 
    Returns:
        True if the values in the array are noisy, False otherwise.
    """
    # Calculate the standard deviation of the array.
    stddev = np.std(array)
    # Calculate the mean of the array.
    mean = np.mean(array)
    # Check if the standard deviation is greater than a certain threshold.
    # This threshold can be adjusted to control how sensitive the function is to noise.
    if stddev > 0.1:
        return True
    else:
        return False