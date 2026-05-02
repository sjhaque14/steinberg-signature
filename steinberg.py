import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx
import scipy.linalg

# rounding results to 4 significant figures

def round_sig(x, sig_figs=4):
    """
    Rounds a number or array to a given number of significant figures.
    """
    def _round(x):
        if x == 0:
            return 0.0
        return round(x, sig_figs - int(np.floor(np.log10(abs(x)))) - 1)
    
    vect = np.vectorize(_round)
    return vect(x).item() if np.ndim(x) == 0 else vect(x)

def random_parameters(min_val=-3,max_val=3,num_params=6,sig_figs=4):
    """
    Randomly samples transition rates for any graph which do not necessarily satisfy detailed balance. These parameters are defined as 10^x, where x is randomly drawn from the uniform distribution on (min_val, max_val).
    
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
             transition rates
    """
    
    params = np.zeros(num_params)
    
    params[:] = 10**(np.random.uniform(min_val,max_val, size = num_params))
    
    return round_sig(params, sig_figs)


# computes cycle affinity for any graph (labels_f and labels_r are computed as lists of lists corresponding to distinct cycles)

def cycle_affinity(labels_f,labels_r):
    """
    Calculates the cycle affinity for any graph.
    """
    aff = np.abs(np.log(np.prod(labels_f)/np.prod(labels_r))).item()
    return aff


# get steady state distribution directly from Laplacian eigenvalues

def pi_dist(lap):
    """
    Computes the steady-state distribution directly from the spectrum of a given Laplacian matrix. This function works for a linear framework graph of any size.
    """
    eigvals, eigvecs = scipy.linalg.eig(lap)
    v = eigvecs[:, np.argmin(np.abs(eigvals))].real
    pi = v / v.sum()
    return pi

# set a range for the tau values to compute autocorrelation functions

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
    tau = np.linspace(0.0, tau_max, num=max_points)
    return tau, tau_max

# analytical autocorrelation function from Eq. 21 in paper

def asymmetric_autocorrelation(signal,lap,tau,alpha=1,beta=3):
    """
    Numerically calculates the asymmetric autocorrelation functions A^{1,3}(tau) and A^{3,1}(tau) for a particular Laplacian matrix. This function works for a linear framework graph of any size.
    
    Parameters
    ----------
    signal : 1D array
        vector of possible values of signal S = (S(1), ..., S(N))
        
    lap : NxN array
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
    pi = np.array(pi_dist(lap))
    delta_u_star = np.diag(pi)
    
    # vectorize the Laplacian matrix multiplied by each value in the vector tau
    list_result = list(map(lambda i: scipy.linalg.expm(lap*i), list(tau)))
    
    # populate arrays with analytical solution to autocorrelation function
    for i in range(len(tau)):
        a_13[i] = ((s_t**beta) @ list_result[i]) @ (delta_u_star @ (s ** alpha))
        a_31[i] = ((s_t**alpha) @ list_result[i]) @ (delta_u_star @ (s ** beta))
        
    return a_13, a_31
    
# numerical area calculation (trapezoidal integration)

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

def numerical_area(t, t_rev, tau):
    """
    Calculates the area between asymmetric autocorrelation functions using Numpy's trapezoidal area formular
    
    Parameters
    ----------
    t : 1D array
        forward autocorrelation function values
    t_rev : 1D array
        reverse autocorrelation function values
    tau : 1D array
        range of intervals between values of signal taken by system
    
    Returns
    -------
    area : 1D array
        numerical area between t and t_rev
    """
    area = np.abs(np.trapezoid(t, tau)-np.trapezoid(t_rev, tau))
    return area.item()

## analytical area calculation ##

def spectrum_any(lap, tol=1e-12):
    """
    Compute the full spectrum of the Laplacian matrix

    Parameters
    ----------
    lap : NxN array
        column-based Laplacian matrix of linear framework graph with N vertices
    
    Returns
    -------
    lambdas : 1D array
        sorted array of eigenvalues of lap from smallest to largest

    w_i : NxN array
        sorted array of left eigenvectors of lap (each eigenvector is a row)

    z_i : NxN array
        sorted array of right eigenvectors of lap (each eigenvector is a column)
    
    """
    symmetric = np.allclose(lap, lap.T, atol=tol)
    
    eigvals_u, l_eigvecs_u, r_eigvecs_u = scipy.linalg.eig(lap, left=True, right=True)

    if symmetric:
        # Clean up tiny numerical imaginary parts for symmetric case
        eigvals_u = eigvals_u.real
        r_eigvecs_u = r_eigvecs_u.real
        l_eigvecs_u = l_eigvecs_u.real
        
    idx = np.argsort(eigvals_u.real)
    lambdas = eigvals_u[idx]
    z_i = r_eigvecs_u[:, idx]
    l_eigvecs_u = l_eigvecs_u[:, idx]

    # Left eigenvectors as rows (conjugate transpose)
    w_i = l_eigvecs_u.conj().T
    
    return lambdas, w_i, z_i

def normalization_factors(w_i, z_i):
    """
    Compute the normalization factors for left and right eigenvectors of lap
    """
    r_i = np.zeros(z_i.shape[1], dtype=complex)  # one factor per eigenpair
    for k in range(z_i.shape[1]):
        r_i[k] = w_i[k,:] @ z_i[:, k]
    return r_i

def projection_matrices(w_i, z_i, r):
    """
    Compute projection matrices
    """
    N = z_i.shape[0]
    m = z_i.shape[1]  # number of eigenpairs
    Lk_list = []
    for k in range(m):
        zk = z_i[:, [k]]   # column vector (N x 1)
        wk = w_i[[k], :]   # column vector (N x 1)
        Lk = (1 / r[k]) * (zk @ wk)  # outer product gives N x N
        Lk_list.append(Lk)
    return Lk_list

def B_matrix(lambdas, Lk_list, delta_u_star):
    """
    Compute B(G) matrix
    """
    N = delta_u_star.shape[0]
    # skip k=0 since lambdas[0] is zero for Laplacian
    Bsum = sum((1/lambdas[k]) * Lk_list[k] for k in range(0, N-1))
    return Bsum @ delta_u_star

def skew_symmetric_area(signal, B, alpha=1,beta=3):
    """
    Computes the Steinberg signature / area for given vectors S_alpha, S_beta and the operator B(G).
    """
    # define the signal vectors
    s_t = np.array([signal],dtype=float) # row vector
    s = s_t.T # column vector
    
    # Form the skew-symmetric combination
    B_skew = B - B.T
    area = (s_t**beta) @ B_skew @ (s ** alpha)
    return area

def steinberg_analytical_area(signal,lap,alpha=1,beta=3):
    """
    Compute the analytical area between two autocorrelation functions from Eq. 33
    """
    lambdas, w_i, z_i = spectrum_any(lap)
    r_i = normalization_factors(w_i, z_i)
    Lk_list = projection_matrices(w_i, z_i, r_i)
    pi = np.array(pi_dist(lap))
    delta_u_star = np.diag(pi)
    B = B_matrix(lambdas, Lk_list, delta_u_star)
    area = skew_symmetric_area(signal, B, alpha=1,beta=3)
    return np.abs(area.real.item())

## Sanity checks ##

def is_zero_at_eq(lap, signal, alpha=1, beta=3,tol=1e-6):
    """
    Equilibrium check: is the area zero when the affinity is zero?
    """
    
    area = steinberg_analytical_area(signal,lap,alpha,beta)
    
    if np.abs(area) > tol:
        raise ValueError(
            f"Steinberg signature is non-zero at equilibrium: {area:.6e}\n"
            f"Check equilibrium initialization and B_matrix range."
        )
    return True

def check_Lk(lap):
    "Sanity check: Does the sum of all L^k matrices = I?"
    lambdas, w_i, z_i = spectrum_any(lap)
    r_i = normalization_factors(w_i, z_i)
    Lk_list = projection_matrices(w_i, z_i, r_i)
    N = len(Lk_list)
    if np.allclose(sum(Lk_list), np.eye(N)):
        return True

def check_analytical_numerical_consistency(signal, lap, tau, alpha=1, beta=3, 
                                           rtol=1e-2, atol=1e-2):
    analytical = steinberg_analytical_area(signal, lap, alpha, beta)
    a_13, a_31 = asymmetric_autocorrelation(signal, lap, tau, alpha, beta)
    numerical = numerical_area(a_13, a_31, tau)
    
    if not np.isclose(analytical, numerical, rtol=rtol, atol=atol):
        raise ValueError(
            f"Analytical and numerical Steinberg signatures disagree:\n"
            f"  analytical = {analytical:.6e}\n"
            f"  numerical  = {numerical:.6e}\n"
            f"  absolute error = {np.abs(analytical - numerical):.3e}"
        )
        return True


