import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx
import scipy.linalg

plt.rc("text", usetex=False)
plt.rc("font", family = "serif",size=14)
plt.rc("figure",figsize=(12,10))
%config InlineBackend.figure_format = 'retina'

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
    # labels_f = k12, k23, k34, k41
    k12, k23, k34, k41 = labels_f[0], labels_f[1], labels_f[2], labels_f[3]
    # labels_r = k21, k32, k43, k14
    k21, k32, k43, k14 = labels_r[0], labels_r[1], labels_r[2], labels_r[3]
    
    lap = np.array([[-k14-k12, k21, 0, k41], [k12, -k21-k23, k32, 0], [0, k23, -k32-k34, k43], [k14, 0, k34, -k41-k43]],dtype=float)
    
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

# most reliable version of the autocorrelation function

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

###### SIMULATION LOOP #######

# equilibrium parameters

labels_f_eq, labels_r_eq = eq_params_k4()
print(labels_f_eq)
print(labels_r_eq)

labels_f_eq1 = labels_f_eq.copy()
labels_r_eq1 = labels_r_eq.copy()

aff = np.abs(np.log(np.prod(labels_f_eq)/np.prod(labels_r_eq)))
print(aff)

labels_f = labels_f_eq1.copy()
labels_r = labels_r_eq1.copy()

# autocorr settings
signal = [3,5,7,9]
alpha, beta  = 1, 3

# arrays for tracking aff, area, and eigs
N = 2000
affinities = np.empty(N, dtype=float)
areas = np.empty(N, dtype=float)
eigvals_all_r = np.zeros((N, 4), dtype=np.float128)
eigvals_all_c = np.zeros((N, 4), dtype=np.float128)

# edge to perturb
edge_idx = 1
perturb_edge = labels_f[edge_idx] # labels_f[1] = k23

for i in tqdm(range(0,N)):
    
    labels_f[edge_idx] = perturb_edge
    
    lap = lap_k4(labels_f, labels_r)
    
    aff = np.abs(np.log(np.prod(labels_f)/np.prod(labels_r)))
    affinities[i] = aff
    
    tau, _ = define_tau_range(lap, max_points=500, cap_factor=10.0)
    a13, a31 = autocorrelation_analytical(signal,lap,tau,alpha=1,beta=3)
    areas[i] = np.abs(np.trapz(a13)-np.trapz(a31))
    
    # Track eigenvalues (real part)
    eigvals = scipy.linalg.eigvals(lap)
    eigvals_all_r[i, :] = np.sort(eigvals.real)
    eigvals_all_c[i, :] = np.sort(eigvals.imag)
    
    perturb_edge *= 1.01
    
# Steinberg curve plot
plt.plot(np.abs(affinities),areas,label=r'$\mathcal{I}^{1,3}(G)$',linewidth=1, color= 'black')
plt.xlabel(r"$\tilde{A}(C)$")
plt.ylabel(r"$\mathcal{I}^{1,3}(G)$")
plt.show()

# Eigenvalues plots
plt.figure(figsize=(12,5))

# Real part
plt.subplot(1,2,1)
for j in range(4):
    plt.plot(affinities, eigvals_all_r[:, j], label=f"Re(λ{j+1})")
plt.xlabel("Affinity")
plt.ylabel("Re(λ)")
plt.title("Real Parts of Eigenvalues")
plt.legend()

# Imaginary part
plt.subplot(1,2,2)
for j in range(4):
    plt.plot(affinities, eigvals_all_c[:, j], label=f"Im(λ{j+1})")
plt.xlabel("Affinity")
plt.ylabel("Im(λ)")
plt.title("Imaginary Parts of Eigenvalues")
plt.legend()

plt.tight_layout()
plt.savefig(f"eigenvalue_evolution_edge_{edge_idx}.png", dpi=300)
plt.show()
