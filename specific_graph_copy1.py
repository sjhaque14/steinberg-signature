import matplotlib.pyplot as plt
from tqdm import tqdm
from general_graph_utils_old import *
from steinberg_utils_3vertex import *

plt.rc("text", usetex=False)
plt.rc("font", family = "serif",size=14)
plt.rc("figure",figsize=(14,12))
%config InlineBackend.figure_format = 'retina'

def autocorrelation_analytical(observable,L,tau_n,alpha=1,beta=3):
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
    f = np.array([observable])
    fstar = f.T
    
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

G = nx.DiGraph()
G.add_node(1)
G.add_node(2)
G.add_node(3)
G.add_node(4)
G.add_edge(1, 2)
G.add_edge(2, 1)
G.add_edge(2, 3)
G.add_edge(3, 2)
G.add_edge(4, 3)
G.add_edge(3, 4)
G.add_edge(1, 4)
G.add_edge(4, 1)
G.add_edge(2, 4)
G.add_edge(4, 2)

G_ud = nx.Graph()
G_ud.add_node(1)
G_ud.add_node(2)
G_ud.add_node(3)
G_ud.add_node(4)
G_ud.add_edge(1, 2)
G_ud.add_edge(2, 1)
G_ud.add_edge(2, 3)
G_ud.add_edge(3, 2)
G_ud.add_edge(4, 3)
G_ud.add_edge(3, 4)
G_ud.add_edge(1, 4)
G_ud.add_edge(4, 1)
G_ud.add_edge(2, 4)
G_ud.add_edge(4, 2)

node_list = get_nodes(G)
print(node_list)

edge_list = get_edges(G)
print(edge_list)

cycle_list = get_cycle_nodes(G_ud)
num_cycles = len(cycle_list)
print(cycle_list)
print(num_cycles)

label_dict = {(1, 2): 1.0, (1, 4): 1.0, (2, 1): 1.0, (2, 3): 1.0, (2, 4): 1.0, (3, 2): 1.0, (3, 4): 1.0, (4, 1): 1.0, (4, 2): 1.0, (4, 3): 1.0}
label_list = np.fromiter(label_dict.values(), dtype=float)
print(label_dict)
print(label_list)
len(label_list)

cycle_edges_forward,cycle_edges_backward,cycle_labels_forward,cycle_labels_backward = get_cycle_labels_edges(cycle_list,label_dict)

print(cycle_edges_forward)
print(cycle_edges_backward)
print(cycle_labels_forward)
print(cycle_labels_backward)

products_f, products_b = calculate_cycle_products(cycle_labels_forward,cycle_labels_backward)
print(products_f)
print(products_b)

total_affinities = calculate_affinities(products_b, products_f, cycle_list)
print(total_affinities)

L = Laplacian_all(edge_list, label_list, node_list)
print(L)

eigvals, eigvecs = scipy.linalg.eig(L)
pi = np.array([eigvecs[:,np.argmin(np.abs(eigvals))].real/sum(eigvecs[:,np.argmin(np.abs(eigvals))].real)]).T
print(pi)
print(eigvals)

cycle_idx = 0
edge_idx = 2

param_choice = cycle_labels_forward[cycle_idx][edge_idx]
print(param_choice)
edge = cycle_edges_forward[cycle_idx][edge_idx]
print(edge)

observable = make_observable(node_list)
alpha, beta = 1, 3
tau_n = np.arange(start=0.01,stop=20.0,step=0.01)

N = 1000
all_affinities = np.zeros((N,num_cycles))
all_areas = np.zeros((N,1))
param_changes = np.zeros((N,1))

for i in tqdm(range(0,N)):
    
    # calculate affinity + record
    products_f, products_b = calculate_cycle_products(cycle_labels_forward,cycle_labels_backward)
    total_affinities = calculate_affinities(products_f,products_b,cycle_list)
    all_affinities[i] = total_affinities
        
    # calculate area + record
    label_dict, label_list = reformat_labels(cycle_list, cycle_labels_forward, edge_tracker, label_dict, label_list)
    L = Laplacian_all(edge_list,label_list,node_list)
    t, t_rev = autocorrelation_analytical(observable,L,tau_n,alpha=1,beta=3)
    all_areas[i] = np.abs(np.trapz(t)-np.trapz(t_rev))
        
    # perturb parameter + record
    cycle_labels_forward[cycle_idx][edge_idx] = cycle_labels_forward[cycle_idx][edge_idx]*1.01
    


