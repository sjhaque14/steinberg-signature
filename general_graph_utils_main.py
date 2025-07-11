import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
from collections import Counter

# general-graph-utils.py

# This library allows the user to greate linear framework graphs (finite, directed graphs with no self-loops and labeled edges). The user can randomly generate a strongly connected and fully reversible linear framework graph G, calculate its Laplacian matrix and its spectrum, and calculate the Steinberg signature from that matrix. The user can also determine how the Steinberg signature changes as a function of increasing entropy production. This software was developed using the NetworkX software package (https://networkx.org/documentation/stable/index.html)

# Note that the user is required to create both a directed graph object and an undirected graph object. This is because some of the functions in this file require the undirected graph object as an argument (particular the cycle-related functions).

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

def random_graph(n):
    """
    Generates a linear framework graph that is strongly connected and fully reversible. The size of the graph is randomly determined from range (3, n), and the edges are added by randomly selecting a pair of nodes in G.
    
    Parameters
    ----------
    n : integer > 3
        the maximum number of vertices
    
    Returns
    -------
    G : NetworkX DiGraph object
        directed graph
    
    G_ud : NetworkX Graph object
        undirected graph
    """
    
    G = nx.DiGraph()
    G_ud = nx.Graph()
    
    # determine the number of nodes in G
    G_size = np.random.choice(np.arange(3,n), size=1)
    
    # add nodes to G and G_ud
    nodes = np.arange(1,G_size+1,step=1)
    G.add_nodes_from(nodes)
    G_ud.add_nodes_from(nodes)
    
    # add edges until the graph is strongly connected
    while nx.is_strongly_connected(G) == False:
        u, v = np.random.choice(nodes, size=2,replace=False)
        G.add_edge(u,v)
        G_ud.add_edge(u,v)
    
    # add edges such that the graph is fully reversible
    for edge in G.edges:
        u = edge[0]
        v = edge[1]
        if G.has_edge(v,u) == False:
            G.add_edge(v,u)
            G_ud.add_edge(v,u)
    
    return G, G_ud

def random_graph_n(n):
    """
    Generates a linear framework graph of size n that is strongly connected and fully reversible.
    
    Parameters
    ----------
    n : integer
        the size of the graph
    
    Returns
    -------
    G : NetworkX DiGraph object
        directed graph
    
    G_ud : NetworkX Graph object
        undirected graph
    """
    
    G = nx.DiGraph()
    G_ud = nx.Graph()
    
    # determine the number of nodes in G
    G_size = n
    
    # add nodes to G and G_ud
    nodes = np.arange(1,G_size+1,step=1)
    G.add_nodes_from(nodes)
    G_ud.add_nodes_from(nodes)
    
    # add edges until the graph is strongly connected
    while nx.is_strongly_connected(G) == False:
        u, v = np.random.choice(nodes, size=2,replace=False)
        G.add_edge(u,v)
        G_ud.add_edge(u,v)
    
    # add edges such that the graph is fully reversible
    for edge in G.edges:
        u = edge[0]
        v = edge[1]
        if G.has_edge(v,u) == False:
            G.add_edge(v,u)
            G_ud.add_edge(v,u)
    
    return G, G_ud

def get_nodes(G):
    """
    Returns an array of nodes in a NetworkX graph object (directed or undirected)
    
    Parameters
    ----------
    G : NetworkX DiGraph object
        directed graph
    
    Returns
    -------
    node_list : NumPy array
        list of nodes
    """
    node_list = np.array(G.nodes)
    return node_list

def get_edges(G):
    """
    Returns a list of edges in a NetworkX DiGraph, each as a (source, sink) tuple.
    
    Parameters
    ----------
    G : networkx.DiGraph
        Directed graph object.
    
    Returns
    -------
    edge_list : list of tuple
        Each tuple is (source, sink).
    """
    return list(G.edges())

def get_cycles(G_ud):
    """
    Returns a list of the cycles which form a basis G (must be undirected). Each element is a list of the nodes connected in a given cycle.
    
    Parameters
    ----------
    G_ud : NetworkX Graph object
        undirected graph
            
    Returns
    -------
    cycle_list : list of lists
        each element is a list of the nodes connected in a given cycle.
    """
    
    cycle_list = [c for c in nx.cycle_basis(G_ud)]
    
    return cycle_list

def get_labels(G):
    """
    Extracts the label information for each edge in a NetworkX graph object. If G not labeled, labels are sampled as 10^x, where x is sampled uniformly between -3 and 3.
    
    Parameters
    ----------
    G : NetworkX DiGraph object
        directed graph
            
    Returns
    -------
    label_dict : dictionary
        keys: edges in G represented as tuple (source,sink), values: edge labels
        
    label_list : 1D numpy array
        list of edge labels in G
        
    """
    
    label_dict = {}
    
    # if G weighted/labeled, extract weight information in dictionary form
    if nx.is_weighted(G)==True:
        for i in range(len(G.edges())):
            label_dict[list(G.edges())[i]] = G.get_edge_data(list(G.edges)[i][0],list(G.edges)[i][1])['weight']
    
    # if G not weighted/labeled, sample new edge label for each edge
    elif nx.is_weighted(G)==False:
        label_dict = {e: round_sig(10**(np.random.uniform(-3,3, size = 1)[0]),sig_figs=4) for e in G.edges}
    
    # create a list of edge labels directly from the dictionary
    label_list = np.fromiter(label_dict.values(), dtype=float)
    
    return label_dict, label_list

def get_labels_ones(G):
    """
    Extracts the label information for each edge in a NetworkX graph object. If G not labeled, all labels are assigned as 1.0.
    
    Parameters
    ----------
    G : NetworkX DiGraph object
        directed graph
            
    Returns
    -------
    label_dict : dictionary
        keys: edges in G represented as tuple (source,sink), values: edge labels
        
    label_list : 1D numpy array
        list of edge labels in G
        
    """
    
    label_dict = {}
    
    # if G weighted/labeled, extract weight information in dictionary form
    if nx.is_weighted(G)==True:
        for i in range(len(G.edges())):
            label_dict[list(G.edges())[i]] = G.get_edge_data(list(G.edges)[i][0],list(G.edges)[i][1])['weight']
    
    # if G not weighted/labeled, sample new edge label for each edge
    elif nx.is_weighted(G)==False:
        label_dict = {e: 1.0 for e in G.edges}
    
    # create a list of edge labels directly from the dictionary
    label_list = np.fromiter(label_dict.values(), dtype=float)
    
    return label_dict, label_list

def get_cycle_labels_edges(cycle_list,label_dict):
    """
    Compartmentalizes, for each cycle, the edges involved and their respective edge labels into separate data structures.
    
    Parameters
    ----------    
    cycle_list : list of lists
        each element is a list of the nodes connected in a given cycle.
        
    label_dict : dictionary
        keys: edges in G represented as tuple (source,sink), values: edge labels
        
    Returns
    -------
    cycle_edges_forward : list of lists
        each element is a list of the edges going around one direction of a given cycle
    
    cycle_edges_backward : list of lists
        each element is a list of the edges going around the opposite direction of a given cycle
    
    cycle_labels_forward : list of lists
        each element is a list of the labels going around one direction of a given cycle
    
    cycle_labels_backward : list of lists
        each element is a list of the labels going around the opposite direction of a given cycle
        
    """
    
    # define number of cycles
    num_cycles = len(cycle_list)
    
    # define arrays for edges in each direction
    cycle_edges_forward = [[] for i in range(num_cycles)]
    cycle_edges_backward = [[] for i in range(num_cycles)]
    
    # define arrays for labels in each direction
    cycle_labels_forward = [[] for i in range(num_cycles)]
    cycle_labels_backward = [[] for i in range(num_cycles)]

    # iterate over each cycle
    for j in range(num_cycles):
        
        # for each node in the cycle
        for i in range(1,len(cycle_list[j])):
            
            # identify what it is connected to
            source = cycle_list[j][i-1]
            sink = cycle_list[j][i]
            
            # edge (source,sink) and accompanying labels
            cycle_labels_forward[j].append(label_dict.get((source,sink)))
            cycle_edges_forward[j].append((source,sink))
            
            #edge (sink, source) and accompanying labels
            cycle_labels_backward[j].append(label_dict.get((sink,source)))
            cycle_edges_backward[j].append((sink,source))
        
        # account for the connection between the last and first elements of each cycle list
        final_source = cycle_list[j][-1]
        final_sink = cycle_list[j][0]

        # edge (final_source, final_sink) and accompanying labels
        cycle_labels_forward[j].append(label_dict.get((final_source,final_sink)))
        cycle_edges_forward[j].append((final_source,final_sink))
        
        # edge (final_sink, final source) and accompanying labels
        cycle_labels_backward[j].append(label_dict.get((final_sink,final_source)))
        cycle_edges_backward[j].append((final_sink,final_source))
        
    return cycle_edges_forward, cycle_edges_backward, cycle_labels_forward, cycle_labels_backward

def calculate_cycle_products(cycle_labels_forward,cycle_labels_backward):
    """
    Calculates the product of edge labels going in forward and reverse directions for each cycle
    
    Parameters
    ----------
    cycle_labels_forward : list of lists
        each element is a list of the labels going around one direction of a given cycle
    
    cycle_labels_backward : list of lists
        each element is a list of the labels going around the opposite direction of a given cycle
        
    Returns
    -------
    products_f : 1D array
        each element is the product of labels corresponding to the forward traversal of each cycle
    
    products_b : 1D array
        each element is the product of labels corresponding to the backward traversal of each cycle
        
    """
    products_f = np.zeros(len(cycle_labels_forward))
    products_b = np.zeros(len(cycle_labels_backward))

    for i in range(len(cycle_labels_forward)):
        products_f[i] = np.prod(cycle_labels_forward[i])
        products_b[i] = np.prod(cycle_labels_backward[i])
    
    return products_f, products_b

def calculate_affinities(products_f, products_b, cycle_list):
    """
    Calculates the cycle affinity (e.g. thermodynamic force) for each cycle in a graph
    
    Parameters
    ----------
    products_f : 1D array
        each element is the product of labels corresponding to the forward traversal of each cycle
    
    products_b : 1D array
        each element is the product of labels corresponding to the backward traversal of each cycle
        
    cycle_list : list of lists
        each element is a list of the nodes connected in a given cycle.
        
    Returns
    -------
    
    total_affinities : 1D array
        each element is the thermodynamic force for each cycle in the graph, corresponding to their order in cycle_list
    
    """
    
    num_cycles = len(cycle_list)
    
    total_affinities = np.zeros(num_cycles)
    
    for i in range(num_cycles):
        total_affinities[i] = np.log(products_f[i]/products_b[i])
    
    return total_affinities

def shared_edges_cycles(cycle_list, 
                        cycle_edges_forward, 
                        cycle_edges_backward):
    """
    Returns a list of all edges that are mutual to more than one cycle in G
    
    Parameters
    ----------
    cycle_list : list of lists
        each element is a list of the nodes connected in a given cycle.
        
    cycle_edges_forward : list of lists
        each element is a list of the edges going around one direction of a given cycle
    
    cycle_edges_backward : list of lists
        each element is a list of the edges going around the opposite direction of a given cycle
        
    Returns
    ----------
    shared_cycle_edges_list : list
        list of all the pairs of reversible edges that are shared between at least 2 cycles in G
    
    all_cycle_edges_forward : list
        list of all the edge tuples that are recorded in cycle_edges_forward
    
    """
    all_cycle_edges = []
    all_cycle_edges_forward = []

    for i in range(len(cycle_list)):
        for j in range(len(cycle_list[i])):
            all_cycle_edges.append(cycle_edges_forward[i][j])
            all_cycle_edges.append(cycle_edges_backward[i][j])
            all_cycle_edges_forward.append(cycle_edges_forward[i][j])
    
    shared_cycle_edges_dict = Counter(all_cycle_edges)
    
    shared_cycle_edges_list = [edge for edge, count in shared_cycle_edges_dict.items() if count >= 2]
    
    return shared_cycle_edges_list,all_cycle_edges_forward

def equilibrium_params(cycle_list,
                         cycle_edges_forward,
                         cycle_labels_forward,
                         cycle_labels_backward,
                         shared_cycle_edges_list):
    """
    Zero the affinity of each fundamental cycle by solving one unshared edge label.
    Returns updated forward labels plus trackers.
    """
    edge_tracker = []
    index_tracker = []

    for i in range(len(cycle_list)):
        # 1) Find candidate forward edges for this cycle:
        candidates = [
            (j, e) for j, e in enumerate(cycle_edges_forward[i])
            if e not in shared_cycle_edges_list and e not in edge_tracker
        ]
        if not candidates:
            raise RuntimeError(f"No available edge to fix cycle {i}")

        # 2) Pick the first candidate (or use random.choice(candidates) if you prefer)
        j, edge = candidates[0]
        old_label = cycle_labels_forward[i][j]

        # 3) Compute the original products for this cycle
        f0 = np.prod(cycle_labels_forward[i])
        b0 = np.prod(cycle_labels_backward[i])

        # 4) Solve for the new forward‐label so that f'/b' = 1
        #    new_label = b0 / (f0 / old_label)
        new_label = b0 / (f0 / old_label)

        # 5) Update *only* the forward label
        cycle_labels_forward[i][j] = new_label

        # 6) Track what we changed
        edge_tracker.append(edge)
        index_tracker.append((i, j))

        # 7) Diagnostic check: should be zero (within machine precision)
        f1 = np.prod(cycle_labels_forward[i])
        b1 = np.prod(cycle_labels_backward[i])
        affinity = np.log(f1 / b1)
        print(f"Cycle {i} affinity after solve:", affinity)

    return cycle_labels_forward, edge_tracker, index_tracker

def reformat_labels(edge_tracker,
                    index_tracker,
                    cycle_labels_forward,
                    label_dict,
                    edge_order):
    """
    Update label_dict with the new forward labels you solved for,
    and rebuild a NumPy label list in the given edge_order.
    """
    
    # 1) Apply solved labels back into the dict
    for (cycle_i, edge_j), edge in zip(index_tracker, edge_tracker):
        label_dict[edge] = cycle_labels_forward[cycle_i][edge_j]

    # 2) Rebuild the label list in a consistent order
    label_list = np.array([ label_dict[edge] for edge in edge_order ], dtype=float)

    return label_dict, label_list

def Laplacian_all(edge_list, label_list, node_list):
    """
    Builds the column-based Laplacian L for a directed graph G, given:
      - edge_list: list of (source, sink) tuples
      - label_list: list of weights for each edge, aligned with edge_list
      - node_list: list/array of the nodes in G (in consistent order)

    L_ij =  k_{ij}   if i≠j  (rate from j→i)
    L_jj = -sum_{i≠j} L_ij

    Parameters
    ----------
    edge_list : sequence of tuples
        Each tuple is (source, sink).
    label_list : 1D array
        Rates for each directed edge, same length and order as edge_list.
    node_list : sequence
        All nodes of G, in the order you want for rows/cols of L.

    Returns
    -------
    L : np.ndarray, shape (n, n)
        The column-based generator (Laplacian) matrix.
    """
    n = len(node_list)
    # Map node label → index in [0, n)
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}

    L = np.zeros((n, n), dtype=float)

    # Fill off-diagonals: for each edge (u→v), add rate k_uv at L[v_index, u_index]?
    # Note: In many CTMC conventions L_{ij} is rate from j→i; adjust if you use L_{ij}=k_{ij}.
    for (u, v), k in zip(edge_list, label_list):
        i = node_to_idx[v]
        j = node_to_idx[u]
        L[i, j] = k

    # Fill diagonals so that each column sums to zero
    # (i.e. L[j, j] = -sum_{i≠j} L[i, j] for each j)
    col_sums = L.sum(axis=0)
    for j in range(n):
        L[j, j] = -col_sums[j]

    return L

def steady_state_spectrum(L):
    """
    Calculates the steady-state distribution for the any linear framework graph by computing the right eigenvector associated with eigenvalue 0 and normalizing it by the sum of all entries
    
    Parameters
    ----------
    L : num_nodes x num_nodes array
        the Laplacian matrix of the graph G
    
    Returns
    -------
    pi : 1D array
         the steady state distribution for a 3-vertex graph K.
    
    """
    
    eigvals, eigvecs = scipy.linalg.eig(L)
    pi_all = eigvecs[:,np.argmin(np.abs(eigvals))].real/sum(eigvecs[:,np.argmin(np.abs(eigvals))].real)
    
    return pi_all

def G_duplicate_data_structures(G,node_list,edge_list,label_list,label_dict,size,L):
    """
    Creates new data structures for a graph G, in case you want to manipulate a graph's topology
    """
    
    node_list_G = node_list
    edge_list_G = edge_list
    label_list_G = label_list
    label_dict_G = label_dict
    size_G = size
    L_G = L
    
    return node_list_G, edge_list_G, label_list_G, label_dict_G, size_G, L_G

def assign_labels(G, label_dict):
    """
    Adds labels to a graph object
    
    Parameters
    ----------
    G : NetworkX graph object (directed)
    
    label_dict : dictionary
        keys: edges in G represented as tuple (source,sink), values: edge labels
        
    Returns
    -------
    
    G : with labels
        
    """
    for e in G.edges:
        G[e[0]][e[1]]['weight'] = label_dict[e]
        
    return G

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
    # define the signal vectors
    s_t = np.array([signal]) # row vector
    s = s_t.T # column vector
    
    # create the diagonal steady state matrix 
    delta_u_star = np.diag(pi)
    
    # vectorize the Laplacian matrix multiplied by each value in the vector tau
    list_result = list(map(lambda i: scipy.linalg.expm(L*i), tau))
    
    # populate arrays with analytical solution to autocorrelation function
    for i in range(len(tau)):
        a_13[i] = (s_t**beta @ list_result[i]) @ (delta_u_star @ s ** alpha)
        a_31[i] = (s_t**alpha @ list_result[i]) @ (delta_u_star @ s ** beta)
        
    return a_13, a_31