import numpy as np
from numba import jit

@jit(nopython=True)
def sample_non_edges_undirected_numba(n_nodes, edge_array, n_samples):
    """Fast Numba sampler for UNDIRECTED graphs"""
    edge_set = set()
    for i in range(edge_array.shape[0]):
        u, v = edge_array[i, 0], edge_array[i, 1]
        # For undirected, we assume the input array is already sorted,
        # but creating the set handles it definitively.
        if u > v:
            u, v = v, u
        edge_set.add((u, v))

    non_edges = []
    attempts = 0
    max_attempts = n_samples * 10

    while len(non_edges) < n_samples and attempts < max_attempts:
        u = np.random.randint(0, n_nodes)
        v = np.random.randint(0, n_nodes)
        if u == v:
            attempts += 1
            continue
        
        # Canonical representation for undirected lookup
        if u > v:
            u, v = v, u
            
        if (u, v) not in edge_set:
            non_edges.append((u, v))
        attempts += 1

    result = np.empty((len(non_edges), 2), dtype=np.int32)
    for i, (u, v) in enumerate(non_edges):
        result[i, 0] = u
        result[i, 1] = v
    return result

@jit(nopython=True)
def sample_non_edges_directed_numba(n_nodes, edge_array, n_samples):
    """Fast Numba sampler for DIRECTED graphs"""
    edge_set = set()
    for i in range(edge_array.shape[0]):
        u, v = edge_array[i, 0], edge_array[i, 1]
        edge_set.add((u, v))

    non_edges = []
    attempts = 0
    max_attempts = n_samples * 10

    while len(non_edges) < n_samples and attempts < max_attempts:
        u = np.random.randint(0, n_nodes)
        v = np.random.randint(0, n_nodes)
        if u == v:
            attempts += 1
            continue
        
        # Direct lookup for directed graphs
        if (u, v) not in edge_set:
            non_edges.append((u, v))
        attempts += 1

    result = np.empty((len(non_edges), 2), dtype=np.int32)
    for i, (u, v) in enumerate(non_edges):
        result[i, 0] = u
        result[i, 1] = v
    return result