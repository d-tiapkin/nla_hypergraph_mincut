import numpy as np
import itertools


def is_cutting_partition(edge, k):
    flag_min = True
    flag_max = True
    for v in edge:
        if v >= k:
            flag_min = False
        if v < k:
            flag_max = False
    return flag_max or flag_min


def generate_H_edges(n, k, p, q):
    size = [n for i in range(k)]
    H = np.zeros(size)
    edges = []
    for edge in itertools.combinations(range(n), k): 
        # the partition is like this: first n / 2 vertices vs last n/2 vertices.
        if is_cutting_partition(edge, n//2):
            coin = np.random.binomial(1, p)
        else:
            coin = np.random.binomial(1, q)
        if coin:
            edges.append(edge)
    return np.array(edges)


def generate_H_matrix(n, k, p, q):
    size = [n for i in range(k)]
    H = np.zeros(size)
    for edge in itertools.combinations(range(n), k): 
        # the partition is like this: first n / 2 vertices vs last n/2 vertices.
        if is_cutting_partition(edge, n/2):
            coin = np.random.binomial(1, p)
        else:
            coin = np.random.binomial(1, q)
        if coin:
            for subedge in itertools.permutations(edge):
                H[subedge] = 1
    return H