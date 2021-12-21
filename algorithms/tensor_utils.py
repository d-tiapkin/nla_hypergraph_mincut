import numpy as np
import numba

@numba.jit(nopython=True)
def L_vec_value(x, edges, weights, rt_degree):
    result = 0
    m, r = edges.shape
    for it_e in range(m):
        for i in range(r):
            for j in range(r):
                v_i = edges[it_e, i]
                v_j = edges[it_e, j]
                result += weights[it_e] * (x[v_i] / rt_degree[v_i] - x[v_j] / rt_degree[v_j]) ** r
    return result


def L_value(X, edges, weights, rt_degree):
    k = X.shape[1]
    return np.sum( [L_vec_value(X[:,i], edges, weights, rt_degree) for i in range(k)]) 


@numba.jit(nopython=True)
def L_vec_grad(x, edges, weights, rt_degree):
    result = np.zeros_like(x)
    m, r = edges.shape
    for it_e in range(m):
        for i in range(r):
            for j in range(r):
                v_i = edges[it_e, i]
                v_j = edges[it_e, j]
                dot_prod = weights[it_e] * (x[v_i] / rt_degree[v_i] - x[v_j] / rt_degree[v_j]) ** (r-1)
                result[v_i] += dot_prod / rt_degree[v_i]
                result[v_j] -= dot_prod / rt_degree[v_j]
    return result


def L_grad(X, edges, weights, rt_degree):
    k = X.shape[1]
    r = edges.shape[1]
    return r * np.array([L_vec_grad(X[:,i], edges, weights, rt_degree) for i in range(k)]).T
