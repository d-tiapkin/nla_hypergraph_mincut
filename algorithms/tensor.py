import numba
import numpy as np
import algorithms.tensor_utils as t_utils
from sklearn.cluster import KMeans


def step(X, edges, weights, rt_degree, tau, rho=0.25, beta=0.5, delta=1e-3):
    n,k = X.shape

    G = t_utils.L_grad(X, edges, weights, rt_degree)
    nabla_f = G - X @ G.T @ X
    E = (np.eye(n) - (1 - 2 * rho) * X @ X.T) @ nabla_f
    W = - (np.eye(n) - X @ X.T) @ E

    value = t_utils.L_value(X, edges, weights, rt_degree)
    der_f_tau = - np.trace(G.T @ E)

    # Initial value of step-size
    tau /= beta

    J_tau = np.eye(k) + tau**2/4 * W.T @ W + tau/2 * X.T @ E
    Y_tau = (2 * X + tau * W) @ np.linalg.inv(J_tau) - X
    # Armijo rule for step-size
    for _ in range(20):
        value_new = t_utils.L_value(Y_tau, edges, weights, rt_degree)
        if value_new <= value + delta * tau * der_f_tau:
            break
        tau *= beta
        J_tau = np.eye(k) + tau**2/4 * W.T @ W + tau/2 * X.T @ E
        Y_tau = (2 * X + tau * W) @ np.linalg.inv(J_tau) - X

    return Y_tau


def laplacian_tensor_clustering(edges, weights, n, k=2, seed=42, rho=0.25, beta=0.5, delta=1e-3, max_iter=500, tol=1e-6):
    """
        args: hypergraph H with m edges, each edge of size r, and n vertices.  
        - edges: np.array of size (m, k)
        - weights: np.array of size (m,)
        - n: int
        - k: int
        output: X - top-k Z-eigenvectors of the normalized Laplacian of hypergraph H
    """
    m,r = edges.shape
    degree = np.zeros(n)
    for it_e in range(m):
        for i in range(r):
            v = edges[it_e, i]
            degree[v] += weights[it_e]
    rt_degree = degree**(1/r)

    # Generate random matrix with orthogonal columns
    np.random.seed(seed)
    X_1 = rt_degree.copy()
    X = np.linalg.qr(np.hstack([X_1.reshape(-1, 1), np.random.rand(n,k-1)]), mode='reduced')[0] 
    f_X = t_utils.L_value(X, edges, weights, rt_degree)
    tau = 0.5

    err_X = []
    err_f = []

    for _ in range(max_iter):
        X_next = step(X, edges, weights, rt_degree, tau, rho=rho, beta=beta, delta=delta)
        f_X_next = t_utils.L_value(X_next, edges, weights, rt_degree)

        err_X.append( np.linalg.norm(X_next - X) / np.sqrt(n) )
        err_f.append( np.abs(f_X_next - f_X) / (np.abs(f_X) + 1) )

        X = X_next
        f_X = f_X_next

        if err_X[-1] < tol or err_f[-1] < tol:
            break
        
    return X, err_X, err_f


def get_clusters(X):
    n,k = X.shape
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    return kmeans.labels_