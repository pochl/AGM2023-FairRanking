import cvxpy as cvx
import numpy as np
from scipy.stats import rankdata
from sklearn.utils import check_random_state
from tqdm import tqdm


def synthesize_rel_mat(
    n_query: int,
    n_doc: int,
    lam: float = 0.0,
    flip_ratio: float = 0.3,
    noise: float = 0.0,
    random_state: int = 12345,
) -> np.ndarray:
    random_ = check_random_state(random_state)

    # generate true relevance matrix
    rel_mat_unif = random_.uniform(0.0, 1.0, size=(n_query, n_doc))
    rel_mat_pop_doc = np.ones_like(rel_mat_unif)
    rel_mat_pop_doc -= (np.arange(n_doc) / n_doc)[np.newaxis, :]
    rel_mat_pop_query = np.ones_like(rel_mat_unif)
    rel_mat_pop_query -= (np.arange(n_query) / n_query)[:, np.newaxis]
    rel_mat_pop = rel_mat_pop_doc * rel_mat_pop_query
    rel_mat_pop *= rel_mat_unif.sum() / rel_mat_pop.sum()
    rel_mat_true = (1.0 - lam) * rel_mat_unif
    rel_mat_true += lam * rel_mat_pop

    # flipping
    flip_docs = random_.choice(
        np.arange(n_doc), size=np.int32(flip_ratio * n_doc), replace=False
    )
    flip_docs.sort()
    rel_mat_true[:, flip_docs] = rel_mat_true[::-1, flip_docs]

    # gen noisy relevance matrix
    if noise > 0.0:
        rel_mat_obs = np.copy(rel_mat_true)
        rel_mat_obs += random_.uniform(-noise, noise, size=(n_query, n_doc))
        rel_mat_obs = np.maximum(rel_mat_obs, 0.001)
    else:
        rel_mat_obs = rel_mat_true

    return rel_mat_true, rel_mat_obs


def exam_func(n_doc: int, K: int, shape: str = "inv") -> np.ndarray:
    assert shape in ["inv", "exp", "log"]
    if shape == "inv":
        v = np.ones(K) / np.arange(1, K + 1)
    elif shape == "exp":
        v = 1.0 / np.exp(np.arange(K))

    return v[:, np.newaxis]


def evaluate_pi(pi: np.ndarray, rel_mat: np.ndarray, v: np.ndarray):
    n_query, n_doc = rel_mat.shape
    expo_mat = (pi * v.T).sum(2)
    click_mat = rel_mat * expo_mat
    user_util = click_mat.sum() / n_query
    item_utils = click_mat.sum(0) / n_query
    nsw = np.power(item_utils.prod(), 1 / n_doc)

    max_envies = np.zeros(n_doc)
    for i in range(n_doc):
        u_d_swap = (expo_mat * rel_mat[:, [i] * n_doc]).sum(0)
        d_envies = u_d_swap - u_d_swap[i]
        max_envies[i] = d_envies.max() / n_query

    return user_util, item_utils, max_envies, nsw


def compute_pi_max(
    rel_mat: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    n_query, n_doc = rel_mat.shape
    K = v.shape[0]

    pi = np.zeros((n_query, n_doc, K))
    for k in np.arange(K):
        pi_at_k = np.zeros_like(rel_mat)
        pi_at_k[rankdata(-rel_mat, axis=1, method="ordinal") == k + 1] = 1
        pi[:, :, k] = pi_at_k

    return pi


def compute_pi_expo_fair(
    rel_mat: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    n_query, n_doc = rel_mat.shape
    K = v.shape[0]
    query_basis = np.ones((n_query, 1))
    am_rel = rel_mat.sum(0)[:, np.newaxis]
    am_expo = v.sum() * n_query * am_rel / rel_mat.sum()

    pi = cvx.Variable((n_query, n_doc * K))
    obj = 0.0
    constraints = []
    for d in np.arange(n_doc):
        pi_d = pi[:, K * d : K * (d + 1)]
        obj += rel_mat[:, d] @ pi_d @ v
        # feasible allocation
        basis_ = np.zeros((n_doc * K, 1))
        basis_[K * d : K * (d + 1)] = 1
        constraints += [pi @ basis_ <= query_basis]
        # amortized exposure
        constraints += [query_basis.T @ pi_d @ v <= am_expo[d]]
    # feasible allocation
    for k in np.arange(K):
        basis_ = np.zeros((n_doc * K, 1))
        basis_[np.arange(n_doc) * K + k] = 1
        constraints += [pi @ basis_ <= query_basis]
    constraints += [pi <= 1.0]
    constraints += [0.0 <= pi]

    prob = cvx.Problem(cvx.Maximize(obj), constraints)
    prob.solve(solver=cvx.SCS)

    pi = pi.value.reshape((n_query, n_doc, K))
    pi = np.clip(pi, 0.0, 1.0)

    return pi


def compute_pi_nsw(
    rel_mat: np.ndarray,
    v: np.ndarray,
    alpha: float = 0.0,
) -> np.ndarray:
    n_query, n_doc = rel_mat.shape  # n_query = U, n_doc = I
    K = v.shape[0]
    query_basis = np.ones((n_query, 1))
    am_rel = rel_mat.sum(0) ** alpha

    # instead of having 3 dims (u,i,k) of pi, they use 2 dim (u, (i,k)) of pi.
    # So for first user (first row), pi of item 1 will be from 0 -> K-1 index,
    # pi of item 1 will be from K -> 2K-1 index, and so on.
    pi = cvx.Variable((n_query, n_doc * K))
    obj = 0.0
    constraints = []
    for d in np.arange(n_doc):
        obj += am_rel[d] * cvx.log(rel_mat[:, d] @ pi[:, K * d : K * (d + 1)] @ v)
        # feasible allocation
        basis_ = np.zeros((n_doc * K, 1))
        basis_[K * d : K * (d + 1)] = 1
        constraints += [pi @ basis_ <= query_basis]
        # basis_ are 1 only for elements corresponding to all ranks for item d. 
        # Thus, pi @ basis_ yields U elements of summing pi over k.
        # Hence, this loop produces first constraint

    # feasible allocation
    for k in np.arange(K):
        basis_ = np.zeros((n_doc * K, 1))
        basis_[np.arange(n_doc) * K + k] = 1
        constraints += [pi @ basis_ <= query_basis]
        # basis_ are 1 only for elements corresponding to all items for rank k.
        # Thus, pi @ basis_ yields U elements of summing pi over i.
        # Hence, this loop produces second constraint

    constraints += [pi <= 1.0]
    constraints += [0.0 <= pi]

    prob = cvx.Problem(cvx.Maximize(obj), constraints)
    prob.solve(solver=cvx.SCS)

    pi = pi.value.reshape((n_query, n_doc, K))
    pi = np.clip(pi, 0.0, 1.0)

    return pi


def compute_pi_nsw_lag1(
    rel_mat: np.ndarray,
    v: np.ndarray,
    alpha: float = 0.0,
    max_iter: int = 10,
    verbose: bool = False
) -> np.ndarray:
    """Computes pi for modified NSW problem where Lagrangian multiplier method is applied
    to the first constraint. The Lagrangian method is done via gradient ascent, following
    the official example from CVXPY's webpage (https://www.cvxpy.org/examples/applications/MM.html).

    Since method from CVXPY's webpage only works when the Lagrangified constraint is an equality constraint,
    we convert our first constraint from inequality to equality by introducing slack variables.
    We also get rid off the lambda >= 0 constraint lambda is free for equality constraint. 
    """

    n_query, n_doc = rel_mat.shape  # n_query = U, n_doc = I
    K = v.shape[0]
    query_basis = np.ones((n_query, 1))
    am_rel = rel_mat.sum(0) ** alpha

    # instead of having 3 dims (u,i,k) of pi, they use 2 dim (u, (i,k)) of pi.
    # So for first user (first row), pi of item 1 will be from 0 -> K-1 index,
    # pi of item 1 will be from K -> 2K-1 index, and so on.
    pi = cvx.Variable((n_query, n_doc * K))
    slack = cvx.Variable((n_query, n_doc))
    lmda = cvx.Parameter((n_query, n_doc))
    lmda.value = np.zeros((n_query, n_doc))

    obj = 0.0
    constraints = []
    resids = []
    rho = 1

    for d in np.arange(n_doc):
        obj += am_rel[d] * cvx.log(rel_mat[:, d] @ pi[:, K * d : K * (d + 1)] @ v)
        
        basis_ = np.zeros((n_doc * K))
        basis_[K * d : K * (d + 1)] = 1
        resids.append(pi @ basis_ + slack[:, d] - 1)  # we create residules for each item
        # basis_ are 1 only for elements corresponding to all ranks for item d. 
        # Thus, pi @ basis_ yields U elements of summing pi over k.
        # Hence, this loop produces first constraint

    # feasible allocation
    for k in np.arange(K):
        basis_ = np.zeros((n_doc * K, 1))
        basis_[np.arange(n_doc) * K + k] = 1
        constraints += [pi @ basis_ <= query_basis]
        # basis_ are 1 only for elements corresponding to all items for rank k.
        # Thus, pi @ basis_ yields U elements of summing pi over i.
        # Hence, this loop produces second constraint

    constraints += [pi <= 1.0]
    constraints += [0.0 <= pi]
    constraints += [0.0 <= slack]

    lagr = 0
    for d in np.arange(n_doc):
        lagr -= lmda[:, d] @ resids[d]
        lagr -= (rho/2)*cvx.sum_squares(resids[d])

    iterator = tqdm(range(max_iter)) if verbose else range(max_iter)
    for _ in iterator:
        prob = cvx.Problem(cvx.Maximize(obj + lagr), constraints)
        prob.solve(solver=cvx.SCS)
        for d in np.arange(n_doc):
            lmda.value[:, d] += resids[d].value

    pi = pi.value.reshape((n_query, n_doc, K))
    pi = np.clip(pi, 0.0, 1.0)

    return pi


def compute_pi_unif(rel_mat: np.ndarray, v: np.ndarray) -> np.ndarray:
    n_query, n_doc = rel_mat.shape
    K = v.shape[0]

    return np.ones((n_query, n_doc, K)) / n_doc


def compute_obj_val(pi: np.ndarray,
                    rel_mat: np.ndarray,
                    v: np.ndarray,
                    alpha: float = 0.0,):

    n_query, n_doc = rel_mat.shape
    am_rel = rel_mat.sum(0) ** alpha
    K = v.shape[0]

    pi = pi.reshape((n_query, n_doc*K))

    obj = 0
    for d in np.arange(n_doc):
        obj += am_rel[d] * np.log(rel_mat[:, d] @ pi[:, K * d : K * (d + 1)] @ v)

    return obj[0]
