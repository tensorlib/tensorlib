"""Tensor factorization."""
import numpy as np
from scipy import linalg
from functools import reduce
from .mathutils import kr, matricize, sign_flip
from .utils import check_random_state, check_tensor


def _random_init(X, n_components, random_state=None):
    rs = check_random_state(random_state)
    return [rs.rand(X.shape[i], n_components) for i in range(len(X.shape))]


def _hosvd_init_op(X, n_components, n):
    XXT = matricize(X, n).dot(matricize(X, n).T)
    _, U = linalg.eigh(XXT, eigvals=(XXT.shape[0] - n_components,
                                     XXT.shape[0] - 1))
    # reverse order of eigenvectors such that eigenvalues are decreasing
    U = U[:, ::-1]
    # flip sign
    U = sign_flip(U)
    return U


def _hosvd_init(X, n_components):
    return [_hosvd_init_op(X, n_components, i)
            for i in range(len(X.shape))]


def _cp3(X, n_components, tol, max_iter, init_type, random_state=None):
    """
    3 dimensional CANDECOMP/PARAFAC decomposition.

    This code is meant to be a tutorial example... in general the _cpN code
    should be more compact and equivalent mathematically.
    """

    if len(X.shape) != 3:
        raise ValueError("CP3 decomposition only supports 3 dimensions!")

    if init_type == "random":
        A, B, C = _random_init(X, n_components, random_state)
    elif init_type == "hosvd":
        A, B, C = _hosvd_init(X, n_components)
    grams = [np.dot(arr.T, arr) for arr in (A, B, C)]
    err = 1E10

    for itr in range(max_iter):
        err_old = err
        A = matricize(X, 0).dot(kr(C, B)).dot(linalg.pinv(grams[1] * grams[2]))
        if itr == 0:
            normalization = np.sqrt((B ** 2).sum(axis=0))
        else:
            normalization = B.max(axis=0)
            normalization[normalization < 1] = 1
        A /= normalization
        grams[0] = np.dot(A.T, A)

        B = matricize(X, 1).dot(kr(C, A)).dot(linalg.pinv(grams[0] * grams[2]))
        if itr == 0:
            normalization = np.sqrt((B ** 2).sum(axis=0))
        else:
            normalization = B.max(axis=0)
            normalization[normalization < 1] = 1
        B /= normalization
        grams[1] = np.dot(B.T, B)

        C = matricize(X, 2).dot(kr(B, A)).dot(linalg.pinv(grams[0] * grams[1]))
        if itr == 0:
            normalization = np.sqrt((C ** 2).sum(axis=0))
        else:
            normalization = C.max(axis=0)
            normalization[normalization < 1] = 1
        C /= normalization
        grams[2] = np.dot(C.T, C)

        err = linalg.norm(matricize(X, 0) - np.dot(A, kr(C, B).T)) ** 2
        thresh = np.abs(err - err_old) / err_old
        if (thresh < tol) or (itr > max_iter):
            break

    return A, B, C


def _cpN(X, n_components, tol, max_iter, init_type, random_state=None):
    """Generalized CANDECOMP/PARAFAC decomposition."""
    if init_type == "random":
        components = _random_init(X, n_components, random_state)
    elif init_type == "hosvd":
        components = _hosvd_init(X, n_components)
    grams = [np.dot(arr.T, arr) for arr in components]
    err = 1E10

    for itr in range(max_iter):
        err_old = err

        for idx in range(len(components)):
            components_sublist = [components[n] for n in range(len(components))
                                  if n != idx]
            grams_sublist = [grams[n] for n in range(len(components))
                             if n != idx]
            p1 = reduce(kr, components_sublist[:-1][::-1],
                        components_sublist[-1])
            p2 = linalg.pinv(reduce(np.multiply, grams_sublist, 1.))
            res = np.dot(matricize(X, idx), p1).dot(p2)
            if itr == 0:
                normalization = np.sqrt((res ** 2).sum(axis=0))
            else:
                normalization = res.max(axis=0)
                normalization[normalization < 1] = 1
            res /= normalization
            components[idx] = res
            grams[idx] = np.dot(res.T, res)

        err = linalg.norm(matricize(X, 0) - np.dot(
            components[0], reduce(kr, components[1:-1][::-1],
                                  components[-1]).T)) ** 2
        thresh = np.abs(err - err_old) / err_old
        if (thresh < tol) or (itr > max_iter):
            break
    return components


def cp(X, n_components=None, tol=1E-4, max_iter=500, init_type="hosvd",
       random_state=None):
    """
    CANDECOMP/PARAFAC decomposition.
    """
    if n_components is None:
        raise ValueError("n_components is a required argument!")

    check_tensor(X)
    return _cpN(X, n_components, tol=tol, max_iter=max_iter,
                init_type=init_type, random_state=random_state)
