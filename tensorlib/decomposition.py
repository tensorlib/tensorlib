"""Tensor factorization."""
import numpy as np
from scipy import linalg
from functools import reduce
from .mathutils import kr, matricize
from .utils import check_random_state, check_tensor


def _cp3(X, n_components, tol=1E-6, max_iter=1000, random_state=None):
    """3 dimensional CANDECOMP/PARFAC decomposition."""
    if len(X.shape) != 3:
        raise ValueError("CP3 decomposition only supports 3 dimensions!")

    rs = check_random_state(random_state)
    A = rs.randn(X.shape[0], n_components)
    B = rs.randn(X.shape[1], n_components)
    C = rs.randn(X.shape[2], n_components)

    SSE = 1E100
    dSSE = 1E100
    itr = 0

    while (dSSE >= tol * SSE) and (itr < max_iter):
        itr += 1
        SSE_old = SSE

        # Symmetry here... could we exploit make calculation faster?
        ATA = np.dot(A.T, A)
        BTB = np.dot(B.T, B)
        CTC = np.dot(C.T, C)

        A = np.dot(matricize(X, 2), kr(C, B)).dot(linalg.pinv(BTB * CTC))
        B = np.dot(matricize(X, 1), kr(C, A)).dot(linalg.pinv(ATA * CTC))
        C = np.dot(matricize(X, 0), kr(B, A)).dot(linalg.pinv(ATA * BTB))

        SSE = linalg.norm(matricize(X, 2) - np.dot(A, kr(C, B).T)) ** 2
        dSSE = SSE_old - SSE
    return A, B, C


def _cpN(X, n_components, tol=1E-6, max_iter=1000, random_state=None):
    """Generalized CANDECOMP/PARFAC decomposition."""

    rs = check_random_state(random_state)
    components = [rs.randn(X.shape[i], n_components)
                  for i in range(len(X.shape))]

    SSE = 1E100
    dSSE = 1E100
    SST = np.sum(X ** 2)
    itr = 0

    while (dSSE >= tol * SSE) and (itr < max_iter):
        itr += 1
        SSE_old = SSE

        # Symmetry here... could we exploit to make calculation faster?
        grams = [np.dot(arr.T, arr) for arr in components]

        updates = []
        for idx in range(len(components)):
            components_sublist = [components[n] for n in range(len(components))
                                  if n != idx]
            grams_sublist = [grams[n] for n in range(len(components))
                             if n != idx]
            p1 = reduce(kr, components_sublist[1:], components_sublist[0])
            p2 = linalg.pinv(reduce(np.multiply, grams_sublist, 1.))
            res = np.dot(matricize(X, -idx - 1), p1)
            updates.append(np.dot(res, p2))

        SSE = np.sum(p2 * np.dot(updates[-1].T, updates[-1])) - 2 * np.sum(
            res * updates[-1]) + SST
        dSSE = SSE_old - SSE
        components = updates
    return components


def cp(X, n_components=None, tol=1E-6, max_iter=1000, random_state=None,
       force_general=False):
    if n_components is None:
        raise ValueError("n_components is a required argument!")

    check_tensor(X)

    if force_general:
        return _cpN(X, n_components, tol, max_iter, random_state)

    elif len(X.shape) == 3:
        return _cp3(X, n_components, tol, max_iter, random_state)
    else:
        return _cpN(X, n_components, tol, max_iter, random_state)
