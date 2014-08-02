"""General math utilities for tensor decomposition."""
# Authors: Kyle Kastner <kastnerkyle@gmail.com>
#          Michael Eickenberg <michael.eickenberg@gmail.com>
# License: BSD 3-Clause
import numpy as np


def kr(B, C):
    """
    Calculate the Khatri-Rao product of 2D matrices. Assumes blocks to
    be the columns of both matrices.

    See
    http://gmao.gsfc.nasa.gov/events/adjoint_workshop-8/present/Friday/Dance.pdf
    for more details.

    Parameters
    ----------
    B : ndarray, shape = [n, p]
    C : ndarray, shape = [m, p]


    Returns
    -------
    A : ndarray, shape = [m * n, p]

    """
    if B.ndim != 2 or C.ndim != 2:
        raise ValueError("B and C must have 2 dimensions")

    n, p = B.shape
    m, pC = C.shape

    if p != pC:
        raise ValueError("B and C must have the same number of columns")

    return np.einsum('ij, kj -> ikj', B, C).reshape(m * n, p)


def _canonical_kr(B, C):
    """
    Internal implementation of vanilla kr product.
    """
    n, p = B.shape
    m, pC = C.shape
    A = np.zeros((n * m, p))
    for k in range(B.shape[-1]):
        A[:, k] = np.kron(B[:, k], C[:, k])
    return A


def matricize(X, axis):
    """
    Returns flattened version of tensor.
    See http://www.graphanalysis.org/SIAM-PP08/Dunlavy.pdf
    for more details.

    Parameters
    ----------
    X : ndarray, shape = [d1, ..., dn]


    Returns
    -------
    X_flat : ndarray, shape = [d_axis, d1 * d2 * ... d_n]

    """
    dims = len(X.shape) - 1
    # If negative axis is passed, convert to equivalent positive form
    if axis < 0:
        axis = dims + axis + 1
    index = dims - axis
    return np.rollaxis(X, index).reshape(X.shape[index], -1)
