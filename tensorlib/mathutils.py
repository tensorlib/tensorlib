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
    axis : int

    Returns
    -------
    X_flat : ndarray, shape = [d_axis, d1 * d2 * ... d_n]

    """
    # If negative axis is passed, convert to equivalent positive form
    if axis < 0:
        axis = X.ndim + axis
    # Need to transpose so that ordering is correct
    # X.shape = (7, 3, 11)
    # matricize(X, 0).shape = (7, 11 * 3)
    # matricize(X, 1).shape = (3, 11 * 7)
    # matricize(X, 2).shape = (11, 3 * 7)
    # Long term, may need an option to specify whether to reverse or not
    not_axis = np.where(np.arange(X.ndim) != axis)[0][::-1]
    return X.transpose(axis, *not_axis).reshape(X.shape[axis], -1)


def unmatricize(X, axis, dims):
    """
    Returns reshaped tensor, reverses matricize operation.
    See http://www.graphanalysis.org/SIAM-PP08/Dunlavy.pdf
    for more details.

    Parameters
    ----------
    X : ndarray, shape = [d_axis, d1 * d2 * ... dn]
    axis : int
    dims : list, [d1, d2, ..., dn]

    Returns
    -------
    X_tensor : ndarray, shape = [d1, d2, ..., d_n]

    """
    # If negative axis is passed, convert to equivalent positive form
    if axis < 0:
        axis = len(dims) + axis
    not_axis = np.where(np.arange(len(dims)) != axis)[0][::-1]
    d = np.array(dims)
    perm = list(range(1, len(dims)))[::-1]
    perm.insert(axis, 0)
    return X.ravel().reshape(d[axis], *d[not_axis]).transpose(*perm)


def sign_flip(X):
    """
    Flip the signs of X so that largest absolute value is positive.

    Parameters
    ----------
    X : array-like

    Returns
    -------
    X_flipped : array-like

    """
    max_abs_cols = np.argmax(np.abs(X), axis=0)
    signs = np.sign(X[max_abs_cols, list(range(X.shape[1]))])
    return signs * X


def tmult(X, M, axis):
    """
    Tensor multiplication (also known as n-mode multiplication)
    Given an array X of shape (n, m, p) and a matrix M of shape (r, m)
    multiplying along the 2nd dimension (axis 1) will result in a new tensor
    of shape (n, r, p).

    Parameters
    ----------
    X : array-like
    M : array-like
    axis : int

    Returns
    -------
    T : ndarray

    """
    tensor_shape = list(X.shape)
    tensor_shape[axis] = M.shape[0]
    return unmatricize(M.dot(matricize(X, axis)), axis, tensor_shape)
