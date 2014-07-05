"""General math utilities for tensor decomposition."""
# Authors: Kyle Kastner <kastnerkyle@gmail.com>
#          Michael Eickenberg <michael.eickenberg@gmail.com>
# License: BSD 3-Clause
import numpy as np


def kr(B, C):
    """Calculate the Khatri-Rao product of 2D matrices. Assumes blocks to
    be the columns of both matrices. 
    See http://en.wikipedia.org/wiki/Kronecker_product#Khatri-Rao_product

    Parameters
    ==========

    B: ndarray, shape=(n, p)
    C: ndarray, shape=(m, p)
"""

    if B.ndim != 2 or C.ndim != 2:
        raise ValueError("B and C must have 2 dimensions")

    n, p = B.shape
    m, pC = C.shape

    if p != pC:
        raise ValueError("B and C must have the same number of columns")

    A = np.zeros((n * m, p))
    for k in range(B.shape[-1]):
        A[:, k] = np.kron(C[:, k], B[:, k])
    return A
