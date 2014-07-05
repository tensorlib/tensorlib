"""General math utilities for tensor decomposition."""
# Authors: Kyle Kastner <kastnerkyle@gmail.com>
#          Michael Eickenberg <michael.eickenberg@gmail.com>
# License: BSD 3-Clause
import numpy as np


def kr(B, C):
    """Calculate the Khatari-Rao produc of 2D matrices."""
    assert(len(B.shape) == len(C.shape) == 2)
    assert(B.shape[-1] == C.shape[-1])
    A = np.zeros((B.shape[0] * C.shape[0], B.shape[-1]))
    for k in range(B.shape[-1]):
        print(np.kron(C[:, k], B[:, k]))
    return A
