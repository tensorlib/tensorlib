"""Tensor factorization."""
import numpy as np
from scipy import linalg
from .mathutils import kr, matricize
from .utils import check_random_state


def cp(X, n_components=None, tol=1E-9, max_iter=1000, random_state=None):
    """Candecomp/PARFAC decomposition."""
    if n_components is None:
        raise ValueError("n_components is a required argument!")

    if len(X.shape) != 3:
        raise ValueError("CP decomposition current supports 3 dimensions only.")

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
