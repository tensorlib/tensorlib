"""
===================================
Comparing SVD and CANDECOMP/PARAFAC
===================================

SVD decomposition can be used to decompose a tensor, but removing the structural
information involved with having multiple axes changes the decomposition.

The tensor decomposition retains structural information that is otherwise lost.

"""
print(__doc__)
from tensorlib.datasets import load_bread
from tensorlib.decomposition import cp
from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt

X, meta = load_bread()
U0, U1, U2 = cp(X, n_components=2, random_state=1999)
X0_flat = X.reshape(X.shape[0], -1)
U, S, V = linalg.svd(X0_flat, full_matrices=False)
svd_proj = V[:, :2]
t1 = np.dot(X0_flat.T, U0).T
t2 = np.dot(X0_flat.T, svd_proj).T
plt.figure()
plt.title("Tensor decomposition of bread data")
plt.scatter(t1[0, :], t1[1, :], color="darkred")
plt.figure()
plt.title("SVD of bread data")
plt.scatter(t2[0, :], t2[1, :], color="steelblue")
plt.show()
