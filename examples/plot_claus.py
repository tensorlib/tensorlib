"""
===========================================================
CANDECOMP/PARAFAC decomposition for analyzing chemical data
===========================================================
"""
print(__doc__)
from tensorlib.datasets import load_claus
from tensorlib.decomposition import cp
import matplotlib.pyplot as plt
import time

X, meta = load_claus()
t0 = time.time()
U0, U1, U2 = cp(X, n_components=2, init_type="hosvd")
plt.title("Decomposition of chemistry data, excitation axis")
plt.plot(U1)
plt.figure()
plt.title("Decomposition of chemistry data, emission axis")
plt.plot(U2)
plt.show()
