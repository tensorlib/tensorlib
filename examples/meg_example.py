"""An example of using tensorlib to decompose a builtin dataset."""
from tensorlib.datasets import fetch_decmeg
from tensorlib.decomposition import cp
import matplotlib.pyplot as plt
import time

X, meta = fetch_decmeg()
X = X[:, :, 125:250]
t0 = time.time()
U0, U1, U2 = cp(X, n_components=10, init_type="hosvd")
plt.plot(U2)
plt.title("Rank 10 decomposition of MEG data, time axis")
plt.show()
