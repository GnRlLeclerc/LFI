"""
Run the experiment

Here, Z is actually 50% X and 50% noise. The hypothesis H0 should be correct.
We compute false negatives
"""

import matplotlib.pyplot as plt
import numpy as np

from kernels import get_optimized_gaussian_kernel
from metrics import Phi_opti
from samples import mixed, noise, signal

n = 100  # Amount of simulated samples for both X and Y
m = 100  # Amount of unknown sampled data for Z
mu = 0.5  # Importance of the signal part in the mixed samples
pi = 0.5  # Tolerance for the kernel classification method

# Generate the data
X = signal(n)
Y = noise(n)
Z = mixed(m, mu)

kernel = get_optimized_gaussian_kernel(sigma=1)

m_range = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
n_range = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

results = np.zeros((len(n_range), len(m_range)))


for i, n in enumerate(n_range):
    for j, m in enumerate(m_range):
        print(i * len(m_range) + j, "/", len(n_range) * len(m_range), end="\r")
        X = signal(n)
        Y = noise(n)
        Z = mixed(m, mu)

        result = Phi_opti(kernel, X, Y, Z, pi=0.5)
        results[i, j] = result
print()

# TODO : do multiple runs, compute the average
plt.imshow(results, cmap="hot", interpolation="nearest")
plt.show()
