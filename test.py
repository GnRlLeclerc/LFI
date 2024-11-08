"""
Run the experiment

Here, Z is actually 50% X and 50% noise. The hypothesis H0 should be correct.
We compute false negatives
"""

import numpy as np

from kernels import get_optimized_gaussian_kernel
from metrics import Phi_opti
from samples import mixed, noise, signal

n = 100  # Amount of simulated samples for both X and Y
m = 100  # Amount of unknown sampled data for Z
mu = 0.5  # Importance of the signal part in the mixed samples
pi = 0.5  # Tolerance for the kernel classification method
N = 200  # Number of attempts to average the tries

# Generate the data
X = signal(n)
Y = noise(n)
Z = mixed(m, mu)

kernel = get_optimized_gaussian_kernel(sigma=1)

m_range = np.arange(10, 201, 5)
n_range = np.arange(10, 201, 5)

results = np.zeros((len(n_range), len(m_range)))


for i, n in enumerate(n_range):
    for j, m in enumerate(m_range):
        print(i * len(m_range) + j, "/", len(n_range) * len(m_range), end="\r")
        for _ in range(N):

            X = signal(n)
            Y = noise(n)
            Z = mixed(m, mu)

            result = Phi_opti(kernel, X, Y, Z, pi=0.5)
            results[i, j] += result
        results[i, j] /= N
print()

np.save("results.npy", results)
