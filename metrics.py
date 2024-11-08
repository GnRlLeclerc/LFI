"""Metrics definitions for mLFHT methods."""

from test import KernelFunction

import numpy as np


class T:
    """Test statistic for mLFHT methods."""

    def __init__(self, kernel: KernelFunction) -> None:
        """Initialize T with a kernel function.
        Args:
            kernel: Kernel function to use.
        """
        self.kernel = kernel

    def __call__(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> float:
        """Compute T(X, Y, Z) for the given data."""

        n = X.shape[0]  # Amount of simulated samples
        m = Z.shape[0]  # Amount of unknown sampled data

        assert X.shape == Y.shape

        # Naive definition (but more readable)
        total = 0
        for i in range(n):
            for j in range(n):
                total += self.kernel(Z[j], Y[i]) - self.kernel(Z[j], X[i])

        return total / (n * m)
