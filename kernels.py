"""Kernels for mLFHT methods."""

from typing import Callable

import numpy as np

KernelFunction = Callable[[np.ndarray, np.ndarray], float]
OptiKernelFunction = Callable[[np.ndarray, np.ndarray], float]


def get_gaussian_kernel(sigma: float) -> KernelFunction:
    """Create a Gaussian kernel with the given sigma."""

    def kernel(x: np.ndarray, y: np.ndarray) -> float:
        """Apply a Gaussian kernel to 2 N-dimensional points."""
        assert x.shape == y.shape

        step = ((x - y) ** 2).sum(axis=-1)
        return np.exp(-step / (2 * sigma**2))

    return kernel


def get_optimized_gaussian_kernel(sigma: float) -> OptiKernelFunction:

    def kernel(x: np.ndarray, y: np.ndarray) -> float:
        """Apply a Gaussian kernel to 2 sets of N-dimensional points.

        # Compute pointwise kernel between all pairs (X_i, Y_j)
        Args:
            X: (n, np.newaxis, d)
            Y: (m, d)

        # Compute the diagonal kernel between all pairs (X_i, X_i)
        Args:
            X: (n, d)
            X: (n, d)
        """
        # (n, m, d) -> (n, m)
        step = ((x - y) ** 2).sum(axis=-1)

        return np.exp(-step / (2 * sigma**2)).sum()

    return kernel
