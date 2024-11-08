"""Kernels for mLFHT methods."""

from typing import Callable

import numpy as np

KernelFunction = Callable[[np.ndarray, np.ndarray], float]


def get_gaussian_kernel(sigma: float) -> KernelFunction:
    """Create a Gaussian kernel with the given sigma."""

    def kernel(x: np.ndarray, y: np.ndarray) -> float:
        """Apply a Gaussian kernel 2 N-dimensional points."""
        assert x.shape == y.shape

        step = ((x - y) ** 2).sum(axis=-1)
        return np.exp(-step / (2 * sigma**2))

    return kernel
