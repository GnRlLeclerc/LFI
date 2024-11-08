"""Metrics definitions for mLFHT methods."""

from typing import Literal

import numpy as np

from kernels import KernelFunction


def T(kernel: KernelFunction, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> float:
    """Compute T(X, Y, Z) for the given data."""

    n = X.shape[0]  # Amount of simulated samples
    m = Z.shape[0]  # Amount of unknown sampled data

    assert X.shape == Y.shape

    # Naive implementation (but more readable)
    total = 0
    for i in range(n):
        for j in range(m):
            total += kernel(Z[j], Y[i]) - kernel(Z[j], X[i])

    return total / (n * m)


def MMD2u(kernel: KernelFunction, X: np.ndarray, Y: np.ndarray) -> float:
    """Compute squared MMDu for the given data."""

    n = X.shape[0]
    m = Y.shape[0]

    total = 0

    # Naive implementations, but readable
    subtotal_x = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                subtotal_x += kernel(X[i], X[j])
    total += subtotal_x / (n * (n - 1))

    subtotal_y = 0
    for i in range(m):
        for j in range(m):
            if i != j:
                subtotal_y += kernel(Y[i], Y[j])
    total += subtotal_y / (m * (m - 1))

    subtotal_mixed = 0
    for i in range(n):
        for j in range(m):
            subtotal_mixed += kernel(X[i], Y[j])
    total -= 2 * subtotal_mixed / (n * m)

    return total


def gamma(kernel: KernelFunction, X: np.ndarray, Y: np.ndarray, pi: float) -> float:
    """Compute gamma for the given data."""

    return pi * MMD2u(kernel, X, Y) + T(kernel, X, Y, X)


def Phi(
    kernel: KernelFunction, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, pi: float
) -> Literal[0, 1]:
    """Compute the Phi hypothesis indicator."""

    if T(kernel, X, Y, Z) >= gamma(kernel, X, Y, pi):
        return 1
    return 0
