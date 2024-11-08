"""
Toy dataset example for the mLFHT methods.

The data is constituted of 2D points.
The probability distributions of both point coordinates are as follows:
- P_X = N(1, 0.5) # Signal
- P_Y = N(0, 1)   # Background noise
- P_Z = mu * P_X + (1 - mu) * P_Y
"""

import numpy as np
import numpy.random as rd


def noise(n_samples: int) -> np.ndarray:
    """Generate noise samples.
    Gaussian centered at 0 with variance 1.
    """
    return rd.normal(0, size=(n_samples, 2))


def signal(n_samples: int) -> np.ndarray:
    """Generate signal samples.
    Offset Gaussian with lower variance than the noise.
    """
    return rd.normal(1, 0.5, size=(n_samples, 2))


def mixed(n_samples: int, mu: float) -> np.ndarray:
    """Generate mixed samples.

    Args:
        mu: importance of the signal part.
    """
    return mu * signal(n_samples) + (1 - mu) * noise(n_samples)
