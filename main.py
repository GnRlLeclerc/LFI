"""
Toy dataset example for the mLFHT methods.

The data is constituted of 2D points.
The probability distributions of both point coordinates are as follows:
- P_X = N(1, 0.5) # Signal
- P_Y = N(0, 1)   # Background noise
- P_Z = mu * P_X + (1 - mu) * P_Y
"""

import matplotlib.pyplot as plt

from samples import mixed, noise, signal

noise = noise(1000)

signal = signal(50)

mixed = mixed(100, 0.4)

plt.scatter(noise[:, 0], noise[:, 1], color="blue", label="Noise")
plt.scatter(mixed[:, 0], mixed[:, 1], color="green", label="Mixed")
plt.scatter(signal[:, 0], signal[:, 1], color="red", label="Signal")
plt.legend()
plt.title("Toy dataset showcase (40% signal, 60% noise)")
plt.axis("equal")
plt.grid()
plt.show()
