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

noise_x = noise(100)
noise_y = noise(100)

signal_x = signal(20)
signal_y = signal(20)

mixed_x = mixed(20, 0.5)
mixed_y = mixed(20, 0.5)

plt.scatter(noise_x, noise_y, color="blue", label="Noise")
plt.scatter(signal_x, signal_y, color="red", label="Signal")
plt.scatter(mixed_x, mixed_y, color="green", label="Mixed")
plt.legend()
plt.title("Toy dataset showcase")
plt.axis("equal")
plt.grid()
plt.show()
