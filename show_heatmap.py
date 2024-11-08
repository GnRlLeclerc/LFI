"""Show a saved heatmap"""

import sys
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from samples import downsample

N_TICKS = 7

m_range = np.arange(10, 201, 5)
n_range = np.arange(10, 201, 5)

if __name__ == "__main__":

    file = sys.argv[1]

    heatmap = np.load(file)

    xticks = downsample(np.arange(heatmap.shape[1]), N_TICKS)
    yticks = downsample(np.arange(heatmap.shape[0]), N_TICKS)
    xlabels: Sequence[str] = downsample(m_range, N_TICKS).astype(str)  # type: ignore
    ylabels: Sequence[str] = downsample(n_range, N_TICKS).astype(str)  # type: ignore

    plt.imshow(heatmap, cmap="hot", interpolation="nearest")
    plt.xlabel("m")
    plt.ylabel("n")
    plt.xticks(xticks, xlabels)
    plt.yticks(yticks, ylabels[::-1])
    plt.title("False negatives in a Z = 50% X + 50% Y setting")
    plt.show()
