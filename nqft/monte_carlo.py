"""Docs
"""

import numpy as np
from scipy.signal import convolve2d


def build_h(shape: tuple[int], mu: float, hops: np.array) -> np.ndarray:
    """Docs
    """
    t, tp, tpp = hops
    sites = shape[0] * shape[1]
    cluster = np.zeros(shape=shape)
    h = np.zeros(shape=(sites, sites))
    t_ij_kernel = np.array([
        [0, 0, tpp, 0, 0],
        [0, tp, t, tp, 0],
        [tpp, t, mu, t, tpp],
        [0, tp, t, tp, 0],
        [0, 0, tpp, 0, 0]
    ])

    column = 0
    for coords in np.ndindex(shape):
        i, j = coords
        cluster[i, j] = 1.0

        h[:, column] = convolve2d(
            cluster, t_ij_kernel, mode='same', boundary='wrap').ravel()

        cluster[i, j] = 0.0
        column += 1

    return h


if __name__ == "__main__":
    # Variables & Parameters
    shape = (3, 3)
    sites = shape[0] * shape[1]
    hoppings = np.array([1, 2, 3])

    # Building Hamiltonian
    H = build_h(shape=shape, mu=0.0, hops=hoppings)
    print(H)
