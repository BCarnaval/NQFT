"""Docs
"""

import numpy as np
from numpy.linalg import eigh
from numpy.random import normal
from scipy.signal import convolve2d

from nqft.functions import timeit

np.set_printoptions(linewidth=300)


@timeit
def build_h(shape: tuple, hops: np.array) -> np.ndarray:
    """Docs
    """
    t, tp, tpp = hops
    sites = shape[0] * shape[1]
    cluster = np.zeros(shape=shape)
    h = np.zeros(shape=(sites, sites))
    t_ij_kernel = np.array([
        [0, 0, tpp, 0, 0],
        [0, tp, t, tp, 0],
        [tpp, t, 0, t, tpp],
        [0, tp, t, tp, 0],
        [0, 0, tpp, 0, 0]
    ])

    colmn = 0
    for coords in np.ndindex(shape):
        i, j = coords
        cluster[i, j] = 1.0

        h[:, colmn] = convolve2d(
            cluster, t_ij_kernel, mode='same', boundary='wrap').ravel()

        cluster[i, j] = 0.0
        colmn += 1

    return h


@timeit
# ------------------
# Return type needed
# ------------------
def monte_carlo(h: np.ndarray, mu: float, std: float):
    """Docs
    """
    length = h.shape[0]
    mus = normal(loc=mu, scale=std, size=(length,))
    epsilon_ij = np.diag(v=mus)
    h += epsilon_ij

    vals, vects = eigh(h)

    return vals, vects


if __name__ == "__main__":
    # Variables & Parameters
    shape = (80, 80)
    sites = shape[0] * shape[1]
    hoppings = np.array([1, -0.3, 0.2])

    # Building Hamiltonian
    H = build_h(shape=shape, hops=hoppings)
    potentials = monte_carlo(h=H, mu=0.0, std=1.0)
