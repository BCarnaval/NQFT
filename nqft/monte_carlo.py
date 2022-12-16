"""Docs
"""

import numpy as np
from numpy.fft import fft2
import matplotlib.pyplot as plt
from numpy.random import normal
from numpy.linalg import eigh, inv
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


def r(idx: int, shape: tuple) -> np.array:
    """Docs
    """
    return np.array([idx % shape[1], idx // shape[1]])


@timeit
# ------------------
# Return type needed
# ------------------
def monte_carlo(h: np.ndarray, shape: tuple, omega: float, eta: float,
                mu: float, std: float):
    """Docs
    """
    # length = h.shape[0]
    # mus = normal(loc=mu, scale=std, size=(length,))
    # epsilon_ij = np.diag(v=mus)
    # h += epsilon_ij

    # vals, vects = eigh(h)

    z = omega + eta * 1j
    G_ij = inv(z - h)
    n_sites = shape[0] * shape[1]
    G_kw = np.zeros(shape=shape, dtype=np.complex128)
    for i in range(n_sites):
        r_i = r(i, shape)
        for j in range(n_sites):
            r_j = r(j, shape)
            delta_r = r_i - r_j
            for k in range(n_sites):
                k_vec = r(k, shape)
                G_kw[k_vec[0], k_vec[1]
                     ] += np.exp(1j * np.dot(k_vec, delta_r)) * G_ij[i, j]

    A_kw = -1 / np.pi * G_kw.imag

    return A_kw


if __name__ == "__main__":
    # Variables & Parameters
    shape = (10, 10)
    hoppings = np.array([1, -0.3, 0.2])

    # Building Hamiltonian
    H = build_h(shape=shape, hops=hoppings)
    spectrum = monte_carlo(h=H, shape=shape, omega=0.0,
                           eta=0.1, mu=0.0, std=1.0)

    k_s = np.linspace(-np.pi, np.pi, shape[0])
    k_x, k_y = np.meshgrid(k_s, k_s)
    plt.contourf(spectrum)
    plt.show()
