"""Hall effect in cuprates with an incommensurate collinear
spin-density wave.
"""

import time
import numpy as np
from rich import print
from matplotlib import cm
from functools import wraps
import matplotlib.pyplot as plt
from scipy.constants import e, pi
from numpy import arange, meshgrid, sin, cos, exp


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__} took {total_time:.4f} seconds")
        return result

    return timeit_wrapper


@timeit
def dE(hop_amps: list, kx: np.ndarray, ky: np.ndarray, mus: np.array) -> tuple:
    """Outputs model's energy and it's derivatives.

    Parameters
    ----------
    hop_amps: list, default=None
        Hopping amplitudes coefficients.

    kx: np.ndarray, shape=(N, N), default=None
        kx space as a 2D array.

    ky: np.ndarray, shape=(N, N), default=None
        ky space as a 2D array.

    mu: np.array, size=M, default=None
        ADD DESCRIPTION

    Returns
    -------
    E, dEx, ddEx, dEy, ddEy, dExEy: tuple, size=6
        Energy and it's derivatives in a tuple.
    """
    t, t1, t2 = hop_amps

    # Energy
    E = []
    a = -2 * t * (cos(kx) + cos(ky))
    b = -4 * t1 * cos(kx) * cos(ky)
    c = -2 * t2 * (cos(2 * kx) + cos(2 * ky))
    for mu in mus:
        E.append(a + b + c - mu)

    # Ex derivatives
    dEx = (
        2 * t * sin(kx) + 2 * t1 * (sin(kx + ky) + sin(kx - ky)) + 4 * t2 * sin(2 * kx)
    )
    ddEx = (
        2 * t * cos(kx) + 2 * t1 * (cos(kx + ky) + cos(kx - ky)) + 8 * t2 * cos(2 * kx)
    )

    # Ey derivatives
    dEy = (
        2 * t * sin(ky) + 2 * t1 * (sin(kx + ky) - sin(kx - ky)) + 4 * t2 * sin(2 * ky)
    )
    ddEy = (
        2 * t * cos(ky) + 2 * t1 * (cos(kx + ky) + cos(kx - ky)) + 8 * t2 * cos(2 * ky)
    )

    # Mixed derivative
    dExEy = 2 * t1 * (cos(kx + ky) - cos(kx - ky))

    return E, dEx, ddEx, dEy, ddEy, dExEy


@timeit
def get_spectral_weight(
    omega: float, eta: float, mu: float, E: np.ndarray
) -> np.ndarray:
    """Ouputs the spectral weight as a 2D numpy array.

    Parameters
    ----------
    omega: float, default=None
        Frequency at which we observe the fermi surface.

    eta: float default=None
        Lorentzian broadening module.

    mu: np.array, size=M, default=None
        ADD DESCRIPTION.

    E: np.ndarray. shape=(N, N), default=None
        Eigenenergies of the system as a 2D numpy array.

    Returns
    -------
    A: np.ndarray, shape=(N, N)
        Spectral weight.
    """
    # Spectral weight
    A = []
    for i, j in zip(E, mu):
        A1 = omega**2 + i**2 + eta**2 + j**2
        A2 = -2 * (omega * i + i * j - omega * j)
        A.append(1 / pi * (eta / (A1 + A2)))

    return A


class Model:
    """Model instance to determine Hall coefficient from
    tight-binding hamiltonian.

    Attributes
    ----------
    hop_amps: list, size=3, default=None
        Hopping amplitude coefficients.

    frequency: float, default=None
        Frequency at which we observe the fermi surface.

    eta: float, default=None
        Lorentzian broadening module.

    mu: np.array, size=M, default=None
        ADD DESCRIPTION.

    V: float, default=None
        Normalization volume.

    k_lims: tuple, size=2
        Wavevectors interval values.

    resolution: int, default=100
        Resolution of phase space (kx, ky).
    """

    def __init__(
        self,
        hop_amps: list,
        frequency: float,
        eta: float,
        mus: np.array,
        V: float,
        beta: float,
        k_lims: tuple,
        resolution=100,
    ) -> None:
        """Initializing Model attributes to actual properties of
        instances.
        """
        # Global attributes
        self.fig = plt.figure("Spectral weight")
        self.omega = frequency
        self.eta = eta
        self.mus = mus
        self.V = V
        self.hops = hop_amps
        self.beta = beta

        # Phase space grid
        dks = (k_lims[1] - k_lims[0]) / resolution
        kx = arange(k_lims[0], k_lims[1], dks)
        self.kx, self.ky = meshgrid(kx, kx)
        self.normalize = 1 / self.kx.shape[0] ** 2

        # Energy derivatives grids
        dEs = dE(hop_amps, self.kx, self.ky, mus)
        self.E, self.dEx, self.ddEx, self.dEy, self.ddEy, self.dExEy = dEs

        # Spectral weight array
        self.A = get_spectral_weight(frequency, eta, mus, self.E)

    def plot_spectral_weight(self) -> plt.figure:
        """Ouputs the spectral weight as a 2D numpy array.

        Returns
        -------
        -: plt.figure
            2D graph of spectral weight.
        """
        # Data for mu = 0
        idx = len(self.A) // 2
        data = self.A[idx]

        ax = self.fig.add_subplot()
        spectral = ax.pcolormesh(self.kx, self.ky, data, cmap=cm.Blues)
        self.fig.colorbar(spectral)

        # Graph format & style
        ax.set_title("$\mu = {:.2f}$".format(self.mus[idx]))
        ax.set_xlabel("$k_x$")
        ax.set_ylabel("$k_y$")

        min, max = self.kx[0, 0], self.kx[-1, -1]
        ax.set_xticks(ticks=[min, 0, max], labels=["$-\pi$", "0", "$\pi$"])
        ax.set_yticks(ticks=[min, 0, max], labels=["$-\pi$", "0", "$\pi$"])
        plt.show()

        return

    def sigma_ii(self, variable: str) -> list:
        """Computing longitudinal conductivity at zero temperature
        in the zero-frequency limit when interband transitions can be
        neglected.

        Parameters
        ----------
        variable: str, default=None
            Axis on which compute conductivity.

        Returns
        -------
        conductivity: float
        """
        conductivity = []
        coeff = e**2 * pi / self.V

        if variable == "x":
            dE = self.dEx

        elif variable == "y":
            dE = self.dEy

        for A in self.A:
            c = coeff * dE**2 * A**2
            conductivity.append(c.sum())

        return conductivity

    def sigma_xy(self) -> list:
        """Computing transversal conductivity at zero temperature
        in the zero-frequency limit when interband transitions can be
        neglected.

        Returns
        -------
        conductivity: float
        """
        conductivity = []
        coeff = e**3 * pi**2 / (3 * self.V)
        c1 = -2 * self.dEx**2 * self.dExEy
        c2 = self.dEx**2 * self.ddEy
        c3 = self.dEy**2 * self.ddEx

        for A in self.A:
            c = coeff * (c1 + c2 + c3) * A**3
            conductivity.append(c.sum())

        return conductivity

    def get_density(self) -> list:
        """Docs"""
        densities = []
        for energies in self.E:
            density = 1.0 / (1.0 + exp(self.beta * energies))
            densities.append(self.normalize * density.sum())

        return densities

    def get_hall_nb(self) -> list:
        """Computes Hall number.

        Returns
        -------
        n_H: float
            Hall number
        """
        n_H = []
        for xx, yy, xy in zip(self.sigma_ii("x"), self.sigma_ii("y"), self.sigma_xy()):
            n_H.append(self.V * xx * yy / (e * xy))

        return n_H


if __name__ == "__main__":
    N = Model(
        hop_amps=[1.0, -0.2, 0.3],
        frequency=0.0,
        eta=0.05,
        mus=np.linspace(-4, 4, 200),
        V=1.0,
        beta=100,
        k_lims=(-pi, pi),
        resolution=1000,
    )

    # Spectral weight
    N.plot_spectral_weight()

    # # Conductivities
    # sigma_xx = N.sigma_ii('x')
    # sigma_yy = N.sigma_ii('y')
    # sigma_xy = N.sigma_xy()

    # # Density
    # n = N.get_density()
    #
    # plt.plot(N.mus, n, label="Conduction electrons density $n(\mu)$")
    # plt.xlabel("$\mu$")
    # plt.ylabel("Density $n$")
    # plt.legend()
    # plt.show()
