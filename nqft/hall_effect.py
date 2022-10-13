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

from nqft.functions import read_fermi_arc, find_nearest


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
    """Outputs model's energies and it's derivatives.

    Parameters
    ----------
    hop_amps: list, default=None
        Hopping amplitudes coefficients.

    kx: np.ndarray, shape=(N, N), default=None
        kx space as a 2D array.

    ky: np.ndarray, shape=(N, N), default=None
        ky space as a 2D array.

    mus: np.array, size=M, default=None
        ADD DESCRIPTION

    Returns
    -------
    E, dEx, ddEx, dEy, ddEy, dExEy: tuple, size=6
        Energies and it's derivatives in a tuple.
    """
    # Energy
    t, t1, t2 = hop_amps
    a = -2 * t * (cos(kx) + cos(ky))
    b = -4 * t1 * cos(kx) * cos(ky)
    c = -2 * t2 * (cos(2 * kx) + cos(2 * ky))

    E = np.array([a + b + c - mu for mu in mus])

    # Ex derivatives
    dEx = 2 * t * sin(kx) + \
        2 * t1 * (sin(kx + ky) + sin(kx - ky)) + \
        4 * t2 * sin(2 * kx)

    ddEx = 2 * t * cos(kx) + \
        2 * t1 * (cos(kx + ky) + cos(kx - ky)) + \
        8 * t2 * cos(2 * kx)

    # Ey derivatives
    dEy = 2 * t * sin(ky) + \
        2 * t1 * (sin(kx + ky) - sin(kx - ky)) + \
        4 * t2 * sin(2 * ky)

    ddEy = 2 * t * cos(ky) + \
        2 * t1 * (cos(kx + ky) + cos(kx - ky)) + \
        8 * t2 * cos(2 * ky)

    # Mixed derivative
    dExEy = 2 * t1 * (cos(kx + ky) - cos(kx - ky))

    return E, dEx, ddEx, dEy, ddEy, dExEy


@timeit
def get_spectral_weight(omega: float, eta: float, E: np.ndarray) -> np.ndarray:
    """Ouputs the spectral weight as a 3D numpy array.

    Parameters
    ----------
    omega: float, default=None
        Frequency at which we observe the fermi surface.

    eta: float default=None
        Lorentzian broadening module.

    E: np.ndarray. shape=(M, N, N), default=None
        Eigenenergies of the system as a 3D numpy array.

    Returns
    -------
    A: np.ndarray, shape=(M, N, N)
        Spectral weight.
    """
    A = -1 / pi * np.array([1 / (omega + eta * 1j - e) for e in E])

    return A.imag


class Model:
    """Model instance to determine Hall coefficient and
    density from tight-binding hamiltonian.

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

    def __init__(self, hop_amps: list, frequency: float, eta: float,
                 mus: np.array, V: float, beta: float, k_lims: tuple,
                 resolution=100) -> None:
        """Initializing Model attributes to actual properties of
        instances.
        """
        # Global attributes
        self.fig, self.axes = plt.subplots(1, 1)
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
        self.norm = 1 / self.kx.shape[0] ** 2

        # Energy derivatives grids
        dEs = dE(hop_amps, self.kx, self.ky, mus)
        self.E, self.dEx, self.ddEx, self.dEy, self.ddEy, self.dExEy = dEs

        # Spectral weights arrays
        self.A_peter = read_fermi_arc()
        self.A = get_spectral_weight(frequency, eta, self.E)

    def plot_spectral_weight(self, mu: float, electron_nb=None) -> plt.figure:
        """Ouputs the spectral weight as a 2D numpy array.

        Returns
        -------
        -: plt.figure
            2D graph of spectral weight.
        """
        # Spectral weight for a given mu
        idx = find_nearest(self.mus, mu)
        spectral_mu = self.A[idx]

        # Plot spectral weight
        title = "$\\mu = {:.2f}$".format(mu)
        spectral = self.axes.pcolormesh(
            self.kx,
            self.ky,
            spectral_mu,
            cmap=cm.Blues,
            label="$\\mu = {:.2f}$".format(self.mus[idx]),
        )
        self.fig.colorbar(spectral)

        # Condition to plot Peter's data over colormesh (with some alpha)
        if electron_nb:
            # Phase space array
            k_arrays = pi * self.A_peter["coords"]
            kx_g, ky_g = meshgrid(k_arrays, k_arrays)
            title += ", Peter's model: {}".format(electron_nb)

            # Plot one of Peter's spectral weight
            self.axes.pcolormesh(
                kx_g,
                ky_g,
                self.A_peter[electron_nb],
                cmap=cm.Oranges,
                alpha=0.7,
                label=f"${electron_nb}$",
            )
        else:
            pass

        # Graph format & style
        self.axes.set_title(title)
        min, max = self.kx[0, 0], self.kx[-1, -1]
        axes_labels = ["$-\\pi$", "$0$", "$\\pi$"]

        # Axes and ticks
        self.axes.set_xlabel("$k_x$")
        self.axes.set_ylabel("$k_y$")
        self.axes.set_xticks(ticks=[min, 0, max], labels=axes_labels)
        self.axes.set_yticks(ticks=[min, 0, max], labels=axes_labels)

        # Show figure's plot
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
        coeff = e**2 * pi / self.V

        if variable == "x":
            dE = self.dEx

        elif variable == "y":
            dE = self.dEy

        sigma_ii = coeff * dE**2 * self.A**2
        conductivity = np.array([sigma.sum() for sigma in sigma_ii])

        return conductivity

    def sigma_ij(self) -> list:
        """Computing transversal conductivity at zero temperature
        in the zero-frequency limit when interband transitions can be
        neglected.

        Returns
        -------
        conductivity: float
        """
        coeff = e**3 * pi**2 / (3 * self.V)
        c1 = -2 * self.dEx**2 * self.dExEy
        c2 = self.dEx**2 * self.ddEy
        c3 = self.dEy**2 * self.ddEx

        sigma_ij = coeff * (c1 + c2 + c3) * self.A**3
        conductivity = np.array([sigma.sum() for sigma in sigma_ij])

        return conductivity

    @timeit
    def get_density(self) -> list:
        """Computes electron density.

        Returns
        -------
        density: np.array, size=M
            Electron density.
        """
        fermi_dirac = 2.0 / (1.0 + exp(self.beta * self.E.astype("float128")))
        density = np.array([self.norm * func.sum() for func in fermi_dirac])

        return density

    @timeit
    def get_hall_nb(self) -> list:
        """Computes Hall number.

        Returns
        -------
        n_H: np.array, size=M
            Hall number.
        """
        s_xy = self.sigma_ij()
        s_xx, s_yy = self.sigma_ii("x"), self.sigma_ii("y")
        n_H = self.norm * self.V * s_xx * s_yy / (e * s_xy)

        return n_H


if __name__ == "__main__":
    N = Model(
        hop_amps=[1.0, -0.2, 0.3],
        frequency=0.0,
        eta=0.1,
        mus=np.linspace(-4, 4, 200),
        V=1.0,
        beta=100,
        k_lims=(-pi, pi),
        resolution=400,
    )

    # Spectral weight
    peter_model, peter_density = "N32", 0.889
    mu_idx = find_nearest(N.get_density(), peter_density)

    N.plot_spectral_weight(mu=-0.1, electron_nb="N32")

    # Density
    # n = N.get_density()

    # Hall number
    # n_H = N.get_hall_nb()
