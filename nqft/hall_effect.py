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
from numpy import arange, meshgrid, sin, cos


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} took {total_time:.4f} seconds\n')
        return result
    return timeit_wrapper


@timeit
def dE(hop_amps: list, kx: np.ndarray, ky: np.ndarray) -> tuple:
    """Outputs model's energy and it's derivatives.

    Parameters
    ----------
    hop_amps: list, default=None
        Hopping amplitudes coefficients.

    kx: np.ndarray, shape=(N, N), default=None
        kx space as a 2D array.

    ky: np.ndarray, shape=(N, N), default=None
        ky space as a 2D array.

    raw: bool, default=True
        If set to False, the function computes all first
        and second derivatives.

    Returns
    -------
    E, dEx, ddEx, dEy, ddEy, dExEy: tuple, size=6
        Energy and it's derivatives in a tuple.
    """
    t, t1, t2 = hop_amps

    # Energy
    a = -2*t*(cos(kx) + cos(ky))
    b = -2*t1*(cos(kx + ky) + cos(kx - ky))
    c = -2*t2*(cos(2*kx) + cos(2*ky))
    E = a + b + c

    # Ex derivatives
    dEx = 2*t*sin(kx) + 2*t1*(sin(kx + ky) + sin(kx - ky)) + 4*t2*sin(2*kx)
    ddEx = 2*t*cos(kx) + 2*t1*(cos(kx + ky) + cos(kx - ky)) + 8*t2*cos(2*kx)

    # Ey derivatives
    dEy = 2*t*sin(ky) + 2*t1*(sin(kx + ky) - sin(kx - ky)) + 4*t2*sin(2*ky)
    ddEy = 2*t*cos(ky) + 2*t1*(cos(kx + ky) + cos(kx - ky)) + 8*t2*cos(2*ky)

    # Mixed derivative
    dExEy = 2*t1*(cos(kx + ky) - cos(kx - ky))

    return E, dEx, ddEx, dEy, ddEy, dExEy

@timeit
def get_spectral_weight(omega: float, eta: float, mu: float,
        E: np.ndarray) -> np.ndarray:
    """Ouputs the spectral weight as a 2D numpy array.

    Parameters
    ----------
    omega: float, default=None
        Frequency at which we observe the fermi surface.
    eta: float default=None
        Lorentzian broadening module.
    mu: float, default=0.0
        ADD DESCRIPTION.
    E: np.ndarray. shape=(N, N), default=None
        Eigenenergies of the system as a 2D numpy array.

    Returns
    -------
    A: np.ndarray, shape=(N, N)
        Spectral weight.
    """
    # Spectral weight
    A1 = omega**2 + E**2 + eta**2 + mu**2
    A2 = -2*(omega*E + E*mu - omega*mu)
    A = 1/pi*(eta/(A1 + A2))

    return A

class Model():
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

    mu: float, default=None
        Constant in dispertion relation.

    k_lims: tuple, size=2
        Wavevectors interval values.

    resolution: int, default=100
        Resolution of phase space (kx, ky).
    """
    def __init__(self, hop_amps: list, frequency: float, eta: float,
            mu: float, V: float, k_lims: tuple, resolution=100) -> None:
        """Initializing Model attributes to actual properties of
        instances.
        """
        # Global attributes
        self.fig = plt.figure("Spectral weight")
        self.omega = frequency
        self.eta = eta
        self.mu = mu
        self.V = V
        self.hops = hop_amps

        # Phase space grid
        kx = arange(k_lims[0], k_lims[1], 1/resolution)
        self.kx, self.ky = meshgrid(kx, kx)

        # Energy derivatives grids
        dEs = dE(hop_amps, self.kx, self.ky)
        self.E, self.dEx, self.ddEx, self.dEy, self.ddEy, self.dExEy = dEs

        # Spectral weight array
        self.A = get_spectral_weight(omega=frequency, eta=eta, mu=mu, E=self.E)

    def plot_spectral_weight(self) -> plt.figure:
        """Ouputs the spectral weight as a 2D numpy array.

        Returns
        -------
        -: plt.figure
            2D graph of spectral weight.
        """
        # Init plot
        ax = self.fig.add_subplot()
        spectral = ax.pcolormesh(self.kx, self.ky, self.A, cmap=cm.Blues)
        self.fig.colorbar(spectral)

        # Graph format & style
        ax.set_title("$\mu = {:.2f}$".format(self.mu))
        ax.set_xlabel("$k_x$")
        ax.set_ylabel("$k_y$")

        min, max = self.kx[0, 0], self.kx[-1, -1]
        ax.set_xticks(ticks=[min, 0, max], labels=["$-\pi$", "0", "$\pi$"])
        ax.set_yticks(ticks=[min, 0, max], labels=["$-\pi$", "0", "$\pi$"])
        plt.show()

        return

    def sigma_ii(self, variable: str) -> np.ndarray:
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
        coeff = e**2*pi/self.V
        if variable == 'x':
            conductivity = coeff*self.dEx**2*self.A**2
        elif variable == 'y':
            conductivity = coeff*self.dEy**2*self.A**2

        return conductivity.sum()

    def sigma_xy(self) -> np.ndarray:
        """Computing transversal conductivity at zero temperature
        in the zero-frequency limit when interband transitions can be
        neglected.

        Returns
        -------
        conductivity: float
        """
        coeff = e**3*pi**2/(3*self.V)

        c1 = -2*self.dEx**2*self.dExEy
        c2 = self.dEx**2*self.ddEy
        c3 = self.dEy**2*self.ddEx
        conductivity = coeff*(c1 + c2 + c3)*self.A**3

        return conductivity.sum()

    def get_hall_nb(self) -> float:
        """Computes Hall number.

        Returns
        -------
        n_H: float
            Hall number
        """
        n_H = self.V*self.sigma_ii('x')*self.sigma_ii('y')/(e*self.sigma_xy())
        return n_H


if __name__ == "__main__":
    N = Model(
            hop_amps=[1, 0.0, 0.0],
            frequency=0.0,
            eta=0.05,
            mu=0.0,
            V=1.0,
            k_lims=(-pi, pi),
            resolution=100
            )

    # Spectral weight
    N.plot_spectral_weight()

    # # Conductivities
    sigma_xx = N.sigma_ii('x')
    sigma_yy = N.sigma_ii('y')
    sigma_xy = N.sigma_xy()

    # Hall number
    n_H = N.get_hall_nb()
    print("n_H = {:.4e}".format(n_H))
