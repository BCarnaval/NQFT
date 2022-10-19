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
from dataclasses import dataclass, field
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
def dE(hop_amps: tuple, kx: np.ndarray, ky: np.ndarray, mu: np.array) -> tuple:
    """Outputs model's energies and it's derivatives.

    Parameters
    ----------
    hop_amps: tuple, default=None
        Hopping amplitudes coefficients.

    kx: np.ndarray, shape=(N, N), default=None
        kx space as a 2D array.

    ky: np.ndarray, shape=(N, N), default=None
        ky space as a 2D array.

    mu: np.array, size=M, default=None
        Chemical potential values array.

    Returns
    -------
    E, dEs: tuple, size=2
        Energies and it's derivatives in a tuple.
    """
    # Energy
    t, t1, t2 = hop_amps
    a = -2 * t * (cos(kx) + cos(ky))
    b = -4 * t1 * cos(kx) * cos(ky)
    c = -2 * t2 * (cos(2 * kx) + cos(2 * ky))

    E = np.array([a + b + c - i for i in mu])

    dEs = {
            'dE_dx': None,
            'ddE_dxx': None,
            'dE_dy': None,
            'ddE_dyy': None,
            'ddE_dxdy': None
            }

    # Ex derivatives
    dEs['dE_dx'] = 2 * t * sin(kx) + \
        2 * t1 * (sin(kx + ky) + sin(kx - ky)) + \
        4 * t2 * sin(2 * kx)

    dEs['ddE_dxx'] = 2 * t * cos(kx) + \
        2 * t1 * (cos(kx + ky) + cos(kx - ky)) + \
        8 * t2 * cos(2 * kx)

    # Ey derivatives
    dEs['dE_dy'] = 2 * t * sin(ky) + \
        2 * t1 * (sin(kx + ky) - sin(kx - ky)) + \
        4 * t2 * sin(2 * ky)

    dEs['ddE_dyy'] = 2 * t * cos(ky) + \
        2 * t1 * (cos(kx + ky) + cos(kx - ky)) + \
        8 * t2 * cos(2 * ky)

    # Mixed derivative
    dEs['ddE_dxdy'] = 2 * t1 * (cos(kx + ky) - cos(kx - ky))

    return E, dEs


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


@dataclass(kw_only=True)
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

    mu_lims: tuple, size=3, default=None
        Chemical potential interval values and
        number of them.

    V: float, default=None
        Normalization volume.

    k_lims: tuple, size=2, default=(-pi, pi)
        Wavevectors interval values.

    resolution: int, default=100
        Resolution of phase space (kx, ky).
    """
    # Non-default attributes
    hopping_amplitudes: tuple[float]
    omega: float
    eta: float
    mu_lims: tuple[float]
    v: float
    beta: float

    # Phase space init
    k_x: np.ndarray = field(init=False, repr=False)
    k_y: np.ndarray = field(init=False, repr=False)

    # Energies init
    mus: np.array = field(init=False, repr=False)
    E: np.ndarray = field(init=False, repr=False)
    dEs: dict[np.ndarray] = field(init=False, repr=False)
    norm: float = field(init=False)

    # Spectral functions init
    A: np.ndarray = field(init=False, repr=False)
    A_Peter: dict[np.ndarray] = field(init=False, repr=False)

    # Matplotlib instances init
    fig: plt.Figure = field(init=False)
    grid: plt.GridSpec = field(init=False)

    # Default attributes
    k_lims: tuple[float] = (-pi, pi)
    resolution: int = 100

    def __post_init__(self) -> None:
        """Computing post-init class attributes.
        """
        # Chemical potential array
        self.mus = np.linspace(*self.mu_lims)

        # Matplotlib instances (figure and subplots)
        self.fig, self.axes = plt.subplots(ncols=2, tight_layout=True)

        # Phase space grid(s)
        dks = (self.k_lims[1] - self.k_lims[0]) / self.resolution
        ks = arange(*self.k_lims, dks)
        self.k_x, self.k_y = meshgrid(ks, ks)
        self.norm = 1 / ks.shape[0] / ks.shape[0]

        # Energies and it's derivatives (firsts and seconds)
        self.E, self.dEs = dE(
                                self.hopping_amplitudes,
                                self.k_x,
                                self.k_y,
                                self.mus
                                )

        # Spectral functions (Non-interacting model + Peter's)
        self.A_Peter = read_fermi_arc()
        self.A = get_spectral_weight(self.omega, self.eta, self.E)

        return

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

        # Fig title
        title = "$\\mu = {:.2f}$".format(mu)
        self.axes[0].set_title(title)

        # Plot spectral weight
        spectral = self.axes[0].pcolormesh(
                    self.k_x,
                    self.k_y,
                    spectral_mu,
                    cmap=cm.Blues,
                    label="$\\mu = {:.2f}$".format(self.mus[idx]),
        )
        self.fig.colorbar(spectral)

        # Condition to plot Peter's data over colormesh (with some alpha)
        if electron_nb:
            # Fig title
            title_peter = "Peter's model: {}".format(electron_nb)
            self.axes[1].set_title(title_peter)

            # Phase space array
            k_arrays = pi * self.A_Peter["coords"]
            kx_g, ky_g = meshgrid(k_arrays, k_arrays)

            # Plot one of Peter's spectral weight
            self.axes[1].pcolormesh(
                                self.k_x,
                                self.k_y,
                                spectral_mu,
                                cmap=cm.Blues
            )
            spectral_peter = self.axes[1].pcolormesh(
                            kx_g,
                            ky_g,
                            self.A_Peter[electron_nb],
                            cmap=cm.Oranges,
                            alpha=0.6
            )
            self.fig.colorbar(spectral_peter)

        # Graph format & style
        min, max = self.k_x[0, 0], self.k_x[-1, -1]
        axes_labels = ["$-\\pi$", "$0$", "$\\pi$"]

        # Axes and ticks
        for idx in range(2):
            self.axes[idx].set_xlabel("$k_x$")
            self.axes[idx].set_ylabel("$k_y$")
            self.axes[idx].set_xticks(ticks=[min, 0, max], labels=axes_labels)
            self.axes[idx].set_yticks(ticks=[min, 0, max], labels=axes_labels)

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
        coeff = e**2 * pi / self.v

        if variable == "x":
            dE = self.dEs['dE_dx']

        elif variable == "y":
            dE = self.dEs['dE_dy']

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
        coeff = e**3 * pi**2 / (3 * self.v)
        c1 = -2 * self.dEs['dE_dx']**2 * self.dEs['ddE_dxdy']
        c2 = self.dEs['dE_dx']**2 * self.dEs['ddE_dyy']
        c3 = self.dEs['dE_dy']**2 * self.dEs['ddE_dxx']

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
        n_H = self.norm * self.v * s_xx * s_yy / (e * s_xy)

        return n_H


if __name__ == "__main__":
    N = Model(
        hopping_amplitudes=(1.0, -0.2, 0.3),
        omega=0.0,
        eta=0.1,
        mu_lims=(-4, 4, 200),
        v=1.0,
        beta=100,
        resolution=200,
    )

    # Spectral weight
    peter_model, peter_density = "N36", 0.889
    mu_idx = find_nearest(N.get_density(), peter_density)
    mu = N.mus[mu_idx]

    N.plot_spectral_weight(mu=-2, electron_nb=peter_model)
