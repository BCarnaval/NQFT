"""Hall effect in cuprates with an incommensurate collinear
spin-density wave.
"""

import time
import numpy as np
from rich import print
from matplotlib import cm
from functools import wraps
import matplotlib.pyplot as plt
from scipy.constants import pi, e
from scipy.optimize import curve_fit
from dataclasses import dataclass, field
from numpy import arange, meshgrid, sin, cos, exp

from nqft.functions import read_fermi_arc, find_nearest, read_locals


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
    t, tp, tpp = hop_amps
    a = -2 * t * (cos(kx) + cos(ky))
    b = -2 * tp * (cos(kx + ky) + cos(kx - ky))
    c = -2 * tpp * (cos(2 * kx) + cos(2 * ky))

    E = np.array([a + b + c - i for i in mu])

    dEs = {
        'dE_dx': None,
        'ddE_dxx': None,
        'dE_dy': None,
        'ddE_dyy': None,
        'ddE_dxdy': None
    }

    # Ex derivatives
    dEs['dE_dx'] = 2 * (t * sin(kx) +
                        tp * (sin(kx - ky) + sin(kx + ky)) +
                        2 * tpp * sin(2 * kx))

    dEs['ddE_dxx'] = 2 * (t * cos(kx) +
                          tp * (cos(kx - ky) + cos(kx + ky)) +
                          4 * tpp * cos(2 * kx))

    # Ey derivatives
    dEs['dE_dy'] = 2 * (t * sin(ky) +
                        tp * (sin(kx + ky) - sin(kx - ky)) +
                        2 * tpp * sin(2 * ky))

    dEs['ddE_dyy'] = 2 * (t * cos(ky) +
                          tp * (cos(kx + ky) + cos(kx - ky)) +
                          4 * tpp * cos(2 * ky))

    # Mixed derivative
    dEs['ddE_dxdy'] = 2 * tp * (cos(kx + ky) - cos(kx - ky))

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
    use_peter: bool

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
        # Matplotlib instances (figure and subplots)
        self.fig, self.axes = plt.subplots(ncols=2, tight_layout=True)
        self.A_Peter = read_fermi_arc()
        self.A_local = read_locals(shape=(3, 4), interaction=8.0)

        # Spectral functions (Non-interacting model + Peter's)
        if self.use_peter:
            dict = read_fermi_arc()
            self.A = np.array([array for array in dict.values()])
            self.mus = np.array([-1.3, -1.0, -0.75, -0.4, 0.0])

            # Phase space grid(s)
            ks = arange(-pi, pi, 2*pi/200)
            self.k_x, self.k_y = meshgrid(ks, ks)
            self.norm = 1 / ks.shape[0] / ks.shape[0]

            # Energies and it's derivatives (firsts and seconds)
            self.E, self.dEs = dE(
                self.hopping_amplitudes,
                self.k_x,
                self.k_y,
                self.mus
            )

        else:
            # Chemical potential array
            self.mus = np.linspace(*self.mu_lims)

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

            self.A = get_spectral_weight(self.omega, self.eta, self.E)

        return

    def plot_spectral_weight(self, mu: float, type: str, key=None) -> None:
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
        spectral = self.axes[0].contourf(
            self.k_x,
            self.k_y,
            spectral_mu,
            cmap=cm.Purples
        )
        self.fig.colorbar(spectral)

        if type == "local":
            imported_A = self.A_local[key]
            imported_title = "Local model 3x4"

        elif type == "peter":
            imported_A = self.A_Peter[key]
            imported_title = "Peter's model: {}".format(key)

        # Condition to plot Peter's data over colormesh (with some alpha)
        self.axes[1].set_title(imported_title)

        # Plot one of Peter's spectral weight
        spectral_peter = self.axes[1].contourf(
            self.k_x,
            self.k_y,
            imported_A,
            cmap=cm.Greens
        )
        self.axes[1].contourf(
            self.k_x,
            self.k_y,
            spectral_mu,
            cmap=cm.Purples,
            alpha=0.6
        )
        self.fig.colorbar(spectral_peter)

        # Graph format & style
        min, max = self.k_x[0, 0], self.k_x[-1, -1]
        axes_labels = ["$-\\pi$", "$0$", "$\\pi$"]

        # Axes and ticks
        self.axes[0].set_ylabel("$k_y$")
        for idx in range(2):
            self.axes[idx].set_xlabel("$k_x$")
            self.axes[idx].set_xticks(ticks=[min, 0, max], labels=axes_labels)
            self.axes[idx].set_yticks(ticks=[min, 0, max], labels=axes_labels)

        # Show figure's plot
        plt.show()

        return

    def sigma_ii(self, variable: str) -> np.array:
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
        coeff = (-e)**2 * pi
        if variable == "x":
            dE = self.dEs['dE_dx']

        elif variable == "y":
            dE = self.dEs['dE_dy']

        sigma_ii = dE**2 * self.A**2
        conductivity = np.array([2 * coeff * sig.sum() for sig in sigma_ii])

        return conductivity

    def sigma_ij(self) -> np.array:
        """Computing transversal conductivity at zero temperature
        in the zero-frequency limit when interband transitions can be
        neglected.

        Returns
        -------
        conductivity: float
        """
        coeff = (-e)**3 * pi**2 / 3
        c1 = -2 * self.dEs['dE_dx'] * self.dEs['dE_dy'] * self.dEs['ddE_dxdy']
        c2 = self.dEs['dE_dx']**2 * self.dEs['ddE_dyy']
        c3 = self.dEs['dE_dy']**2 * self.dEs['ddE_dxx']

        sigma_ij = (c1 + c2 + c3) * self.A**3
        conductivity = np.array([2 * coeff * sig.sum() for sig in sigma_ij])

        return conductivity

    @timeit
    def get_density(self) -> np.array:
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
    def get_hall_nb(self) -> np.array:
        """Computes Hall number.

        Returns
        -------
        n_H: np.array, size=M
            Hall number.
        """
        s_xy = self.sigma_ij()
        s_xx, s_yy = self.sigma_ii("x"), self.sigma_ii("y")
        n_H = self.norm * s_xx * s_yy / (e * s_xy)

        return n_H


def fit_lin(x: np.array, a: float, b: float) -> np.array:
    """Linear function.
    """
    return a * x + b


if __name__ == "__main__":
    N = Model(
        hopping_amplitudes=(1.0, -0.3, 0.2),
        omega=0.0,
        eta=0.1,
        mu_lims=(-4.0, 4.0, 200),
        v=1.0,
        beta=100,
        resolution=200,
        use_peter=False
    )
    # Plot spectral weight
    # N.plot_spectral_weight(mu=-0.4, type='peter', key='N32')

    # Plot Hall number
    fig, ax = plt.subplots()
    hall_nb = N.get_hall_nb()
    # p_densities = 1 - N.get_density()
    p_densities = 1 - np.array([0.66666666666, 0.77777777777, 0.83333333333,
                                0.88888888888, 1.0])

    # Fitting data
    p_fit = np.linspace(-0.025, 0.35, 400)

    popt_s, pcov_s = curve_fit(fit_lin, p_densities[2:], hall_nb[2:])
    popt_e, pcov_e = curve_fit(fit_lin, p_densities[:2], hall_nb[:2])

    fitted_start = fit_lin(p_fit, *popt_s)
    fitted_end = fit_lin(p_fit, *popt_e)

    # Plots
    ax.plot(p_fit, fitted_start, ".", color="#5fc75d", markersize=0.5,
            label="$y_1(x) = {:.2f}x + {:.2f}$ ".format(*popt_s))

    ax.plot(p_fit, fitted_end, ".", color="#36868f", markersize=0.5,
            label="$y_2(x) = {:.2f}x + {:.2f}$".format(*popt_e))

    ax.plot(p_fit, fitted_start + 1/2, ".", color="#203671", markersize=0.5,
            label="$y_3(x) = y_1(x) + 1/2$".format(*popt_s))

    ax.plot(p_densities, hall_nb, ".-", color="#0f052d", label="$n_H(p)$")

    # Annotations
    ax.annotate(text='', xy=(p_fit[50], fitted_start[50]),
                xytext=(p_fit[50], fitted_start[50] + 1/2),
                arrowprops=dict(arrowstyle='<->'))

    ax.annotate(text='', xy=(p_fit[350], fitted_start[350]),
                xytext=(p_fit[350], fitted_start[350] + 1/2),
                arrowprops=dict(arrowstyle='<->'))

    ax.annotate(text='$1/2$', xy=(p_fit[55], fitted_start[55] + 0.25))
    ax.annotate(text='$1/2$', xy=(p_fit[355], fitted_start[355] + 0.25))
    ax.annotate(text="$y_1(x)$", xy=(p_fit[-4], fitted_start[-4] + 0.05))
    ax.annotate(text="$y_2(x)$", xy=(p_fit[-4], fitted_end[-4] + 0.05))
    ax.annotate(text="$y_3(x)$", xy=(p_fit[-4], fitted_start[-4] + 0.55))

    # Labels & legend
    ax.set_xlabel("$p$")
    ax.set_ylabel("$n_H$")
    ax.set_ylim([0, 2])
    ax.set_xlim([p_densities[4] - 0.05, p_densities[0] + 0.05])

    plt.legend()
    plt.show()
