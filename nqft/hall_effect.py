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
from numpy import arange, meshgrid, sin, cos, exp, linspace

from nqft.functions import read_fermi_arc, find_nearest


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__} took {total_time:.2f} seconds.")
        return result
    return timeit_wrapper


@timeit
def get_energies(hops: tuple, kx: np.ndarray, ky: np.ndarray,
                 mus: np.array) -> tuple:
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
    t, tp, tpp = hops
    a = -2 * t * (cos(kx) + cos(ky))
    b = -2 * tp * (cos(kx + ky) + cos(kx - ky))
    c = -2 * tpp * (cos(2 * kx) + cos(2 * ky))

    E = np.array([a + b + c - i for i in mus])

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
def get_spectral_weight(omega: float, eta: float, E: np.ndarray,
                        filter=False) -> np.ndarray:
    """Ouputs the spectral weight as a 3D numpy array.

    Parameters
    ----------
    omega: float, default=None
        Frequency at which we observe the fermi surface.

    eta: float default=None
        Lorentzian broadening module.

    E: np.ndarray. shape=(M, N, N), default=None
        Eigenenergies of the system as a 3D numpy array.

    filter: bool, default=False
        Determines if we use diagonal filter over spectral weights.

    Returns
    -------
    A: np.ndarray, shape=(M, N, N)
        Spectral weight.
    """
    dim = E.shape[1]
    diag_filter = np.ones(shape=(dim, dim))
    if filter:
        buffer = np.linspace(-pi, pi, dim)
        for i in range(dim):
            for j in range(dim):
                if abs(buffer[i]) + abs(buffer[j]) <= pi:
                    diag_filter[i, j] = 1.0
                else:
                    diag_filter[i, j] = 0.0
    else:
        pass

    A = -1 / pi * np.array([1 / (omega + eta * 1j - e) for e in E])

    return diag_filter * A.imag


class Model:
    """Model instance to determine Hall coefficient and
    density from tight-binding hamiltonian.

    Attributes
    ----------
    hoppings: tuple, size=3, default=None
        Hopping amplitude coefficients.

    broadening: float, default=None
        Lorentzian broadening module.

    omega: float, default=None
        Frequency at which we observe the fermi surface.

    mus: tuple, size=3, default=(-4, 4, 0.02)
        Chemical potential interval values and the interval between each
        element of an hypothetical array.

    resolution: int, default=4
        Resolution of phase space (k_x, k_y).

    use_peters: bool, default=False
        Determines if model's based on Peter R. spectral functions (fermi arcs)
    """

    def __init__(self, hoppings: tuple[float], broadening: float, omega=0.0,
                 mus=(-4, 4, 0.02), resolution=400, use_peters=False,
                 use_filter=False) -> None:
        """Docs
        """
        self.w = omega
        self.use_peters = use_peters

        if use_peters:
            self.eta = 0.1
            self.norm = 1 / 200**2
            self.hops = (1.0, -0.3, 0.2)
            self.mus = np.array([-1.3, -1.0, -0.75, -0.4, 0.0])

            k_s = linspace(-pi, pi, 200)
            self.k_x, self.k_y = meshgrid(k_s, k_s)

            self.E, self.dEs = get_energies(
                    hops=hoppings, kx=self.k_x, ky=self.k_y, mus=self.mus)

            self.A = np.array([array for array in read_fermi_arc().values()])

        else:
            self.hops = hoppings
            self.eta = broadening
            self.mus = arange(*mus)
            self.norm = 1 / resolution**2

            k_s = linspace(-pi, pi, resolution)
            self.k_x, self.k_y = meshgrid(k_s, k_s)

            self.E, self.dEs = get_energies(
                    hops=hoppings, kx=self.k_x, ky=self.k_y, mus=self.mus)

            self.A = get_spectral_weight(
                    omega=omega, eta=broadening, E=self.E, filter=use_filter)

        return

    def plot_spectral_weight(self, mu: float, key=None) -> plt.Figure:
        """Ouputs the spectral weight as a 2D numpy array.

        Returns
        -------
        -: plt.figure
            2D graph of spectral weight.
        """
        # Figure settings
        fig, axes = plt.subplots(ncols=2, tight_layout=True, figsize=(9, 4))
        axes[0].set(adjustable='box', aspect='equal')
        axes[1].set(adjustable='box', aspect='equal')

        # Spectral weight for a given mu
        idx = find_nearest(self.mus, mu)
        spectral_mu = self.A[idx]

        # Fig title
        title = "$\\mu = {:.2f}$".format(mu)
        axes[0].set_title(title)

        # Plot spectral weight
        spectral = axes[0].contourf(
            self.k_x,
            self.k_y,
            spectral_mu,
            cmap=cm.Purples,
            extend='max'
        )
        fig.colorbar(spectral)

        peters_spectrum = read_fermi_arc()[key]
        peters_title = "Peter's model: {}".format(key)
        k_x, k_y = meshgrid(linspace(-pi, pi, 200), linspace(-pi, pi, 200))

        # Condition to plot Peter's data over colormesh (with some alpha)
        axes[1].set_title(peters_title)

        # Plot one of Peter's spectral weight
        spectral_peter = axes[1].contourf(
            k_x,
            k_y,
            peters_spectrum,
            cmap=cm.Greens,
            extend='max'
        )
        axes[1].contourf(
            self.k_x,
            self.k_y,
            spectral_mu,
            cmap=cm.Purples,
            alpha=0.6
        )
        fig.colorbar(spectral_peter)

        # Graph format & style
        min, max = self.k_x[0, 0], self.k_x[-1, -1]
        axes_labels = ["$-\\pi$", "$0$", "$\\pi$"]

        # Axes and ticks
        axes[0].set_ylabel("$k_y$")
        for idx in range(2):
            axes[idx].set_xlabel("$k_x$")
            axes[idx].set_xticks(ticks=[min, 0, max], labels=axes_labels)
            axes[idx].set_yticks(ticks=[min, 0, max], labels=axes_labels)

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
        beta = 100
        fermi_dirac = 2.0 / (1.0 + exp(beta * self.E.astype("float128")))
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

    def plot_hall(self) -> plt.Figure:
        """Outputs a plot of the Hall coefficient as a
        function of doping (1 - density).
        """
        _, ax = plt.subplots()

        if self.use_peters:
            doping = 1 - np.array(
                    [0.66666666666, 0.77777777777, 0.83333333333,
                     0.88888888888, 1.0]
            )
            ax.set_ylim([0, 2])

        else:
            doping = 1 - self.get_density()
            ax.set_ylim([-2, 2])

        hall_coeffs = self.get_hall_nb()
        ax.plot(doping, hall_coeffs, ".-", label="$n_H(p)$")
        ax.set_xlabel("$p$")
        ax.set_ylabel("$n_H$")
        ax.legend()
        plt.show()

        return


def fit_lin(x: np.array, a: float, b: float) -> np.array:
    """Linear function.
    """
    return a * x + b


if __name__ == "__main__":
    N = Model(
        hoppings=(1.0, -0.3, 0.2),
        broadening=0.1,
        use_peters=False,
        use_filter=True
    )

    N.plot_spectral_weight(mu=-0.4, key="N32")

    # # Plot Hall number
    # fig, ax = plt.subplots()
    # hall_nb = N.get_hall_nb()
    # # p_densities = 1 - N.get_density()
    # p_densities = 1 - np.array([0.66666666666, 0.77777777777, 0.83333333333,
    #                             0.88888888888, 1.0])
    #
    # # Fitting data
    # p_fit = np.linspace(-0.025, 0.35, 400)
    #
    # popt_s, pcov_s = curve_fit(fit_lin, p_densities[2:], hall_nb[2:])
    # popt_e, pcov_e = curve_fit(fit_lin, p_densities[:2], hall_nb[:2])
    #
    # fitted_start = fit_lin(p_fit, *popt_s)
    # fitted_end = fit_lin(p_fit, *popt_e)
    #
    # # Plots
    # ax.plot(p_fit, fitted_start, ".", color="#5fc75d", markersize=0.5,
    #         label="$y_1(x) = {:.2f}x + {:.2f}$ ".format(*popt_s))
    #
    # ax.plot(p_fit, fitted_end, ".", color="#36868f", markersize=0.5,
    #         label="$y_2(x) = {:.2f}x + {:.2f}$".format(*popt_e))
    #
    # ax.plot(p_fit, fitted_start + 1/2, ".", color="#203671", markersize=0.5,
    #         label="$y_3(x) = y_1(x) + 1/2$".format(*popt_s))
    #
    # ax.plot(p_densities, hall_nb, ".-", color="#0f052d", label="$n_H(p)$")
    #
    # # Annotations
    # ax.annotate(text='', xy=(p_fit[50], fitted_start[50]),
    #             xytext=(p_fit[50], fitted_start[50] + 1/2),
    #             arrowprops=dict(arrowstyle='<->'))
    #
    # ax.annotate(text='', xy=(p_fit[350], fitted_start[350]),
    #             xytext=(p_fit[350], fitted_start[350] + 1/2),
    #             arrowprops=dict(arrowstyle='<->'))
    #
    # ax.annotate(text='$1/2$', xy=(p_fit[55], fitted_start[55] + 0.25))
    # ax.annotate(text='$1/2$', xy=(p_fit[355], fitted_start[355] + 0.25))
    # ax.annotate(text="$y_1(x)$", xy=(p_fit[-4], fitted_start[-4] + 0.05))
    # ax.annotate(text="$y_2(x)$", xy=(p_fit[-4], fitted_end[-4] + 0.05))
    # ax.annotate(text="$y_3(x)$", xy=(p_fit[-4], fitted_start[-4] + 0.55))
    #
    # # Labels & legend
    # ax.set_xlabel("$p$")
    # ax.set_ylabel("$n_H$")
    # ax.set_ylim([0, 2])
    # ax.set_xlim([p_densities[4] - 0.05, p_densities[0] + 0.05])
    #
    # plt.legend()
    # plt.show()
