"""Hall effect in cuprates with an incommensurate collinear
spin-density wave.
"""

import time
import numpy as np
from rich import print
from functools import wraps
from scipy.constants import pi
import matplotlib.pyplot as plt
from numpy import arange, meshgrid, sin, cos, exp, linspace

from nqft.functions import read_fermi_arc, find_nearest, make_cmap


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
    hops: tuple, default=None
        Hopping amplitudes coefficients.

    kx: np.ndarray, shape=(N, N), default=None
        kx space as a 2D array.

    ky: np.ndarray, shape=(N, N), default=None
        ky space as a 2D array.

    mus: np.array, size=M, default=None
        Chemical potential values array.

    Returns
    -------
    E, dEs: tuple[np.ndarray, dict], size=2
        Energies and it's derivatives in a tuple.
    """
    # Energy
    t, tp, tpp = hops
    a = -2 * t * (cos(kx) + cos(ky))
    b = -2 * tp * (cos(kx + ky) + cos(kx - ky))
    c = -2 * tpp * (cos(2 * kx) + cos(2 * ky))

    E = (a + b + c)[..., None] - mus

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
                        filter=False) -> tuple[np.ndarray]:
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
        Determines if we use diamond filter over spectral weights.

    Returns
    -------
    A, diag_line: tuple[np.ndarray], size=2
        Spectral weight and diamond line array to plot over.
    """
    dim = E.shape[1]
    diag_filter = np.ones(shape=(dim, dim))
    diag_line = np.ones(shape=(dim, dim))
    buffer = np.linspace(-pi, pi, dim)
    for i in range(dim):
        for j in range(dim):
            if round(abs(buffer[i]) + abs(buffer[j]), 2) == round(pi, 2):
                diag_line[i, j] = 1.0
            else:
                diag_line[i, j] = 0.0

    if filter:
        for i in range(dim):
            for j in range(dim):
                if abs(buffer[i]) + abs(buffer[j]) <= pi:
                    diag_filter[i, j] = 1.0
                else:
                    diag_filter[i, j] = 0.0
    else:
        pass

    A = -1 / pi * (1 / (omega + eta * 1j - E))

    return diag_filter[..., None] * A.imag, diag_line


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

    resolution: int, default=600
        Resolution of phase space (k_x, k_y).

    use_peters: int, default=None
        Determines if model's based on Peter R. spectral functions. User must
        choose between (None, 36, 64) where 36, 64 represent the number of
        sites of studied model.

    use_filter: bool, default=False
        Determines if spectral weights will be filtered using diamond shape
        filter to create artificial Fermi arcs.
    """

    def __init__(self, hoppings: tuple[float], broadening: float, omega=0.0,
                 mus=(-4, 4, 0.02), resolution=600, use_peters=None,
                 use_filter=False) -> None:
        """Initializing specified attributes.
        """
        self.w = omega
        self.use_peters = use_peters

        if use_peters:
            self.norm = 1 / 200**2
            self.hops = (1.0, -0.3, 0.2)

            if self.use_peters == 36:
                self.eta = 0.1
                self.mus = np.array([-1.3, -1.3, -1.0, -0.75, -0.4, -0.4, 0.0])

            elif self.use_peters == 64:
                self.eta = 0.05
                self.mus = np.array([-1.3, -0.8, -0.55, -0.1, 0.0])

            k_s = linspace(-pi, pi, 200)
            self.k_x, self.k_y = meshgrid(k_s, k_s)

            self.E, self.dEs = get_energies(
                hops=hoppings, kx=self.k_x, ky=self.k_y, mus=self.mus)

            self.A = np.array(
                [array for array in read_fermi_arc(size=use_peters).values()]
            ).T

        else:
            self.hops = hoppings
            self.eta = broadening
            self.mus = arange(*mus)
            self.norm = 1 / resolution**2

            k_s = linspace(-pi, pi, resolution)
            self.k_x, self.k_y = meshgrid(k_s, k_s)

            self.E, self.dEs = get_energies(
                hops=hoppings, kx=self.k_x, ky=self.k_y, mus=self.mus)

            self.A, self.diamond = get_spectral_weight(
                omega=omega, eta=broadening, E=self.E, filter=use_filter)

        return

    def plot_spectral_weight(self, mu: float, size=36, key=None) -> plt.Figure:
        """Ouputs a matplotlib figure containing 3 subplots. Left one
        represents the spectral function of non-interacting model. Center one
        the spectrum comming from Peter's article. Right one the superposition
        of the firsts.

        Parameters
        ----------
        mu: float, default=None
            Chemical potential at which we observe spectral function.

        size: int, default=36
            Size of Peter's model used to compare with non-interacting model.

        key: str, default=None
            Key of the dictionnary containing Peter's spectrums. For example,
            the 64 sites model has ('N48', 'N52', 'N56', 'N60' and 'N64').

        Returns
        -------
        -: plt.figure
            2D graphs of spectral weights.
        """
        # Figure settings
        fig, axes = plt.subplots(ncols=3, tight_layout=True, figsize=(10, 3))
        axes[0].set(adjustable='box', aspect='equal')
        axes[1].set(adjustable='box', aspect='equal')
        axes[2].set(adjustable='box', aspect='equal')

        # Spectral weight for a given mu
        idx = find_nearest(self.mus, mu)
        spectral_mu = self.A[:, :, idx]

        # Fig title
        title = "$\\mu = {:.2f}$".format(mu)
        axes[0].set_title(title)

        # Plot spectral weight

        axes[0].contour(self.k_x, self.k_y, self.diamond, linewidths=0.6)
        spectral = axes[0].contourf(
            self.k_x,
            self.k_y,
            spectral_mu,
            cmap=make_cmap(['#FFFFFF', '#ae6a47']),
            extend='both',
        )
        fig.colorbar(spectral)

        peters_spectrum = read_fermi_arc(size=size)[key]
        peters_title = "Peter's model: {}".format(key)
        k_x, k_y = meshgrid(linspace(-pi, pi, 200), linspace(-pi, pi, 200))

        axes[1].set_title(peters_title)
        # Plot one of Peter's spectral weight
        axes[1].contour(self.k_x, self.k_y, self.diamond, linewidths=0.6)
        spectral_peter = axes[1].contourf(
            k_x,
            k_y,
            peters_spectrum,
            cmap=make_cmap(['#FFFFFF', '#8b4049']),
            extend='both'
        )
        fig.colorbar(spectral_peter)

        axes[2].contour(self.k_x, self.k_y, self.diamond, linewidths=0.6)
        spectral = axes[2].contourf(
            self.k_x,
            self.k_y,
            spectral_mu,
            cmap=make_cmap(['#FFFFFF', '#ae6a47']),
            extend='both',
        )
        spectral_peter = axes[2].contourf(
            k_x,
            k_y,
            peters_spectrum,
            cmap=make_cmap(['#FFFFFF', '#543344']),
            alpha=0.6,
            extend='both'
        )
        fig.colorbar(spectral_peter)

        # Graph format & style
        min, max = self.k_x[0, 0], self.k_x[-1, -1]
        axes_labels = ["$-\\pi$", "$0$", "$\\pi$"]

        # Axes and ticks
        axes[0].set_ylabel("$k_y$")
        for idx in range(3):
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
            Axis on which compute conductivity. (ex: 'x' or 'y')

        Returns
        -------
        conductivity: np.array, size=M
        """
        if variable == "x":
            dE = self.dEs['dE_dx']

        elif variable == "y":
            dE = self.dEs['dE_dy']

        sigma_ii = (dE**2)[..., None] * self.A**2
        conductivity = -1 * sigma_ii.sum(axis=1).sum(axis=0)

        return conductivity

    def sigma_ij(self) -> np.array:
        """Computing transversal conductivity at zero temperature
        in the zero-frequency limit when interband transitions can be
        neglected.

        Returns
        -------
        conductivity: np.array, size=M
        """
        c1 = -2 * self.dEs['dE_dx'] * self.dEs['dE_dy'] * self.dEs['ddE_dxdy']
        c2 = self.dEs['dE_dx']**2 * self.dEs['ddE_dyy']
        c3 = self.dEs['dE_dy']**2 * self.dEs['ddE_dxx']

        sigma_ij = (c1 + c2 + c3)[..., None] * self.A**3
        conductivity = -1 * sigma_ij.sum(axis=1).sum(axis=0)

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
        density = self.norm * fermi_dirac.sum(axis=1).sum(axis=0)

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
        n_H = 6 * self.norm * s_xx * s_yy / s_xy

        return n_H

    def plot_hall(self) -> plt.Figure:
        """Outputs a plot of the Hall coefficient as a
        function of doping (1 - density).

        Returns
        -------
        -: plt.Figure
            Graph of Hall number as a function of hole doping.
        """
        _, ax = plt.subplots()

        if self.use_peters:
            if self.use_peters == 36:
                ax.set_ylim([0, 2])
                doping = 1 - np.array(
                    [0.66666666666, 0.72222222222, 0.77777777777,
                     0.83333333333, 0.88888888888, 0.94444444444, 1.0]
                )
            elif self.use_peters == 64:
                ax.set_ylim([0, 1.5])
                doping = 1 - np.array([0.75, 0.8125, 0.875, 0.9375, 1.0])

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


if __name__ == "__main__":
    N = Model(
        hoppings=(1.0, -0.3, 0.2),
        broadening=0.05,
        mus=(-4, 4, 0.05),
        use_peters=None,
        use_filter=False
    )

    N.plot_hall()
