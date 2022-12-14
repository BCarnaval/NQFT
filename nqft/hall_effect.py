"""Hall effect in cuprates with an incommensurate collinear
spin-density wave.
"""

import numpy as np
from rich import print
from scipy.constants import pi
import matplotlib.pyplot as plt
from numpy import arange, meshgrid, sin, cos, exp, linspace

from nqft.functions import read_fermi_arc, find_nearest, make_cmap, timeit


@timeit
def get_energies(hops: tuple[float], kx: np.ndarray, ky: np.ndarray,
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

    Examples
    --------
    >>> hops = (1.0, -0.3, 0.2)
    >>> ks, mus = np.linspace(-np.pi, np.pi, 2), np.linspace(-4, 4, 4)
    >>> kx, ky = np.meshgrid(ks, ks)
    >>> get_energies(hops, kx, ky, mus)
    (
        array([[[8.4       , 8.4       ],
            [8.4       , 8.4       ]],
           [[5.73333333, 5.73333333],
            [5.73333333, 5.73333333]],
           [[3.06666667, 3.06666667],
            [3.06666667, 3.06666667]],
           [[0.4       , 0.4       ],
            [0.4       , 0.4       ]]]),
        {
            'dE_dx': array([[-1.95943488e-16,  1.95943488e-16],
           [-1.95943488e-16,  1.95943488e-16]]),
            'ddE_dxx': array([[-1.6, -1.6],
           [-1.6, -1.6]]),
            'dE_dy': array([[-1.95943488e-16, -1.95943488e-16],
           [ 1.95943488e-16,  1.95943488e-16]]),
            'ddE_dyy': array([[-1.6, -1.6],
           [-1.6, -1.6]]),
            'ddE_dxdy': array([[-0., -0.],
           [-0., -0.]])
        }
    )
    """
    # Energy
    t, tp, tpp = hops
    a = -2 * t * (cos(kx) + cos(ky))
    b = -2 * tp * (cos(kx + ky) + cos(kx - ky))
    c = -2 * tpp * (cos(2 * kx) + cos(2 * ky))

    E = ((a + b + c)[..., None] - mus).T

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

    Examples
    --------
    >>> hops = (1.0, -0.3, 0.2)
    >>> ks, mus = np.linspace(-np.pi, np.pi, 2), np.linspace(-4, 4, 4)
    >>> kx, ky = np.meshgrid(ks, ks)
    >>> E, dEs = get_energies(hops, kx, ky, mus)
    >>> get_spectral_weight(0.0, 0.05, E)
    (
        array([[[0.00022555, 0.00022555],
            [0.00022555, 0.00022555]],
           [[0.00048414, 0.00048414],
            [0.00048414, 0.00048414]],
           [[0.00169189, 0.00169189],
            [0.00169189, 0.00169189]],
           [[0.0979415 , 0.0979415 ],
            [0.0979415 , 0.0979415 ]]]),
        array([[0., 0.],
           [0., 0.]])
    )
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

    return diag_filter[None, ...] * A.imag, diag_line


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

    use_peters: tuple[int], default=(None, 200)
        Determines if model's based on Peter R. spectral functions. User must
        choose between None, 36 and 64 for the first element representing the
        number of sites and between 200 and 500 for the second element
        representing the resolution of momentum space.

        (Note: User must let first element as 'None' to use non-interacting
        spectrums normally.)

    use_filter: bool, default=False
        Determines if spectral weights will be filtered using diamond shape
        filter to create artificial Fermi arcs.

        (Note: Can only be used when use_peters[0] is None)
    """

    def __init__(self, hoppings: tuple[float], broadening: float, omega=0.0,
                 mus=(-4, 4, 0.02), resolution=600, use_peters=(None, 200),
                 use_filter=False) -> None:
        """Initializing specified attributes.
        """
        self.w = omega
        self.use_peters = use_peters
        peter_sites, peter_dim = use_peters

        if peter_sites:
            self.hops = (1.0, -0.3, 0.2)

            if peter_sites == 36:
                self.eta = 0.1
                self.norm = 1 / 200**2
                self.mus = np.array([-1.3, -1.3, -1.0, -0.75, -0.4, -0.4, 0.0])
                self.peter_density = np.array(
                    [0.6666, 0.7222, 0.7777, 0.8333, 0.8888, 0.9444, 1.0])

            elif peter_sites == 64:

                self.mus = np.array([-1.3, -0.8, -0.55, -0.1, 0.0])
                self.peter_density = np.array(
                    [0.75, 0.8125, 0.875, 0.9375, 1.0])

                if peter_dim == 200:
                    self.eta = 0.05
                    self.norm = 1 / 200**2

                elif peter_dim == 500:
                    self.eta = 0.1
                    self.norm = 1 / 500**2

            k_s = linspace(-pi, pi, peter_dim)
            self.k_x, self.k_y = meshgrid(k_s, k_s)

            self.E, self.dEs = get_energies(
                hops=hoppings, kx=self.k_x, ky=self.k_y, mus=self.mus)

            self.A = np.array(
                [array for array in read_fermi_arc(size=peter_sites,
                                                   res=peter_dim).values()])

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
        spectral_mu = self.A[idx, :, :]

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

        dim = self.use_peters[1]
        peters_spectrum = read_fermi_arc(size=size, res=dim)[key]
        peters_title = "Peter's model: {}".format(key)
        axes[1].set_title(peters_title)
        k_x, k_y = meshgrid(linspace(-pi, pi, dim), linspace(-pi, pi, dim))

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

        # Plot the superposition of the firsts
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

        sigma_ii = (dE**2)[None, ...] * self.A**2
        conductivity = -1 * sigma_ii.sum(axis=1).sum(axis=1)

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

        sigma_ij = (c1 + c2 + c3)[None, ...] * self.A**3
        conductivity = -1 * sigma_ij.sum(axis=1).sum(axis=1)

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
        density = self.norm * fermi_dirac.sum(axis=1).sum(axis=1)

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

    def plot_hall(self, save_path=None) -> plt.Figure:
        """Outputs a plot of the Hall coefficient as a
        function of doping (1 - density).

        Parameters
        ----------
        save_path: str, default=None
            Determines the path in which save n_H(p) as a text file.

        Returns
        -------
        -: plt.Figure
            Graph of Hall number as a function of hole doping.
        """
        _, ax = plt.subplots()

        if self.use_peters[0]:
            doping = 1 - self.peter_density

        else:
            doping = 1 - self.get_density()
            ax.set_ylim([-2, 2])

        hall_coeffs = self.get_hall_nb()

        if save_path:
            np.savetxt(
                fname=save_path,
                X=np.transpose([doping, hall_coeffs]),
                delimiter=' ',
                header='doping hall_coeff')
        else:
            pass

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
        use_peters=(36, 200)
    )

    N.plot_hall()
