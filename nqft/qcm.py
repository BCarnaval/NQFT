"""This module is dedicated to 'pyqcm'
experimentation.
"""

import os
import sys
import numpy as np
from rich import print
import importlib as iplib
from matplotlib import cm
from numpy import sin, cos
from scipy.constants import pi
import matplotlib.pyplot as plt

from pyqcm import (
    averages,
    new_cluster_model,
    add_cluster,
    lattice_model,
    interaction_operator,
    hopping_operator,
    sectors,
    cluster_averages,
    new_model_instance,
    set_parameters
)
from pyqcm.spectral import mdc

from nqft.functions import read_fermi_arc


def build_matrix(shape: tuple) -> list:
    """Gives a coordinates matrix of a cluster having
    shape[0]*shape[1] sites.

    Parameters
    ----------
    shape: tuple, shape=(2, 1), default=None
        Shape of sites network.

    Returns
    -------
    array: list, shape=(*shape)
        Nested lists of coordinates.

    Examples
    --------
    >>> build_matrix(shape=(2, 2))
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]
    """
    array, idty = [], np.identity(3, dtype=np.int16)
    for i in range(shape[0]):
        for j in range(shape[1]):
            elem = i * idty[1] + j * idty[0]
            array.append(elem.tolist())

    return array


class QcmModel:
    """QcmModel instance to make the usage of 'pyqcm' easier.

    Attributes
    ----------
    shape: tuple[int], size=2, default=None
        Shape of the source cluster as: (rows, columns).

    filling: int, default=None
        Number of electrons inside each cluster.

    interaction: float, default=None
        Coefficient of interation operator.

    hoppings: tuple[float], default=None
        Hopping amplitudes coefficients.

    broadening: float, default=None
        Lorentzian broadening module.

    w: float, default=None
        Frequency at which we observe the fermi surfaces.

    mu: float, default=None
        Chemical potential.

    resolution: int, default=None
        Resolution of phase space (k_x, k_y).

    tiling_shift: bool, default=None
        Determines if super-vectors are shifted or exactly orthogonals.

    show_spectrum: bool, default=False
        Determines if 'pyqcm.spectral.mdc' displays spectral function when it
        computes it.

    overwrite: bool, default=False
        Determines if the script reuses an already computed model to do further
        calcultations or if the script computes it from scratch.
    """

    def __init__(self, shape: tuple[int], filling: int, interaction: float,
                 hoppings: tuple[float], broadening: float, w: float,
                 mu: float, resolution: int, tiling_shift: bool,
                 show_spectrum=False, overwrite=False) -> None:
        """Initialiazing specified attributes.
        """
        # Cluster geometry related attributes
        self.shape = shape
        self.filling = filling
        self.density = filling / (shape[0] * shape[1])

        # Path related attributes
        u_f = str(interaction).split('.')
        u_str = "".join(u_f if u_f[-1] != '0' else u_f[:-1])

        # Global paths & file names
        self.model_path = f'./nqft/Data/model_{shape[0]}x{shape[1]}'
        self.file_name = f'model_{shape[0]}x{shape[1]}_n{filling}_U{u_str}'
        self.module_name = '.'.join(self.model_path.split('/')[1:])

        # Creating or using relevant directory
        try:
            os.makedirs(self.model_path)
        except FileExistsError:
            print(f'Using dir: {self.model_path}/')

        # Density of states & spectrum file names
        self.dos_file = f'dos/dos_n{filling}_U{u_str}.tsv'
        self.spectrum_file = f'spectrums/spectrum_n{filling}_U{u_str}'

        # Import model's module if it has already been computed
        if overwrite:
            # Building cluster
            new_cluster_model(name="clus", n_sites=shape[0] * shape[1])
            add_cluster(name="clus", pos=[0, 0, 0], sites=build_matrix(shape))

            # Initialiazing lattice using built cluster
            if tiling_shift:
                super_vecs = [[shape[1], 0, 0], [1, shape[0], 0]]
            else:
                super_vecs = [[shape[1], 0, 0], [0, shape[0], 0]]

            lattice_model(name=self.file_name, superlattice=super_vecs)

            # Interaction operator U
            interaction_operator(name="U")

            # Hopping operators (t, tp, tpp)
            hopping_operator(name="t", link=[1, 0, 0], amplitude=-1)
            hopping_operator(name="t", link=[0, 1, 0], amplitude=-1)

            hopping_operator(name="tp", link=[1, 1, 0], amplitude=-1)
            hopping_operator(name="tp", link=[-1, 1, 0], amplitude=-1)

            hopping_operator(name="tpp", link=[2, 0, 0], amplitude=-1)
            hopping_operator(name="tpp", link=[0, 2, 0], amplitude=-1)

            # Setting target sectors
            sectors(R=0, N=filling, S=0)

            # Setting operators parameters
            set_parameters(
                f"""
                U = {interaction}
                t = {hoppings[0]}
                tp = {hoppings[1]}
                tpp = {hoppings[2]}
                mu = {mu}
                """
            )

            # Instancing lattice model
            model = new_model_instance(record=True)
            model.print(filename=f'{self.model_path}/{self.file_name}.py')
            print(f"Module '{self.file_name}' saved in: {self.model_path}")

        else:
            try:
                iplib.import_module(
                    name=f'{self.module_name}.{self.file_name}')

                print(f"Module '{self.file_name}' has been imported.\n")

            except ModuleNotFoundError:
                print(
                    f"Module '{self.file_name}' not found. Consider using "
                    "'overwrite=True' in model definition."
                )
                exit(1)

            except TypeError as type_error:
                print(type_error)
                exit(1)

        # 'pyqcm' Parameters
        self.u = interaction
        self.w = w
        self.mu = mu
        self.t, self.tp, self.tpp = hoppings
        self.eta = broadening
        self.res = resolution

        # Computing spectral weight
        spectral = mdc(
            freq=w,
            nk=resolution,
            eta=broadening,
            sym='RXY',
            # data_file=f'{self.model_path}/{self.spectrum_file}',
            data_file=None,
            show=show_spectrum
        )

        self.spectrum = spectral / pi

        return

    def get_lattice_averages(self, operators: list[str]) -> dict:
        """Computes lattice operator averages.

        Parameters
        ----------
        operators: list, default=None
            Operators on which get average.

        Returns
        -------
        _: dict
            Dictionnary containing operator names as keys and average as
            values.
        """
        return averages(ops=operators)

    def get_cluster_averages(self, operators: list[str]) -> dict:
        """Computes single cluster operator averages.

        Parameters
        ----------
        operators: list, default=None
            Operators to keep in output dictionnary.

        Returns
        -------
        out_dict: dict
            Dictionnary containing operator names as keys and tuples as values
            representing operator average and it's variance.
        """
        avgs = cluster_averages()
        out_dict = {i: j for i, j in avgs.items() if i in operators}

        return out_dict

    def plot_spectrums(self, peter_key: str, type='contourf',
                       save=False) -> plt.Figure:
        """Opens spectrums from (2x2, 3x4, 4x3) models and Peters spectrums
        array to compare the plot for given parameters.

        Parameters
        ----------
        peter_key: str, default=None
            Determines which of Peter's array to compare.
            ('N24', 'N26', 'N28', 'N30', 'N32', 'N34', 'N36')

        type: str, default='contourf'
            Spectral plot type (contourf, pcolormesh).

        save: bool, default=False
            Saves of not the output plot.

        Returns
        -------
        _: matplotlib.pyplot.Figure object
        """
        fig, axes = plt.subplots(ncols=2, tight_layout=True, figsize=(9, 4))
        axes[0].set(adjustable='box', aspect='equal')
        axes[1].set(adjustable='box', aspect='equal')

        # Momentum space grids
        momentums = np.linspace(-np.pi, np.pi, self.res)
        k_x, k_y = np.meshgrid(momentums, momentums)

        # Momentum space grids (Peter data (200, 200))
        momentums = np.linspace(-np.pi, np.pi, 200)
        k_x_p, k_y_p = np.meshgrid(momentums, momentums)

        # Get spectral functions paths
        peter_array = read_fermi_arc()[peter_key]

        # Fig title
        title = "{}/{} fill, $U=${}, $t=$[{}, {}, {}], $\eta=${}".format(
            self.filling,
            self.shape[0]*self.shape[1],
            self.u,
            self.t, self.tp, self.tpp,
            self.eta
        )
        axes[0].set_title(title)

        # Plot spectral weight
        if type == 'contourf':
            low_interaction = axes[0].contourf(
                k_x,
                k_y,
                self.spectrum,
                cmap=cm.Purples,
                extend='max'
            )

        elif type == 'pcolormesh':
            low_interaction = axes[0].pcolormesh(
                k_x,
                k_y,
                self.spectrum,
                cmap=cm.Purples,
            )

        fig.colorbar(low_interaction)

        # Fig title
        title_p = "{}/36 fill, $U=$8.0, $t=$[1, -0.3, 0.2], $\eta=$0.1".format(
            peter_key[1:]
        )

        axes[1].set_title(title_p)

        # Plot one of Peter's spectral weight
        if type == 'contourf':
            peter_spectral = axes[1].contourf(
                k_x_p,
                k_y_p,
                peter_array,
                cmap=cm.Greens,
                alpha=1.0,
                extend='max'
            )

            axes[1].contourf(
                k_x,
                k_y,
                self.spectrum,
                cmap=cm.Purples,
                alpha=0.6
            )

        elif type == 'pcolormesh':
            peter_spectral = axes[1].pcolormesh(
                k_x_p,
                k_y_p,
                peter_array,
                cmap=cm.Greens,
                alpha=1.0,
            )

            axes[1].pcolormesh(
                k_x,
                k_y,
                self.spectrum,
                cmap=cm.Purples,
                alpha=0.6
            )

        fig.colorbar(peter_spectral)

        # Graph format & style
        min, max = k_x[0, 0], k_x[-1, -1]
        axes_labels = ["$-\\pi$", "$0$", "$\\pi$"]

        # Axes and ticks
        axes[0].set_ylabel("$k_y$")
        for idx in range(2):
            axes[idx].set_xlabel("$k_x$")
            axes[idx].set_xticks(ticks=[min, 0, max], labels=axes_labels)
            axes[idx].set_yticks(ticks=[min, 0, max], labels=axes_labels)

        # Saving figure's plot
        if save:
            plt.savefig(f"{self.model_path}/figs/{self.file_name}.pdf")
        else:
            pass

        plt.show()

        return


def get_hall_coeff(spectral_weight: np.ndarray, hoppings: tuple[float],
                   x_coord=None, file="./nqft/Data/hall.txt") -> float:
    """Computes Hall coefficient for given parameters and writes it to a file
    using specified x coordinate.

    Parameters
    ----------
    spectral_weight: np.ndarray, shape=(N, M), default=None
        Spectral function used to compute Hall coefficient.

    hoppings: tuple[float], size=3, default=None
        Hopping amplitudes corresponding to spectral function.

    x_coord: float/int, default=None
        X coordinate to use if writting Hall coefficient to a file.
        (Makes plotting easier and faster)

    file: str, default="./nqft/Data/hall.txt"
        Path to file in which write Hall coefficient and given x coord.

    Returns
    -------
    n_h: float
        Hall coefficient as a float.
    """
    # Model's energy derivatives
    t, tp, tpp = hoppings
    normalize = 1 / spectral_weight.shape[0]**2
    momentum = np.linspace(-pi, pi, spectral_weight.shape[0])
    k_x, k_y = np.meshgrid(momentum, momentum)

    # Ex derivatives
    dE_dx = 2 * (t * sin(k_x) +
                 tp * (sin(k_x - k_y) + sin(k_x + k_y)) +
                 2 * tpp * sin(2 * k_x))

    ddE_dxx = 2 * (t * cos(k_x) +
                   tp * (cos(k_x - k_y) + cos(k_x + k_y)) +
                   4 * tpp * cos(2 * k_x))

    # Ey derivatives
    dE_dy = 2 * (t * sin(k_y) +
                 tp * (sin(k_x + k_y) - sin(k_x - k_y)) +
                 2 * tpp * sin(2 * k_y))

    ddE_dyy = 2 * (t * cos(k_y) +
                   tp * (cos(k_x + k_y) + cos(k_x - k_y)) +
                   4 * tpp * cos(2 * k_y))

    # Mixed derivative
    ddE_dxdy = 2 * tp * (cos(k_x + k_y) - cos(k_x - k_y))

    # Conductivities (ii)
    sigma_xx = -(dE_dx**2 * spectral_weight**2).sum()
    sigma_yy = -(dE_dy**2 * spectral_weight**2).sum()

    # Conductivity (ij)
    xy_1 = -2 * dE_dx * dE_dy * ddE_dxdy
    xy_2 = (dE_dx**2 * ddE_dyy) + (dE_dy**2 * ddE_dxx)
    sigma_xy = -((xy_1 + xy_2) * spectral_weight**3).sum()

    # Hall coefficient
    n_h = 6 * normalize * sigma_xx * sigma_yy / sigma_xy

    if file and x_coord:
        with open(file, "a") as file:
            file.write(f'{x_coord} {n_h}\n')
            file.close()
    else:
        print(f"User must give 'x coordinate' to write data in: {file}")

    return n_h


if __name__ == "__main__":
    u = 2.0
    mu = float(sys.argv[1])
    fill = int(sys.argv[2])
    hops = (1.0, -0.3, 0.2)

    lattice = QcmModel(
        shape=(2, 2),
        filling=fill,
        interaction=u,
        hoppings=hops,
        broadening=0.1,
        w=0.0,
        mu=mu,
        resolution=400,
        tiling_shift=True,
        show_spectrum=False,
        overwrite=True
    )

    # Plots
    # lattice.plot_spectrums(peter_key="N24", type='pcolormesh')

    # Averages
    # cluster_avgs = lattice.get_cluster_averages(operators=['mu'])
    # d_clus = 1.0 - cluster_avgs['mu'][0]

    # lattice_avgs = lattice.get_lattice_averages(operators=['mu'])
    # d_latt = 1.0 - lattice_avgs['mu']

    # Hall
    peter = read_fermi_arc()["N24"]
    n_h = get_hall_coeff(
        peter,
        hops,
        # x_coord=d_latt,
        # file=f"./nqft/Data/hall_3x4/n_h_3x4_n{fill}_U2_eta01.txt"
    )
