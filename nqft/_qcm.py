
"""This module is dedicated to 'pyqcm'
experimentation.
"""

import os
import numpy as np
from rich import print
import importlib as iplib
from matplotlib import cm
from numpy import sin, cos
from scipy.constants import pi
import matplotlib.pyplot as plt

from pyqcm import (
    new_cluster_model,
    add_cluster,
    lattice_model,
    interaction_operator,
    hopping_operator,
    sectors
)
from pyqcm.spectral import DoS, mdc
from pyqcm import new_model_instance, set_parameters

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
    >>> [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]
    """
    array, idty = [], np.identity(3, dtype=np.int16)
    for i in range(shape[0]):
        for j in range(shape[1]):
            elem = i * idty[1] + j * idty[0]
            array.append(elem.tolist())

    return array


class QcmModel:
    """Docs
    """

    def __init__(self, shape: tuple[int], filling: int, interaction: float,
                 hoppings: tuple[float], broadening: float, resolution: int,
                 tiling_shift: bool, overwrite=False) -> None:
        """Docs
        """
        # Cluster geometry related attributes
        self.shape = shape
        self.filling = filling
        self.norm = 1 / resolution ** 2
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
            print(f'Storing model inside dir: {self.model_path}/')

        # Density of states & spectrum file names
        self.dos_file = f'dos/dos_n{filling}_U{u_str}.tsv'
        self.spectrum_file = f'spectrums/spectrum_n{filling}_U{u_str}.npy'

        # Import model's module if it has already been computed
        if overwrite:
            # Building cluster
            new_cluster_model(
                name="clus",
                n_sites=shape[0] * shape[1],
                n_bath=0
            )
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
                """
            )

            # Instancing lattice model
            model = new_model_instance(record=True)
            model.print(filename=f'{self.model_path}/{self.file_name}.py')
            print(f"Module '{self.file_name}' saved in: {self.model_path}\n")

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
        self.t, self.tp, self.tpp = hoppings
        self.eta = broadening
        self.res = resolution

        return

    def get_spectrum(self, frequency=0.0, interp_density=None, show=True):
        """Docs
        """
        if interp_density:
            dos_path = f'{self.model_path}/{self.dos_file}'

            if os.path.isfile(dos_path):
                data = np.loadtxt(dos_path, skiprows=1, dtype=np.complex64)
                freqs, cumul_density = data[:, 0], data[:, 3]

            else:
                freqs, cumul_density = DoS(w=40.0, eta=self.eta,
                                           data_file=dos_path
                                           )

            freq = np.interp(interp_density, np.real(cumul_density),
                             np.real(freqs)
                             )
        else:
            freq = frequency

        spectrum = mdc(
            freq=freq,
            nk=self.res,
            eta=self.eta,
            sym='RXY',
            data_file=f'{self.model_path}/{self.spectrum_file}',
            show=show
        )

        return spectrum

    def plot_spectrums(self, peter_key: str, save=False):
        """Docs
        """

        fig, axes = plt.subplots(ncols=2, tight_layout=True, figsize=(10, 4))
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
        spectral = np.load(f'{self.model_path}/{self.spectrum_file}')

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
        low_interaction = axes[0].contourf(
            k_x,
            k_y,
            spectral,
            cmap=cm.Purples,
            extend='max'
        )
        fig.colorbar(low_interaction)

        # Fig title
        title_p = "{}/36 fill, $U=$8.0, $t=$[1, -0.3, 0.2], $\eta=$0.1".format(
            peter_key[1:]
        )

        axes[1].set_title(title_p)

        # Plot one of Peter's spectral weight
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
            spectral,
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

        # Show figure's plot
        if save:
            plt.savefig(f"{self.model_path}/figs/{self.file_name}.pdf")
        else:
            pass

        plt.show()

        return

    def get_hall_coeff(self, spectral_weight: np.ndarray):
        """Docs
        """
        # Computing model's energy derivatives
        momentum = np.linspace(-pi, pi, self.res)
        k_x, k_y = np.meshgrid(momentum, momentum)

        dEs = {
            'dE_dx': None,
            'ddE_dxx': None,
            'dE_dy': None,
            'ddE_dyy': None,
            'ddE_dxdy': None
        }

        # Ex derivatives
        dEs['dE_dx'] = 2 * self.t * sin(k_x) + \
            4 * self.tp * sin(k_x) * cos(k_y) + \
            4 * self.tpp * sin(2 * k_x)

        dEs['ddE_dxx'] = 2 * self.t * cos(k_x) + \
            4 * self.tp * cos(k_x) * cos(k_y) + \
            8 * self.tpp * cos(2 * k_x)

        # Ey derivatives
        dEs['dE_dy'] = 2 * self.t * sin(k_y) + \
            4 * self.tp * cos(k_x) * cos(k_y) + \
            4 * self.tpp * sin(2 * k_y)

        dEs['ddE_dyy'] = 2 * self.t * cos(k_y) + \
            4 * self.tp * cos(k_x) * cos(k_y) + \
            8 * self.tpp * cos(2 * k_y)

        # Mixed derivative
        dEs['ddE_dxdy'] = -4 * self.tp * sin(k_x) * sin(k_y)

        # Computing conductivities (ii)
        sigma_xx = (dEs['dE_dx']**2 * spectral_weight**2).sum()
        sigma_yy = (dEs['dE_dy']**2 * spectral_weight**2).sum()

        # Computing conductivity (ij)
        coeff = 1 / 3
        xy_1 = -2 * dEs['dE_dx'] * dEs['dE_dy'] * dEs['ddE_dxdy']
        xy_2 = dEs['dE_dx']**2 * dEs['ddE_dyy'] + \
            dEs['dE_dy']**2 * dEs['ddE_dxx']
        sigma_xy = coeff * ((xy_1 + xy_2) * spectral_weight**3).sum()

        # Computing Hall coefficient
        n_h = self.norm * sigma_xx * sigma_yy / sigma_xy

        return n_h


if __name__ == "__main__":
    lattice = QcmModel(
        shape=(4, 4),
        filling=16,
        interaction=0.0,
        hoppings=(-1.0, 0.3, -0.2),
        broadening=0.05,
        resolution=200,
        tiling_shift=False,
        overwrite=True
    )

    A = lattice.get_spectrum()
    n_h = lattice.get_hall_coeff(spectral_weight=A)

    print(n_h)
