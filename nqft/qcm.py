"""This module is dedicated to 'pyqcm'
experimentation.
"""

import os
import sys
import numpy as np
from rich import print
import importlib as iplib
from matplotlib import cm
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


def setup_model(shape: tuple[int], e_nbr: int, U: float, hops: list[float],
                shift=False, overwrite=False) -> str:
    """Building lattice model using 'pyqcm' functions for given input
    parameters.

    Parameters
    ----------
    shape: tuple[int], size=2, default=None
        Shape of source cluster.
    e_nbr: int, default=None
        Number of fermions in the sites system.
    U: float, Default=None
        Interaction operator amplitude.
    hops: list[float], size=3, default=None
        Hopping amplitudes (t, t', t'').
    shift: bool, default=False
        Determines if clusters are shifted when building lattice.
    overwrite: bool, default=False
        Determines if overwritting model file in model directory.

    Returns
    -------
    density: float
        Fermionic density of the system (nbr of fermions / nbr of sites).
    model_path: str
        Model directory in which store related data files (spectrum, freqs)

    Examples
    --------
    >>> setup_model(shape=(3, 4), e_nbr=10, U=1.0, hops=[1, -0.3, 0.2])
    >>> Number of OpenMP threads = 8
    Storing model inside dir: ./nqft/Data/model_3x4/
    (0.8333333333333334, './nqft/Data/model_3x4')
    """
    matrix = build_matrix(shape)
    elem_nb = shape[0] * shape[1]
    density = e_nbr / elem_nb

    # Make model directory if it doesn't exist
    U_f_to_str = str(U).split('.')
    U_str = "".join(U_f_to_str if U_f_to_str[-1] != '0' else U_f_to_str[:-1])
    model_path = f'./nqft/Data/model_{shape[0]}x{shape[1]}'
    try:
        os.makedirs(model_path)
    except FileExistsError:
        print(f'Storing model inside dir: {model_path}/')

    # Module related paths
    file = f'model_{shape[0]}x{shape[1]}_n{e_nbr}_U{U_str}'
    module_format = '.'.join(model_path.split('/')[1:])

    if overwrite:
        # Building cluster
        new_cluster_model(name="clus", n_sites=elem_nb, n_bath=0)
        add_cluster(name="clus", pos=[0, 0, 0], sites=matrix)

        # Initialiazing lattice using built cluster
        if shift:
            super_vecs = [[shape[1], 0, 0], [1, shape[0], 0]]
        else:
            super_vecs = [[shape[1], 0, 0], [0, shape[0], 0]]

        lattice_model(name=file, superlattice=super_vecs)

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
        sectors(R=0, N=e_nbr, S=0)

        # Setting operators parameters
        t, tp, tpp = hops
        set_parameters(
            f"""
            U = {U}
            t = {t}
            tp = {tp}
            tpp = {tpp}
            """
        )

        # Instancing lattice model
        model = new_model_instance(record=True)
        model.print(filename=f'{model_path}/{file}.py')
    else:
        try:
            # Import module if not overwriting model
            iplib.import_module(name=f'{module_format}.{file}')
        except ModuleNotFoundError as e:
            print(e)

    return density, model_path


def run_model(model_path: str, densities: tuple[int], U: float, eta: float,
              overwrite=False) -> None:
    """Calls and computed model's quantities such as the density of states
    and the spectral weight in the Brillouin zone.

    Parameters
    ----------
    model_path: str, default=None
        Model directory in which store related data files (spectrum, freqs)
    density: float, default=None
        Fermionic density of the system (nbr of fermions / nbr of sites).
    eta: float, default=None
        Lorentzian broadening of the system.

    Returns
    -------
    None
    """
    density = densities[0] / densities[1]

    # Initializing data files path
    U_f_to_str = str(U).split('.')
    U_str = "".join(U_f_to_str if U_f_to_str[-1] != '0' else U_f_to_str[:-1])
    dos_file = f'{model_path}/dos/dos_n{densities[0]}_U{U_str}.tsv'
    spectrum_file = f'{model_path}/spectrums/spectrum_n{densities[0]}_U{U_str}'

    if overwrite or not os.path.isfile(dos_file):
        # Finding chemical potential using DoS (density of states)
        freqs, cumul_density = DoS(w=30.0, eta=eta, data_file=dos_file)
    else:
        data = np.loadtxt(dos_file, skiprows=1, dtype=np.complex64)
        freqs, cumul_density = data[:, 0], data[:, 3]

    # Interpolate to find wanted frequency associated with density
    # w_re = np.interp(1/2 * density, np.real(cumul_density), np.real(freqs))
    # w_im = np.interp(1/2 * density, np.real(cumul_density), np.imag(freqs))

    w_re = np.interp(1/2 * density, np.real(cumul_density), np.real(freqs))

    # Computing spectral weight at given frequency
    # mdc(freq=w_re + w_im*1j, eta=0.12, sym="RXY", data_file=spectrum_file)
    mdc(freq=w_re, eta=eta, sym="RXY", data_file=spectrum_file)

    return


def plot_spectrum(shape: tuple[int], electrons: int, hops: list[float],
                  U: float, eta: float, peters: str, save=False) -> None:
    """Docs
    """
    # Init matplotlib figure
    U_f_to_str = str(U).split('.')
    U_str = "".join(U_f_to_str if U_f_to_str[-1] != '0' else U_f_to_str[:-1])
    fig, axes = plt.subplots(ncols=2, tight_layout=True, figsize=(9, 5))

    # Momentum space grids
    momentums = np.linspace(-np.pi, np.pi, 200)
    k_x, k_y = np.meshgrid(momentums, momentums)

    # Get spectral functions paths
    peter_array = read_fermi_arc()[peters]
    model = f'model_{shape[0]}x{shape[1]}'
    file = f'spectrums/spectrum_n{electrons}_U{U_str}.npy'

    # Load array
    path_to_file = f'./nqft/Data/{model}/{file}'
    spectral = np.load(path_to_file)

    # Fig title
    title = "{}/{} fill, $U=${}, $t=$[{}, {}, {}], $\eta=${}".format(
        electrons,
        shape[0]*shape[1],
        U,
        *hops,
        eta
    )
    axes[0].set_title(title)

    # Plot spectral weight
    low_interaction = axes[0].contourf(
        k_x,
        k_y,
        spectral,
        cmap=cm.Purples
    )
    fig.colorbar(low_interaction)

    # Fig title
    title_peter = "{}/36 fill, $U=$8.0, $t=$[1, -0.3, 0.2], $\eta=$0.1".format(
        peters[1:]
    )

    axes[1].set_title(title_peter)

    # Plot one of Peter's spectral weight
    peter_spectral = axes[1].contourf(
        k_x,
        k_y,
        peter_array,
        cmap=cm.Greens,
        alpha=1.0
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
        dim = f"{shape[0]}x{shape[1]}"
        plt.savefig(
            f"./nqft/Data/model_{dim}/figs/{dim}_{electrons}n_U{U_str}.pdf")

    # plt.show()

    return


if __name__ == "__main__":
    # Model params
    shift = False
    eta, hoppings = 0.1, [1.0, -0.3, 0.2]
    shape, e_number, interaction = (2, 2), 4, 1.0

    # # Build model frame
    # density, model_path = setup_model(
    #     shape=shape,
    #     e_nbr=e_number,
    #     U=interaction,
    #     hops=hoppings,
    #     shift=shift,
    #     overwrite=True
    # )
    #
    # # Actual computing
    # run_model(
    #     model_path=model_path,
    #     densities=(e_number, shape[0] * shape[1]),
    #     U=interaction,
    #     eta=eta,
    #     overwrite=True
    # )

    # Compare low interaction to Peter's
    for i in [0.5, 1.0, 2.0, 8.0]:
        plot_spectrum(
            shape=shape,
            electrons=e_number,
            hops=hoppings,
            U=i,
            eta=eta,
            peters='N32',
            save=True
        )
