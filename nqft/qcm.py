"""This module is dedicated to 'pyqcm'
experimentation.
"""

import os
import numpy as np
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
from pyqcm.draw_operator import draw_operator
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
    """
    matrix = build_matrix(shape)
    elem_nb = shape[0] * shape[1]
    density = e_nbr / elem_nb

    # Make model directory if it doesn't exist
    model_path = f'./nqft/Data/model_{shape[0]}x{shape[1]}'
    try:
        os.makedirs(model_path)
    except FileExistsError:
        print(f'Storing model inside dir: {model_path}/')

    # Module related paths
    file = f'model_{shape[0]}x{shape[1]}_n{e_nbr}'
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

        lattice_model(name=f"2D_{elem_nb}_sites_{e_nbr}_fermions",
                      superlattice=super_vecs)

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


def run_model(model_path: str, densities: tuple[int], eta: float,
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
    dos_file = f'{model_path}/dos_n{densities[0]}.tsv'
    spectrum_file = f'{model_path}/spectrum_n{densities[0]}'

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
                  U: float, eta: float, peters: str) -> None:
    """Docs
    """
    # Init matplotlib figure
    fig, axes = plt.subplots(ncols=2, tight_layout=True, figsize=(9, 5))

    # Momentum space grids
    momentums = np.linspace(-np.pi, np.pi, 200)
    k_x, k_y = np.meshgrid(momentums, momentums)

    # Get spectral functions arrays
    peter_array = read_fermi_arc()[peters]
    file = f'model_{shape[0]}x{shape[1]}'
    path_to_file = f'./nqft/Data/{file}/spectrum_n{electrons}.npy'
    spectral = np.load(path_to_file)

    # Fig title
    title = "{}/{} fill, U={}, t=[{}, {}, {}], $\eta$={}".format(
        electrons,
        shape[0]*shape[1],
        U,
        *hops,
        eta
    )
    axes[0].set_title(title)

    # Plot spectral weight
    low_interaction = axes[0].pcolormesh(
        k_x,
        k_y,
        spectral,
        cmap=cm.Blues
    )
    fig.colorbar(low_interaction)

    # Fig title
    title_peter = f"{peters[1:]}/36 fill, U=8.0, t=[1, -0.3, 0.2], $\eta=0.1$"

    axes[1].set_title(title_peter)

    # Plot one of Peter's spectral weight
    axes[1].pcolormesh(
        k_x,
        k_y,
        spectral,
        cmap=cm.Blues
    )
    peter_spectral = axes[1].pcolormesh(
        k_x,
        k_y,
        peter_array,
        cmap=cm.Oranges,
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
    dim = f"{shape[0]}x{shape[1]}"
    plt.savefig(f"./nqft/Data/model_{dim}/figs/{dim}_{electrons}n_{U}_U.png")
    plt.show()

    return


if __name__ == "__main__":
    eta, model_shape, electrons = 0.1, (3, 4), 12
    U, hoppings = 1.0, [1.0, -0.3, 0.2]

    # Build model frame
    density, model_path = setup_model(
        shape=model_shape,
        e_nbr=electrons,
        U=U,
        hops=hoppings,
        shift=True,
        overwrite=True
    )

    # Actual computing
    run_model(
        model_path=model_path,
        densities=(electrons, model_shape[0] * model_shape[1]),
        eta=eta,
        overwrite=True
    )

    # Compare low interaction to Peter's
    plot_spectrum(
        shape=model_shape,
        electrons=electrons,
        hops=hoppings,
        U=U,
        eta=eta,
        peters='N32')
