"""This module is dedicated to 'pyqcm'
experimentation.
"""

import os
import numpy as np
import importlib as iplib

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
    """Docs
    """
    # Model's attributes
    matrix = build_matrix(shape)
    elem_nb = shape[0] * shape[1]
    density = e_nbr / elem_nb

    # Make model directory
    model_path = f'./nqft/Data/model_{shape[0]}x{shape[1]}'
    try:
        os.makedirs(model_path)
    except FileExistsError:
        print(f'Storing model inside dir: {model_path}/')

    # Module related paths
    file = f'model_{shape[0]}x{shape[1]}'
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
        draw_operator('t')

        hopping_operator(name="tp", link=[1, 1, 0], amplitude=-1)
        hopping_operator(name="tp", link=[-1, 1, 0], amplitude=-1)
        draw_operator('tp')

        hopping_operator(name="tpp", link=[2, 0, 0], amplitude=-1)
        hopping_operator(name="tpp", link=[0, 2, 0], amplitude=-1)
        draw_operator('tpp')

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

        try:
            # Delete old data files
            os.remove(f'{model_path}/dos.tsv')
            os.remove(f'{model_path}/spectrum.npy')
        except FileNotFoundError:
            pass
    else:
        try:
            # Import module
            iplib.import_module(name=f'{module_format}.{file}')
        except Exception as e:
            print(e)

    return density, model_path


def run_model(model_path: str, density: float, eta: float) -> None:
    """Docs
    """
    # Initializing data files path
    dos_file = f'{model_path}/dos.tsv'
    spectrum_file = f'{model_path}/spectrum'

    if not os.path.isfile(dos_file):
        # Finding chemical potential using DoS (density of states)
        freqs, cumul_density = DoS(w=30.0, eta=eta, data_file=dos_file)
    else:
        data = np.loadtxt(dos_file, skiprows=1, dtype=np.complex64)
        freqs, cumul_density = data[:, 0], data[:, 3]

    # Interpolate to find wanted frequency associated with density
    w_re = np.interp(1/2 * density, np.real(cumul_density), np.real(freqs))
    w_im = np.interp(1/2 * density, np.real(cumul_density), np.imag(freqs))

    # w_re = np.interp(1/2 * density, np.real(cumul_density), np.real(freqs))

    # Computing spectral weight at given frequency
    mdc(freq=w_re + w_im*1j, eta=0.12, sym="RXY", data_file=spectrum_file)
    # mdc(freq=w_re, eta=eta, sym="RXY", data_file=spectrum_file)

    return


if __name__ == "__main__":
    model_shape, electrons = (3, 4), 10
    U, hoppings = 8.0, [1.0, -0.3, 0.2]

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
    eta = 0.12
    run_model(model_path=model_path, density=density, eta=eta)
